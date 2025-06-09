# Push prototypes to the nearest training image patches.

import torch
import numpy as np
import os
import copy
import time
from dataclasses import dataclass
import cv2
import matplotlib.pyplot as plt

from src.utils.receptive_field import compute_rf_prototype
from src.utils.helpers import makedir, find_high_activation_crop


# push each prototype to the nearest patch in the training set
def push_prototypes(
    dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
    prototype_network_parallel,  # pytorch network with prototype_vectors
    preprocess_input_function=None,  # normalize if needed
    log=print,
):
    prototype_network_parallel.eval()
    log("\tpush")

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]]
    )
    ph, pw = dataloader.dataset.patch_size
    global_proto_regions = np.zeros(
        [n_prototypes, ph, pw, 24]
    )  # stores the receptive field of each prototype
    global_index_location = np.zeros(n_prototypes, dtype=int)  # original index of the patch in our dataset
    global_proto_bounds = np.zeros(
        [n_prototypes, 4], dtype=int
    )  # stores the bounding box of the prototype region w.r.t. the patch

    for push_iter, (search_batch_input, seg_mask, original_index_batch) in enumerate(dataloader):
        update_prototypes_on_batch(
            search_batch_input,
            original_index_batch,
            prototype_network_parallel,
            global_min_proto_dist,
            global_min_fmap_patches,
            global_proto_regions,
            global_index_location,
            global_proto_bounds,
            seg_mask=seg_mask,
            preprocess_input_function=preprocess_input_function,
        )

    log("\tExecuting push ...")
    # Log distances between each prototype and its best matching feature map patch
    os.makedirs("prototypes", exist_ok=True)
    if os.path.exists("prototypes"):
        for file in os.listdir("prototypes"):
            if file.endswith(".npz"):
                os.remove(os.path.join("prototypes", file))
    for proto_idx, (min_dist, proto_region) in enumerate(zip(global_min_proto_dist, global_proto_regions)):
        log(f"\tPrototype {proto_idx}: min distance = {min_dist:.4f}")
        # save proto region as npz file
        np.savez(
            os.path.join("prototypes", f"prototype_{proto_idx:02d}.npz"),
            region=proto_region,
        )
    with open("prototypes/prototype_bounds.txt", "w") as f:
        for proto_idx, bounds in enumerate(global_proto_bounds):
            f.write(f"Prototype {proto_idx}: index={global_index_location[proto_idx]}, bounds={bounds.tolist()}\n")
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32).cuda()
    )
    end = time.time()
    log("\tpush time: \t{0}".format(end - start))


# update each prototype for current search batch
def update_prototypes_on_batch(
    search_batch_input,
    original_index_batch,
    prototype_network_parallel,
    global_min_proto_dist,  # this will be updated
    global_min_fmap_patches,  # this will be updated
    global_proto_regions,  # this will be updated
    global_index_location, # this will be updated
    global_proto_bounds, # this will be updated
    seg_mask=None,
    preprocess_input_function=None,
):
    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        protoL_input_torch, proto_dist_torch = (
            prototype_network_parallel.module.push_forward(search_batch)
        )

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    # infer spatial stride between prototype-distance grid and the feature‑map
    feature_map_h, feature_map_w = protoL_input_.shape[2], protoL_input_.shape[3]
    dist_h, dist_w = proto_dist_.shape[2], proto_dist_.shape[3]
    stride_h = feature_map_h // dist_h
    stride_w = feature_map_w // dist_w

    mask_down = torch.nn.functional.interpolate(
        seg_mask,
        size=(proto_dist_.shape[2], proto_dist_.shape[3]),
        mode="nearest",
    ).long()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    for j in range(n_prototypes):
        target_class = torch.argmax(
            prototype_network_parallel.module.prototype_class_identity[j]
        ).item()
        invalid = (mask_down != target_class).cpu().numpy()
        proto_dist_j = np.where(
            invalid,
            np.inf,
            proto_dist_[:, j, :, :],
        )

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(
                np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)
            )

            img_index_in_batch = batch_argmin_proto_dist_j[0]
            # original index of the image in our dataset
            original_index = original_index_batch[img_index_in_batch]
            global_index_location[j] = original_index
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * stride_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * stride_w

            # clamp indices so we never step outside the feature‑map
            fmap_height_start_index = min(
                fmap_height_start_index, feature_map_h - proto_h
            )
            fmap_width_start_index = min(
                fmap_width_start_index, feature_map_w - proto_w
            )

            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[
                img_index_in_batch,
                :,
                fmap_height_start_index:fmap_height_end_index,
                fmap_width_start_index:fmap_width_end_index,
            ]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # clamp spatial indices so they never exceed the proto‑layer grid
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            h_idx_clamped = min(batch_argmin_proto_dist_j[1], protoL_rf_info[0] - 1)
            w_idx_clamped = min(batch_argmin_proto_dist_j[2], protoL_rf_info[0] - 1)

            rf_prototype_j = compute_rf_prototype(
                search_batch.size(2),
                [batch_argmin_proto_dist_j[0], h_idx_clamped, w_idx_clamped],
                protoL_rf_info
            )

            # whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]

            # crop out the receptive field
            rf_img_j = original_img_j[
                rf_prototype_j[1] : rf_prototype_j[2],
                rf_prototype_j[3] : rf_prototype_j[4],
                :,
            ]
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            proto_act_img_j = np.log(
                (proto_dist_img_j + 1)
                / (proto_dist_img_j + prototype_network_parallel.module.epsilon)
            )
            upsampled_act_img_j = cv2.resize(
                proto_act_img_j,
                dsize=(original_img_size, original_img_size),
                interpolation=cv2.INTER_CUBIC,
            )
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            global_proto_bounds[j] = proto_bound_j
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[
                proto_bound_j[0] : proto_bound_j[1],
                proto_bound_j[2] : proto_bound_j[3],
                :,
            ]
            # copy into top left corner
            global_proto_regions[j].fill(0)
            global_proto_regions[j, :proto_img_j.shape[0], :proto_img_j.shape[1], :] = proto_img_j