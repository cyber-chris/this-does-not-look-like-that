# Push prototypes to the nearest training image patches.

import torch
import numpy as np
import os
import copy
import time

from src.utils.receptive_field import compute_rf_prototype
from src.utils.helpers import makedir, find_high_activation_crop


# push each prototype to the nearest patch in the training set
def push_prototypes(
    dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
    prototype_network_parallel,  # pytorch network with prototype_vectors
    class_specific=True,
    preprocess_input_function=None,  # normalize if needed
    prototype_layer_stride=1,
    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved here
    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    proto_bound_boxes_filename_prefix=None,
    save_prototype_class_identity=True,  # which class the prototype image comes from
    log=print,
    prototype_activation_function_in_numpy=None,
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

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, seg_mask) in enumerate(dataloader):
        """
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        """
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(
            search_batch_input,
            start_index_of_search_batch,
            prototype_network_parallel,
            global_min_proto_dist,
            global_min_fmap_patches,
            None,
            None,
            class_specific=class_specific,
            seg_mask=seg_mask,
            num_classes=num_classes,
            preprocess_input_function=preprocess_input_function,
        )

    log("\tExecuting push ...")
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32).cuda()
    )
    # prototype_network_parallel.cuda()
    end = time.time()
    log("\tpush time: \t{0}".format(end - start))


# update each prototype for current search batch
def update_prototypes_on_batch(
    search_batch_input,
    start_index_of_search_batch,
    prototype_network_parallel,
    global_min_proto_dist,  # this will be updated
    global_min_fmap_patches,  # this will be updated
    proto_rf_boxes,  # this will be updated (unused)
    proto_bound_boxes,  # this will be updated (unused)
    class_specific=True,
    seg_mask=None,  # required if class_specific == True
    num_classes=None,  # required if class_specific == True
    preprocess_input_function=None,
    dir_for_saving_prototypes=None,  # removed
    prototype_img_filename_prefix=None,  # removed
    prototype_self_act_filename_prefix=None,  # removed
    prototype_activation_function_in_numpy=None,  # removed
):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    # infer spatial stride between prototype-distance grid and the feature‑map
    feature_map_h, feature_map_w = protoL_input_.shape[2], protoL_input_.shape[3]
    dist_h, dist_w = proto_dist_.shape[2], proto_dist_.shape[3]
    stride_h = feature_map_h // dist_h
    stride_w = feature_map_w // dist_w

    if class_specific:
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
        if class_specific:
            target_class = torch.argmax(
                prototype_network_parallel.module.prototype_class_identity[j]
            ).item()
            invalid = (mask_down != target_class).cpu().numpy()
            proto_dist_j = np.where(
                invalid,
                np.inf,
                proto_dist_[:, j, :, :],
            )
        else:
            proto_dist_j = proto_dist_[:, j, :, :]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(
                np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)
            )

            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * stride_h
            fmap_width_start_index  = batch_argmin_proto_dist_j[2] * stride_w

            # clamp indices so we never step outside the feature‑map
            fmap_height_start_index = min(fmap_height_start_index, feature_map_h - proto_h)
            fmap_width_start_index  = min(fmap_width_start_index,  feature_map_w - proto_w)

            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_end_index  = fmap_width_start_index  + proto_w

            batch_min_fmap_patch_j = protoL_input_[
                img_index_in_batch,
                :,
                fmap_height_start_index:fmap_height_end_index,
                fmap_width_start_index:fmap_width_end_index,
            ]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
