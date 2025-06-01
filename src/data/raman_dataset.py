import logging
import os
from typing import Any
import hashlib

from PIL import Image
import numpy as np
import zarr
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import squidpy as sq
except ImportError as e:
    print(
        "WARNING: SquidPy package not installed. You won't be able to use segmentation masks in InMemoryStainData."
    )

try:
    import cv2
except ImportError as e:
    print(
        "WARNING: OpenCV (cv2) package not installed. You won't be able to use InMemoryStainData."
    )


def quantile_norm(array, q):
    array = array - np.quantile(array, q)
    array = array / np.quantile(array, 1 - q)
    # In case division by zero for very skewed distributions
    if isinstance(array, torch.Tensor):
        array[torch.isnan(array)] = 0
    elif isinstance(array, np.ndarray):
        array[np.isnan(array)] = 0
    array[array < 0] = 0
    array[array > 1] = 1
    return array


def transform_image_to_fixed_clusters(image, cluster_centroids):
    h, w, c = image.shape
    pixels = image.reshape(-1, 3)

    # Compute the distance from each pixel to each predefined cluster
    distances = np.linalg.norm(pixels[:, None] - cluster_centroids[None, :], axis=2)
    labels = np.argmin(distances, axis=1)  # Find the nearest cluster for each pixel

    # Map each label back to the centroid for visualization
    clustered_image = cluster_centroids[labels].reshape(h, w, 3)

    return labels.reshape(h, w), clustered_image


# To make it easier to mimic wandb config dicts when necessary
# Example: AttrDict({'a': 1, 'hhh':4}).hhh
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class InMemoryRamanMaskData(Dataset):
    def _patchify(self, big_img: torch.Tensor) -> torch.Tensor:
        # h, w, c = big_img.shape
        ph, pw = self.patch_size
        sh, sw = self.stride_size
        tmp_img = big_img.unfold(0, ph, sh).unfold(1, pw, sw)  # N1, N2, C, H, W
        tmp_img = tmp_img.reshape(
            tmp_img.shape[0] * tmp_img.shape[1], tmp_img.shape[2], ph, pw
        )
        return tmp_img

    def __init__(
        self,
        ids: list[str],
        raman_paths: list[dict[str, str]],
        mask_paths: list[str],
        conf_train: dict[str, Any],
        conf_data: dict[str, Any],
        transforms=None,
        wandb_conf=None,
    ):
        logging.info(
            f"InMemoryRamanMaskData: conf_data={conf_data}, conf_train={conf_train}, wandb_conf={wandb_conf}"
        )

        self.raman_imgs = []
        self.mask_imgs = []
        self.patch_size = conf_train["patch_size"]
        self.stride_size = conf_train["stride"]
        self.transforms = transforms

        for id_i, raman_path, mask_path in zip(ids, raman_paths, mask_paths):
            logging.debug(f"Processing {id_i}...")
            # Load Raman
            if conf_data["use_srs_norm"]:
                c_1, c_2 = 0, 21
            else:
                c_1, c_2 = conf_data["acdc_channels"]
            tmp_srs_norm = np.load(raman_path["srs_norm"])
            tmp_srs = np.load(raman_path["acdc_orig"])

            assert mask_path is not None
            tmp_mask = np.load(mask_path)
            tmp_mask = cv2.resize(
                tmp_mask, (tmp_srs.shape[1], tmp_srs.shape[0]), cv2.INTER_NEAREST
            )
            tmp_mask = torch.tensor(
                tmp_mask.reshape(tmp_mask.shape[0], tmp_mask.shape[1], 1),
                dtype=torch.float32,
            )
            mask_patches = self._patchify(tmp_mask)
            # convert to binary classification: map classes 1 and 4 to 1, all others to 0
            mask_patches[mask_patches == 1] = 1
            mask_patches[mask_patches == 4] = 1
            mask_patches[mask_patches != 1] = 0

            # Filter out background patches (using just 3 channels to judge)
            srs_patches = self._patchify(
                torch.tensor(tmp_srs_norm[..., 0:3], dtype=torch.float32)
            )
            all_zero = torch.all(srs_patches == 0, dim=(1, 2, 3))
            ids_to_include = torch.where(~all_zero)[0]
            self.mask_imgs.append(mask_patches[ids_to_include, ...])

            # Processing Raman
            # DC
            tmp_dc = np.load(raman_path["dc_orig"])
            if not conf_data["use_srs_norm"]:
                filt_ini = np.where(tmp_dc[..., c_1:c_2] < 0)
                tmp_srs[..., c_1:c_2][filt_ini] = np.median(tmp_srs[..., c_1:c_2])
            else:
                tmp_srs = tmp_srs_norm
            tmp_dc = np.mean(tmp_dc, axis=2)
            tmp_dc = (tmp_dc - tmp_dc.min()) / (tmp_dc.max() - tmp_dc.min())
            tmp_dc = torch.tensor(tmp_dc[..., None], dtype=torch.float32)

            # tmp_srs = tmp_srs[..., [0, 4, 10, 19]] # when use_srs: lipids, paraffin, proteins, and extra channels
            merged_raman = [
                torch.tensor(tmp_srs[..., c_1:c_2], dtype=torch.float32),
                tmp_dc,
            ]
            # AUX
            if conf_data["use_srs_norm"]:
                tmp_aux = torch.tensor(tmp_srs_norm[..., -2:], dtype=torch.float32)
            else:
                tmp_aux = torch.tensor(
                    np.load(raman_path["aux_orig"]), dtype=torch.float32
                )
            merged_raman.append(tmp_aux)
            logging.debug(
                f"merged_raman.shapes - {merged_raman[0].shape}, {merged_raman[1].shape}, {merged_raman[2].shape}"
            )
            if wandb_conf is None:
                # Using Preprocessed SRS - no need to change it
                merged_raman = torch.concat(merged_raman, dim=2)
                raman_imgs_tmp = self._patchify(merged_raman)[ids_to_include, ...]
                logging.debug(f"raman_imgs_tmp.shape={raman_imgs_tmp.shape}")
                self.raman_imgs.append(raman_imgs_tmp)  # [8, ...][None, ...])
            else:
                # Using raw ACDC - preprocessing it
                merged_raman = torch.concat(merged_raman, dim=2)
                raman_imgs_tmp = self._patchify(merged_raman)[ids_to_include, ...]

                if wandb_conf.remove_negatives:
                    logging.debug("Removing negative values...")
                    raman_imgs_tmp[raman_imgs_tmp < 0] = 0

                if wandb_conf.preprocess_norm == "perpixel_minmax":
                    logging.debug("PREPROCESS: perpixel_minmax...")
                    # raman_imgs_tmp: BS x Raman+DC+AUX x 256 x 256
                    sliced_raman = raman_imgs_tmp[:, : (c_2 - c_1), ...]
                    min_val = sliced_raman.min(dim=1).values[:, None, ...]
                    max_val = sliced_raman.max(dim=1).values[:, None, ...]
                    raman_imgs_tmp[:, : (c_2 - c_1), ...] = (sliced_raman - min_val) / (
                        max_val - min_val
                    )

                    # For AUX, just having a simple minmax with 1% quantile
                    for idx in range(raman_imgs_tmp.shape[0]):
                        raman_imgs_tmp[idx, -2, ...] = quantile_norm(
                            raman_imgs_tmp[idx, -2, ...], 0.001
                        )
                        raman_imgs_tmp[idx, -1:, ...] = quantile_norm(
                            raman_imgs_tmp[idx, -1, ...], 0.001
                        )
                else:
                    if wandb_conf.preprocess_norm == "perpatch_minmax_01":
                        logging.debug("PREPROCESS: perpatch_minmax_01...")
                    elif wandb_conf.preprocess_norm == "perpatch_channel_minmax_01":
                        logging.debug("PREPROCESS: perpatch_channel_minmax_01...")
                    else:
                        logging.error("PREPROCESS: something weird went wrong!!")

                    for idx in range(raman_imgs_tmp.shape[0]):
                        sliced_raman = raman_imgs_tmp[idx, : (c_2 - c_1), ...]
                        if wandb_conf.preprocess_norm == "perpatch_minmax_01":
                            raman_imgs_tmp[idx, : (c_2 - c_1), ...] = quantile_norm(
                                sliced_raman, 0.01
                            )
                        elif wandb_conf.preprocess_norm == "perpatch_channel_minmax_01":
                            sliced_raman = sliced_raman.reshape(
                                sliced_raman.shape[0], -1
                            )
                            sliced_raman = (
                                sliced_raman
                                - np.quantile(sliced_raman, q=0.01, axis=1)[:, None]
                            )
                            sliced_raman = (
                                sliced_raman
                                / np.quantile(sliced_raman, q=0.99, axis=1)[:, None]
                            )
                            # When there were divisions by zero - it could happen when only border of sample in patch
                            #  Eg for E12198-23_sample1_half1, patch 12
                            sliced_raman[torch.isnan(sliced_raman)] = 0
                            sliced_raman[sliced_raman < 0] = 0
                            sliced_raman[sliced_raman > 1] = 1
                            raman_imgs_tmp[idx, : (c_2 - c_1), ...] = (
                                sliced_raman.reshape(
                                    sliced_raman.shape[0],
                                    self.patch_size[0],
                                    self.patch_size[1],
                                )
                            )
                        # For AUX, just having a simple minmax with 1% quantile
                        raman_imgs_tmp[idx, -2:, ...] = quantile_norm(
                            raman_imgs_tmp[idx, -2:, ...], 0.01
                        )

                if torch.isnan(raman_imgs_tmp).any():
                    # weird assert failures in cuda, so moving torch.where calculations to cpu
                    where_nan = torch.where(torch.isnan(raman_imgs_tmp.cpu()))
                    logging.debug(f"raman_imgs_tmp had {len(where_nan[0])} NaNs!")
                    logging.debug(where_nan)
                    raman_imgs_tmp[torch.isnan(raman_imgs_tmp)] = 0
                logging.debug(f"raman_imgs_tmp.shape={raman_imgs_tmp.shape}")
                self.raman_imgs.append(raman_imgs_tmp)

        # Putting it all together
        self.raman_imgs = torch.concat(self.raman_imgs)
        self.mask_imgs = torch.concat(self.mask_imgs)
        logging.info(f"Final Raman shape: {self.raman_imgs.shape}")

        if self.raman_imgs[:, 0, ...].shape != self.mask_imgs[:, 0, ...].shape:
            raise ValueError(
                "Raman and binary mask must have the same initial shape. "
                f"Instead, they have {self.raman_imgs.shape} and {self.mask_imgs.shape} respectively"
            )

    def __len__(self):
        return self.raman_imgs.shape[0]

    def __getitem__(self, idx):
        if self.transforms:
            raman_i, mask_i = self.raman_imgs[idx], self.mask_imgs[idx]
            # Concatenating them together such that same transformation is done to both Raman and Mask
            mask_and_raman = torch.cat([mask_i, raman_i], dim=0)

            out = self.transforms(mask_and_raman)
            new_mask = out[:1, ...]
            new_raman = out[1:, ...]
            return new_raman, new_mask
        else:
            return self.raman_imgs[idx], self.mask_imgs[idx]


class ZarrRamanMaskData(Dataset):
    def _patchify(self, big_img: torch.Tensor) -> torch.Tensor:
        # h, w, c = big_img.shape
        ph, pw = self.patch_size
        sh, sw = self.stride_size
        tmp_img = big_img.unfold(0, ph, sh).unfold(1, pw, sw)  # N1, N2, C, H, W
        tmp_img = tmp_img.reshape(
            tmp_img.shape[0] * tmp_img.shape[1], tmp_img.shape[2], ph, pw
        )
        return tmp_img

    def __init__(
        self,
        ids: list[str],
        raman_paths: list[dict[str, str]],
        mask_paths: list[str],
        conf_train: dict[str, Any],
        conf_data: dict[str, Any],
        transforms=None,
        wandb_conf=None,
        raman_zarr_path: str = "/tmp/raman_data.zarr",
        mask_zarr_path: str = "/tmp/mask_data.zarr",
        hash_path: str = "/tmp/zarr_hash",
    ):
        logging.info(
            f"ZarrRamanMaskData: conf_data={conf_data}, conf_train={conf_train}, wandb_conf={wandb_conf}"
        )

        self.patch_size = conf_train["patch_size"]
        self.stride_size = conf_train["stride"]
        self.transforms = transforms
        self.raman_zarr_path = raman_zarr_path
        self.mask_zarr_path = mask_zarr_path

        def stable_hash(ids) -> str:
            """Generate a stable hash for the list of ids."""
            m = hashlib.md5(usedforsecurity=False)
            for id_i in sorted(ids):
                m.update(id_i.encode("utf-8"))
            for id_i in sorted(conf_data["extra_filtering"]):
                m.update(id_i.encode("utf-8"))
            # also include important configuration parameters
            m.update(str(conf_data["acdc_channels"]).encode("utf-8"))
            m.update(str(conf_data["use_srs_norm"]).encode("utf-8"))
            m.update(str(conf_train["patch_size"]).encode("utf-8"))
            m.update(str(conf_train["stride"]).encode("utf-8"))
            return m.hexdigest()

        # Check if there exists a file at hash_path and check if the contents match
        expected_hash = stable_hash(ids)
        if os.path.exists(hash_path) and open(hash_path).read() == expected_hash:
            logging.info(
                "Zarr files already exist and match the expected hash. Skipping processing."
            )
            self.raman_zarr = zarr.open(self.raman_zarr_path, mode="r")
            self.mask_zarr = zarr.open(self.mask_zarr_path, mode="r")
            logging.info(
                f"Final Raman shape: {self.raman_zarr.shape}, Mask shape: {self.mask_zarr.shape}"
            )
            # Fast shape check without loading data slices: compare batch, height, and width dimensions
            r_shape = self.raman_zarr.shape
            m_shape = self.mask_zarr.shape
            if r_shape[0] != m_shape[0] or r_shape[2] != m_shape[2] or r_shape[3] != m_shape[3]:
                raise ValueError(
                    "Raman and binary mask must have the same shape (excluding channels). "
                    f"Instead, they have {r_shape} and {m_shape} respectively"
                )
            return
        else:
            self.raman_zarr = None
            self.mask_zarr = None

        for id_i, raman_path, mask_path in zip(ids, raman_paths, mask_paths):
            logging.debug(f"Processing {id_i}...")
            # Load Raman data
            if conf_data["use_srs_norm"]:
                c_1, c_2 = 0, 21
            else:
                c_1, c_2 = conf_data["acdc_channels"]
            tmp_srs_norm = np.load(raman_path["srs_norm"])
            tmp_srs = np.load(raman_path["acdc_orig"])

            # Load and process mask
            assert mask_path is not None
            tmp_mask = np.load(mask_path)
            tmp_mask = cv2.resize(
                tmp_mask, (tmp_srs.shape[1], tmp_srs.shape[0]), cv2.INTER_NEAREST
            )
            tmp_mask = torch.tensor(
                tmp_mask.reshape(tmp_mask.shape[0], tmp_mask.shape[1], 1),
                dtype=torch.float32,
            )
            mask_patches = self._patchify(tmp_mask)

            # Filter out background patches using srs_norm channels
            srs_patches = self._patchify(
                torch.tensor(tmp_srs_norm[..., 0:3], dtype=torch.float32)
            )
            all_zero = torch.all(srs_patches == 0, dim=(1, 2, 3))
            ids_to_include = torch.where(~all_zero)[0]

            if conf_data["extra_filtering"] and id_i in conf_data["extra_filtering"]:
                logging.info(f"Applying extra filtering for {id_i}...")
                # Only keep patches that have some non-zero values in the mask
                mask_has_nonzero = torch.any(mask_patches != 0, dim=(1, 2, 3))
                ids_to_include = ids_to_include[mask_has_nonzero[ids_to_include]]

            mask_patches = mask_patches[ids_to_include, ...]
            # Convert to binary: map classes 1 and 4 to 1, all others to 0
            mask_patches[mask_patches == 1] = 1
            mask_patches[mask_patches == 4] = 1
            mask_patches[mask_patches != 1] = 0

            # Process Raman data
            tmp_dc = np.load(raman_path["dc_orig"])
            if not conf_data["use_srs_norm"]:
                filt_ini = np.where(tmp_dc[..., c_1:c_2] < 0)
                tmp_srs[..., c_1:c_2][filt_ini] = np.median(tmp_srs[..., c_1:c_2])
            else:
                tmp_srs = tmp_srs_norm
            tmp_dc = np.mean(tmp_dc, axis=2)
            tmp_dc = (tmp_dc - tmp_dc.min()) / (tmp_dc.max() - tmp_dc.min())
            tmp_dc = torch.tensor(tmp_dc[..., None], dtype=torch.float32)

            # tmp_srs = tmp_srs[..., [0, 4, 10, 19]] # when use_srs: lipids, paraffin, proteins, and extra channels
            merged_raman = [
                torch.tensor(tmp_srs[..., c_1:c_2], dtype=torch.float32),
                tmp_dc,
            ]
            if conf_data["use_srs_norm"]:
                tmp_aux = torch.tensor(tmp_srs_norm[..., -2:], dtype=torch.float32)
            else:
                tmp_aux = torch.tensor(
                    np.load(raman_path["aux_orig"]), dtype=torch.float32
                )
            merged_raman.append(tmp_aux)

            logging.debug(
                f"merged_raman.shapes - {merged_raman[0].shape}, {merged_raman[1].shape}, {merged_raman[2].shape}"
            )
            merged_raman = torch.concat(merged_raman, dim=2)
            raman_patches = self._patchify(merged_raman)[ids_to_include, ...]

            # Convert patches to NumPy arrays.
            raman_np = (
                raman_patches.cpu().numpy()
                if torch.is_tensor(raman_patches)
                else raman_patches
            )
            mask_np = (
                mask_patches.cpu().numpy()
                if torch.is_tensor(mask_patches)
                else mask_patches
            )

            # Initialize the Zarr arrays if they have not been created yet.
            if self.raman_zarr is None:
                # Delete any hash at hash_path, to avoid accidentally using partially created files.
                if os.path.exists(hash_path):
                    os.remove(hash_path)

                # Determine shape from the first sample: (0, channels, patch_height, patch_width).
                channels = raman_np.shape[1]
                ph, pw = self.patch_size
                self.raman_zarr = zarr.open(
                    self.raman_zarr_path,
                    mode="w",
                    shape=(0, channels, ph, pw),
                    chunks=(1, channels, ph, pw),
                    dtype="float32",
                )
            if self.mask_zarr is None:
                channels = mask_np.shape[1]
                ph, pw = self.patch_size
                self.mask_zarr = zarr.open(
                    self.mask_zarr_path,
                    mode="w",
                    shape=(0, channels, ph, pw),
                    chunks=(1, channels, ph, pw),
                    dtype="float32",
                )

            # Append the current sample's patches directly to the Zarr arrays.
            self.raman_zarr.append(raman_np)
            self.mask_zarr.append(mask_np)

        # Store the hash at hash_path
        with open(hash_path, "w") as f:
            f.write(expected_hash)
        logging.info(
            f"Zarr files created at {self.raman_zarr_path} and {self.mask_zarr_path}"
        )

        logging.info(
            f"Final Raman shape: {self.raman_zarr.shape}, Mask shape: {self.mask_zarr.shape}"
        )
        if self.raman_zarr[:, 0, ...].shape != self.mask_zarr[:, 0, ...].shape:
            raise ValueError(
                "Raman and binary mask must have the same initial shape. "
                f"Instead, they have {self.raman_zarr.shape} and {self.mask_zarr.shape} respectively"
            )

    def __len__(self):
        return self.raman_zarr.shape[0]

    def __getitem__(self, idx):
        # Load the patch on demand from the Zarr array and convert to torch tensor.
        raman_sample = torch.from_numpy(self.raman_zarr[idx])
        mask_sample = torch.from_numpy(self.mask_zarr[idx])
        if self.transforms:
            # Concatenate the mask and raman data for joint transformations.
            combined = torch.cat([mask_sample, raman_sample], dim=0)
            out = self.transforms(combined)
            new_mask = out[:1, ...]
            new_raman = out[1:, ...]
            return new_raman, new_mask
        else:
            return raman_sample, mask_sample


def get_new_raman_path(raman_id: str) -> dict[str, str]:
    """
    Returns the new raman path, with the patient inferred from the raman_id.
    :param raman_id:
    :return:
    """
    patient = raman_id.split("_")[0]
    return_dict = {
        "srs_norm": os.path.join(
            "/home/ct678/code/charm_data", patient, f"{raman_id}_SRS_corrected.npy"
        ),
        "dc_orig": os.path.join(
            "/home/ct678/code/charm_data", patient, f"{raman_id}_dc_orig.npy"
        ),
        "acdc_orig": os.path.join(
            "/home/ct678/code/charm_data", patient, f"{raman_id}_acdc_orig.npy"
        ),
        "aux_orig": os.path.join(
            "/home/ct678/code/charm_data", patient, f"{raman_id}_aux_orig.npy"
        ),
    }

    return return_dict


def create_raman_mask_dataloaders_from_ids(
    ids, conf, transforms=None, shuffle=False, in_memory=False, is_train=True, be_fast=True,
):
    logging.info(f"Creating Raman/Mask data loader for IDs: {ids}")
    raman_paths = []
    mask_paths = []
    for idx_name in ids:
        raman_paths.append(get_new_raman_path(idx_name))
        # HACK: temporary way to do both preliminary and random mask training
        possible_mask_path = f"/home/ct678/aligned_masks/{idx_name}.npy"
        mask_paths.append(
            possible_mask_path if os.path.exists(possible_mask_path) else None
        )
    tmp_ds = (
        InMemoryRamanMaskData(
            ids,
            raman_paths,
            mask_paths,
            transforms=transforms,
            conf_train=conf["training"],
            conf_data=conf["data"],
        )
        if in_memory
        else ZarrRamanMaskData(
            ids,
            raman_paths,
            mask_paths,
            transforms=transforms,
            conf_train=conf["training"],
            conf_data=conf["data"],
            raman_zarr_path=f"/tmp/raman_data_{'train' if is_train else 'val'}.zarr",
            mask_zarr_path=f"/tmp/mask_data_{'train' if is_train else 'val'}.zarr",
            hash_path=f"/tmp/zarr_hash_{'train' if is_train else 'val'}.zarr",
        )
    )
    return DataLoader(
        tmp_ds,
        batch_size=conf["hyperparams"]["batch_size"],
        shuffle=shuffle,
        num_workers=16 if be_fast else 8,
        pin_memory=True,
        pin_memory_device="cuda" if torch.cuda.is_available() else "cpu",
        persistent_workers=be_fast,
        prefetch_factor=2 if be_fast else 1,
    )
