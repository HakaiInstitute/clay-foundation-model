r"""
We will create chips of size `224 x 224` to feed them to the model, feel
free to experiment with other chip sizes as well.
   Run the script as follows:
   python preprocess_data.py <data_dir> <output_dir> <chip_size> <chip_stride? (defaults to chip_size)>

   Example:
   python preprocess_data.py data/ps_8b/ data/ps_8b_tiled 224
"""  # noqa E501

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm.auto import tqdm

NUM_BANDS = 8
MAX_NODATA_PROPORTION = 0.0

LABELS = {
    "water": 0,
    "kelp": 1,
    "land": 2,
    "nodata": 3,
    "noise": 4,
}


class PlanetRasterDataset_SR_8b(RasterDataset):
    filename_glob = "**/*_AnalyticMS_SR_8b_*.tif"
    filename_regex = r"^(?P<date>\d{8})_.+"
    date_format = "%Y%m%d"
    is_image = True
    separate_files = False

    rgb_indices = (5, 3, 1)

    @classmethod
    def plot(cls, sample, ax=None):
        image = sample[cls.rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 3000, min=0, max=1).numpy()

        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(image)
        return ax


class KelpLabelsDataset(RasterDataset):
    filename_glob = "**/*_AnalyticMS_SR_8b_*.tif"
    filename_regex = r"^(?P<date>\d{8})_.+"
    date_format = "%Y%m%d"
    is_image = False

    bg_value = 0

    @classmethod
    def plot(cls, sample, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        mask = np.ma.masked_where(sample == cls.bg_value, sample)
        ax.imshow(mask, cmap="summer", alpha=0.5)
        return ax


def create_chips(out_root, name, dset, chip_size=224, chip_stride=224):
    out_dir = out_root / name
    out_dir.mkdir(exist_ok=True, parents=True)

    sampler = GridGeoSampler(dset, size=chip_size, stride=chip_stride)
    dataloader = DataLoader(
        dset,
        sampler=sampler,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=stack_samples,
    )

    for i, batch in enumerate(tqdm(dataloader, desc=name)):
        img = batch["image"]
        label = batch["mask"]

        kelp_pixels = (label == LABELS["kelp"]).sum()
        land_pixels = (label == LABELS["land"]).sum()
        height = img.shape[2]
        width = img.shape[3]

        if (
            (
                name == "train" and kelp_pixels == 0 and land_pixels == 0
            )  # Train chips must have some kelp or land present
            or height < chip_size
            or width < chip_size
        ):
            continue

        # Convert to numpy arrays
        img_array = img[0].numpy().astype(np.float32)  # Convert to float32
        label_array = label.numpy().astype(
            np.uint8
        )  # Keep labels as uint8 for efficiency

        # Save the image as npz
        np.savez_compressed(out_dir / f"{i}.npz", image=img_array, label=label_array)


def load_dataset(data_dir):
    images = PlanetRasterDataset_SR_8b(data_dir / "images")
    labels = KelpLabelsDataset(data_dir / "labels")

    return images & labels


def main():
    """
    Main function to process files and create chips.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir", type=Path, help="Directory containing the input GeoTIFF files."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save the output chips."
    )
    parser.add_argument(
        "--size", type=int, default=224, help="Size of the square chips."
    )
    parser.add_argument("--stride", type=int, default=168, help="Stride for the chips.")

    args = parser.parse_args()

    train_ds = load_dataset(args.data_dir / "train")
    create_chips(args.output_dir, "train", train_ds, args.size, args.stride)

    val_ds = load_dataset(args.data_dir / "val")
    create_chips(args.output_dir, "val", val_ds, args.size, args.stride)

    test_ds = load_dataset(args.data_dir / "test")
    create_chips(args.output_dir, "test", test_ds, args.size, args.stride)


if __name__ == "__main__":
    main()
