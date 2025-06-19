r"""
We will create chips of size `224 x 224` to feed them to the model, feel
free to experiment with other chip sizes as well.
   Run the script as follows:
   python preprocess_data.py <data_dir> <output_dir> <chip_size> <chip_stride? (defaults to chip_size)>

   Example:
   python preprocess_data.py data/ps_8b/ data/ps_8b_tiled 224
"""  # noqa E501

import sys
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
    (out_dir / "images").mkdir(exist_ok=True, parents=True)
    (out_dir / "labels").mkdir(exist_ok=True, parents=True)

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

        black_pixels = (img == torch.zeros((1, NUM_BANDS, 1, 1))).sum()
        bad_data = (label > 2).sum()
        kelp_pixels = (label == 1).sum()
        total_pixels = img.numel() / NUM_BANDS
        height = img.shape[2]
        width = img.shape[3]

        if (
            (bad_data > 0)
            or (kelp_pixels == 0)
            or (black_pixels / total_pixels) > MAX_NODATA_PROPORTION
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
        np.savez_compressed(out_dir / "images" / f"{i}.npz", data=img_array)

        # Save the label as npz
        np.savez_compressed(out_dir / "labels" / f"{i}.npz", data=label_array)


def load_dataset(data_dir):
    images = PlanetRasterDataset_SR_8b(data_dir / "images")
    labels = KelpLabelsDataset(data_dir / "labels")

    return images & labels


def main():
    """
    Main function to process files and create chips.
    Expects three command line arguments:
        - data_dir: Directory containing the input GeoTIFF files.
        - output_dir: Directory to save the output chips.
        - chip_size: Size of the square chips.
    """
    if 4 > len(sys.argv) > 5:  # noqa: PLR2004
        print(
            "Usage: python script.py <data_dir> <output_dir> <chip_size> <chip_stride=chip_size>"
        )
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    chip_size = int(sys.argv[3])
    chip_stride = int(sys.argv[4]) if len(sys.argv) > 4 else chip_size

    train_ds = load_dataset(data_dir / "train")
    create_chips(output_dir, "train", train_ds, chip_size, chip_stride)

    val_ds = load_dataset(data_dir / "val")
    create_chips(output_dir, "val", val_ds, chip_size, chip_stride)

    test_ds = load_dataset(data_dir / "test")
    create_chips(output_dir, "test", test_ds, chip_size, chip_stride)


if __name__ == "__main__":
    main()
