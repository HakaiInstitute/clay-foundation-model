"""
DataModule for the Chesapeake Bay dataset for segmentation tasks.

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a segmentation model.

Dataset citation:
Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition
(CVPR 2019).

Dataset URL: https://lila.science/datasets/chesapeakelandcover
"""

from pathlib import Path

import albumentations as A
import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset


class PSKelpDataset(Dataset):
    """
    Dataset class for the Chesapeake Bay segmentation dataset.

    Args:
        chip_dir (str): Directory containing the image chips.
        label_dir (str): Directory containing the labels.
        transform (Callable): Albumentations transforms for this dataset.
    """

    def __init__(self, chip_dir, transform):
        self.chip_dir = Path(chip_dir)
        self.transform = transform

        # Load chip file names
        self.chips = [chip_path.name for chip_path in self.chip_dir.glob("*.npz")]

    @staticmethod
    def create_train_transforms(mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return A.Compose(
            [
                A.D4(),  # Random flip/rotation combinations
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0, always_apply=True),
                # A.MotionBlur(
                #     blur_limit=(3, 5),      # Very subtle blur, 3-5 pixel kernel
                #     allow_shifted=True,     # Allows directional blur
                #     p=0.15                  # Low probability - most images are sharp
                # ),
                # A.GaussNoise(
                #     var_limit=(0.0001, 0.001),  # Very low noise
                #     mean=0,
                #     per_channel=True,
                #     p=0.3
                # ),
                A.ToTensorV2(),
            ]
        )

    @staticmethod
    def create_test_transforms(mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return A.Compose(
            [
                A.D4(),  # Random flip/rotation combinations
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0, always_apply=True),
                A.ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, label, and additional information.
        """
        chip_name = self.chip_dir / self.chips[idx]
        data = np.load(chip_name)
        chip = np.moveaxis(data["image"], 0, -1).astype(
            np.float32
        )  # Move channel dimension to last position
        label = np.moveaxis(
            data["label"], 0, -1
        )  # Move channel dimension to last position

        # Remap labels to match desired classes
        label_mapping = {
            0: 0,
            1: 1,
            2: 0,
            3: 0,
            4: 0,
        }
        remapped_label = np.vectorize(label_mapping.get)(label)

        augmented = self.transform(image=chip, mask=remapped_label)
        sample = {
            "pixels": augmented["image"],
            "label": augmented["mask"].squeeze(-1),
            "time": torch.zeros(4),  # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for latlon information
        }
        return sample


class PSKelpDataModule(L.LightningDataModule):
    """
    DataModule class for the Chesapeake Bay dataset.

    Args:
        train_chip_dir (str): Directory containing training image chips.
        train_label_dir (str): Directory containing training labels.
        val_chip_dir (str): Directory containing validation image chips.
        val_label_dir (str): Directory containing validation labels.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(  # noqa: PLR0913
        self,
        train_chip_dir,
        val_chip_dir,
        test_chip_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
    ):
        super().__init__()
        self.train_chip_dir = train_chip_dir
        self.val_chip_dir = val_chip_dir
        self.test_chip_dir = test_chip_dir
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        mean = list(self.metadata[self.platform].bands.mean.values())
        std = list(self.metadata[self.platform].bands.std.values())

        train_transforms = PSKelpDataset.create_train_transforms(mean=mean, std=std)
        test_transforms = PSKelpDataset.create_test_transforms(mean=mean, std=std)

        if stage in {"fit", None}:
            self.trn_ds = PSKelpDataset(
                chip_dir=self.train_chip_dir,
                transform=train_transforms,
            )
            self.val_ds = PSKelpDataset(
                chip_dir=self.val_chip_dir,
                transform=test_transforms,
            )
        if stage in {"test", None}:
            self.test_ds = PSKelpDataset(
                chip_dir=self.test_chip_dir,
                transform=test_transforms,
            )

    def train_dataloader(self):
        """
        Create DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Create DataLoader for test data.

        Returns:
            DataLoader: DataLoader for test dataset.
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
