import os
from typing import Any, Dict, Literal, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from numpy.random import default_rng
from torch.utils.data import Dataset

from flag_classifier.augmentations import (
    RandomResize,
    RandomShadows,
    RandomWarp,
    get_contour_masks,
)


class FlagReaderDataset(Dataset):
    """
    Dataset for retrieving training data for the flag classifier.

    Data needs to be prepared using FlagGeneratorDataset.
    """

    def __init__(self, config: Dict[str, Any], mode: Literal["train", "val", "test"]):
        """
        Initialize FlagReaderDataset.

        :param config: Global configuration
        :param mode: Dataset mode
        """
        super().__init__()

        self.config = config
        self.mode = mode

        self.epoch = None
        self.dataset_dir = None
        self.data_index = None

        if self.mode == "train":
            self.set_up_epoch(0)
        else:
            self.set_up_val_test(mode)

    def set_up_epoch(self, epoch: int) -> None:
        """
        Set up dataset for epoch.

        :param epoch: Epoch
        """
        self.epoch = epoch
        self.dataset_dir = os.path.join(
            self.config["dataset_dir"], f"epoch_{self.epoch:05}"
        )
        self.data_index = pd.read_csv(os.path.join(self.dataset_dir, "data_index.csv"))

    def set_up_val_test(self, mode: Literal["val", "test"]) -> None:
        """
        Set up dataset for validation or testing.
        """
        self.dataset_dir = os.path.join(self.config["dataset_dir"], mode)
        self.data_index = pd.read_csv(os.path.join(self.dataset_dir, "data_index.csv"))

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data_index.iloc[index]
        img = torch.load(os.path.join(self.dataset_dir, item.file_name))

        return {"img": img, "target": item.target}


class FlagGeneratorDataset(Dataset):
    """
    Dataset for generating training data for the flag classifier.

    Data is created synthetically by combining flag images and random samples from the
    Places365 datasets as backgrounds.
    """

    def __init__(self, config: Dict[str, Any], mode: Literal["train", "val", "test"]):
        """
        Initialize FlagGeneratorDataset.

        :param config: Global configuration
        """
        super().__init__()

        self.config = config
        self.mode = mode

        # Store some of the config entries explicitly
        self.input_size = self.config["input_size"]
        self.data_dir_flags = self.config["data_dir_flags"]
        self.data_dir_places = self.config["data_dir_places"]

        # Load and prepare flags data index
        self.data_index_flags = pd.read_csv(self.config["data_index_flags"])
        self.data_index_flags.file_name += ".png"
        self.data_index_flags["target"] = np.arange(len(self.data_index_flags))

        # Load places data index
        self.data_index_places = pd.read_csv(
            self.config["data_index_places"], names=["file_name"]
        )

        # Random number generator for sampling places
        self.rng = None
        self.set_seed(self.config["global_seed"])

        # Create image transforms
        self.augment_flag = A.Compose(
            [
                RandomResize(self.rng, size_limit=(100, 200)),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.3,
                    hue=0.05,
                    always_apply=True,
                ),
                A.Flip(),  # this will necessarily introduce ambiguities
            ]
        )
        self.augment_place = A.Compose(
            [
                A.Resize(int(self.input_size * 1.4), int(self.input_size * 1.4)),
                A.Rotate(always_apply=True),
                A.Flip(p=0.5),
                A.Transpose(p=0.5),
                A.ColorJitter(p=0.5),
            ]
        )
        self.augment_img = A.Compose(
            [
                A.Rotate(always_apply=True),
                A.Affine(shear={"x": (-45, 45), "y": 0}, p=0.75),
                A.CenterCrop(self.input_size, self.input_size),
                RandomShadows(self.rng),
                RandomWarp(self.rng),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(9, 21)),
                        A.Downscale(
                            scale_min=0.25,
                            scale_max=0.50,
                            interpolation={
                                "downscale": cv2.INTER_AREA,
                                "upscale": cv2.INTER_NEAREST,
                            },
                        ),
                        A.ImageCompression(quality_lower=25, quality_upper=50),
                    ],
                    p=0.75,
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05,
                    always_apply=True,
                ),
                A.ToFloat(max_value=255.0),
                A.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.0
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.data_index_flags) * self.config["n_per_flag"][self.mode]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data_index_flags.iloc[index % len(self.data_index_flags)]

        flag = self._read_image(os.path.join(self.data_dir_flags, item.file_name))
        place = self._read_image(
            os.path.join(
                self.data_dir_places, self.rng.choice(self.data_index_places.file_name)
            )
        )

        # Apply separate augmentation
        flag = self.augment_flag(image=flag)["image"]
        place = self.augment_place(image=place)["image"]

        # Combine flag and place
        img = self._combine_images(flag, place)

        # Apply final augmentation
        img = self.augment_img(image=img)["image"]

        return {"img": img, "target": item.target}

    def _read_image(self, file_name: str) -> np.ndarray:
        img = cv2.imread(file_name)

        if img is None:
            raise IOError(f"Image {file_name} could not be loaded")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _combine_images(
        self,
        flag: np.ndarray,
        place: np.ndarray,
        alpha_limit: Tuple[float, float] = (0.75, 1.0),
    ) -> np.ndarray:
        img = place.copy()

        # Generate random overlay opacity
        alpha = self.rng.uniform(*alpha_limit)

        # Calculate coordinates of top left edge when centering flag
        pos = [int((p - f) / 2) for p, f in zip(place.shape, flag.shape)]
        slack = [int(0.5 * p) for p in pos]

        n_segments = self.rng.integers(1, 6)
        if n_segments > 1:
            masks = get_contour_masks(
                rng=self.rng,
                edge_size=flag.shape[:2],
                n_segments=n_segments,
                rotate=self.rng.uniform() > 0.5,
            )
            for mask in masks:
                offset_0 = self.rng.integers(-slack[0], slack[0])
                offset_1 = self.rng.integers(-slack[1], slack[1])
                eff_pos = [offset_0 + pos[0], offset_1 + pos[1]]

                pad_0 = (eff_pos[0], img.shape[0] - mask.shape[0] - eff_pos[0])
                pad_1 = (eff_pos[1], img.shape[1] - mask.shape[1] - eff_pos[1])

                dst_mask = np.pad(mask, (pad_0, pad_1), constant_values=False)

                img[dst_mask] = alpha * flag[mask] + (1 - alpha) * img[dst_mask]
        else:
            offset_0 = self.rng.integers(-slack[0], slack[0])
            offset_1 = self.rng.integers(-slack[1], slack[1])
            idx = (
                slice(offset_0 + pos[0], offset_0 + pos[0] + flag.shape[0]),
                slice(offset_1 + pos[1], offset_1 + pos[1] + flag.shape[1]),
            )
            img[idx] = alpha * flag + (1 - alpha) * img[idx]

        return img.astype(np.uint8)

    def set_seed(self, seed: int) -> None:
        """
        Set seed of random number generator.

        :param seed: Seed
        """
        self.rng = default_rng(seed)
