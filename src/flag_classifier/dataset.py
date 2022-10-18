import os
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import skimage.transform
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from numpy.random import Generator, default_rng
from torch.utils.data import Dataset


class RandomResize(ImageOnlyTransform):
    """
    Resize augmentation with independent, random target width and height.
    """

    def __init__(
        self,
        rng: Generator,
        size_limit: Tuple[float, float] = (100, 200),
        always_apply: bool = False,
        p: float = 1.0,
    ):
        """
        Initialize RandomResize.

        :param rng: Random number generator
        :param size_limit: Target edge size limits
        :param always_apply: Whether to always apply transform
        :param p: Probability of applying transform
        """
        super().__init__(always_apply, p)
        self.size_limit = size_limit

        self.rng = rng

    def apply(self, img: np.array, **kwargs: Any) -> np.array:
        """
        Apply transform to image.

        :param img: Target image
        """
        dsize = self.rng.integers(*self.size_limit, size=2)
        return cv2.resize(img, dsize=tuple(dsize))


class RandomShadows(ImageOnlyTransform):
    """
    Add random shadow patches aligned with image edges.
    """

    def __init__(
        self,
        rng: Generator,
        n_segments: int = 4,
        p_per_segment: float = 0.2,
        p_highlight: float = 0.3,
        alpha_limit: Tuple[float, float] = (0.1, 0.4),
        always_apply: bool = False,
        p: float = 1.0,
    ):
        """
        Initialize RandomShadows.

        :param rng: Random number generator
        :param n_segments: Number of segments per axis/dimension
        :param p_per_segment: Probability of drawing each segment
        :parram p_highlight: Probability of using highlight vs. shadow
        :param alpha_limit: Alpha limits for shadow/highlight segments
        :param always_apply: Whether to always apply transform
        :param p: Probability of applying transform
        """
        super().__init__(always_apply, p)
        self.n_segments = n_segments
        self.p_per_segment = p_per_segment
        self.p_highlight = p_highlight
        self.alpha_limit = alpha_limit

        self.rng = rng

    def _get_contours(
        self, edge_size: int, rotate: bool = False
    ) -> List[List[Tuple[int, int]]]:
        points_t = [
            0,
            *sorted(
                (self.rng.uniform(size=self.n_segments - 1) * edge_size).astype(int)
            ),
            edge_size - 1,
        ]
        points_b = [
            0,
            *sorted(
                (self.rng.uniform(size=self.n_segments - 1) * edge_size).astype(int)
            ),
            edge_size - 1,
        ]

        contours = []
        for i in range(self.n_segments):
            if rotate:
                contours.append(
                    [
                        (points_t[i], 0),
                        (points_t[i + 1], 0),
                        (points_b[i + 1], edge_size - 1),
                        (points_b[i], edge_size - 1),
                    ]
                )
            else:
                contours.append(
                    [
                        (0, points_t[i]),
                        (0, points_t[i + 1]),
                        (edge_size - 1, points_b[i + 1]),
                        (edge_size - 1, points_b[i]),
                    ]
                )

        return contours

    def apply(self, img: np.array, **kwargs: Any) -> np.array:
        """
        Apply transform to image.

        :param img: Target image
        """
        contours = self._get_contours(img.shape[0], rotate=False) + self._get_contours(
            img.shape[1], rotate=True
        )

        mask = np.ones(img.shape[:-1])

        do_draws = self.rng.uniform(size=len(contours)) < self.p_per_segment
        factor = np.where(
            self.rng.uniform(size=len(contours)) < self.p_highlight, 1, -1
        )
        alphas = 1 + factor * self.rng.uniform(*self.alpha_limit, size=len(contours))
        for do_draw, contour, alpha in zip(do_draws, contours, alphas):
            if not do_draw:
                continue

            tmp_mask = np.ones(img.shape[:-1])
            cv2.drawContours(
                tmp_mask, np.array([contour]), contourIdx=0, color=alpha, thickness=-1
            )
            mask *= tmp_mask

        return np.clip(img * mask[..., None], 0, 1)


class RandomWarp(ImageOnlyTransform):
    """
    Distortion augmentation.
    """

    def __init__(
        self,
        rng: Generator,
        n_points: int = 30,
        shadow_intensity: float = 0.3,
        dmp_coefs: Tuple[float, float] = (0.4, 5),
        always_apply: bool = False,
        p: float = 1.0,
    ):
        """
        Initialize RandomResize.

        :param rng: Random number generator
        :param n_points: Number of interpolation points
        :param shadow_intensity: Overlay shadow intensity
        :param dmp_coefs: Dampening coefficients (trans/long)
        :param always_apply: Whether to always apply transform
        :param p: Probability of applying transform
        """
        super().__init__(always_apply, p)
        self.n_points = n_points
        self.shadow_intensity = shadow_intensity
        self.dmp_coefs = dmp_coefs

        self.rng = rng

    def _get_base_points(self, edge_size: int) -> np.array:
        """
        Get base point grid (first column all zeros, second all ones, ...).

        :param edge_size: Image edge size
        :return: Base point grid
        """
        return np.repeat(
            np.linspace(0, edge_size, self.n_points), self.n_points
        ).reshape(self.n_points, self.n_points)

    def _get_wave(self, n_waves: int = 3) -> np.array:
        """
        Generate random wave function from sine waves.

        :param n_waves: Numbers of sine waves to add
        :return: Random wave function samples
        """
        wave = np.zeros(self.n_points)
        for _ in range(n_waves):
            n_phases = 3 * self.rng.integers(1, 5)
            amp = self.n_points / n_phases * self.rng.uniform(1, 2)
            phase = self.rng.uniform(0, 2 * np.pi)
            wave += amp * np.sin(
                phase + np.linspace(0, 2 * n_phases * np.pi, self.n_points)
            )
        return wave

    def _get_transversal_points(self, edge_size: int, n_waves: int = 3) -> np.array:
        """
        Get point field displaced like a transversal wave.

        :param edge_size: Image edge size
        :param n_waves: Numbers of sine waves to add
        :return: Transversally displaced point field
        """
        wave = self._get_wave(n_waves)

        multiplier = 1 - np.abs(np.linspace(-1, 1, self.n_points))  # triangle
        multiplier *= self.rng.uniform(self.dmp_coefs[0], 1)  # apply dampening
        wave_field = wave.reshape(1, -1) * multiplier.reshape(-1, 1)

        point_field = self._get_base_points(edge_size)
        point_field = np.clip(point_field + wave_field, 0, edge_size)

        return point_field

    def _get_longitudinal_points(self, edge_size: int, n_waves: int = 3) -> np.array:
        """
        Get point field displaced like a longitudinal wave.

        :param edge_size: Image edge size
        :param n_waves: Numbers of sine waves to add
        :return: Longitudinally displaced point field
        """
        wave = self._get_wave(n_waves)
        wave += self.rng.uniform(0, self.dmp_coefs[1])  # apply dampening
        wave = np.cumsum(wave - wave.min())
        wave -= wave[0]
        wave /= wave[-1]

        point_field = self._get_base_points(edge_size)
        point_field = np.clip(point_field + wave.reshape(-1, 1), 0, edge_size)

        return point_field

    def apply(self, img: np.array, **kwargs: Any) -> np.array:
        """
        Apply transform to image.

        :param img: Target image
        """
        edge_size = img.shape[0]
        assert img.shape[1] == edge_size

        base_points = self._get_base_points(edge_size=edge_size)
        trans_points = self._get_transversal_points(edge_size=edge_size)
        long_points = self._get_longitudinal_points(edge_size=edge_size)

        src = np.stack([base_points.flat, base_points.T.flat], axis=-1)

        trans_first = self.rng.uniform() > 0.5
        if trans_first:
            dst = np.stack([trans_points.flat, long_points.T.flat], axis=-1)
        else:
            dst = np.stack([long_points.flat, trans_points.T.flat], axis=-1)

        tform = skimage.transform.PiecewiseAffineTransform()
        tform.estimate(src, dst)

        # Warp image
        img = skimage.transform.warp(img, tform, output_shape=(edge_size, edge_size))

        # Calculate shadow
        if self.shadow_intensity > 0:
            wave_diff = trans_points - base_points
            shadow = wave_diff / np.abs(wave_diff).max() * self.shadow_intensity

            if trans_first:
                shadow = shadow.T

            if self.rng.uniform() > 0.5:
                shadow *= -1

            shadow = cv2.resize(shadow, dsize=(edge_size, edge_size))[..., None]
        else:
            shadow = 0

        return np.clip(img + shadow, 0, 1)


class FlagDataset(Dataset):
    """
    Dataset for training the flag classifier.

    Data is created synthetically by combining flag images and random samples from the
    Places365 datasets as backgrounds.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FlagDataset.

        :param config: Global configuration
        """
        super().__init__()

        self.config = config

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
        self.rng = default_rng(self.config["global_seed"])

        # Create image transforms
        self.augment_flag = A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 9), sigma_limit=0),
                        A.Downscale(
                            scale_min=0.25,
                            scale_max=0.75,
                            interpolation={
                                "downscale": cv2.INTER_AREA,
                                "upscale": cv2.INTER_NEAREST,
                            },
                        ),
                        A.ImageCompression(quality_lower=50, quality_upper=95),
                    ],
                    p=0.75,
                ),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.05,
                    hue=0.02,
                    always_apply=True,
                ),
                RandomResize(self.rng, size_limit=(100, 200)),
            ]
        )
        self.augment_place = A.Compose(
            [
                A.Rotate(always_apply=True),
                A.Flip(p=0.5),
                A.Transpose(p=0.5),
                A.ColorJitter(p=0.5),
            ]
        )
        self.augment_img = A.Compose(
            [
                A.Rotate(always_apply=True),
                A.CenterCrop(self.input_size, self.input_size),
                RandomShadows(self.rng),
                RandomWarp(self.rng),
                A.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.0
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.data_index_flags)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data_index_flags.iloc[index]

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
        img = self._combine_images(flag, place).astype(np.float32) / 255.0

        # Apply final augmentation
        img = self.augment_img(image=img)["image"]

        return {"img": img, "target": item.target}

    def _read_image(self, file_name: str) -> np.array:
        img = cv2.imread(file_name)

        if img is None:
            raise IOError(f"Image {file_name} could not be loaded")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _combine_images(
        self,
        flag: np.array,
        place: np.array,
        alpha_limit: Tuple[float, float] = (0.8, 1.0),
    ) -> np.array:
        img = place.copy()
        pos = [int((p - f) / 2) for p, f in zip(place.shape, flag.shape)]
        idx = (
            slice(pos[0], pos[0] + flag.shape[0]),
            slice(pos[1], pos[1] + flag.shape[1]),
        )
        alpha = self.rng.uniform(*alpha_limit)
        img[idx] = alpha * flag + (1 - alpha) * img[idx]
        return img
