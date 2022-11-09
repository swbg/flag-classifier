from typing import Any, List, Tuple

import cv2
import numpy as np
import skimage.transform
from albumentations.core.transforms_interface import ImageOnlyTransform
from numpy.random import Generator


def get_contours(
    rng: Generator, edge_size: Tuple[int, int], n_segments: int, rotate: bool = False
) -> List[List[Tuple[int, int]]]:
    """
    Generate random, wedge-shaped contours.

    :param rng: Random number generator
    :param edge_size: Edge size of target image
    :param n_segments: How many wedges to generate
    :param rotate: Whether to rotate (divide along dimension 0)
    """
    if rotate:
        edge_size = edge_size[::-1]

    # Cuts are placed along dimension 1 (if not rotated)
    points_t = [
        0,
        *sorted((rng.uniform(size=n_segments - 1) * edge_size[1]).astype(int)),
        edge_size[1] - 1,
    ]
    points_b = [
        0,
        *sorted((rng.uniform(size=n_segments - 1) * edge_size[1]).astype(int)),
        edge_size[1] - 1,
    ]

    contours = []
    for i in range(n_segments):
        if rotate:
            contours.append(
                [
                    (points_t[i], 0),
                    (points_b[i], edge_size[0] - 1),
                    (points_b[i + 1], edge_size[0] - 1),
                    (points_t[i + 1], 0),
                ]
            )
        else:
            contours.append(
                [
                    (0, points_t[i]),
                    (0, points_t[i + 1]),
                    (edge_size[0] - 1, points_b[i + 1]),
                    (edge_size[0] - 1, points_b[i]),
                ]
            )

    return contours


def get_contour_masks(
    rng: Generator, edge_size: Tuple[int, int], n_segments: int, rotate: bool = False
) -> List[np.ndarray]:
    """
    Generate random, wedge-shaped masks.

    :param rng: Random number generator
    :param edge_size: Edge size of target image
    :param n_segments: How many wedges to generate
    :param rotate: Whether to rotate (divide along dimension 0)
    """
    contours = get_contours(rng, edge_size, n_segments, rotate)
    masks = []

    xx = np.arange(edge_size[1]).reshape(1, -1)
    yy = np.arange(edge_size[0]).reshape(-1, 1)

    for contour in contours:
        a, b, c, d = contour
        base_mask = np.ones(edge_size)

        # Top horizontal
        if rotate:
            fnr = a[0] + (b[0] - a[0]) / (b[1] - a[1]) * xx
            base_mask[yy < fnr] = 0

        # Right vertical
        if not rotate:
            fnr = b[1] + (c[1] - b[1]) / (c[0] - b[0]) * yy
            base_mask[xx > fnr] = 0

        # Bottom horizontal
        if rotate:
            fnr = d[0] + (c[0] - d[0]) / (c[1] - d[1]) * xx
            base_mask[yy > fnr] = 0

        # Left vertical
        if not rotate:
            fnr = a[1] + (d[1] - a[1]) / (d[0] - a[0]) * yy
            base_mask[xx < fnr] = 0

        masks.append(base_mask > 0)

    return masks


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

    def apply(self, img: np.ndarray, **kwargs: Any) -> np.ndarray:
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

    def apply(self, img: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply transform to image.

        :param img: Target image
        """
        contours = get_contours(
            rng=self.rng,
            edge_size=img.shape[:2],
            n_segments=self.n_segments,
            rotate=False,
        ) + get_contours(
            rng=self.rng,
            edge_size=img.shape[:2],
            n_segments=self.n_segments,
            rotate=True,
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

        return np.clip(img * mask[..., None], 0, 255).astype(np.uint8)


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

    def _get_base_points(self, edge_size: int) -> np.ndarray:
        """
        Get base point grid (first column all zeros, second all ones, ...).

        :param edge_size: Image edge size
        :return: Base point grid
        """
        return np.repeat(
            np.linspace(0, edge_size, self.n_points), self.n_points
        ).reshape(self.n_points, self.n_points)

    def _get_wave(self, n_waves: int = 3) -> np.ndarray:
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

    def _get_transversal_points(self, edge_size: int, n_waves: int = 3) -> np.ndarray:
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

    def _get_longitudinal_points(self, edge_size: int, n_waves: int = 3) -> np.ndarray:
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

    def apply(self, img: np.ndarray, **kwargs: Any) -> np.ndarray:
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

        # Warp image, converts to double
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

        return (255 * np.clip(img + shadow, 0, 1)).astype(np.uint8)
