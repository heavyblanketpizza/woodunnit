"""Deterministic, full-field OpenCV preparation for training and inference."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


class PreprocessingError(ValueError):
    """The image or preprocessing configuration violates the input contract."""


def _channel_values(value: Any, name: str, *, positive: bool) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 3:
        raise PreprocessingError(f"{name} must contain three finite numbers")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise PreprocessingError(f"{name} must contain three finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise PreprocessingError(f"{name} must contain three finite numbers")
        if positive and number <= 0:
            raise PreprocessingError("std values must be positive")
        if not positive and not 0 <= number <= 1:
            raise PreprocessingError("mean values must be between zero and one")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Match mean/std to the downstream model's documented input contract.

    Defaults are ImageNet RGB normalization. Serialization contains only tunable
    values; ``OpenCVPreprocessor.description()`` records the fixed operations.
    """

    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if type(self.image_size) is not int or self.image_size <= 0:
            raise PreprocessingError("image_size must be a positive integer")
        object.__setattr__(self, "mean", _channel_values(self.mean, "mean", positive=False))
        object.__setattr__(self, "std", _channel_values(self.std, "std", positive=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PreprocessConfig:
        if not isinstance(value, Mapping) or set(value) != {"image_size", "mean", "std"}:
            raise PreprocessingError("Preprocessing config requires exactly image_size, mean, std")
        return cls(image_size=value["image_size"], mean=value["mean"], std=value["std"])


class OpenCVPreprocessor:
    """Load a single stored-orientation image without cropping or augmentation.

    ``rgb_to_tensor`` applies the same geometry and normalization to an already
    decoded uint8 RGB image. Raw images are only read; no generated file is saved.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        if not isinstance(config, PreprocessConfig):
            raise PreprocessingError("config must be a PreprocessConfig")
        self.config = config
        self.padding_rgb = tuple(round(channel * 255) for channel in config.mean)
        self._mean = torch.tensor(config.mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(config.std, dtype=torch.float32).view(3, 1, 1)

    def description(self) -> dict[str, Any]:
        """A serializable operation contract to save with an experiment."""
        return {
            "version": "opencv_full_field_v1",
            "config": self.config.to_dict(),
            "decode": "cv2.imdecode(IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION)",
            "channel_conversion": "BGR_to_RGB",
            "orientation": "stored_pixels_ignore_exif",
            "resize": "longest_side_to_image_size_preserve_aspect_ratio",
            "resize_rounding": "nearest_integer_ties_to_even_minimum_one_pixel",
            "interpolation": "cv2.INTER_CUBIC",
            "padding": "center_extra_pixel_on_bottom_or_right",
            "padding_rgb_uint8": list(self.padding_rgb),
            "normalization": "(float32_RGB_CHW / 255 - mean) / std",
            "augmentation": "none",
        }

    def __call__(self, path: Path) -> torch.Tensor:
        path = Path(path)
        try:
            encoded = path.read_bytes()
        except OSError as error:
            raise PreprocessingError(f"Cannot read image: {path}") from error
        if not encoded:
            raise PreprocessingError(f"Cannot decode empty image: {path}")
        try:
            with Image.open(io.BytesIO(encoded)) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise PreprocessingError(f"Multiframe image is not an approved still: {path}")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
            if isinstance(error, PreprocessingError):
                raise
            raise PreprocessingError(f"Cannot inspect image: {path}") from error
        try:
            bgr = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
        except cv2.error as error:
            raise PreprocessingError(f"Cannot decode image with OpenCV: {path}") from error
        if bgr is None:
            raise PreprocessingError(f"Cannot decode image with OpenCV: {path}")
        return self.rgb_to_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def rgb_to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or rgb.shape[0] == 0
            or rgb.shape[1] == 0
        ):
            raise PreprocessingError("Expected a nonempty HWC uint8 RGB array")
        height, width = rgb.shape[:2]
        size = self.config.image_size
        scale = size / max(height, width)
        resized_width = max(1, min(size, round(width * scale)))
        resized_height = max(1, min(size, round(height * scale)))
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
        prepared = np.empty((size, size, 3), dtype=np.uint8)
        prepared[:] = self.padding_rgb
        top = (size - resized_height) // 2
        left = (size - resized_width) // 2
        prepared[top : top + resized_height, left : left + resized_width] = resized
        tensor = torch.from_numpy(prepared.transpose(2, 0, 1).copy()).to(dtype=torch.float32)
        return tensor.div_(255).sub_(self._mean).div_(self._std)
