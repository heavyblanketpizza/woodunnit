import json

import pytest
from PIL import Image

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
preprocessing = pytest.importorskip("woodunnit.preprocessing")
OpenCVPreprocessor = preprocessing.OpenCVPreprocessor
PreprocessConfig = preprocessing.PreprocessConfig
PreprocessingError = preprocessing.PreprocessingError


def unnormalized(size):
    return OpenCVPreprocessor(PreprocessConfig(image_size=size, mean=(0, 0, 0), std=(1, 1, 1)))


def test_file_decode_preserves_rgb_order_and_raw_bytes(tmp_path):
    path = tmp_path / "색상.png"
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [51, 102, 153]]], dtype=np.uint8)
    Image.fromarray(rgb).save(path)
    original_bytes = path.read_bytes()
    output = unnormalized(2)(path)
    expected = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255
    assert torch.equal(output, expected)
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert output.device.type == "cpu"
    assert path.read_bytes() == original_bytes


def test_full_field_is_center_padded_without_cropping():
    # Distinct edges survive; this detects accidental center cropping or stretching.
    rgb = np.full((2, 4, 3), 100, dtype=np.uint8)
    rgb[:, 0] = (255, 0, 0)
    rgb[:, -1] = (0, 0, 255)
    result = unnormalized(4).rgb_to_tensor(rgb)
    assert result.shape == (3, 4, 4)
    assert torch.equal(result[:, 1:3], torch.from_numpy(rgb.transpose(2, 0, 1).copy()) / 255)
    assert torch.count_nonzero(result[:, 0]) == 0
    assert torch.count_nonzero(result[:, 3]) == 0


def test_padding_uses_rounded_mean_rgb_and_odd_extra_pixel_at_right():
    config = PreprocessConfig(image_size=4, mean=(0.5, 0.25, 0.75), std=(0.2, 0.5, 0.25))
    prep = OpenCVPreprocessor(config)
    result = prep.rgb_to_tensor(np.full((4, 1, 3), 255, dtype=np.uint8))
    expected_padding = (
        torch.tensor([128, 64, 191], dtype=torch.float32) / 255 - torch.tensor(config.mean)
    ) / torch.tensor(config.std)
    assert torch.allclose(result[:, :, 0], expected_padding[:, None].expand(3, 4))
    assert torch.allclose(result[:, :, 2:], expected_padding[:, None, None].expand(3, 4, 2))
    expected_image = (1 - torch.tensor(config.mean)) / torch.tensor(config.std)
    assert torch.allclose(result[:, :, 1], expected_image[:, None].expand(3, 4))


def test_resize_preserves_aspect_ratio():
    result = unnormalized(8).rgb_to_tensor(np.full((2, 4, 3), 255, dtype=np.uint8))
    assert torch.all(result[:, 2:6] == 1)
    assert torch.count_nonzero(result[:, :2]) == 0
    assert torch.count_nonzero(result[:, 6:]) == 0


def test_extreme_aspect_ratio_remains_at_least_one_pixel():
    result = unnormalized(4).rgb_to_tensor(np.full((1, 100, 3), 255, dtype=np.uint8))
    assert torch.all(result[:, 1] == 1)
    assert torch.count_nonzero(result) == 12


def test_grayscale_file_is_replicated_into_three_rgb_channels(tmp_path):
    path = tmp_path / "gray.png"
    gray = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    Image.fromarray(gray).save(path)
    result = unnormalized(2)(path)
    assert torch.equal(result[0], torch.from_numpy(gray).float() / 255)
    assert torch.equal(result[0], result[1])
    assert torch.equal(result[1], result[2])


def test_stored_orientation_is_used_even_with_exif_rotation(tmp_path):
    path = tmp_path / "rotated.jpg"
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)
    rgb[:, :4] = (255, 0, 0)
    rgb[:, 4:] = (0, 0, 255)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(rgb).save(path, exif=exif, quality=100, subsampling=0)
    result = unnormalized(8)(path)
    assert torch.count_nonzero(result[:, :2]) == 0
    assert torch.count_nonzero(result[:, 6:]) == 0
    assert result[0, 3, 1] > 0.95
    assert result[2, 3, 6] > 0.95


def test_corrupt_missing_and_empty_files_are_rejected(tmp_path):
    prep = unnormalized(4)
    path = tmp_path / "bad.png"
    with pytest.raises(PreprocessingError, match="Cannot read image"):
        prep(path)
    path.write_bytes(b"")
    with pytest.raises(PreprocessingError, match="empty image"):
        prep(path)
    path.write_bytes(b"This is not a microscopy image")
    with pytest.raises(PreprocessingError, match="Cannot inspect image"):
        prep(path)


def test_multiframe_tiff_is_rejected(tmp_path):
    path = tmp_path / "frames.tiff"
    Image.new("RGB", (4, 4), "red").save(
        path, save_all=True, append_images=[Image.new("RGB", (4, 4), "blue")]
    )
    with pytest.raises(PreprocessingError, match="Multiframe image"):
        unnormalized(4)(path)


def test_configuration_roundtrip_and_preprocessing_are_deterministic():
    config = PreprocessConfig(image_size=5)
    decoded = PreprocessConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert decoded == config
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    original = rgb.copy()
    first = OpenCVPreprocessor(config).rgb_to_tensor(rgb)
    second = OpenCVPreprocessor(decoded).rgb_to_tensor(rgb)
    assert torch.equal(first, second)
    assert np.array_equal(rgb, original)
    description = OpenCVPreprocessor(config).description()
    assert json.loads(json.dumps(description)) == description
    assert description["augmentation"] == "none"
    assert description["config"] == config.to_dict()


@pytest.mark.parametrize("size", [0, -1, 4.5, "224", True])
def test_invalid_image_size_is_rejected(size):
    with pytest.raises(PreprocessingError, match="positive integer"):
        PreprocessConfig(image_size=size)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean": (0.5, 0.5)},
        {"mean": "rgb"},
        {"mean": (True, 0.5, 0.5)},
        {"mean": (-0.1, 0.5, 0.5)},
        {"mean": (1.1, 0.5, 0.5)},
        {"mean": (float("nan"), 0.5, 0.5)},
        {"std": (0, 0.2, 0.3)},
        {"std": (-0.1, 0.2, 0.3)},
        {"std": (float("inf"), 0.2, 0.3)},
    ],
)
def test_invalid_normalization_is_rejected(kwargs):
    with pytest.raises(PreprocessingError):
        PreprocessConfig(**kwargs)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"image_size": 224},
        {"image_size": 224, "mean": [0, 0, 0], "std": [1, 1, 1], "crop": True},
    ],
)
def test_deserialized_configuration_rejects_missing_or_unknown_fields(value):
    with pytest.raises(PreprocessingError, match="requires exactly"):
        PreprocessConfig.from_dict(value)


@pytest.mark.parametrize(
    "array",
    [
        None,
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((0, 2, 3), dtype=np.uint8),
    ],
)
def test_non_rgb_arrays_are_rejected(array):
    with pytest.raises(PreprocessingError, match="uint8 RGB array"):
        unnormalized(4).rgb_to_tensor(array)
