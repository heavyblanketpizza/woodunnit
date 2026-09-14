import hashlib
import json

import pytest
from PIL import Image

from woodunnit.cli import main, parser


def test_public_commands_cover_data_preparation_only():
    help_text = parser().format_help()
    for command in ("ingest", "validate", "report", "release", "split", "preprocess"):
        assert command in help_text
    for command in ("train", "predict", "benchmark"):
        with pytest.raises(SystemExit) as exc:
            parser().parse_args([command])
        assert exc.value.code == 2


def test_preprocess_exports_tensor_and_path_free_provenance(tmp_path, capsys):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    source = tmp_path / "private-source.png"
    Image.new("RGB", (8, 4), (255, 0, 0)).save(source)
    original = source.read_bytes()
    output = tmp_path / "derived" / "example.npz"
    args = ["preprocess", "--image", str(source), "--output", str(output), "--image-size", "8"]
    assert main(args) == 0
    assert source.read_bytes() == original
    with np.load(output, allow_pickle=False) as artifact:
        tensor = artifact["image"]
        assert tensor.shape == (3, 8, 8)
        assert tensor.dtype == np.float32
        assert np.isfinite(tensor).all()
        np.testing.assert_allclose(tensor[0, 2:6], (1 - 0.485) / 0.229)
    metadata_text = output.with_suffix(".json").read_text()
    metadata = json.loads(metadata_text)
    assert metadata["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["preprocessing"]["augmentation"] == "none"
    assert str(source) not in metadata_text
    assert source.name not in metadata_text
    assert json.loads(capsys.readouterr().out) == metadata
    artifact_bytes = output.read_bytes()
    assert main(args) == 2
    assert "already exist" in capsys.readouterr().err
    assert output.read_bytes() == artifact_bytes


def test_preprocess_rejects_wrong_output_type_before_writing(tmp_path, capsys):
    pytest.importorskip("woodunnit.preprocessing")
    output = tmp_path / "wrong.csv"
    assert main(["preprocess", "--image", "missing.png", "--output", str(output)]) == 2
    assert ".npz extension" in capsys.readouterr().err
    assert not output.exists()
