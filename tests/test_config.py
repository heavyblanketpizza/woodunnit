"""Portable path resolution and protection of immutable acquisition folders."""

import shutil

import pytest

from woodunnit.config import IngestionConfig, load_config


@pytest.fixture
def configured_project(tmp_path):
    project = tmp_path / "first-location"
    (project / "config").mkdir(parents=True)
    (project / "raw").mkdir()
    (project / "raw/image_inventory.csv").write_text("collection,path\n", encoding="utf-8")
    path = project / "config/ingestion.toml"
    path.write_text('raw_root = "../raw"\noutput_root = "../catalogs"\n', encoding="utf-8")
    return project, path


def test_relative_configuration_relocates_without_edits(configured_project, tmp_path, monkeypatch):
    project, path = configured_project
    monkeypatch.chdir(tmp_path)
    original = load_config(path)
    assert original.raw_root == project / "raw"
    assert original.output_root == project / "catalogs"
    moved = tmp_path / "second-location"
    shutil.copytree(project, moved)
    relocated = load_config(moved / "config/ingestion.toml")
    assert relocated.raw_root == moved / "raw"
    assert relocated.output_root == moved / "catalogs"
    assert relocated.task_groups == ["fungi", "oomycetes"]


@pytest.mark.parametrize("output", ["../raw", "../raw/derived", "../raw/../raw/nested"])
def test_output_must_not_modify_acquisition_tree(configured_project, output):
    _, path = configured_project
    path.write_text(f'raw_root = "../raw"\noutput_root = "{output}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="outside the original dataset tree"):
        load_config(path)


def test_symlink_alias_cannot_bypass_raw_output_protection(configured_project):
    project, path = configured_project
    (project / "alias").symlink_to(project / "raw", target_is_directory=True)
    path.write_text('raw_root = "../raw"\noutput_root = "../alias/generated"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="outside the original dataset tree"):
        load_config(path)


@pytest.mark.parametrize(
    "groups",
    [
        '["bacteria"]',
        '["nematodes"]',
        '["fungi", "oomycetes", "bacteria"]',
        '["fungi", "oomycetes", "nematodes"]',
        '["unknown"]',
        '["fungi", "fungi"]',
        '["oomycetes", "fungi"]',
        '["oomycetes"]',
        '["fungi"]',
        "[]",
    ],
)
def test_configuration_does_not_silently_expand_product_taxonomy(configured_project, groups):
    _, path = configured_project
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"task_groups = {groups}\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_raster_verification_cannot_be_disabled(configured_project):
    _, path = configured_project
    with path.open("a", encoding="utf-8") as stream:
        stream.write("verify_images = false\n")
    with pytest.raises(ValueError, match="verify_images"):
        load_config(path)


def test_unmounted_or_missing_raw_root_has_actionable_error(configured_project):
    project, path = configured_project
    shutil.rmtree(project / "raw")
    with pytest.raises(ValueError, match="Dataset root is unavailable"):
        load_config(path)


def test_missing_inventory_is_rejected(configured_project):
    project, path = configured_project
    (project / "raw/image_inventory.csv").unlink()
    with pytest.raises(ValueError, match="image_inventory.csv is missing"):
        load_config(path)


def test_programmatic_config_also_normalizes_raw_output_aliases(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="outside the original dataset tree"):
        IngestionConfig(raw_root=raw, output_root=raw / ".." / "raw" / "nested")
