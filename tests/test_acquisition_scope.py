"""Acquisition maintenance checks use temporary files and never contact repositories."""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def acquisition_scripts():
    scripts = Path(__file__).resolve().parents[1] / "scripts" / "acquisition"
    modules = {}
    for name in ("download_collections", "finalize_downloads"):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def test_removed_collection_cannot_be_restored_by_figshare_helper(tmp_path, acquisition_scripts):
    downloader = acquisition_scripts["download_collections"]
    with pytest.raises(ValueError, match="outside the retained"):
        downloader.figshare(tmp_path, "04_usda_turfgrass_nematodes", 27244674)
    assert not list(tmp_path.iterdir())


def test_idphy_resume_preserves_different_saved_selection(tmp_path, acquisition_scripts):
    downloader = acquisition_scripts["download_collections"]
    saved = tmp_path / "02_idphy_microscopy" / "metadata" / "selection_manifest.json"
    saved.parent.mkdir(parents=True)
    saved.write_text('{"images": [], "source": "original"}\n')
    original = saved.read_bytes()
    incoming = tmp_path / "new_selection.json"
    incoming.write_text('{"images": [], "source": "replacement"}\n')
    with pytest.raises(ValueError, match="existing selection preserved"):
        downloader.idphy(tmp_path, incoming)
    assert saved.read_bytes() == original


def test_optional_cache_only_fills_missing_provenance(tmp_path, acquisition_scripts):
    finalizer = acquisition_scripts["finalize_downloads"]
    saved, cache = tmp_path / "saved", tmp_path / "cache"
    saved.mkdir()
    cache.mkdir()
    (saved / "selection_manifest.json").write_text('{"images": []}\n')
    (saved / "IDPHY_SELECTION.md").write_text("original selection notes")
    (cache / "IDPHY_SELECTION.md").write_text("different cache notes")
    (cache / "discover_idphy.py").write_text("cached discovery source")
    for directory in (saved, cache):
        (directory / "idphy_metadata").mkdir()
        (directory / "idphy_metadata" / "page.html").write_text(directory.name)
    (cache / "idphy_metadata" / "missing.html").write_text("missing source page")
    summary = finalizer.preserve_idphy_provenance(saved, cache)
    assert (saved / "IDPHY_SELECTION.md").read_text() == "original selection notes"
    assert (saved / "idphy_metadata" / "page.html").read_text() == "saved"
    assert (saved / "idphy_metadata" / "missing.html").read_text() == "missing source page"
    assert (saved / "discover_idphy.py").read_text() == "cached discovery source"
    assert summary["source_html_cache_present"] is True


@pytest.mark.parametrize("write_reports", [False, True])
def test_finalizer_works_without_temp_cache_and_inventories_only_retained_sources(
    tmp_path, monkeypatch, acquisition_scripts, write_reports
):
    finalizer = acquisition_scripts["finalize_downloads"]
    assert set(finalizer.COLLECTIONS) == {
        "01_tgfc",
        "02_idphy_microscopy",
        "03_soil_fungi_phytophthora",
    }
    assert sum(info["expected_images"] for info in finalizer.COLLECTIONS.values()) == 6166
    base, workspace = tmp_path / "datasets", tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    source_records = {}
    for slug in finalizer.COLLECTIONS:
        root = base / slug
        (root / "images").mkdir(parents=True)
        Image.new("RGB", (4, 3), "red").save(root / "images" / "example.png")
        source = root / "download_manifest.json"
        source.write_text(json.dumps({"downloads": [{"filename": "example.png"}]}))
        source_records[source] = source.read_bytes()
    metadata = base / "02_idphy_microscopy" / "metadata"
    metadata.mkdir()
    selection = metadata / "selection_manifest.json"
    selection.write_text('{"images": [{"filename": "example.png"}]}\n')
    source_records[selection] = selection.read_bytes()
    removed_source = base / "04_usda_turfgrass_nematodes"
    removed_source.mkdir()
    sentinel = removed_source / "untouched.txt"
    sentinel.write_text("Deletion is a separate operation, not a finalizer side effect.")
    monkeypatch.setattr(
        finalizer,
        "COLLECTIONS",
        {slug: info | {"expected_images": 1} for slug, info in finalizer.COLLECTIONS.items()},
    )
    arguments = ["--destination", str(base)]
    if write_reports:
        arguments.extend(["--report-dir", str(workspace)])
    finalizer.main(arguments)
    report = json.loads((base / "acquisition_summary.json").read_text())
    assert report["total_image_files"] == 3
    assert report["destination"] == "."
    if write_reports:
        assert {path.name for path in workspace.iterdir()} == {
            "ACQUISITION_SUMMARY.json",
            "DATA_DOWNLOADS.md",
        }
        assert all(str(tmp_path) not in path.read_text() for path in workspace.iterdir())
    else:
        assert not list(workspace.iterdir())
    assert report["image_decode_failures"] == []
    assert report["idphy_provenance"]["source_html_cache_present"] is False
    assert {item["collection"] for item in report["collections"]} == set(finalizer.COLLECTIONS)
    assert "usda" not in (base / "image_inventory.csv").read_text().lower()
    assert sentinel.is_file()
    assert all(path.read_bytes() == original for path, original in source_records.items())


@pytest.mark.parametrize("script", ["download_collections", "finalize_downloads"])
def test_destination_is_required(script, acquisition_scripts):
    with pytest.raises(SystemExit) as error:
        acquisition_scripts[script].main([])
    assert error.value.code == 2


@pytest.mark.parametrize("script", ["download_collections", "finalize_downloads"])
@pytest.mark.parametrize("invalid", ["missing", "file", "home", "filesystem"])
def test_invalid_destination_fails_before_writing_or_network(
    tmp_path, monkeypatch, acquisition_scripts, script, invalid
):
    home = tmp_path / "synthetic-home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    destination = {
        "missing": tmp_path / "missing",
        "file": tmp_path / "file.txt",
        "home": home,
        "filesystem": Path(tmp_path.anchor),
    }[invalid]
    if invalid == "file":
        destination.write_text("preserve this file")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    arguments = ["--destination", str(destination)]
    if script == "download_collections":
        arguments.extend(["--collection", "repositories"])
    with pytest.raises(ValueError, match="existing directory|dedicated"):
        acquisition_scripts[script].main(arguments)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not (tmp_path / "missing").exists()
    assert not list(home.iterdir())


def test_downloader_uses_explicit_portable_destination(tmp_path, monkeypatch, acquisition_scripts):
    downloader = acquisition_scripts["download_collections"]
    base = tmp_path / "chosen datasets"
    base.mkdir()
    calls = []

    def fake_figshare(root, slug, article_id):
        calls.append((root, slug, article_id))
        return {"downloads": []}

    def fake_soil(root):
        calls.append((root, "soil"))
        return {"downloads": []}

    monkeypatch.setattr(downloader, "figshare", fake_figshare)
    monkeypatch.setattr(downloader, "soil", fake_soil)
    with pytest.raises(SystemExit) as error:
        downloader.main(["--destination", str(base), "--collection", "repositories"])
    assert error.value.code == 0
    assert calls == [(base, "01_tgfc", 28855910), (base, "soil")]
    assert json.loads((base / "acquisition_repositories_status.json").read_text())["failures"] == []


@pytest.mark.parametrize("nested", [False, True])
def test_finalizer_report_directory_cannot_overwrite_raw_metadata(
    tmp_path, acquisition_scripts, nested
):
    base = tmp_path / "datasets"
    base.mkdir()
    reports = base / "reports" if nested else base
    reports.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="outside the dataset"):
        acquisition_scripts["finalize_downloads"].main(
            ["--destination", str(base), "--report-dir", str(reports)]
        )
    assert not any(path.is_file() for path in base.rglob("*"))


def test_finalizer_missing_selection_never_uses_implicit_cache(tmp_path, acquisition_scripts):
    with pytest.raises(RuntimeError, match="Missing IDphy selection provenance"):
        acquisition_scripts["finalize_downloads"].preserve_idphy_provenance(tmp_path / "metadata")
    assert not list(tmp_path.iterdir())
