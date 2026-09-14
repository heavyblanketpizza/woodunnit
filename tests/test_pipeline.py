"""Exercise ingestion integrity and lineage on isolated synthetic acquisitions."""

import copy
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from woodunnit import pipeline
from woodunnit.config import IngestionConfig
from woodunnit.io import sha256_file
from woodunnit.taxonomy import derive_taxonomy, hierarchy_targets


@pytest.fixture
def acquisition(tmp_path, monkeypatch):
    """Use real tiny image files while isolating the source-specific adapters."""
    raw = tmp_path / "raw"
    specifications = [
        ("01_tgfc", "extracted/train/images/fungus.png", "red"),
        ("01_tgfc", "extracted/train/images/second.png", "green"),
        ("02_idphy_microscopy", "images/oomycete.png", "red"),
        ("03_soil_fungi_phytophthora", "images/Phytophtora_1.png", "blue"),
        ("03_soil_fungi_phytophthora", "images/Phytophtora_1_crop.png", "orange"),
        ("03_soil_fungi_phytophthora", "images/Phytophtora_64_crop.png", "purple"),
    ]
    rows = []
    for collection, relative, color in specifications:
        path = raw / collection / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 32), color).save(path)
        rows.append(
            {
                "collection": collection,
                "path": path.relative_to(raw).as_posix(),
                "bytes": str(path.stat().st_size),
                "sha256": sha256_file(path),
                "width": "24",
                "height": "32",
                "format": "PNG",
            }
        )
    with (raw / "image_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(reversed(rows))
    for collection in pipeline.COLLECTIONS:
        downloads = [
            {"filename": Path(row["path"]).name} for row in rows if row["collection"] == collection
        ]
        (raw / collection / "download_manifest.json").write_text(
            json.dumps({"downloads": downloads, "expected_files": len(downloads), "errors": []}),
            encoding="utf-8",
        )

    class FixtureAdapters:
        def __init__(self, raw_root):
            assert raw_root == raw
            self.sources = {
                source: {"source_id": source, "training_permission_verified": False}
                for source, _ in pipeline.COLLECTIONS.values()
            }

        def adapt(self, row):
            source = pipeline.COLLECTIONS[row["collection"]][0]
            labels = {
                "tgfc": ["MixedClass"],
                "idphy": ["Phytophthora example"],
                "soil": ["Phytophtora"],
            }
            groups = {
                "tgfc": ["fungi"],
                "idphy": ["oomycetes"],
                "soil": ["oomycetes"],
            }
            result = {
                "source_id": source,
                "source_labels": labels[source],
                "candidate_groups": groups[source],
                "taxonomy": derive_taxonomy(source, labels[source], groups[source]),
                "source_split": "train" if source == "tgfc" else None,
                "image_role": "unknown",
                "source_metadata": {"original_filename": Path(row["path"]).name},
                "usage_permission_ref": f"sources.json#{source}",
                "lineage": {"specimen_id": None},
            }
            if source == "soil":
                stem = Path(row["path"]).stem
                parent = stem.removesuffix("_crop")
                result["image_role"] = "crop" if stem.endswith("_crop") else "full_field"
                result["lineage"]["source_parent_id"] = parent
                result["relationship_keys"] = [f"soil:parent:{parent}"]
            return copy.deepcopy(result)

    monkeypatch.setattr(pipeline, "AdapterSet", FixtureAdapters)
    return IngestionConfig(raw_root=raw, output_root=tmp_path / "catalogs"), rows


def catalog_records(result):
    path = Path(result["catalog_dir"]) / "images.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_repeated_ingestion_reuses_immutable_deterministic_catalog(acquisition):
    config, _ = acquisition
    first = pipeline.ingest(config, progress=lambda _: None)
    directory = Path(first["catalog_dir"])
    snapshot = {path.name: path.read_bytes() for path in directory.iterdir()}
    second = pipeline.ingest(config, progress=lambda _: None)
    assert first == second
    assert snapshot == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert list(config.output_root.iterdir()) == [directory]
    validated = pipeline.validate_catalog(directory, config.raw_root)
    assert validated["catalog_id"] == first["catalog_id"]

    other = config.model_copy(update={"output_root": config.output_root.parent / "elsewhere"})
    regenerated = pipeline.ingest(other, progress=lambda _: None)
    assert regenerated["catalog_id"] == first["catalog_id"]
    assert snapshot == {
        path.name: path.read_bytes() for path in Path(regenerated["catalog_dir"]).iterdir()
    }


def test_catalog_preserves_source_labels_without_inventing_review_or_project_splits(acquisition):
    config, _ = acquisition
    result = pipeline.ingest(config, progress=lambda _: None)
    records = catalog_records(result)
    assert len(records) == 6
    assert result["summary"]["candidate_group_counts"] == {
        "fungi": 2,
        "oomycetes": 4,
    }
    assert result["summary"]["reviewed_training_images"] == 0
    assert result["summary"]["independent_specimens"] is None
    assert result["summary"]["missing_task_groups"] == []
    assert result["schema_version"] == "3.0"
    assert result["task_groups"] == ["fungi", "oomycetes"]
    for record in records:
        assert record["review_status"] == "unreviewed"
        assert record["disposition"] is None
        assert record["group_label"] is None
        assert record["split"] is None
        assert record["lineage"]["specimen_id"] is None
        assert record["image_mode"] == "RGB"
        assert record["frame_count"] == 1
        assert len(record["pixel_sha256"]) == 64
        assert len(record["lossless_transform_sha256"]) == 64
        assert len(record["perceptual_dhash64"]) == 16
        assert not Path(record["relative_path"]).is_absolute()
        assert "native_dimension_below_224" in record["import_flags"]
    fungi = [record for record in records if record["source_id"] == "tgfc"]
    assert all(record["source_labels"] == ["MixedClass"] for record in fungi)
    assert all(record["source_split"] == "train" for record in fungi)
    soil = [record for record in records if record["source_id"] == "soil"]
    assert all(record["source_labels"] == ["Phytophtora"] for record in soil)


def test_exact_duplicate_links_entire_sources_without_collapsing_original_records(acquisition):
    config, _ = acquisition
    result = pipeline.ingest(config, progress=lambda _: None)
    records = catalog_records(result)
    source_groups = {
        source: {row["leakage_group_id"] for row in records if row["source_id"] == source}
        for source, _ in pipeline.COLLECTIONS.values()
    }
    assert all(len(groups) == 1 for groups in source_groups.values())
    assert source_groups["tgfc"] == source_groups["idphy"]
    assert source_groups["soil"].isdisjoint(source_groups["tgfc"])
    assert result["summary"]["conservative_leakage_groups"] == 2
    assert result["summary"]["unique_image_hashes"] == 5
    duplicates = json.loads(
        (Path(result["catalog_dir"]) / "duplicate_groups.json").read_text(encoding="utf-8")
    )
    assert len(duplicates) == 1
    assert len(duplicates[0]) == 2
    assert len({row["image_id"] for row in records}) == 6
    excluded = [row for row in records if row["selection_status"] == "excluded_redundant"]
    assert len(excluded) == 1
    assert excluded[0]["exclusion_reason"] == "exact_file_duplicate"
    canonical = next(row for row in records if row["image_id"] == excluded[0]["canonical_image_id"])
    assert canonical["selection_status"] == "included"
    assert canonical["leakage_group_id"] == excluded[0]["leakage_group_id"]
    assert "duplicate_label_conflict" in canonical["import_flags"]
    assert "duplicate_label_conflict" in excluded[0]["import_flags"]


def test_working_image_label_views_agree_and_omit_redundant_records(acquisition):
    config, _ = acquisition
    result = pipeline.ingest(config, progress=lambda _: None)
    directory = Path(result["catalog_dir"])
    source_records = catalog_records(result)
    examples = [
        json.loads(line)
        for line in (directory / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with (directory / "examples.csv").open(encoding="utf-8", newline="") as stream:
        csv_examples = list(csv.DictReader(stream))
    label_map = json.loads((directory / "label_map.json").read_text(encoding="utf-8"))
    taxonomy_map = json.loads((directory / "taxonomy_map.json").read_text(encoding="utf-8"))
    assert label_map == {"fungi": 0, "oomycetes": 1}
    assert len(examples) == len(csv_examples) == 5
    assert {row["image_id"] for row in examples} == {
        row["image_id"] for row in source_records if row["selection_status"] == "included"
    }
    assert Counter(row["candidate_group"] for row in examples) == {"fungi": 2, "oomycetes": 3}
    sources_by_id = {row["image_id"]: row for row in source_records}
    csv_by_id = {row["image_id"]: row for row in csv_examples}
    for example in examples:
        source, flat = sources_by_id[example["image_id"]], csv_by_id[example["image_id"]]
        assert example["class_id"] == label_map[example["candidate_group"]]
        assert example["canonical_image_id"] == example["image_id"]
        assert example["relative_path"] == source["relative_path"]
        assert example["sha256"] == source["sha256"]
        assert example["label_origin"] == "publisher_mapping"
        assert example["review_status"] == "unreviewed"
        assert example["split"] is None
        assert example["import_flags"] == source["import_flags"]
        for field in ("source_labels", "taxonomy", "annotations", "source_metadata"):
            assert example[field] == source[field]
            assert json.loads(flat[field]) == source[field]
        expected = hierarchy_targets(source["taxonomy"], taxonomy_map)
        for field in ("targets", "target_mask"):
            assert example[field] == expected[field]
            assert json.loads(flat[field]) == expected[field]
        assert flat["relative_path"] == example["relative_path"]
        assert int(flat["class_id"]) == example["class_id"]
        assert flat["candidate_group"] == example["candidate_group"]
        assert flat["label_origin"] == "publisher_mapping"
        assert flat["review_status"] == "unreviewed"
        assert flat["split"] == ""
        assert json.loads(flat["import_flags"]) == example["import_flags"]
        assert (config.raw_root / example["relative_path"]).is_file()


def test_soil_parent_links_retain_missing_source_reference(acquisition):
    config, _ = acquisition
    records = catalog_records(pipeline.ingest(config, progress=lambda _: None))
    soil = {Path(row["relative_path"]).stem: row for row in records if row["source_id"] == "soil"}
    full, present, missing = (
        soil[name]
        for name in (
            "Phytophtora_1",
            "Phytophtora_1_crop",
            "Phytophtora_64_crop",
        )
    )
    assert full["lineage"]["parent_image_id"] is None
    assert present["lineage"]["parent_image_id"] == full["image_id"]
    assert missing["lineage"]["parent_image_id"] is None
    assert missing["lineage"]["source_parent_id"] == "Phytophtora_64"
    assert "parent_image_missing" in missing["import_flags"]
    assert "parent_image_missing" not in present["import_flags"]
    assert len({row["leakage_group_id"] for row in soil.values()}) == 1


def test_same_length_changed_bytes_are_rejected(acquisition):
    config, rows = acquisition
    target = config.raw_root / rows[0]["path"]
    damaged = bytearray(target.read_bytes())
    damaged[-1] ^= 1
    target.write_bytes(damaged)
    with pytest.raises(ValueError, match="Image checksum changed"):
        pipeline.ingest(config, progress=lambda _: None)
    assert not config.output_root.exists()


def test_unlisted_image_is_rejected_instead_of_silently_dropped(acquisition):
    config, rows = acquisition
    listed = config.raw_root / rows[0]["path"]
    shutil.copyfile(listed, listed.with_name("unlisted.PNG"))
    with pytest.raises(ValueError, match="1 unlisted images"):
        pipeline.ingest(config, progress=lambda _: None)


def test_missing_inventory_image_is_rejected(acquisition):
    config, rows = acquisition
    (config.raw_root / rows[0]["path"]).unlink()
    with pytest.raises(ValueError, match="Missing source image"):
        pipeline.ingest(config, progress=lambda _: None)


def test_image_validation_detects_changed_source_after_catalog_creation(acquisition):
    config, rows = acquisition
    result = pipeline.ingest(config, progress=lambda _: None)
    with (config.raw_root / rows[0]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Image size changed"):
        pipeline.validate_catalog(Path(result["catalog_dir"]), config.raw_root)


def test_modified_catalog_is_not_silently_reused(acquisition):
    config, _ = acquisition
    result = pipeline.ingest(config, progress=lambda _: None)
    report = Path(result["catalog_dir"]) / "report.md"
    report.write_text("edited outside pipeline", encoding="utf-8")
    with pytest.raises(ValueError, match="Catalog file checksum mismatch"):
        pipeline.ingest(config, progress=lambda _: None)
    assert not list(config.output_root.glob(".ingesting-*"))


@pytest.mark.parametrize("reference_kind", ["source_metadata", "annotation_file"])
def test_changed_original_metadata_invalidates_catalog(acquisition, monkeypatch, reference_kind):
    config, _ = acquisition
    original = config.raw_root / "01_tgfc" / "original-annotation.txt"
    original.write_text("original source label", encoding="utf-8")
    reference = {
        "path": original.relative_to(config.raw_root).as_posix(),
        "sha256": sha256_file(original),
    }
    base = pipeline.AdapterSet

    class ReferencedAdapters(base):
        def __init__(self, raw_root):
            super().__init__(raw_root)
            if reference_kind == "source_metadata":
                self.sources["tgfc"]["metadata_files"] = [reference]

        def adapt(self, row):
            result = super().adapt(row)
            if result["source_id"] == "tgfc" and reference_kind == "annotation_file":
                result["source_metadata"]["annotation_file"] = reference
            return result

    monkeypatch.setattr(pipeline, "AdapterSet", ReferencedAdapters)
    result = pipeline.ingest(config, progress=lambda _: None)
    original.write_text("changed source label", encoding="utf-8")
    with pytest.raises(ValueError, match="Acquisition metadata changed"):
        pipeline.validate_catalog(Path(result["catalog_dir"]), config.raw_root, verify_images=False)


def test_unresolved_acquisition_failure_prevents_catalog_creation(acquisition):
    config, _ = acquisition
    path = config.raw_root / "02_idphy_microscopy/download_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["errors"] = [{"filename": "missing.png", "error": "download failed"}]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved download failures"):
        pipeline.ingest(config, progress=lambda _: None)
    assert not config.output_root.exists()


def test_old_catalog_schema_is_rejected_before_use(tmp_path):
    catalog = tmp_path / "old-catalog"
    catalog.mkdir()
    (catalog / "catalog.json").write_text('{"schema_version": "1.0"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported catalog schema version"):
        pipeline.validate_catalog(catalog, tmp_path)


def test_removed_collection_cannot_reenter_through_inventory(acquisition):
    config, rows = acquisition
    removed = {**rows[0], "collection": "04_usda_turfgrass_nematodes"}
    inventory = config.raw_root / "image_inventory.csv"
    with inventory.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=list(rows[0])).writerow(removed)
    with pytest.raises(ValueError, match="Unknown inventory collection"):
        pipeline.ingest(config, progress=lambda _: None)


def test_similarity_candidates_are_flagged_without_exclusion(acquisition):
    config, rows = acquisition
    with Image.new("RGB", (24, 32)) as image:
        image.putdata([(40 if x % 6 < 3 else 220,) * 3 for _y in range(32) for x in range(24)])
        for index, row in enumerate(rows[:2]):
            if index:
                image.putpixel((5, 5), (221, 220, 220))
            path = config.raw_root / row["path"]
            image.save(path)
            row.update(bytes=str(path.stat().st_size), sha256=sha256_file(path))
    inventory = config.raw_root / "image_inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = pipeline.ingest(config, progress=lambda _: None)
    records = catalog_records(result)
    fungi = [record for record in records if record["source_id"] == "tgfc"]
    assert len({record["lossless_transform_sha256"] for record in fungi}) == 2
    assert all(record["selection_status"] == "included" for record in fungi)
    assert all("near_duplicate_candidate" in record["import_flags"] for record in fungi)
    directory = Path(result["catalog_dir"])
    pairs = [
        json.loads(line)
        for line in (directory / "near_duplicate_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        {pair["image_id_a"], pair["image_id_b"]} == {record["image_id"] for record in fungi}
        for pair in pairs
    )
    curation = json.loads((directory / "curation.json").read_text(encoding="utf-8"))
    assert curation["near_duplicate_search"]["automatically_excluded_by_similarity"] == 0
    assert curation["near_duplicate_search"]["review_completed"] is False
