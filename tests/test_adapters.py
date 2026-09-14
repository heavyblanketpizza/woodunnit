"""Synthetic metadata fixtures exercise imports without copying source images."""

import hashlib
import json
from pathlib import Path

import pytest

from woodunnit.adapters import COLLECTIONS, AdapterSet


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    for collection, source_id in COLLECTIONS.items():
        write_json(
            tmp_path / collection / "download_manifest.json",
            {
                "source_url": f"https://example.org/{source_id}",
                "version": 1,
                "license": {"name": "Synthetic fixture license"},
                "downloads": [],
            },
        )
        metadata_name = (
            "selection_manifest.json" if source_id == "idphy" else "repository_record.json"
        )
        write_json(
            tmp_path / collection / "metadata" / metadata_name, {"title": "Synthetic fixture"}
        )
    mapping = tmp_path / "01_tgfc/extracted/main/data.yaml"
    mapping.parent.mkdir(parents=True)
    mapping.write_text(
        "nc: 3\nnames: ['Colletotrichum siamense', 'Olivea tectonae', 'Neopestalotiopsis sp.']\n",
        encoding="utf-8",
    )
    return tmp_path


def row(collection: str, path: str, digest: str = "a" * 64) -> dict[str, str]:
    return {
        "collection": collection,
        "path": path,
        "bytes": "12",
        "sha256": digest,
        "width": "10",
        "height": "10",
        "format": "JPEG",
    }


def add_download(raw: Path, collection: str, filename: str, **metadata) -> dict[str, str]:
    path = raw / collection / "download_manifest.json"
    document = json.loads(path.read_text())
    document["downloads"].append(
        {
            "filename": filename,
            "bytes": 12,
            "sha256": "a" * 64,
            "source_url": f"https://example.org/{filename}",
            **metadata,
        }
    )
    write_json(path, document)
    return row(collection, f"{collection}/images/{filename}")


def test_source_license_and_metadata_hash_are_preserved(raw: Path) -> None:
    sources = AdapterSet(raw).sources
    assert set(sources) == {"tgfc", "idphy", "soil"}
    assert sources["tgfc"]["license"] == {"name": "Synthetic fixture license"}
    assert sources["tgfc"]["training_permission_verified"] is False
    ref = sources["tgfc"]["metadata_files"][0]
    assert ref["sha256"] == hashlib.sha256((raw / ref["path"]).read_bytes()).hexdigest()


def test_tgfc_preserves_boxes_split_and_export_relationship(raw: Path) -> None:
    stem = "Example_1_jpg.rf." + "b" * 32
    label_path = raw / f"01_tgfc/extracted/main/train/labels/{stem}.txt"
    label_path.parent.mkdir(parents=True)
    label_text = "0 0.5 0.5 0.2 0.3\n1 0.2 0.2 0.1 0.1\n"
    label_path.write_text(label_text)
    result = AdapterSet(raw).adapt(
        row("01_tgfc", f"01_tgfc/extracted/main/train/images/{stem}.jpg")
    )
    assert result["source_labels"] == ["Colletotrichum siamense", "Olivea tectonae"]
    assert result["candidate_groups"] == ["fungi"]
    assert result["source_split"] == "train"
    assert result["taxonomy"]["status"] == "multiple_taxa"
    assert result["taxonomy"]["group"] == "fungi"
    assert result["taxonomy"]["genus"] is None
    assert {taxon["species"] for taxon in result["taxonomy"]["taxa"]} == {
        "Colletotrichum siamense",
        "Olivea tectonae",
    }
    assert result["source_metadata"]["annotation_text"] == label_text
    assert result["annotations"][0] == {
        "label": "Colletotrichum siamense",
        "class_id": 0,
        "format": "yolo_cxcywh_normalized",
        "coordinates": [0.5, 0.5, 0.2, 0.3],
    }
    assert result["relationship_keys"] == ["tgfc:export_stem:Example_1_jpg"]
    assert result["microscopy"]["objective_magnification"] is None
    assert "group_label" not in result and "split" not in result and "disposition" not in result


@pytest.mark.parametrize(
    "text",
    ["3 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2", "0 0.5 0.5 1.2 0.2", "0 0.5 0.5"],
)
def test_bad_yolo_boxes_fail_instead_of_becoming_labels(raw: Path, text: str) -> None:
    path = raw / "01_tgfc/extracted/main/train/labels/image.txt"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    with pytest.raises(ValueError, match="YOLO"):
        AdapterSet(raw).adapt(row("01_tgfc", "01_tgfc/extracted/main/train/images/image.jpg"))


def test_missing_tgfc_labels_fail(raw: Path) -> None:
    with pytest.raises(ValueError, match="Missing TgFC annotation"):
        AdapterSet(raw).adapt(row("01_tgfc", "01_tgfc/extracted/main/test/images/missing.jpg"))


def test_mixed_fungal_taxa_are_one_candidate_group(raw: Path) -> None:
    result = AdapterSet(raw).adapt(row("01_tgfc", "01_tgfc/extracted/MixedClass/Mixed_Class 1.jpg"))
    assert result["candidate_groups"] == ["fungi"]
    assert result["source_labels"] == ["MixedClass"]
    assert result["annotations"] == []
    assert result["source_split"] is None
    assert result["taxonomy"]["status"] == "group_only"
    assert result["taxonomy"]["genus"] is None and result["taxonomy"]["species"] is None
    assert "disposition" not in result


def test_soil_crops_link_to_parent_without_claiming_specimen_identity(raw: Path) -> None:
    full = add_download(
        raw,
        "03_soil_fungi_phytophthora",
        "Fusarium_1.png",
        source_label="Fusarium",
        parent_image_id="Fusarium_1",
        image_role="full_field",
    )
    crop = add_download(
        raw,
        "03_soil_fungi_phytophthora",
        "Fusarium_1_2.png",
        source_label="Fusarium",
        parent_image_id="Fusarium_1",
        image_role="crop",
    )
    adapters = AdapterSet(raw)
    parent, child = adapters.adapt(full), adapters.adapt(crop)
    assert parent["relationship_keys"] == child["relationship_keys"]
    assert child["source_metadata"]["parent_relative_path"] == full["path"]
    assert child["lineage"]["source_parent_id"] == "Fusarium_1"
    assert child["lineage"]["parent_image_id"] is None
    assert child["lineage"]["specimen_id"] is None
    assert "crop_coordinates_unavailable" in child["import_flags"]


def test_soil_missing_parent_and_original_spelling_are_preserved(raw: Path) -> None:
    crop = add_download(
        raw,
        "03_soil_fungi_phytophthora",
        "Phytophtora_8_2.png",
        source_label="Phytophtora",
        parent_image_id="Phytophtora_8",
        image_role="crop",
    )
    result = AdapterSet(raw).adapt(crop)
    assert result["source_labels"] == ["Phytophtora"]
    assert result["candidate_groups"] == ["oomycetes"]
    assert result["taxonomy"]["genus"] == "Phytophthora"
    assert result["taxonomy"]["species"] is None
    assert result["taxonomy"]["taxa"][0]["source_label"] == "Phytophtora"
    assert "mapped explicitly" in result["taxonomy"]["taxa"][0]["mapping_note"]
    assert "parent_image_not_in_release" in result["import_flags"]


def test_idphy_retains_conflicting_labels_and_rights(raw: Path) -> None:
    metadata = {
        "species_labels": ["Phytophthora examplea", "Phytophthora exampleb"],
        "source_label_ambiguity": True,
        "caption": "Synthetic caption with multiple isolates",
        "photographer": "Fixture author",
        "gallery_records": [{"entity_id": "1"}, {"entity_id": "2"}],
        "rights_status": "text-only rights check",
        "rights_exception_evidence": ["Review embedded credit"],
    }
    input_row = add_download(raw, "02_idphy_microscopy", "reference.jpg", source_metadata=metadata)
    result = AdapterSet(raw).adapt(input_row)
    assert result["source_labels"] == metadata["species_labels"]
    assert result["candidate_groups"] == ["oomycetes"]
    assert "conflicting_source_taxon_labels" in result["import_flags"]
    assert result["taxonomy"]["genus"] == "Phytophthora"
    assert result["taxonomy"]["species"] is None
    assert result["taxonomy"]["status"] == "multiple_taxa"
    assert result["source_metadata"]["gallery_records"] == metadata["gallery_records"]
    assert (
        result["source_metadata"]["rights_exception_evidence"]
        == metadata["rights_exception_evidence"]
    )
    assert result["lineage"]["isolate_id"] is None
    assert result["image_role"] == "unknown"


def test_idphy_links_every_caption_culture_without_claiming_one_isolate(raw: Path) -> None:
    metadata = {
        "species_labels": ["Phytophthora examplea"],
        "caption": "Oogonia of CPHST BL 55G paired with tester CPHST BL 33.",
        "linked_entity_image_captions": ["Other panel (CPHST BL 55G)."],
        "gallery_records": [
            {
                "caption": "Ex-type CPHST BL 123 = CBS 135746.",
                "linked_entity_image_captions": ["Detail (ex-type CBS 135746)."],
            }
        ],
    }
    input_row = add_download(raw, "02_idphy_microscopy", "reference.jpg", source_metadata=metadata)
    result = AdapterSet(raw).adapt(input_row)
    cultures = {key for key in result["relationship_keys"] if ":culture:" in key}
    assert cultures == {
        "idphy:culture:CPHST_BL:55G",
        "idphy:culture:CPHST_BL:33",
        "idphy:culture:CPHST_BL:123",
        "idphy:culture:CBS:135746",
    }
    assert len(result["relationship_keys"]) == len(set(result["relationship_keys"]))
    assert result["lineage"]["isolate_id"] is None
    assert result["lineage"]["specimen_id"] is None
    assert "caption_culture_references_unreviewed" in result["import_flags"]
    assert result["source_metadata"]["caption"] == metadata["caption"]
    assert result["source_metadata"]["gallery_records"] == metadata["gallery_records"]
    evidence = result["source_metadata"]["observed_culture_references"]
    assert len(evidence) == 6
    assert {reference["source_field"] for reference in evidence} == {
        "caption",
        "linked_entity_image_captions[0]",
        "gallery_records[0].caption",
        "gallery_records[0].linked_entity_image_captions[0]",
    }
    assert evidence[0]["matched_text"] == "CPHST BL 55G"


def test_idphy_keeps_culture_namespaces_and_identifier_suffixes_distinct(raw: Path) -> None:
    metadata = {
        "species_labels": ["Phytophthora examplea"],
        "caption": (
            "Selected specimens CPHST BL 168, CPHST BL 168G, CPHST BL 150a; "
            "Ph168, P168 WPC, SE 166, CH05NSU11, GF145, Toku1; "
            "isolate VI 3-100B9F, AKWA 72.3-0708 + AKWA58.1-0708; "
            "ex-type RHS 53593.1."
        ),
    }
    input_row = add_download(raw, "02_idphy_microscopy", "reference.jpg", source_metadata=metadata)
    result = AdapterSet(raw).adapt(input_row)
    assert {key for key in result["relationship_keys"] if ":culture:" in key} == {
        "idphy:culture:CPHST_BL:168",
        "idphy:culture:CPHST_BL:168G",
        "idphy:culture:CPHST_BL:150a",
        "idphy:culture:PH:168",
        "idphy:culture:P:168",
        "idphy:culture:SE:166",
        "idphy:culture:CH:05NSU11",
        "idphy:culture:GF:145",
        "idphy:culture:TOKU:1",
        "idphy:culture:VI:3-100B9F",
        "idphy:culture:AKWA:72.3-0708",
        "idphy:culture:AKWA:58.1-0708",
        "idphy:culture:RHS:53593.1",
    }


def test_idphy_does_not_extract_lineage_from_filenames_entities_or_bare_numbers(raw: Path) -> None:
    metadata = {
        "species_labels": ["Phytophthora examplea"],
        "caption": "Panels A1 and A2, 20 μm; observed in 2023 on V8 medium.",
        "gallery_records": [{"entity_id": "CPHST BL 123", "filename": "P123-4.jpg"}],
    }
    input_row = add_download(raw, "02_idphy_microscopy", "P168-4.jpg", source_metadata=metadata)
    result = AdapterSet(raw).adapt(input_row)
    assert result["relationship_keys"] == ["idphy:asset:https://example.org/P168-4.jpg"]
    assert "observed_culture_references" not in result["source_metadata"]
    assert all(result["lineage"][field] is None for field in ("isolate_id", "specimen_id"))


def test_unknown_taxon_not_guessed(raw: Path) -> None:
    input_row = add_download(
        raw,
        "03_soil_fungi_phytophthora",
        "Unexpected_1.png",
        source_label="Unexpected",
        parent_image_id="Unexpected_1",
        image_role="full_field",
    )
    result = AdapterSet(raw).adapt(input_row)
    assert result["candidate_groups"] == []
    assert result["taxonomy"]["status"] == "unresolved"
    assert "unmapped_source_label" in result["import_flags"]


def test_download_hash_mismatch_fails(raw: Path) -> None:
    input_row = add_download(
        raw,
        "03_soil_fungi_phytophthora",
        "Fusarium_1.png",
        source_label="Fusarium",
        parent_image_id="Fusarium_1",
        image_role="full_field",
    )
    input_row["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Inventory differs"):
        AdapterSet(raw).adapt(input_row)


def test_removed_usda_collection_is_rejected(raw: Path) -> None:
    input_row = row(
        "04_usda_turfgrass_nematodes",
        "04_usda_turfgrass_nematodes/extracted/Nematodes/"
        "20x objective/Trichodorus/Trichodorus_20x_Obj_8.jpg",
    )
    with pytest.raises(ValueError, match="Unsupported collection"):
        AdapterSet(raw).adapt(input_row)


@pytest.mark.parametrize(
    "relative",
    ["/etc/passwd", "../outside.jpg", "01_tgfc/../../outside.jpg", "01_tgfc\\outside.jpg"],
)
def test_adapter_rejects_paths_outside_source_root(raw: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="Unsafe source path"):
        AdapterSet(raw).adapt(row("01_tgfc", relative))


def test_adapter_rejects_escaping_symlink(raw: Path, tmp_path_factory) -> None:
    outside = tmp_path_factory.mktemp("outside") / "image.jpg"
    outside.write_bytes(b"fixture")
    (raw / "01_tgfc/link.jpg").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes raw root"):
        AdapterSet(raw).adapt(row("01_tgfc", "01_tgfc/link.jpg"))
