"""Synthetic fixtures verify admission rules; they are not biological labels."""

import csv
import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image

from woodunnit.release import (
    ReleaseError,
    build_release,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from woodunnit.taxonomy import derive_taxonomy


def make_inputs(tmp_path, count=2):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    records = []
    for index in range(count):
        image_path = raw_root / f"{index}.png"
        Image.new("RGB", (8, 6), (index, 10, 20)).save(image_path)
        records.append(
            {
                "image_id": f"image-{index}",
                "source_id": f"source-{index}",
                "relative_path": image_path.name,
                "sha256": sha256_file(image_path),
                "width_px": 8,
                "height_px": 6,
                "source_labels": ["untrusted source label"],
                "candidate_groups": ["fungi"],
                "leakage_group_id": f"group-{index}",
                "review_status": "unreviewed",
                "group_label": None,
                "disposition": None,
                "lineage": {},
                "relationship_keys": [],
            }
        )
    save_catalog(catalog, records)
    reviews = tmp_path / "reviews.jsonl"
    write_jsonl(reviews, [review("image-0")])
    assignments = tmp_path / "assignments.csv"
    save_assignments(assignments, [("group-0", "train")])
    return catalog, raw_root, reviews, assignments, tmp_path / "release", "v1"


def save_catalog(path, records, schema_version="2.0"):
    write_jsonl(path / "images.jsonl", records)
    write_json(
        path / "catalog.json",
        {
            "schema_version": schema_version,
            "catalog_id": "catalog-synthetic",
            "task_groups": ["fungi", "oomycetes"],
            "file_sha256": {"images.jsonl": sha256_file(path / "images.jsonl")},
        },
    )


def save_assignments(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["leakage_group_id", "split"])
        writer.writerows(rows)


def review(image_id, **overrides):
    index = int(image_id.split("-")[1]) if image_id.startswith("image-") else 0
    image_bytes = BytesIO()
    Image.new("RGB", (8, 6), (index, 10, 20)).save(image_bytes, format="PNG")
    result = {
        "image_id": image_id,
        "annotation_version": 1,
        "review_status": "adjudicated",
        "image_sha256": hashlib.sha256(image_bytes.getvalue()).hexdigest(),
        "disposition": "eligible_single_group",
        "group_label": "fungi",
        "visible_evidence": "Synthetic test evidence, not a biological assessment.",
        "annotator_id": "synthetic-a",
        "reviewer_id": "synthetic-b",
        "usage_approved": True,
        "scope_approved": True,
        "usage_permission_ref": "synthetic-fixture-created-for-tests",
        "second_review_confirmed": True,
    }
    result.update(overrides)
    return result


def test_only_reviewed_labels_enter_release_and_absent_class_ids_stay_stable(tmp_path):
    args = make_inputs(tmp_path)
    # The highest numbered review wins even when the file is not version-sorted.
    write_jsonl(
        args[2],
        [
            review("image-0", annotation_version=3, group_label="oomycetes"),
            review("image-0", annotation_version=1),
        ],
    )
    summary = build_release(*args)
    rows = read_jsonl(args[4] / "manifest.jsonl")
    assert len(rows) == 1
    assert rows[0]["group_label"] == "oomycetes"
    assert rows[0]["class_id"] == 1
    assert rows[0]["annotation_version"] == 3
    assert summary["class_map"] == {"fungi": 0, "oomycetes": 1}
    assert summary["missing_classes"] == ["fungi"]
    assert summary["class_split_counts"]["fungi"]["test"] == 0
    assert summary["validation"]["diagnostic_validation"] is False
    assert summary["exclusion_reason_counts"] == {"unreviewed": 1}
    assert summary["input_sha256"]["reviews.jsonl"] == sha256_file(args[2])
    assert read_jsonl(args[4] / "exclusions.jsonl") == [
        {"image_id": "image-1", "reasons": ["unreviewed"]}
    ]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"usage_approved": False}, "usage_not_approved"),
        ({"scope_approved": False}, "scope_not_approved"),
        ({"usage_permission_ref": None}, "missing_usage_permission_ref"),
        ({"visible_evidence": " "}, "missing_visible_evidence"),
        ({"group_label": "nematodes"}, "unsupported_or_missing_group"),
        ({"group_label": "bacteria"}, "unsupported_or_missing_group"),
        ({"group_label": None}, "unsupported_or_missing_group"),
        ({"disposition": "mixed"}, "disposition:mixed"),
        ({"disposition": "unresolved"}, "disposition:unresolved"),
        ({"disposition": "no_target_visible"}, "disposition:no_target_visible"),
        ({"disposition": "unusable"}, "disposition:unusable"),
        ({"disposition": "out_of_scope"}, "disposition:out_of_scope"),
    ],
)
def test_eligibility_gates_never_promote_source_labels(tmp_path, changes, reason):
    args = make_inputs(tmp_path)
    write_jsonl(args[2], [review("image-0", **changes)])
    with pytest.raises(ReleaseError, match=reason):
        build_release(*args)
    assert not args[4].exists()


def test_later_review_can_revoke_previous_admission(tmp_path):
    args = make_inputs(tmp_path)
    write_jsonl(
        args[2],
        [
            review("image-0"),
            review("image-0", annotation_version=2, disposition="unresolved"),
        ],
    )
    with pytest.raises(ReleaseError, match="No eligible reviewed images"):
        build_release(*args)


def test_review_is_bound_to_image_bytes_not_only_stable_id(tmp_path):
    args = make_inputs(tmp_path)
    write_jsonl(args[2], [review("image-0", image_sha256="0" * 64)])
    with pytest.raises(ReleaseError, match="Review image_sha256 does not match"):
        build_release(*args)


def test_latest_review_can_replace_a_historical_review_of_old_bytes(tmp_path):
    args = make_inputs(tmp_path)
    write_jsonl(
        args[2],
        [
            review("image-0", image_sha256="0" * 64),
            review("image-0", annotation_version=2),
        ],
    )
    assert build_release(*args)["image_count"] == 1


@pytest.mark.parametrize(
    "reviews,match",
    [
        ([review("image-0"), review("image-0")], "Duplicate review version"),
        ([review("missing")], "Orphan review"),
        ([review("image-0", annotation_version=True)], "invalid review"),
        ([review("image-0", annotation_version=0)], "invalid review"),
        ([review("image-0", usage_approved="true")], "invalid review"),
        ([review("image-0", review_status="unreviewed")], "invalid review"),
        ([review("image-0", unknown_field=True)], "invalid review"),
    ],
)
def test_review_contract_rejects_invalid_or_ambiguous_records(tmp_path, reviews, match):
    args = make_inputs(tmp_path)
    write_jsonl(args[2], reviews)
    with pytest.raises(ReleaseError, match=match):
        build_release(*args)


def test_duplicate_json_keys_do_not_silently_override_reviews(tmp_path):
    args = make_inputs(tmp_path)
    text = json.dumps(review("image-0"))
    args[2].write_text(text[:-1] + ', "usage_approved": false}\n')
    with pytest.raises(ReleaseError, match="Duplicate JSON object key"):
        build_release(*args)


@pytest.mark.parametrize(
    "rows,match",
    [
        ([], "no split assignment"),
        ([("group-0", "train"), ("group-0", "test")], "Duplicate or conflicting"),
        ([("group-0", "train"), ("group-0", "train")], "Duplicate or conflicting"),
        ([("image-0", "train")], "Unknown leakage_group_id"),
        ([("group-0", "tune")], "Invalid split"),
        ([("group-0", "calibration")], "Invalid split"),
    ],
)
def test_assignments_require_whole_known_groups(tmp_path, rows, match):
    args = make_inputs(tmp_path)
    save_assignments(args[3], rows)
    with pytest.raises(ReleaseError, match=match):
        build_release(*args)


@pytest.mark.parametrize(
    "relation",
    [
        "source",
        "hash",
        "pixel_sha256",
        "lossless_transform_sha256",
        "canonical_image_id",
        "specimen",
        "parent",
        "key",
        "video_id",
        "isolate_id",
    ],
)
def test_rechecks_leakage_even_if_catalog_groups_were_changed(tmp_path, relation):
    args = make_inputs(tmp_path)
    records = read_jsonl(args[0] / "images.jsonl")
    if relation == "source":
        records[1]["source_id"] = records[0]["source_id"]
    elif relation == "hash":
        records[1]["sha256"] = records[0]["sha256"]
    elif relation in ("pixel_sha256", "lossless_transform_sha256"):
        records[0][relation] = "a" * 64
        records[1][relation] = "a" * 64
    elif relation == "canonical_image_id":
        records[1][relation] = records[0]["image_id"]
    elif relation == "specimen":
        records[0]["lineage"] = {"specimen_id": "shared"}
        records[1]["lineage"] = {"specimen_id": "shared"}
    elif relation == "parent":
        records[1]["lineage"] = {"parent_image_id": records[0]["image_id"]}
    elif relation in ("video_id", "isolate_id"):
        records[0]["lineage"] = {relation: "shared"}
        records[1]["lineage"] = {relation: "shared"}
    else:
        records[0]["relationship_keys"] = ["shared-crop-parent"]
        records[1]["relationship_keys"] = ["shared-crop-parent"]
    save_catalog(args[0], records)
    save_assignments(args[3], [("group-0", "train"), ("group-1", "test")])
    # Image 1 is deliberately unreviewed; it must still participate in leakage checks.
    with pytest.raises(ReleaseError, match="Split leakage"):
        build_release(*args)


@pytest.mark.parametrize(
    "changes",
    [
        {"second_review_confirmed": False},
        {"reviewer_id": "synthetic-a"},
    ],
)
def test_held_out_requires_recorded_second_review(tmp_path, changes):
    args = make_inputs(tmp_path)
    write_jsonl(args[2], [review("image-0", **changes)])
    save_assignments(args[3], [("group-0", "test")])
    with pytest.raises(ReleaseError, match="second_review_confirmed"):
        build_release(*args)


def test_conservative_source_group_can_be_assigned_to_validation(tmp_path):
    args = make_inputs(tmp_path)
    records = read_jsonl(args[0] / "images.jsonl")
    records[0]["import_flags"] = ["lineage_unrecoverable"]
    save_catalog(args[0], records)
    save_assignments(args[3], [("group-0", "validation")])
    result = build_release(*args)
    assert result["split_counts"]["validation"] == 1


def test_catalog_checksum_tampering_is_rejected(tmp_path):
    args = make_inputs(tmp_path)
    path = args[0] / "images.jsonl"
    path.write_text(path.read_text().replace("unreviewed", "tampered"))
    with pytest.raises(ReleaseError, match="Checksum mismatch"):
        build_release(*args)


def test_raw_image_tampering_is_rejected(tmp_path):
    args = make_inputs(tmp_path)
    Image.new("RGB", (8, 6), (250, 250, 250)).save(args[1] / "0.png")
    with pytest.raises(ReleaseError, match="Image checksum mismatch"):
        build_release(*args)


def test_wrong_catalog_dimensions_prevent_release(tmp_path):
    args = make_inputs(tmp_path)
    records = read_jsonl(args[0] / "images.jsonl")
    records[0]["width_px"] = 100
    save_catalog(args[0], records)
    with pytest.raises(ReleaseError, match="Image dimensions mismatch"):
        build_release(*args)


def test_multiframe_images_require_explicit_reviewed_still_extraction(tmp_path):
    args = make_inputs(tmp_path)
    animated_path = args[1] / "animated.gif"
    Image.new("RGB", (8, 6), "red").save(
        animated_path,
        save_all=True,
        append_images=[Image.new("RGB", (8, 6), "blue")],
    )
    records = read_jsonl(args[0] / "images.jsonl")
    records[0]["relative_path"] = animated_path.name
    records[0]["sha256"] = sha256_file(animated_path)
    save_catalog(args[0], records)
    write_jsonl(args[2], [review("image-0", image_sha256=records[0]["sha256"])])
    with pytest.raises(ReleaseError, match="Multiframe image requires"):
        build_release(*args)


@pytest.mark.parametrize("path", ["../escape.png", "/escape.png", "C:/escape.png", "."])
def test_unsafe_paths_are_rejected_even_on_excluded_images(tmp_path, path):
    args = make_inputs(tmp_path)
    records = read_jsonl(args[0] / "images.jsonl")
    records[1]["relative_path"] = path
    save_catalog(args[0], records)
    with pytest.raises(ReleaseError, match="Unsafe relative path"):
        build_release(*args)


def test_symlink_escape_is_rejected(tmp_path):
    args = make_inputs(tmp_path)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 6)).save(outside)
    link = args[1] / "escape.png"
    link.symlink_to(outside)
    records = read_jsonl(args[0] / "images.jsonl")
    records[1]["relative_path"] = "escape.png"
    save_catalog(args[0], records)
    with pytest.raises(ReleaseError, match="inside its configured root"):
        build_release(*args)


def test_existing_release_is_immutable(tmp_path):
    args = make_inputs(tmp_path)
    build_release(*args)
    before = (args[4] / "release.json").read_bytes()
    with pytest.raises(ReleaseError, match="already exists"):
        build_release(*args)
    assert (args[4] / "release.json").read_bytes() == before


@pytest.mark.parametrize("target", ["raw", "catalog", "raw_alias", "catalog_alias"])
def test_release_output_cannot_modify_raw_storage_or_immutable_catalog(tmp_path, target):
    args = list(make_inputs(tmp_path))
    root = args[1] if target.startswith("raw") else args[0]
    if target.endswith("alias"):
        alias = tmp_path / target
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    args[4] = root / "release"
    with pytest.raises(ReleaseError, match="outside the raw root and catalog"):
        build_release(*args)
    assert not args[4].exists()


def test_source_parent_identifiers_are_namespaced_by_collection(tmp_path):
    args = make_inputs(tmp_path)
    records = read_jsonl(args[0] / "images.jsonl")
    for record in records:
        record["lineage"] = {"source_parent_id": "local-parent-1"}
    save_catalog(args[0], records)
    save_assignments(args[3], [("group-0", "train"), ("group-1", "test")])
    assert build_release(*args)["image_count"] == 1


def test_failed_export_removes_staging_directory(tmp_path, monkeypatch):
    args = make_inputs(tmp_path)

    def fail(*_args):
        raise OSError("synthetic write failure")

    monkeypatch.setattr("woodunnit.release.write_jsonl", fail)
    with pytest.raises(OSError, match="synthetic write failure"):
        build_release(*args)
    assert not args[4].exists()
    assert not list(tmp_path.glob(".release.staging-*"))


def test_release_bound_files_match_output(tmp_path):
    args = make_inputs(tmp_path)
    result = build_release(*args)
    for name, expected in result["file_sha256"].items():
        assert sha256_file(args[4] / name) == expected
    assert json.loads((args[4] / "label_map.json").read_text()) == result["class_map"]


@pytest.mark.parametrize(
    "reason",
    ["exact_file_duplicate", "exact_pixel_duplicate", "lossless_transform_duplicate"],
)
def test_redundant_images_stay_excluded_even_after_an_eligible_review(tmp_path, reason):
    args = make_inputs(tmp_path)
    records = read_jsonl(args[0] / "images.jsonl")
    records[1].update(
        selection_status="excluded_redundant",
        exclusion_reason=reason,
        canonical_image_id="image-0",
    )
    save_catalog(args[0], records)
    write_jsonl(args[2], [review("image-0"), review("image-1")])
    summary = build_release(*args)
    assert [row["image_id"] for row in read_jsonl(args[4] / "manifest.jsonl")] == ["image-0"]
    assert read_jsonl(args[4] / "exclusions.jsonl") == [
        {"image_id": "image-1", "reasons": [f"redundant:{reason}"]}
    ]
    assert summary["exclusion_reason_counts"] == {f"redundant:{reason}": 1}


@pytest.mark.parametrize(
    "duplicate_relation", ["sha256", "pixel_sha256", "lossless_transform_sha256", "canonical"]
)
def test_excluded_duplicate_bridges_remain_in_split_checks(tmp_path, duplicate_relation):
    args = make_inputs(tmp_path, count=3)
    records = read_jsonl(args[0] / "images.jsonl")
    records[1].update(
        selection_status="excluded_redundant",
        exclusion_reason="exact_file_duplicate",
        canonical_image_id="image-0",
        relationship_keys=["bridge-to-third-image"],
    )
    records[2]["relationship_keys"] = ["bridge-to-third-image"]
    if duplicate_relation != "canonical":
        records[0][duplicate_relation] = records[0]["sha256"]
        records[1][duplicate_relation] = records[0]["sha256"]
        # Make the canonical reference point to image 2, so this test relies on
        # the byte/pixel/transform relation to connect image 0 to the bridge.
        records[1]["canonical_image_id"] = "image-2"
    save_catalog(args[0], records)
    save_assignments(args[3], [("group-0", "train"), ("group-2", "test")])
    with pytest.raises(ReleaseError, match="Split leakage"):
        build_release(*args)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"selection_status": "maybe"}, "Invalid selection_status"),
        ({"selection_status": "included", "exclusion_reason": "bad"}, "Included image"),
        (
            {"selection_status": "excluded_redundant", "exclusion_reason": None},
            "Invalid redundant exclusion_reason",
        ),
        (
            {"selection_status": "excluded_redundant", "exclusion_reason": "exact_file_duplicate"},
            "needs a different canonical_image_id",
        ),
        ({"pixel_sha256": "not-a-sha256"}, "Invalid pixel_sha256"),
        ({"lossless_transform_sha256": 123}, "Invalid lossless_transform_sha256"),
        ({"canonical_image_id": " "}, "Invalid canonical_image_id"),
        ({"canonical_image_id": "unknown"}, "Unknown canonical_image_id"),
    ],
)
def test_curation_metadata_is_validated_before_export(tmp_path, changes, match):
    args = make_inputs(tmp_path)
    records = read_jsonl(args[0] / "images.jsonl")
    records[1].update(changes)
    save_catalog(args[0], records)
    with pytest.raises(ReleaseError, match=match):
        build_release(*args)


def test_canonical_references_cannot_end_at_an_excluded_record(tmp_path):
    args = make_inputs(tmp_path, count=3)
    records = read_jsonl(args[0] / "images.jsonl")
    for index in (1, 2):
        records[index].update(
            selection_status="excluded_redundant",
            exclusion_reason="exact_pixel_duplicate",
            canonical_image_id=f"image-{index - 1}",
        )
    save_catalog(args[0], records)
    with pytest.raises(ReleaseError, match="must reference an included image"):
        build_release(*args)


@pytest.mark.parametrize("groups", [["oomycetes", "fungi"], ["fungi"], ["fungi", "bacteria"]])
def test_binary_class_order_cannot_change_silently(tmp_path, groups):
    args = make_inputs(tmp_path)
    path = args[0] / "catalog.json"
    metadata = json.loads(path.read_text())
    metadata["task_groups"] = groups
    write_json(path, metadata)
    with pytest.raises(ReleaseError, match="stable order"):
        build_release(*args)


def test_old_catalog_schema_is_rejected_without_rewriting_it(tmp_path):
    args = make_inputs(tmp_path)
    path = args[0] / "catalog.json"
    metadata = json.loads(path.read_text())
    metadata["schema_version"] = "1.0"
    write_json(path, metadata)
    before = path.read_bytes()
    with pytest.raises(ReleaseError, match="Unsupported catalog schema_version"):
        build_release(*args)
    assert path.read_bytes() == before
    assert not args[4].exists()


def test_validation_partition_supports_a_single_recorded_review(tmp_path):
    args = make_inputs(tmp_path)
    write_jsonl(args[2], [review("image-0", second_review_confirmed=False)])
    save_assignments(args[3], [("group-0", "validation")])
    result = build_release(*args)
    assert result["schema_version"] == "2.0"
    assert result["split_counts"] == {"train": 0, "validation": 1, "test": 0}


def annotated_record(args):
    record = read_jsonl(args[0] / "images.jsonl")[0]
    record.update(
        schema_version="3.0",
        source_id="tgfc",
        source_labels=["Colletotrichum siamense"],
        annotations=[
            {
                "label": "Colletotrichum siamense",
                "class_id": 0,
                "format": "yolo_cxcywh_normalized",
                "coordinates": [0.5, 0.5, 0.25, 0.25],
            }
        ],
        source_metadata={"annotation_text": "0 0.5 0.5 0.25 0.25\n"},
    )
    record["taxonomy"] = derive_taxonomy(
        record["source_id"], record["source_labels"], record["candidate_groups"]
    )
    return record


def test_v3_annotations_roundtrip_without_promoting_fine_labels_to_reviewed_targets(tmp_path):
    args = make_inputs(tmp_path, count=1)
    record = annotated_record(args)
    save_catalog(args[0], [record], schema_version="3.0")
    # Deliberate source/review disagreement proves the review remains authoritative
    # for its broad target only; imported species and boxes stay unreviewed.
    write_jsonl(args[2], [review("image-0", group_label="oomycetes")])
    result = build_release(*args)
    row = read_jsonl(args[4] / "manifest.jsonl")[0]
    assert result["schema_version"] == "2.0"
    assert result["catalog_schema_version"] == "3.0"
    assert row["group_label"] == "oomycetes"
    assert row["class_id"] == 1
    assert row["source_annotation_origin"] == "publisher_mapping_unreviewed"
    assert row["taxonomy"]["species"] == "Colletotrichum siamense"
    with (args[4] / "manifest.csv").open(newline="") as stream:
        csv_row = next(csv.DictReader(stream))
    for field in ("source_labels", "taxonomy", "annotations", "source_metadata"):
        assert row[field] == record[field]
        assert json.loads(csv_row[field]) == record[field]
    assert csv_row["source_annotation_origin"] == "publisher_mapping_unreviewed"
    assert "taxonomy" not in read_jsonl(args[4] / "reviews.jsonl")[0]
    assert read_jsonl(args[0] / "images.jsonl") == [record]


@pytest.mark.parametrize("labels", [["Neopestalotiopsis sp."], []])
def test_legacy_catalog_derives_only_supported_ranks_and_keeps_inputs_intact(tmp_path, labels):
    args = make_inputs(tmp_path, count=1)
    record = read_jsonl(args[0] / "images.jsonl")[0]
    record["source_id"] = "tgfc"
    record["source_labels"] = labels
    save_catalog(args[0], [record])
    before = (args[0] / "images.jsonl").read_bytes()
    build_release(*args)
    row = read_jsonl(args[4] / "manifest.jsonl")[0]
    assert row["taxonomy"]["genus"] == ("Neopestalotiopsis" if labels else None)
    assert row["taxonomy"]["species"] is None
    assert row["source_labels"] == labels
    assert row["annotations"] == []
    assert (args[0] / "images.jsonl").read_bytes() == before


@pytest.mark.parametrize("change", ["missing", "different_labels", "invalid_consensus"])
def test_v3_taxonomy_must_be_present_and_match_source_labels(tmp_path, change):
    args = make_inputs(tmp_path, count=1)
    record = annotated_record(args)
    if change == "missing":
        del record["taxonomy"]
    elif change == "different_labels":
        record["source_labels"] = ["Olivea tectonae"]
    else:
        record["taxonomy"]["genus"] = "Olivea"
    save_catalog(args[0], [record], schema_version="3.0")
    with pytest.raises(ReleaseError, match="Invalid source taxonomy"):
        build_release(*args)
    assert not args[4].exists()


def test_catalog_record_version_cannot_disagree_with_catalog_version(tmp_path):
    args = make_inputs(tmp_path, count=1)
    save_catalog(args[0], [annotated_record(args)], schema_version="2.0")
    with pytest.raises(ReleaseError, match="schema_version disagrees"):
        build_release(*args)
