"""Freeze an explicitly reviewed, group-assigned research dataset release.

Source labels are never training labels. This module performs no model fitting,
image transforms, or automatic biological annotation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .schema import SCHEMA_VERSION as CATALOG_SCHEMA_VERSION
from .schema import Taxonomy
from .taxonomy import derive_taxonomy, taxonomy_consensus

SCHEMA_VERSION = "2.0"
CATALOG_SCHEMA_VERSIONS = ("2.0", CATALOG_SCHEMA_VERSION)
SOURCE_ANNOTATION_ORIGIN = "publisher_mapping_unreviewed"
SOURCE_JSON_COLUMNS = ("source_labels", "taxonomy", "annotations", "source_metadata")
SPLITS = ("train", "validation", "test")
TARGET_GROUPS = ("fungi", "oomycetes")
REDUNDANCY_REASONS = (
    "exact_file_duplicate",
    "exact_pixel_duplicate",
    "lossless_transform_duplicate",
)
MANIFEST_COLUMNS = (
    "image_id",
    "source_id",
    "relative_path",
    "sha256",
    "width_px",
    "height_px",
    "group_label",
    "class_id",
    "leakage_group_id",
    "split",
    "annotation_version",
    "source_annotation_origin",
    *SOURCE_JSON_COLUMNS,
)


class ReleaseError(ValueError):
    """An input does not satisfy the reviewed-release contract."""


class Review(BaseModel):
    """An append-only expert review; booleans require explicit JSON booleans."""

    model_config = ConfigDict(extra="forbid", strict=True)

    image_id: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_version: int = Field(gt=0)
    review_status: Literal["adjudicated"]
    disposition: Literal[
        "eligible_single_group",
        "mixed",
        "unresolved",
        "no_target_visible",
        "unusable",
        "out_of_scope",
    ]
    group_label: str | None
    visible_evidence: str
    annotator_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    usage_approved: bool
    scope_approved: bool
    usage_permission_ref: str | None = None
    supporting_evidence_ref: str | None = None
    second_review_confirmed: bool = False
    quality_flags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("image_id", "annotator_id", "reviewer_id")
    @classmethod
    def nonblank_identifier(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("identifier must be nonblank and have no outer whitespace")
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative_path: str) -> Path:
    """Resolve a portable relative POSIX path without following an escape."""
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise ReleaseError(f"Invalid relative path: {relative_path!r}")
    relative = PurePosixPath(relative_path)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or ":" in relative.parts[0]
        or "\x00" in relative_path
        or relative.as_posix() != relative_path
    ):
        raise ReleaseError(f"Unsafe relative path: {relative_path!r}")
    root = root.resolve(strict=True)
    resolved = (root / relative_path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ReleaseError(f"Path is not a file inside its configured root: {relative_path}")
    return resolved


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ReleaseError(f"{path.name}:{line_number}: blank JSONL record")
            try:
                record = json.loads(line, object_pairs_hook=_unique_json_keys)
            except json.JSONDecodeError as exc:
                raise ReleaseError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ReleaseError(f"{path.name}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _unique_json_keys(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def verify_bound_files(directory: Path, metadata: dict, required_file: str) -> None:
    hashes = metadata.get("file_sha256")
    if not isinstance(hashes, dict) or required_file not in hashes:
        raise ReleaseError(f"Metadata must bind {required_file} in file_sha256")
    for filename, expected in hashes.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ReleaseError(f"Invalid SHA-256 for {filename}")
        actual = sha256_file(safe_path(directory, filename))
        if actual != expected:
            raise ReleaseError(f"Checksum mismatch: {filename}")


def _read_catalog(catalog_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((catalog_dir / "catalog.json").read_text(encoding="utf-8"))
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") not in CATALOG_SCHEMA_VERSIONS
    ):
        raise ReleaseError("Unsupported catalog schema_version")
    if not isinstance(summary.get("catalog_id"), str) or not summary["catalog_id"]:
        raise ReleaseError("Catalog must have a nonempty catalog_id")
    groups = summary.get("task_groups")
    if groups != list(TARGET_GROUPS):
        raise ReleaseError("Catalog task_groups must be ['fungi', 'oomycetes'] in stable order")
    verify_bound_files(catalog_dir, summary, "images.jsonl")
    records = read_jsonl(catalog_dir / "images.jsonl")
    image_ids = set()
    paths = set()
    for record in records:
        if record.get("schema_version", summary["schema_version"]) != summary["schema_version"]:
            raise ReleaseError("Catalog image schema_version disagrees with catalog metadata")
        for field in ("image_id", "source_id", "relative_path", "sha256", "leakage_group_id"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ReleaseError(f"Catalog image requires nonempty {field}")
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ReleaseError(f"Invalid image checksum: {record['image_id']}")
        for hash_field in ("pixel_sha256", "lossless_transform_sha256"):
            value = record.get(hash_field)
            if value is not None and (
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            ):
                raise ReleaseError(f"Invalid {hash_field}: {record['image_id']}")
        canonical_id = record.get("canonical_image_id")
        if canonical_id is not None and (
            not isinstance(canonical_id, str)
            or not canonical_id.strip()
            or canonical_id != canonical_id.strip()
        ):
            raise ReleaseError(f"Invalid canonical_image_id: {record['image_id']}")
        status = record.get("selection_status", "included")
        reason = record.get("exclusion_reason")
        if status not in ("included", "excluded_redundant"):
            raise ReleaseError(f"Invalid selection_status: {record['image_id']}")
        if status == "excluded_redundant":
            if reason not in REDUNDANCY_REASONS:
                raise ReleaseError(f"Invalid redundant exclusion_reason: {record['image_id']}")
            if canonical_id is None or canonical_id == record["image_id"]:
                raise ReleaseError(
                    f"Excluded redundant image needs a different canonical_image_id: "
                    f"{record['image_id']}"
                )
        elif reason is not None:
            raise ReleaseError(f"Included image cannot have exclusion_reason: {record['image_id']}")
        if record["image_id"] in image_ids or record["relative_path"] in paths:
            raise ReleaseError(f"Duplicate catalog image ID or path: {record['image_id']}")
        image_ids.add(record["image_id"])
        paths.add(record["relative_path"])
        for field in ("width_px", "height_px"):
            if type(record.get(field)) is not int or record[field] <= 0:
                raise ReleaseError(f"Invalid {field}: {record['image_id']}")
        record.update(_source_annotations(record, summary["schema_version"]))
    by_image_id = {record["image_id"]: record for record in records}
    for record in records:
        canonical_id = record.get("canonical_image_id")
        if canonical_id is not None and canonical_id not in image_ids:
            raise ReleaseError(f"Unknown canonical_image_id: {record['image_id']}")
        if (
            canonical_id is not None
            and by_image_id[canonical_id].get("selection_status") == "excluded_redundant"
        ):
            raise ReleaseError(
                f"canonical_image_id must reference an included image: {canonical_id}"
            )
    return summary, records


def _source_annotations(record: dict, catalog_version: str) -> dict:
    """Preserve publisher evidence separately from the adjudicated broad target."""
    labels = record.get("source_labels", [])
    annotations = record.get("annotations", [])
    metadata = record.get("source_metadata", {})
    groups = record.get("candidate_groups", [])
    if (
        not isinstance(labels, list)
        or not all(isinstance(label, str) and label for label in labels)
        or len(set(labels)) != len(labels)
        or not isinstance(groups, list)
        or not all(isinstance(group, str) and group in TARGET_GROUPS for group in groups)
        or len(set(groups)) != len(groups)
        or not isinstance(annotations, list)
        or not all(isinstance(annotation, dict) for annotation in annotations)
        or not isinstance(metadata, dict)
    ):
        raise ReleaseError(f"Invalid source annotation data: {record['image_id']}")
    try:
        if record["source_id"] in {"tgfc", "idphy", "soil"}:
            taxonomy = derive_taxonomy(record["source_id"], labels, groups)
        else:
            # Legacy catalogs can name other collections. Keep their labels without
            # inventing a taxonomic interpretation or borrowing the reviewed target.
            taxonomy = taxonomy_consensus(
                [
                    {
                        "source_label": label,
                        "group": None,
                        "genus": None,
                        "species": None,
                        "rank": "unresolved",
                        "mapping_note": "Legacy source has no supported taxonomy mapping.",
                    }
                    for label in sorted(labels)
                ]
            )
        if catalog_version == CATALOG_SCHEMA_VERSION:
            if record["source_id"] not in {"tgfc", "idphy", "soil"}:
                raise ValueError("Unsupported taxonomy source")
            supplied = Taxonomy.model_validate(record.get("taxonomy")).model_dump()
            if supplied != taxonomy:
                raise ValueError("Taxonomy conflicts with source labels")
    except ValueError as exc:
        raise ReleaseError(f"Invalid source taxonomy: {record['image_id']}: {exc}") from exc
    return {
        "source_annotation_origin": SOURCE_ANNOTATION_ORIGIN,
        "source_labels": labels,
        "taxonomy": taxonomy,
        "annotations": annotations,
        "source_metadata": metadata,
    }


def _latest_reviews(path: Path, image_ids: set[str]) -> dict[str, Review]:
    latest = {}
    seen = set()
    for line_number, row in enumerate(read_jsonl(path), 1):
        try:
            review = Review.model_validate(row)
        except ValidationError as exc:
            raise ReleaseError(f"{path.name}:{line_number}: invalid review: {exc}") from exc
        if review.image_id not in image_ids:
            raise ReleaseError(f"Orphan review image_id: {review.image_id}")
        key = (review.image_id, review.annotation_version)
        if key in seen:
            raise ReleaseError(f"Duplicate review version: {key}")
        seen.add(key)
        previous = latest.get(review.image_id)
        if previous is None or review.annotation_version > previous.annotation_version:
            latest[review.image_id] = review
    return latest


def _assignments(path: Path, known_groups: set[str]) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["leakage_group_id", "split"]:
            raise ReleaseError("Assignments CSV headers must be leakage_group_id,split")
        for line_number, row in enumerate(reader, 2):
            if None in row or None in row.values():
                raise ReleaseError(f"Assignments line {line_number}: malformed CSV row")
            group, split = row["leakage_group_id"], row["split"]
            if group not in known_groups:
                raise ReleaseError(f"Unknown leakage_group_id in assignments: {group}")
            if group in result:
                raise ReleaseError(f"Duplicate or conflicting assignment for group: {group}")
            if split not in SPLITS:
                raise ReleaseError(f"Invalid split {split!r}; use one of {SPLITS}")
            result[group] = split
    return result


def _relation_tokens(record: dict) -> set[tuple]:
    # A source-wide token preserves the deliberately conservative public-catalog policy.
    tokens = {("source", record["source_id"]), ("sha256", record["sha256"])}
    if record.get("pixel_sha256") is not None:
        tokens.add(("pixel_sha256", record["pixel_sha256"]))
    if record.get("lossless_transform_sha256") is not None:
        tokens.add(("lossless_transform_sha256", record["lossless_transform_sha256"]))
    # Include each image's own ID so a reference to a canonical image joins it too.
    tokens.add(("canonical_image_id", record["image_id"]))
    if record.get("canonical_image_id") is not None:
        tokens.add(("canonical_image_id", record["canonical_image_id"]))
    relationship_keys = record.get("relationship_keys", [])
    if not isinstance(relationship_keys, list):
        raise ReleaseError(f"Invalid relationship_keys for {record['image_id']}")
    for key in relationship_keys:
        if not isinstance(key, str) or not key:
            raise ReleaseError(f"Invalid relationship key for {record['image_id']}")
        tokens.add(("relationship", key))
    lineage = record.get("lineage", {})
    if not isinstance(lineage, dict):
        raise ReleaseError(f"Invalid lineage for {record['image_id']}")
    for field in (
        "tree_id",
        "case_id",
        "specimen_id",
        "slide_id",
        "parent_image_id",
        "video_id",
        "isolate_id",
        "source_parent_id",
    ):
        value = lineage.get(field) or record.get(field)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ReleaseError(f"Invalid {field} for {record['image_id']}")
            if field == "source_parent_id":
                tokens.add((field, record["source_id"], value))
            else:
                tokens.add((field, value))
    # A parent ID and the parent's own ID must share a token.
    tokens.add(("parent_image_id", record["image_id"]))
    return tokens


def _guard_leakage(records: list[dict], assignments: dict[str, str]) -> None:
    # Union all catalog groups, including ineligible images which can bridge two groups.
    parents = {record["leakage_group_id"]: record["leakage_group_id"] for record in records}

    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    first_seen = {}
    for record in records:
        group = record["leakage_group_id"]
        for token in _relation_tokens(record):
            if token in first_seen:
                left, right = find(group), find(first_seen[token])
                if left != right:
                    parents[right] = left
            else:
                first_seen[token] = group
    component_splits = {}
    for group, split in assignments.items():
        component = find(group)
        previous = component_splits.setdefault(component, split)
        if previous != split:
            raise ReleaseError(
                f"Split leakage: related source material or duplicate images cross partitions "
                f"at {group}. Rebuild or regroup the catalog before exporting."
            )


def _exclusion_reasons(review: Review | None, task_groups: list[str]) -> list[str]:
    if review is None:
        return ["unreviewed"]
    reasons = []
    if review.disposition != "eligible_single_group":
        reasons.append(f"disposition:{review.disposition}")
    if review.group_label not in task_groups:
        reasons.append("unsupported_or_missing_group")
    if not review.visible_evidence.strip():
        reasons.append("missing_visible_evidence")
    if not review.usage_approved:
        reasons.append("usage_not_approved")
    if not review.usage_permission_ref or not review.usage_permission_ref.strip():
        reasons.append("missing_usage_permission_ref")
    if not review.scope_approved:
        reasons.append("scope_not_approved")
    return reasons


def build_release(
    catalog_dir: Path,
    raw_root: Path,
    reviews_path: Path,
    assignments_path: Path,
    output_dir: Path,
    release_version: str,
) -> dict:
    """Validate and atomically create a reviewed release; never overwrite one.

    A release can be incomplete in class or split coverage. Such gaps are reported,
    never repaired by promoting source labels or dividing biological groups.
    """
    catalog_dir, raw_root = Path(catalog_dir).resolve(), Path(raw_root).resolve(strict=True)
    reviews_path, assignments_path = Path(reviews_path), Path(assignments_path)
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(raw_root) or output_dir.is_relative_to(catalog_dir):
        raise ReleaseError("Release output must be outside the raw root and catalog directory")
    if output_dir.exists() or output_dir.is_symlink():
        raise ReleaseError(f"Release output already exists: {output_dir}")
    if not release_version or release_version != release_version.strip():
        raise ReleaseError("release_version must be a nonempty, explicit version")
    summary, records = _read_catalog(catalog_dir)
    input_hashes = {
        "catalog.json": sha256_file(catalog_dir / "catalog.json"),
        "reviews.jsonl": sha256_file(reviews_path),
        "assignments.csv": sha256_file(assignments_path),
    }
    latest = _latest_reviews(reviews_path, {row["image_id"] for row in records})
    for record in records:
        review = latest.get(record["image_id"])
        if review is not None and review.image_sha256 != record["sha256"]:
            raise ReleaseError(
                f"Review image_sha256 does not match catalog image: {record['image_id']}; "
                "review the current image bytes before releasing"
            )
    assignments = _assignments(assignments_path, {row["leakage_group_id"] for row in records})
    _guard_leakage(records, assignments)
    class_map = {group: index for index, group in enumerate(summary["task_groups"])}
    manifest, exclusions, selected_reviews = [], [], []
    for record in sorted(records, key=lambda row: row["image_id"]):
        image_path = safe_path(raw_root, record["relative_path"])
        review = latest.get(record["image_id"])
        reasons = _exclusion_reasons(review, summary["task_groups"])
        if record.get("selection_status") == "excluded_redundant":
            reasons.insert(0, f"redundant:{record['exclusion_reason']}")
        if reasons:
            exclusions.append({"image_id": record["image_id"], "reasons": reasons})
            continue
        group = record["leakage_group_id"]
        if group not in assignments:
            raise ReleaseError(f"Eligible image group has no split assignment: {group}")
        split = assignments[group]
        if split == "test" and (
            not review.second_review_confirmed or review.annotator_id == review.reviewer_id
        ):
            raise ReleaseError(
                f"{record['image_id']}: {split} requires second_review_confirmed=true "
                "and distinct annotator_id/reviewer_id"
            )
        if sha256_file(image_path) != record["sha256"]:
            raise ReleaseError(f"Image checksum mismatch: {record['image_id']}")
        try:
            with Image.open(image_path) as original:
                if original.size != (record["width_px"], record["height_px"]):
                    raise ReleaseError(f"Image dimensions mismatch: {record['image_id']}")
                if getattr(original, "n_frames", 1) != 1:
                    raise ReleaseError(
                        f"Multiframe image requires an explicitly extracted, reviewed frame: "
                        f"{record['image_id']}"
                    )
                original.load()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ReleaseError(f"Image cannot be decoded: {record['image_id']}: {exc}") from exc
        row = {field: record[field] for field in MANIFEST_COLUMNS[:6]}
        row.update(
            {
                "group_label": review.group_label,
                "class_id": class_map[review.group_label],
                "leakage_group_id": group,
                "split": split,
                "annotation_version": review.annotation_version,
                "source_annotation_origin": SOURCE_ANNOTATION_ORIGIN,
                **{field: record[field] for field in SOURCE_JSON_COLUMNS},
            }
        )
        manifest.append(row)
        selected_reviews.append(review.model_dump())
    if not manifest:
        counts = Counter(reason for row in exclusions for reason in row["reasons"])
        raise ReleaseError(
            "No eligible reviewed images; no model-ready release was written. "
            "Supply adjudicated image-level evidence, supported-group scope approval, "
            f"documented usage approval, and group split assignments. Exclusions: {dict(counts)}"
        )

    # Recheck inputs after validation, before freezing copies (detect concurrent edits).
    if (
        sha256_file(catalog_dir / "catalog.json") != input_hashes["catalog.json"]
        or sha256_file(reviews_path) != input_hashes["reviews.jsonl"]
        or sha256_file(assignments_path) != input_hashes["assignments.csv"]
    ):
        raise ReleaseError("Inputs changed during release validation; retry with frozen inputs")
    verify_bound_files(catalog_dir, summary, "images.jsonl")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        write_jsonl(staging / "manifest.jsonl", manifest)
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in manifest:
                writer.writerow(
                    {
                        **row,
                        **{
                            field: json.dumps(row[field], ensure_ascii=False, sort_keys=True)
                            for field in SOURCE_JSON_COLUMNS
                        },
                    }
                )
        write_jsonl(staging / "reviews.jsonl", selected_reviews)
        write_jsonl(staging / "exclusions.jsonl", exclusions)
        shutil.copyfile(assignments_path, staging / "split_assignments.csv")
        write_json(staging / "label_map.json", class_map)
        by_class = Counter(row["group_label"] for row in manifest)
        by_split = Counter(row["split"] for row in manifest)
        by_source = Counter(row["source_id"] for row in manifest)
        release = {
            "schema_version": SCHEMA_VERSION,
            "kind": "reviewed_research_release",
            "status": "approved_for_research",
            "release_version": release_version,
            "created_at": datetime.now(UTC).isoformat(),
            "catalog_id": summary["catalog_id"],
            "catalog_schema_version": summary["schema_version"],
            "source_annotation_origin": SOURCE_ANNOTATION_ORIGIN,
            "class_map": class_map,
            "image_count": len(manifest),
            "excluded_image_count": len(exclusions),
            "leakage_group_count": len({row["leakage_group_id"] for row in manifest}),
            "class_counts": {group: by_class[group] for group in class_map},
            "split_counts": {split: by_split[split] for split in SPLITS},
            "source_counts": dict(sorted(by_source.items())),
            "class_split_counts": {
                group: {
                    split: sum(
                        row["group_label"] == group and row["split"] == split for row in manifest
                    )
                    for split in SPLITS
                }
                for group in class_map
            },
            "missing_classes": [group for group in class_map if not by_class[group]],
            "missing_splits": [split for split in SPLITS if not by_split[split]],
            "exclusion_reason_counts": dict(
                Counter(reason for row in exclusions for reason in row["reasons"])
            ),
            "input_sha256": input_hashes,
            "split_version": "sha256:" + input_hashes["assignments.csv"],
            "review_snapshot_version": "sha256:" + input_hashes["reviews.jsonl"],
            "file_sha256": {path.name: sha256_file(path) for path in sorted(staging.iterdir())},
            "image_policy": {
                "originals_preserved": True,
                "color_decode": "RGB",
                "orientation": "stored_pixel_order_no_exif_transpose",
                "frame_policy": "single_frame_images_only",
                "resize": None,
                "normalization": None,
                "augmentation": None,
            },
            "validation": {
                "catalog_and_image_checksums_verified": True,
                "image_decode_and_dimensions_verified": True,
                "known_relationship_split_check": "passed",
                "near_duplicate_audit_completed": False,
                "diagnostic_validation": False,
                "model_trained": False,
            },
            "limitations": [
                "Research manifest; no diagnostic performance or validity is established.",
                "Unknown biological lineage is conservatively grouped by entire source.",
                "Source identity and capture protocol can be confounded with organism group.",
                "Exact byte/pixel hashes and known links do not rule out visual near-duplicates.",
                "Confidence calibration and deployment thresholds are deferred.",
                "Review approvals are assertions; software cannot verify expertise or rights.",
                "Missing classes and partitions must be resolved before corresponding evaluation.",
            ],
        }
        write_json(staging / "release.json", release)
        if output_dir.exists() or output_dir.is_symlink():
            raise ReleaseError(f"Release output appeared during validation: {output_dir}")
        os.rename(staging, output_dir)
        return release
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
