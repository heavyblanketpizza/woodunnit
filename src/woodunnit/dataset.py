"""Readers for reviewed releases and explicitly unreviewed hierarchical splits."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from .release import (
    SCHEMA_VERSION,
    SPLITS,
    TARGET_GROUPS,
    ReleaseError,
    _relation_tokens,
    read_jsonl,
    safe_path,
    sha256_file,
    verify_bound_files,
)
from .taxonomy import RANKS, build_taxonomy_map, derive_taxonomy


def _validate_hierarchical_groups(directory: Path, rows: list[dict], metadata: dict) -> None:
    """Check frozen full-catalog membership and relationships visible in the manifest.

    Excluded image IDs resolve through the bound group inventory. Their original
    metadata is not copied here, so this does not repeat the complete catalog audit.
    """
    memberships, eligible, group_ids = {}, set(), set()
    for group in read_jsonl(directory / "groups.jsonl"):
        group_id = group.get("experiment_group_id")
        members, selected = group.get("catalog_image_ids"), group.get("eligible_image_ids")
        partition = group.get("split")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in group_ids
            or not isinstance(members, list)
            or not members
            or not all(isinstance(item, str) and item for item in members)
            or len(set(members)) != len(members)
            or not isinstance(selected, list)
            or not all(isinstance(item, str) and item for item in selected)
            or len(set(selected)) != len(selected)
            or not set(selected).issubset(members)
            or (partition not in SPLITS if selected else partition is not None)
        ):
            raise ReleaseError("Invalid hierarchical group inventory")
        group_ids.add(group_id)
        for image_id in members:
            if image_id in memberships:
                raise ReleaseError("Repeated catalog image in hierarchical group inventory")
            memberships[image_id] = (group_id, partition)
        eligible.update(selected)
    if (
        len(memberships) != metadata.get("catalog_images")
        or len(group_ids) != metadata.get("catalog_observed_groups")
        or eligible != {row["image_id"] for row in rows}
    ):
        raise ReleaseError("Hierarchical group inventory coverage mismatch")
    tokens = {
        (kind, image_id): partition
        for image_id, (_, partition) in memberships.items()
        for kind in ("canonical_image_id", "parent_image_id")
    }
    for row in rows:
        group_id, partition = memberships[row["image_id"]]
        if (
            row["experiment_group_id"] != group_id
            or row.get("leakage_group_id") != group_id
            or row["split"] != partition
        ):
            raise ReleaseError("Manifest group or split differs from frozen group inventory")
        for field in ("canonical_image_id", "parent_image_id"):
            value = row.get("lineage", {}).get(field) or row.get(field)
            if value is not None and value not in memberships:
                raise ReleaseError("Hierarchical relationship references an unknown catalog image")
        for token in _relation_tokens(row):
            # Exploratory groups retain observed relationships, without the reviewed
            # export's deliberately source-wide grouping rule.
            if token[0] == "source":
                continue
            if token in tokens and tokens[token] != partition:
                raise ReleaseError("Known image relationships cross hierarchical partitions")
            tokens[token] = partition


class ManifestDataset:
    """Return ``(RGB image or transform result, stable integer class ID)``.

    Reads only a bound reviewed release, never a raw ingestion catalog. Pixels
    retain stored orientation and size. A future experiment supplies transforms.
    ``verify_hash=False`` skips per-access image hashes only; release metadata and
    manifest checksums remain mandatory.
    """

    def __init__(
        self,
        manifest_path: Path,
        raw_root: Path,
        split: str | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        verify_hash: bool = True,
    ) -> None:
        path = Path(manifest_path)
        if path.is_dir():
            path = path / "manifest.jsonl"
        if path.name != "manifest.jsonl":
            raise ReleaseError("Use a reviewed release directory or its manifest.jsonl")
        self.raw_root = Path(raw_root).resolve(strict=True)
        self.transform = transform
        self.verify_hash = verify_hash
        self.manifest_path = path
        release_path = path.parent / "release.json"
        if not release_path.is_file():
            raise ReleaseError("Missing release.json; raw catalogs are not model-ready releases")
        release = json.loads(release_path.read_text(encoding="utf-8"))
        if (
            not isinstance(release, dict)
            or release.get("schema_version") != SCHEMA_VERSION
            or release.get("kind") != "reviewed_research_release"
            or release.get("status") != "approved_for_research"
        ):
            raise ReleaseError("Unsupported or unapproved release metadata")
        verify_bound_files(path.parent, release, "manifest.jsonl")
        if split is not None and split not in SPLITS:
            raise ReleaseError(f"Invalid split: {split}")
        self.class_map = release.get("class_map")
        expected_map = {group: index for index, group in enumerate(TARGET_GROUPS)}
        if (
            not isinstance(self.class_map, dict)
            or self.class_map != expected_map
            or any(type(value) is not int for value in self.class_map.values())
        ):
            raise ReleaseError("Invalid stable class map")
        self.release_metadata = release
        rows = read_jsonl(path)
        seen = set()
        for row in rows:
            image_id = row.get("image_id")
            if not isinstance(image_id, str) or not image_id or image_id in seen:
                raise ReleaseError("Invalid or duplicate manifest image_id")
            seen.add(image_id)
            if (
                row.get("group_label") not in self.class_map
                or type(row.get("class_id")) is not int
                or self.class_map[row["group_label"]] != row["class_id"]
                or row.get("split") not in SPLITS
            ):
                raise ReleaseError(f"Invalid class ID or split in manifest: {image_id}")
            safe_path(self.raw_root, row.get("relative_path"))
        if release.get("image_count") != len(rows) or not rows:
            raise ReleaseError("Release image count does not match its nonempty manifest")
        self.records = [row for row in rows if split is None or row["split"] == split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self.records[index]
        path = safe_path(self.raw_root, row["relative_path"])
        if self.verify_hash and sha256_file(path) != row["sha256"]:
            raise ReleaseError(f"Image checksum mismatch: {row['image_id']}")
        with Image.open(path) as original:
            if original.size != (row["width_px"], row["height_px"]):
                raise ReleaseError(f"Image dimensions changed: {row['image_id']}")
            if getattr(original, "n_frames", 1) != 1:
                raise ReleaseError(f"Multiframe image is not an approved still: {row['image_id']}")
            image = original.convert("RGB")
            image.load()
        if self.transform is not None:
            image = self.transform(image)
        return image, row["class_id"]


class HierarchicalManifestDataset:
    """Read a frozen exploratory split as ``(image, {targets, target_mask})``.

    Explicit ``allow_unreviewed=True`` acknowledges publisher-derived targets.
    Missing ranks have ID -1. A known holdout label can retain its ID while its
    mask is false because there is insufficient grouped evaluation support.
    ``transform`` receives the original image Path, so OpenCVPreprocessor can
    use exactly the same decoding and preparation during training and inference.
    Without a transform this reader returns a stored-orientation PIL RGB image.
    Original boxes/labels and mask reasons remain accessible in ``records``.
    """

    def __init__(
        self,
        manifest_path: Path,
        raw_root: Path,
        split: str | None = None,
        transform: Callable[[Path], Any] | None = None,
        verify_hash: bool = True,
        *,
        allow_unreviewed: bool = False,
    ) -> None:
        from .experiment_split import label_support, split_targets
        from .io import object_hash

        if allow_unreviewed is not True:
            raise ReleaseError("Publisher labels require explicit allow_unreviewed=True")
        path = Path(manifest_path)
        if path.is_dir():
            path /= "manifest.jsonl"
        if path.name != "manifest.jsonl" or not (path.parent / "split.json").is_file():
            raise ReleaseError("Use a frozen hierarchical split directory or its manifest.jsonl")
        metadata = json.loads((path.parent / "split.json").read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version") != "3.0"
            or metadata.get("kind") != "exploratory_hierarchical_split"
            or metadata.get("status") != "publisher_labels_unreviewed"
        ):
            raise ReleaseError("Unsupported hierarchical split metadata")
        identity = {key: value for key, value in metadata.items() if key != "split_version"}
        if metadata.get("split_version") != "split-" + object_hash(identity)[:16]:
            raise ReleaseError("Hierarchical split identity mismatch")
        required = {
            "manifest.jsonl",
            "groups.jsonl",
            "taxonomy_map.json",
            "label_support.json",
            "source_usage.json",
        }
        if not required.issubset(metadata.get("file_sha256", {})):
            raise ReleaseError("Incomplete hierarchical split file inventory")
        verify_bound_files(path.parent, metadata, "manifest.jsonl")
        if split is not None and split not in SPLITS:
            raise ReleaseError(f"Invalid split: {split}")
        self.raw_root = Path(raw_root).resolve(strict=True)
        self.transform, self.verify_hash = transform, verify_hash
        self.manifest_path, self.split_metadata = path, metadata
        self.taxonomy_map = json.loads((path.parent / "taxonomy_map.json").read_text())
        self.label_support = json.loads((path.parent / "label_support.json").read_text())
        rows = read_jsonl(path)
        if not rows or len(rows) != metadata.get("eligible_images"):
            raise ReleaseError("Hierarchical manifest image count mismatch")
        groups, assignments = {}, {}
        for row in rows:
            image_id, group = row.get("image_id"), row.get("experiment_group_id")
            if not isinstance(image_id, str) or not image_id or image_id in groups:
                raise ReleaseError("Invalid or duplicate hierarchical image_id")
            if not isinstance(group, str) or not group or row.get("split") not in SPLITS:
                raise ReleaseError("Invalid hierarchical group or split")
            if group in assignments and assignments[group] != row["split"]:
                raise ReleaseError("Related images cross partitions")
            groups[image_id], assignments[group] = group, row["split"]
            safe_path(self.raw_root, row.get("relative_path"))
            expected = derive_taxonomy(
                row["source_id"], row["source_labels"], [row["candidate_group"]]
            )
            if row.get("taxonomy") != expected or row.get("review_status") != "unreviewed":
                raise ReleaseError("Hierarchical taxonomy differs from unreviewed source labels")
        _validate_hierarchical_groups(path.parent, rows, metadata)
        if not isinstance(self.taxonomy_map, dict) or any(
            not isinstance(self.taxonomy_map.get(rank), dict)
            or any(type(value) is not int for value in self.taxonomy_map[rank].values())
            for rank in RANKS
        ):
            raise ReleaseError("Hierarchical taxonomy map requires integer IDs")
        if self.taxonomy_map != build_taxonomy_map(rows):
            raise ReleaseError("Hierarchical taxonomy map mismatch")
        if self.label_support != label_support(rows, groups, assignments):
            raise ReleaseError("Hierarchical label support mismatch")
        for row in rows:
            expected = split_targets(
                row["taxonomy"], self.taxonomy_map, self.label_support, row["split"]
            )
            if any(row.get(key) != value for key, value in expected.items()):
                raise ReleaseError("Hierarchical targets or masks disagree with frozen support")
            if (
                type(row.get("class_id")) is not int
                or row["class_id"] != expected["targets"]["group"]
            ):
                raise ReleaseError("Hierarchical broad class ID differs from group target")
            if any(
                type(row["targets"][rank]) is not int or type(row["target_mask"][rank]) is not bool
                for rank in RANKS
            ):
                raise ReleaseError("Hierarchical targets require integer IDs and boolean masks")
        self.records = [row for row in rows if split is None or row["split"] == split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, dict]:
        row = self.records[index]
        path = safe_path(self.raw_root, row["relative_path"])
        if self.verify_hash and sha256_file(path) != row["sha256"]:
            raise ReleaseError(f"Image checksum mismatch: {row['image_id']}")
        with Image.open(path) as original:
            if original.size != (row["width_px"], row["height_px"]):
                raise ReleaseError(f"Image dimensions changed: {row['image_id']}")
            if getattr(original, "n_frames", 1) != 1:
                raise ReleaseError(f"Multiframe image is not a still: {row['image_id']}")
            image = original.convert("RGB") if self.transform is None else None
        if self.transform is not None:
            image = self.transform(path)
        return image, {key: dict(row[key]) for key in ("targets", "target_mask")}
