"""Read-only importers for the three retained public collections.

Candidate mappings describe supplied labels, never reviewed image evidence. The
adapters do not assign project targets, review dispositions, or experiment splits.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from woodunnit.taxonomy import derive_taxonomy

COLLECTIONS = {
    "01_tgfc": "tgfc",
    "02_idphy_microscopy": "idphy",
    "03_soil_fungi_phytophthora": "soil",
}
TGFC_GROUPS = {
    "Colletotrichum siamense": "fungi",
    "Olivea tectonae": "fungi",
    "Neopestalotiopsis sp.": "fungi",
}
SOIL_GROUPS = {
    "Fusarium": "fungi",
    "Trichoderma": "fungi",
    "Verticillium": "fungi",
    # Preserve the release's original spelling; map it explicitly, do not rename files.
    "Phytophtora": "oomycetes",
    "Phytophthora": "oomycetes",
}

# Explicit identifier namespaces observed in publisher captions. A match is a
# conservative co-occurrence link, never an adjudicated specimen assignment.
# In particular, PH168, P168 and CPHST BL168 are different references.
_IDPHY_CULTURE_PATTERNS = (
    ("CPHST_BL", r"\bCPHST\s+BL\s*(?:[-:]\s*)?(\d+[A-Za-z]?)\b"),
    ("CBS", r"\bCBS\s+(\d+(?:\.\d+)?)\b"),
    ("P", r"\bP(\d+)\b"),
    ("PH", r"\bPH\s?(\d+)\b"),
    ("SE", r"\bSE\s+(\d+)\b"),
    ("CH", r"\bCH(\d+[A-Z]+\d+)\b"),
    ("GF", r"\bGF(\d+)\b"),
    ("TOKU", r"\bToku(\d+)\b"),
    ("VI", r"\bVI\s+(\d+-[A-Z0-9]+)\b"),
    ("AKWA", r"\bAKWA\s*(\d+(?:\.\d+)?-\d+)\b"),
    ("RHS", r"\bRHS\s+(\d+(?:\.\d+)?)\b"),
)


def _idphy_culture_references(metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Retain every explicit caption reference, including tester and alias codes."""
    captions = []

    def collect(item: dict, prefix: str) -> None:
        if isinstance(item.get("caption"), str):
            captions.append((prefix + "caption", item["caption"]))
        linked = item.get("linked_entity_image_captions", [])
        if isinstance(linked, list):
            captions.extend(
                (f"{prefix}linked_entity_image_captions[{index}]", caption)
                for index, caption in enumerate(linked)
                if isinstance(caption, str)
            )

    collect(metadata, "")
    galleries = metadata.get("gallery_records", [])
    if isinstance(galleries, list):
        for index, gallery in enumerate(galleries):
            if isinstance(gallery, dict):
                collect(gallery, f"gallery_records[{index}].")
    references = []
    for source_field, caption in captions:
        for namespace, pattern in _IDPHY_CULTURE_PATTERNS:
            for match in re.finditer(pattern, caption, flags=re.IGNORECASE):
                identifier = match.group(1)
                references.append(
                    {
                        "namespace": namespace,
                        "identifier": identifier,
                        "relationship_key": f"idphy:culture:{namespace}:{identifier}",
                        "source_field": source_field,
                        "matched_text": match.group(0),
                    }
                )
    return references


def _path(root: Path, relative: str) -> Path:
    """Resolve a source reference without allowing traversal or escaping symlinks."""
    item = PurePosixPath(relative)
    if not relative or item.is_absolute() or ".." in item.parts or "\\" in relative:
        raise ValueError(f"Unsafe source path: {relative!r}")
    result = root.joinpath(*item.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes raw root: {relative!r}")
    return result


def _reference(root: Path, path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _read_object(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return result


class AdapterSet:
    """Cache collection metadata once, then adapt rows from image_inventory.csv."""

    def __init__(self, raw_root: Path):
        self.raw_root = Path(raw_root).resolve()
        self.sources: dict[str, dict[str, Any]] = {}
        self._manifests: dict[str, dict[str, Any]] = {}
        self._downloads: dict[str, dict[str, dict[str, Any]]] = {}
        for collection, source_id in COLLECTIONS.items():
            manifest_path = _path(self.raw_root, f"{collection}/download_manifest.json")
            manifest = _read_object(manifest_path)
            if manifest.get("errors"):
                raise ValueError(f"Acquisition errors remain for {source_id}")
            self._manifests[source_id] = manifest
            downloads: dict[str, dict[str, Any]] = {}
            for item in manifest.get("downloads", []):
                filename = item["filename"]
                if filename in downloads:
                    raise ValueError(f"Duplicate download filename in {source_id}: {filename}")
                downloads[filename] = item
            self._downloads[source_id] = downloads
            refs = [_reference(self.raw_root, manifest_path)]
            metadata_name = (
                "selection_manifest.json" if source_id == "idphy" else "repository_record.json"
            )
            metadata_path = _path(self.raw_root, f"{collection}/metadata/{metadata_name}")
            # The acquisition metadata is required: access alone is not permission.
            metadata = _read_object(metadata_path)
            refs.append(_reference(self.raw_root, metadata_path))
            license_value = manifest.get("license")
            if not license_value:
                raise ValueError(f"Missing saved license statement for {source_id}")
            self.sources[source_id] = {
                "source_id": source_id,
                "collection": collection,
                "title": manifest.get("title") or metadata.get("title") or metadata.get("name"),
                "version": str(
                    manifest.get("version")
                    or manifest.get("doi")
                    or manifest.get("downloaded_at")
                    or "unknown"
                ),
                "doi": manifest.get("doi"),
                "source_url": manifest.get("source_url"),
                "license": copy.deepcopy(license_value),
                "authors": copy.deepcopy(manifest.get("authors", [])),
                "downloaded_at": manifest.get("downloaded_at"),
                "rights_status": "site_default_with_image_exceptions_review_required"
                if source_id == "idphy"
                else "publisher_license_recorded",
                "training_permission_verified": False,
                "usage_permission_ref": f"sources.json#{source_id}",
                "metadata_files": refs,
                "download_manifest_ref": refs[0]["path"],
                "download_manifest_sha256": refs[0]["sha256"],
                "archive_downloads": [
                    copy.deepcopy(item)
                    for item in downloads.values()
                    if item["filename"].lower().endswith(".zip")
                ],
                "source_description": metadata.get("description")
                or metadata.get("metadata", {}).get("description"),
            }
            if source_id == "idphy":
                self.sources[source_id]["rights"] = {
                    "copyright_url": metadata.get("copyright_url"),
                    "site_rights_statement": metadata.get("site_rights_statement"),
                    "default_image_credit": metadata.get("default_image_credit"),
                    "per_image_exceptions": (
                        "Preserved in image source_metadata; embedded markings need review."
                    ),
                }

        candidates = sorted(_path(self.raw_root, "01_tgfc/extracted").rglob("data.yaml"))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one TgFC data.yaml class mapping")
        self._tgfc_yaml = candidates[0]
        if not self._tgfc_yaml.resolve().is_relative_to(self.raw_root):
            raise ValueError("TgFC data.yaml escapes raw root")
        schema = yaml.safe_load(self._tgfc_yaml.read_text(encoding="utf-8"))
        if not isinstance(schema, dict):
            raise ValueError("TgFC data.yaml must contain a mapping")
        names = schema.get("names")
        if isinstance(names, list):
            self._tgfc_labels = dict(enumerate(names))
        elif isinstance(names, dict):
            self._tgfc_labels = {int(key): value for key, value in names.items()}
        else:
            raise ValueError("Missing TgFC class names")
        if set(self._tgfc_labels) != set(range(len(self._tgfc_labels))):
            raise ValueError("TgFC class IDs must be consecutive from zero")
        if schema.get("nc") != len(self._tgfc_labels):
            raise ValueError("TgFC class count and names disagree")
        if not all(isinstance(name, str) and name for name in self._tgfc_labels.values()):
            raise ValueError("Invalid TgFC class names")
        self.sources["tgfc"]["metadata_files"].append(_reference(self.raw_root, self._tgfc_yaml))
        self.sources["tgfc"]["class_names"] = {
            str(key): value for key, value in self._tgfc_labels.items()
        }
        self._soil_parents = {
            item["parent_image_id"]: item["filename"]
            for item in self._downloads["soil"].values()
            if item.get("image_role") == "full_field"
        }

    def adapt(self, row: dict[str, str]) -> dict[str, Any]:
        collection = row["collection"]
        if collection not in COLLECTIONS:
            raise ValueError(f"Unsupported collection: {collection}")
        path = _path(self.raw_root, row["path"])
        if PurePosixPath(row["path"]).parts[0] != collection:
            raise ValueError("Inventory collection and path disagree")
        source_id = COLLECTIONS[collection]
        result: dict[str, Any] = {
            "source_id": source_id,
            "source_labels": [],
            "candidate_groups": [],
            "source_split": None,
            "source_url": self.sources[source_id]["source_url"],
            "image_role": "unknown",
            "source_metadata": {"original_filename": path.name},
            "annotations": [],
            "microscopy": {
                "modality": None,
                "objective_magnification": None,
                "total_magnification": None,
                "microns_per_pixel": None,
                "microscope_id": None,
                "camera_id": None,
                "preparation_type": None,
                "stain": None,
            },
            "lineage": {
                key: None
                for key in (
                    "parent_image_id",
                    "tree_id",
                    "case_id",
                    "specimen_id",
                    "slide_id",
                    "video_id",
                    "isolate_id",
                )
            },
            "relationship_keys": [],
            "import_flags": ["source_label_unreviewed", "biological_lineage_unrecoverable"],
            "usage_permission_ref": f"sources.json#{source_id}",
        }
        getattr(self, f"_adapt_{source_id}")(row, path, result)
        result["source_labels"] = sorted(set(result["source_labels"]))
        result["candidate_groups"] = sorted(set(result["candidate_groups"]))
        result["taxonomy"] = derive_taxonomy(
            source_id, result["source_labels"], result["candidate_groups"]
        )
        result["import_flags"] = sorted(set(result["import_flags"]))
        return result

    def _download(self, source_id: str, path: Path, row: dict[str, str]) -> dict[str, Any]:
        try:
            item = self._downloads[source_id][path.name]
        except KeyError as exc:
            raise ValueError(f"No acquisition metadata for {source_id}/{path.name}") from exc
        if item.get("sha256") != row["sha256"] or item.get("bytes") != int(row["bytes"]):
            raise ValueError(f"Inventory differs from acquired file metadata: {path.name}")
        return item

    def _adapt_tgfc(self, row: dict[str, str], path: Path, result: dict[str, Any]) -> None:
        del row
        result["microscopy"].update(
            {
                "preparation_type": "wet_mount",
                "stain": "none_reported",
                "reported_magnification": "40x; objective versus total not specified",
                "source_scope": "collection_description",
            }
        )
        result["import_flags"].append("magnification_kind_unspecified")
        metadata = result["source_metadata"]
        metadata["class_mapping_ref"] = self._tgfc_yaml.relative_to(self.raw_root).as_posix()
        metadata["context"] = {
            "host_taxon": "Tectona grandis",
            "tissue_type": "leaf",
            "source_scope": "collection_description",
        }
        if "MixedClass" in path.parts:
            result["source_labels"] = ["MixedClass"]
            result["candidate_groups"] = ["fungi"]
            result["import_flags"].extend(
                [
                    "multiple_fungal_taxa_source_subset",
                    "per_image_taxa_unavailable",
                    "source_boxes_unavailable",
                ]
            )
            metadata["label_mapping_basis"] = (
                "MixedClass supplementary fungal collection; individual taxa are not assigned "
                "from collection membership."
            )
            return
        if path.parent.name != "images" or path.parent.parent.name not in {
            "train",
            "valid",
            "test",
        }:
            raise ValueError(f"Unexpected TgFC image location: {path}")
        result["source_split"] = path.parent.parent.name
        label_path = _path(
            self.raw_root,
            (path.parent.parent / "labels" / f"{path.stem}.txt")
            .relative_to(self.raw_root)
            .as_posix(),
        )
        if not label_path.is_file():
            raise ValueError(f"Missing TgFC annotation file: {label_path}")
        label_text = label_path.read_text(encoding="utf-8")
        metadata["annotation_file"] = _reference(self.raw_root, label_path)
        metadata["annotation_text"] = label_text
        for line_number, line in enumerate(label_text.splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Expected YOLO class and four coordinates: {label_path}:{line_number}"
                )
            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
                label = self._tgfc_labels[class_id]
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Invalid YOLO annotation: {label_path}:{line_number}") from exc
            if (
                not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates)
                or coordinates[2] <= 0
                or coordinates[3] <= 0
            ):
                raise ValueError(f"Invalid normalized YOLO box: {label_path}:{line_number}")
            result["annotations"].append(
                {
                    "label": label,
                    "class_id": class_id,
                    "format": "yolo_cxcywh_normalized",
                    "coordinates": coordinates,
                }
            )
            result["source_labels"].append(label)
        if not result["annotations"]:
            result["import_flags"].append("empty_source_annotations")
        for label in result["source_labels"]:
            if label in TGFC_GROUPS:
                result["candidate_groups"].append(TGFC_GROUPS[label])
            else:
                result["import_flags"].append("unmapped_source_label")
        # Roboflow's export basename identifies a source filename, not a specimen.
        match = re.fullmatch(r"(.+)\.rf\.[0-9a-fA-F]{32}", path.stem)
        if match:
            metadata["source_export_stem"] = match.group(1)
            result["relationship_keys"].append(f"tgfc:export_stem:{match.group(1)}")
        result["import_flags"].append("source_split_not_project_split")

    def _adapt_idphy(self, row: dict[str, str], path: Path, result: dict[str, Any]) -> None:
        item = self._download("idphy", path, row)
        metadata = item.get("source_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing IDphy source metadata: {path.name}")
        result["source_metadata"].update(copy.deepcopy(metadata))
        result["source_metadata"]["acquisition_file"] = {
            key: value for key, value in item.items() if key != "source_metadata"
        }
        result["source_metadata"]["checksum_provenance_note"] = (
            "The acquisition record's publisher_checksum field contains a locally computed "
            "SHA256 for IDphy; it is not a publisher-supplied digest."
        )
        labels = metadata.get("species_labels")
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) for label in labels)
        ):
            raise ValueError(f"Missing IDphy source species labels: {path.name}")
        result["source_labels"] = labels.copy()
        if all(label.startswith("Phytophthora ") for label in labels):
            result["candidate_groups"] = ["oomycetes"]
        else:
            result["import_flags"].append("unmapped_source_label")
        result["source_url"] = item["source_url"]
        result["relationship_keys"].append(f"idphy:asset:{item['source_url']}")
        result["import_flags"].extend(
            ["image_role_requires_review", "image_rights_markings_require_review"]
        )
        if metadata.get("source_label_ambiguity") or len(labels) > 1:
            result["import_flags"].append("conflicting_source_taxon_labels")
        references = _idphy_culture_references(metadata)
        if references:
            result["source_metadata"]["observed_culture_references"] = references
            result["relationship_keys"].extend(
                sorted({reference["relationship_key"] for reference in references})
            )
            result["import_flags"].append("caption_culture_references_unreviewed")
        # Keep every named culture (including testers), without claiming that one
        # identifier is the image's isolate or that co-occurring aliases are equal.

    def _adapt_soil(self, row: dict[str, str], path: Path, result: dict[str, Any]) -> None:
        item = self._download("soil", path, row)
        label = item.get("source_label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"Missing soil source label: {path.name}")
        result["source_labels"] = [label]
        if label in SOIL_GROUPS:
            result["candidate_groups"] = [SOIL_GROUPS[label]]
        else:
            result["import_flags"].append("unmapped_source_label")
        result["source_url"] = item["source_url"]
        result["source_metadata"]["acquisition_file"] = copy.deepcopy(item)
        role = item.get("image_role")
        if role not in {"crop", "full_field"}:
            raise ValueError(f"Missing/invalid soil image role: {path.name}")
        result["image_role"] = role
        parent = item.get("parent_image_id")
        if not isinstance(parent, str) or not parent:
            raise ValueError(f"Missing soil source parent identifier: {path.name}")
        result["lineage"]["source_parent_id"] = parent
        result["relationship_keys"].append(f"soil:source_parent:{parent}")
        if role == "crop":
            result["import_flags"].append("crop_coordinates_unavailable")
            if parent in self._soil_parents:
                result["source_metadata"]["parent_relative_path"] = (
                    f"03_soil_fungi_phytophthora/images/{self._soil_parents[parent]}"
                )
            else:
                result["import_flags"].append("parent_image_not_in_release")
        if label == "Phytophtora":
            result["source_metadata"]["normalized_source_taxon"] = "Phytophthora"
            result["import_flags"].append("source_spelling_preserved")


def load_sources(raw_root: Path) -> dict[str, dict[str, Any]]:
    """Load source and permission references; prefer AdapterSet for batch ingestion."""
    return AdapterSet(raw_root).sources


def adapt(
    raw_root: Path, inventory_row: dict[str, str], sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Convenience single-row API. AdapterSet avoids repeated metadata reads."""
    adapters = AdapterSet(raw_root)
    if sources != adapters.sources:
        raise ValueError("Provided source registry differs from saved acquisition metadata")
    return adapters.adapt(inventory_row)
