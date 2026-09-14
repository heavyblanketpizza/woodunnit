"""Build an immutable, reproducible catalog without modifying source images."""

import csv
import importlib.metadata
import json
import platform
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from PIL import Image

from . import __version__
from .adapters import AdapterSet
from .config import IngestionConfig
from .curation import curate, fingerprints, near_duplicate_candidates
from .io import canonical_json, object_hash, read_json, safe_path, sha256_file, write_json
from .schema import CANDIDATE_GROUPS, PROJECT_GROUPS, SCHEMA_VERSION, ImageRecord
from .taxonomy import RANKS, build_taxonomy_map, hierarchy_targets

LABEL_MAP = {group: index for index, group in enumerate(PROJECT_GROUPS)}
EXAMPLE_COLUMNS = [
    "image_id",
    "source_id",
    "relative_path",
    "sha256",
    "pixel_sha256",
    "canonical_image_id",
    "width_px",
    "height_px",
    "candidate_group",
    "class_id",
    "label_origin",
    "review_status",
    "leakage_group_id",
    "split",
    "import_flags",
    "source_labels",
    "taxonomy",
    "annotations",
    "source_metadata",
    "targets",
    "target_mask",
]

COLLECTIONS = {
    "01_tgfc": ("tgfc", "extracted"),
    "02_idphy_microscopy": ("idphy", "images"),
    "03_soil_fungi_phytophthora": ("soil", "images"),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}


def image_id(source_id: str, relative_path: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"woodunnit:{source_id}:{relative_path}"))


def load_inventory(raw_root: Path) -> list[dict[str, str]]:
    with (raw_root / "image_inventory.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"collection", "path", "bytes", "sha256", "width", "height", "format"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Acquisition inventory requires columns {sorted(required)}")
        rows = list(reader)
    paths = set()
    for row in rows:
        collection = row["collection"]
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown inventory collection: {collection}")
        prefix = f"{collection}/{COLLECTIONS[collection][1]}/"
        if not row["path"].startswith(prefix):
            raise ValueError(f"Image path does not belong to its collection: {row['path']}")
        path = safe_path(raw_root, row["path"])
        if row["path"] in paths:
            raise ValueError(f"Duplicate inventory path: {row['path']}")
        paths.add(row["path"])
        if not path.is_file():
            raise ValueError(f"Missing source image: {row['path']}")
    if not rows:
        raise ValueError("Acquisition inventory is empty")
    actual = set()
    for collection, (_, image_dir) in COLLECTIONS.items():
        root = raw_root / collection / image_dir
        if not root.is_dir():
            raise ValueError(f"Missing source image folder: {root}")
        actual.update(
            p.relative_to(raw_root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if actual != paths:
        raise ValueError(
            f"Inventory coverage mismatch: {len(actual - paths)} unlisted images, "
            f"{len(paths - actual)} missing/unsupported images"
        )
    return sorted(rows, key=lambda row: row["path"])


def verify_image(raw_root: Path, row: dict, decode: bool = True) -> tuple[str, int, dict]:
    path = safe_path(raw_root, row["path"])
    if path.stat().st_size != int(row["bytes"]):
        raise ValueError(f"Image size changed: {row['path']}")
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"Image checksum changed: {row['path']}")
    if not decode:
        raise ValueError("Catalog ingestion requires raster decoding for duplicate curation")
    with Image.open(path) as image:
        image.load()
        if image.size != (int(row["width"]), int(row["height"])):
            raise ValueError(f"Image dimensions changed: {row['path']}")
        if image.format != row["format"]:
            raise ValueError(f"Image format changed: {row['path']}")
        return image.mode, getattr(image, "n_frames", 1), fingerprints(image)


def group_records(records: list[ImageRecord]) -> list[list[str]]:
    """Union entire sources, then proven equivalence and lineage across sources.

    None of the acquisitions establishes independent specimen IDs for every image.
    Source-wide groups are intentionally conservative, not biological case counts.
    """
    parent = {record.source_id: record.source_id for record in records}

    def find(source):
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    hashes: dict[str, list[ImageRecord]] = defaultdict(list)
    relations: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        hashes[record.sha256].append(record)
        if record.frame_count == 1:
            relations[f"rgb:{record.pixel_sha256}"].append(record)
            relations[f"lossless:{record.lossless_transform_sha256}"].append(record)
        relations[f"image:{record.image_id}"].append(record)
        if parent_id := record.lineage.get("parent_image_id"):
            relations[f"image:{parent_id}"].append(record)
        for key in record.relationship_keys:
            relations[key].append(record)
        for field in ("tree_id", "case_id", "specimen_id", "slide_id", "video_id", "isolate_id"):
            if value := record.lineage.get(field):
                relations[f"{field}:{value}"].append(record)
    for connected in [*hashes.values(), *relations.values()]:
        for record in connected[1:]:
            a, b = find(connected[0].source_id), find(record.source_id)
            parent[max(a, b)] = min(a, b)
    members = defaultdict(list)
    for source in sorted(parent):
        members[find(source)].append(source)
    for record in records:
        record.leakage_group_id = "lg-" + object_hash(members[find(record.source_id)])[:16]
    return [[r.image_id for r in group] for group in hashes.values() if len(group) > 1]


def connect_parents(records: list[ImageRecord]) -> None:
    full_fields = {}
    for record in records:
        if record.source_id == "soil" and record.image_role == "full_field":
            full_fields[Path(record.relative_path).stem] = record.image_id
    for record in records:
        if record.source_id == "soil":
            if record.image_role == "full_field":
                record.lineage["parent_image_id"] = None
            elif record.image_role == "crop":
                key = record.lineage.get("source_parent_id")
                record.lineage["parent_image_id"] = full_fields.get(key)
                if key and key not in full_fields:
                    record.import_flags = sorted(
                        set(record.import_flags) | {"parent_image_missing"}
                    )


def environment_record() -> dict:
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
    return {
        "python": platform.python_version(),
        "woodunnit": __version__,
        "uv_lock_sha256": sha256_file(lockfile) if lockfile.is_file() else None,
        "packages": {
            name: importlib.metadata.version(name) for name in ("pillow", "pydantic", "pyyaml")
        },
        "implementation_sha256": object_hash(
            {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
    }


def summarize(records: list[ImageRecord], task_groups: list[str], duplicates: list) -> dict:
    source_counts = Counter(r.source_id for r in records)
    group_counts = Counter(group for r in records for group in r.candidate_groups)
    by_source = {}
    for source in sorted(source_counts):
        subset = [r for r in records if r.source_id == source]
        by_source[source] = {
            "images": len(subset),
            "candidate_groups": dict(Counter(g for r in subset for g in r.candidate_groups)),
            "image_roles": dict(Counter(r.image_role for r in subset)),
            "source_splits": dict(Counter(r.source_split or "unspecified" for r in subset)),
            "source_annotation_boxes": sum(len(r.annotations) for r in subset),
        }
    selected = [r for r in records if r.selection_status == "included"]
    selected_counts = Counter(group for r in selected for group in r.candidate_groups)
    return {
        "images": len(records),
        "unique_image_hashes": len({r.sha256 for r in records}),
        "selected_images": len(selected),
        "excluded_redundant_images": len(records) - len(selected),
        "selected_candidate_group_counts": {g: selected_counts[g] for g in CANDIDATE_GROUPS},
        "selected_taxonomy_counts": {
            rank: dict(
                sorted(
                    Counter(
                        getattr(r.taxonomy, rank) for r in selected if getattr(r.taxonomy, rank)
                    ).items()
                )
            )
            for rank in RANKS
        },
        "selected_missing_rank_counts": {
            rank: sum(getattr(r.taxonomy, rank) is None for r in selected) for rank in RANKS
        },
        "sources": by_source,
        "candidate_group_counts": {g: group_counts[g] for g in CANDIDATE_GROUPS},
        "missing_task_groups": [g for g in task_groups if not group_counts[g]],
        "conservative_leakage_groups": len({r.leakage_group_id for r in records}),
        "independent_specimens": None,
        "reviewed_training_images": 0,
        "duplicate_groups": len(duplicates),
        "flags": dict(Counter(f for r in records for f in r.import_flags)),
        "label_status": "Publisher-derived candidates; all records remain unreviewed.",
        "split_status": "No project split assigned; original source splits retained as metadata.",
        "limitations": [
            "Source-wide groups prevent claiming unverified images are independent specimens.",
            "Source style and candidate group are strongly associated; "
            "pooled accuracy is not workflow validation.",
            "Genus/species targets exist only where publisher labels supply those ranks; "
            "they remain unreviewed within the fungi/oomycetes scope.",
            "Exact RGB and lossless-transform equivalents were curated; "
            "approximate similarity candidates still require visual review.",
        ],
    }


def write_review_queue(path: Path, records: list[ImageRecord]) -> None:
    columns = [
        "image_id",
        "relative_path",
        "sha256",
        "canonical_image_id",
        "selection_status",
        "exclusion_reason",
        "source_id",
        "source_labels",
        "candidate_groups",
        "image_role",
        "width_px",
        "height_px",
        "leakage_group_id",
        "import_flags",
        "review_status",
        "disposition",
        "group_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = record.model_dump(include=set(columns))
            for name in ("source_labels", "candidate_groups", "import_flags"):
                row[name] = canonical_json(row[name])
            writer.writerow(row)


def working_examples(records: list[ImageRecord]) -> list[dict]:
    """Expose source-labelled RGB image candidates without inventing adjudication."""
    examples = []
    taxonomy_map = build_taxonomy_map([record.model_dump() for record in records])
    for record in records:
        if record.selection_status != "included":
            continue
        candidate = record.candidate_groups[0] if len(record.candidate_groups) == 1 else None
        row = record.model_dump(include=set(EXAMPLE_COLUMNS))
        row.update(
            candidate_group=candidate,
            class_id=LABEL_MAP.get(candidate),
            label_origin="publisher_mapping",
            **hierarchy_targets(row["taxonomy"], taxonomy_map),
        )
        examples.append(row)
    return examples


def write_examples(directory: Path, records: list[ImageRecord]) -> None:
    examples = working_examples(records)
    with (directory / "examples.jsonl").open("w", encoding="utf-8") as stream:
        for row in examples:
            stream.write(canonical_json(row) + "\n")
    with (directory / "examples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXAMPLE_COLUMNS)
        writer.writeheader()
        for row in examples:
            writer.writerow(
                {
                    key: canonical_json(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def curate_and_flag(records: list[ImageRecord]) -> tuple[dict, list[dict]]:
    for record in records:
        record.import_flags = sorted(
            set(record.import_flags) - {"duplicate_label_conflict", "near_duplicate_candidate"}
        )
    curation = curate(records)
    similarity_pairs = near_duplicate_candidates(records)
    curation["near_duplicate_review_pairs"] = len(similarity_pairs)
    curation["near_duplicate_search"] = {
        "method": "64-bit horizontal difference hash, hamming distance <= 2",
        "excluded_from_search": "noncanonical images and hashes with <4 or >60 set bits",
        "automatically_excluded_by_similarity": 0,
        "review_completed": False,
    }
    flagged_ids = {pair[key] for pair in similarity_pairs for key in ("image_id_a", "image_id_b")}
    for record in records:
        if record.image_id in flagged_ids:
            record.import_flags = sorted(set(record.import_flags) | {"near_duplicate_candidate"})
    return curation, similarity_pairs


def ingest(config: IngestionConfig, progress=print) -> dict:
    rows = load_inventory(config.raw_root)
    adapters = AdapterSet(config.raw_root)
    records = []
    for index, row in enumerate(rows, 1):
        mode, frames, pixel_fingerprints = verify_image(config.raw_root, row, config.verify_images)
        adapted = adapters.adapt(row)
        source_id = COLLECTIONS[row["collection"]][0]
        if adapted["source_id"] != source_id:
            raise ValueError(f"Adapter source mismatch for {row['path']}")
        record = ImageRecord(
            image_id=image_id(source_id, row["path"]),
            relative_path=row["path"],
            sha256=row["sha256"],
            bytes=int(row["bytes"]),
            width_px=int(row["width"]),
            height_px=int(row["height"]),
            file_format=row["format"],
            image_mode=mode,
            frame_count=frames,
            leakage_group_id="pending",
            **pixel_fingerprints,
            **adapted,
        )
        if min(record.width_px, record.height_px) < 224:
            record.import_flags = sorted(set(record.import_flags) | {"native_dimension_below_224"})
        if frames and frames > 1:
            record.import_flags = sorted(set(record.import_flags) | {"multi_frame_asset"})
        records.append(record)
        if index % 1000 == 0:
            progress(f"Verified and catalogued {index:,}/{len(rows):,} images")
    connect_parents(records)
    curation, similarity_pairs = curate_and_flag(records)
    duplicates = group_records(records)
    inputs = {"image_inventory.csv": sha256_file(config.raw_root / "image_inventory.csv")}
    for source in adapters.sources.values():
        for reference in source.get("metadata_files", []):
            inputs[reference["path"]] = reference["sha256"]
    for record in records:
        if reference := record.source_metadata.get("annotation_file"):
            inputs[reference["path"]] = reference["sha256"]
    for collection in COLLECTIONS:
        manifest = config.raw_root / collection / "download_manifest.json"
        acquisition = read_json(manifest)
        if acquisition.get("errors"):
            raise ValueError(f"Source has unresolved download failures: {collection}")
        if acquisition.get("expected_files", len(acquisition["downloads"])) != len(
            acquisition["downloads"]
        ):
            raise ValueError(f"Source download manifest is incomplete: {collection}")
        inputs[manifest.relative_to(config.raw_root).as_posix()] = sha256_file(manifest)
    summary = summarize(records, config.task_groups, duplicates)
    config.output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ingesting-", dir=config.output_root))
    try:
        with (staging / "images.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(canonical_json(record.model_dump()) + "\n")
        write_json(staging / "sources.json", adapters.sources)
        write_json(staging / "image.schema.json", ImageRecord.model_json_schema())
        write_json(staging / "duplicate_groups.json", duplicates)
        write_json(staging / "curation.json", curation)
        write_json(staging / "label_map.json", LABEL_MAP)
        write_json(
            staging / "taxonomy_map.json",
            build_taxonomy_map([record.model_dump() for record in records]),
        )
        with (staging / "near_duplicate_candidates.jsonl").open("w", encoding="utf-8") as stream:
            for pair in similarity_pairs:
                stream.write(canonical_json(pair) + "\n")
        write_examples(staging, records)
        write_review_queue(staging / "review_queue.csv", records)
        report = [
            "# Woodunnit ingestion report",
            "",
            f"Source image files: **{len(records):,}**. "
            f"Working selection: **{summary['selected_images']:,}** images.",
            "",
            "| Source | Images | Candidate groups |",
            "| --- | ---: | --- |",
        ]
        for source, info in summary["sources"].items():
            report.append(f"| {source} | {info['images']:,} | {info['candidate_groups']} |")
        report.extend(
            [
                "",
                "All labels are publisher-derived candidates awaiting image-level review.",
                "Reviewed training images: **0**. No project split or model was created.",
                f"Excluded proven redundant records: {curation['excluded_images']:,}. "
                f"Similarity pairs awaiting review: {len(similarity_pairs):,}.",
                "examples.jsonl and examples.csv contain the working image/label view, "
                "with publisher-derived candidate labels and unassigned splits.",
                "",
                *summary["limitations"],
                "",
                "Copy review_queue.csv for inspection; catalog files are immutable.",
                "Original images are referenced by relative path and have not been transformed.",
            ]
        )
        (staging / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        file_hashes = {p.name: sha256_file(p) for p in sorted(staging.iterdir())}
        identity = {
            "schema_version": SCHEMA_VERSION,
            "task_groups": config.task_groups,
            "inputs": inputs,
            "environment": environment_record(),
            "verification": {"sha256": True, "first_frame_decode": config.verify_images},
            "file_sha256": file_hashes,
        }
        catalog_id = "catalog-" + object_hash(identity)[:16]
        result = {**identity, "catalog_id": catalog_id, "summary": summary}
        write_json(staging / "catalog.json", result)
        destination = config.output_root / catalog_id
        if destination.exists():
            previous = validate_catalog(destination, config.raw_root, verify_images=False)
            if previous != result:
                raise ValueError(f"Existing catalog metadata differs: {destination}")
            shutil.rmtree(staging)
            progress(f"Reused identical catalog: {destination}")
        else:
            staging.rename(destination)
            progress(f"Created catalog: {destination}")
        return {"catalog_dir": str(destination), **result}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_catalog(catalog_dir: Path, raw_root: Path, verify_images: bool = True) -> dict:
    catalog = read_json(catalog_dir / "catalog.json")
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported catalog schema version")
    if catalog.get("task_groups") != list(PROJECT_GROUPS):
        raise ValueError("Catalog requires the frozen fungi/oomycetes task groups")
    required_files = {
        "images.jsonl",
        "sources.json",
        "image.schema.json",
        "duplicate_groups.json",
        "review_queue.csv",
        "report.md",
        "curation.json",
        "label_map.json",
        "taxonomy_map.json",
        "examples.jsonl",
        "examples.csv",
        "near_duplicate_candidates.jsonl",
    }
    if set(catalog.get("file_sha256", {})) != required_files:
        raise ValueError("Catalog file inventory is incomplete or unexpected")
    for name, expected in catalog["file_sha256"].items():
        if sha256_file(safe_path(catalog_dir, name)) != expected:
            raise ValueError(f"Catalog file checksum mismatch: {name}")
    identity = {
        key: catalog[key]
        for key in (
            "schema_version",
            "task_groups",
            "inputs",
            "environment",
            "verification",
            "file_sha256",
        )
    }
    if catalog["catalog_id"] != "catalog-" + object_hash(identity)[:16]:
        raise ValueError("Catalog identity mismatch")
    for name, digest in catalog["inputs"].items():
        if sha256_file(safe_path(raw_root, name)) != digest:
            raise ValueError(f"Acquisition metadata changed: {name}")
    records = []
    with (catalog_dir / "images.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank catalog records are not permitted")
            record = ImageRecord.model_validate_json(line)
            expected_id = image_id(record.source_id, record.relative_path)
            if record.image_id != expected_id:
                raise ValueError(f"Image identity mismatch: {record.relative_path}")
            if verify_images:
                mode, frames, actual_fingerprints = verify_image(
                    raw_root,
                    {
                        "path": record.relative_path,
                        "bytes": record.bytes,
                        "sha256": record.sha256,
                        "width": record.width_px,
                        "height": record.height_px,
                        "format": record.file_format,
                    },
                    True,
                )
                if (
                    mode != record.image_mode
                    or frames != record.frame_count
                    or any(
                        getattr(record, key) != digest
                        for key, digest in actual_fingerprints.items()
                    )
                ):
                    raise ValueError(f"Image raster fingerprint mismatch: {record.relative_path}")
            records.append(record)
    if len({r.image_id for r in records}) != len(records):
        raise ValueError("Duplicate image IDs in catalog")
    inventory = load_inventory(raw_root)
    if [r.relative_path for r in records] != [row["path"] for row in inventory]:
        raise ValueError("Catalog image coverage/order differs from acquisition inventory")
    original = [record.model_dump() for record in records]
    curation, similarity_pairs = curate_and_flag(records)
    if original != [record.model_dump() for record in records]:
        raise ValueError("Catalog curation assignments or review flags are inconsistent")
    if curation != read_json(catalog_dir / "curation.json"):
        raise ValueError("Catalog curation report mismatch")
    with (catalog_dir / "near_duplicate_candidates.jsonl").open(encoding="utf-8") as stream:
        if similarity_pairs != [json.loads(line) for line in stream]:
            raise ValueError("Catalog similarity review pairs mismatch")
    if read_json(catalog_dir / "label_map.json") != LABEL_MAP:
        raise ValueError("Catalog label map mismatch")
    if read_json(catalog_dir / "taxonomy_map.json") != build_taxonomy_map(
        [record.model_dump() for record in records]
    ):
        raise ValueError("Catalog taxonomy map mismatch")
    with tempfile.TemporaryDirectory(prefix="woodunnit-validate-") as temporary:
        directory = Path(temporary)
        write_examples(directory, records)
        write_review_queue(directory / "review_queue.csv", records)
        for name in ("examples.jsonl", "examples.csv", "review_queue.csv"):
            if sha256_file(directory / name) != catalog["file_sha256"][name]:
                raise ValueError(f"Catalog derived record view mismatch: {name}")
    old_groups = [r.leakage_group_id for r in records]
    duplicates = group_records(records)
    if old_groups != [r.leakage_group_id for r in records]:
        raise ValueError("Catalog leakage groups do not preserve source/duplicate relationships")
    if duplicates != read_json(catalog_dir / "duplicate_groups.json"):
        raise ValueError("Catalog duplicate groups mismatch")
    if summarize(records, catalog["task_groups"], duplicates) != catalog["summary"]:
        raise ValueError("Catalog summary mismatch")
    return catalog
