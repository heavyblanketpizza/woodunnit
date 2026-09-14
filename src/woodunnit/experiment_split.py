"""Frozen three-way exploratory splits that preserve observed image families."""

import csv
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .io import canonical_json, object_hash, read_json, safe_path, sha256_file, write_json
from .taxonomy import build_taxonomy_map, derive_taxonomy, hierarchy_targets

LABEL_MAP = {"fungi": 0, "oomycetes": 1}
SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
POLICY_VERSION = "exploratory-hierarchical-observed-families-threeway-v2"
RANKS = ("group", "genus", "species")


def read_jsonl(path: Path) -> list[dict]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: canonical_json(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def observed_groups(records: list[dict], near_pairs: list[dict]) -> dict[str, str]:
    """Join all records before eligibility filtering, including excluded bridges.

    Taxon labels and species-page IDs are classification metadata, not image
    relationships. Only recorded lineage, source asset/export keys and observed
    equivalence join images; those links still do not prove specimen independence.
    """
    ids = [row["image_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalog image IDs must be unique")
    parent = {image_id: image_id for image_id in ids}

    def find(image_id):
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def join(left, right):
        if left not in parent or right not in parent:
            raise ValueError("Image relationship contains an unknown image ID")
        a, b = sorted((find(left), find(right)))
        parent[b] = a

    seen = {}
    for row in sorted(records, key=lambda item: item["image_id"]):
        image_id, source = row["image_id"], row["source_id"]
        keys = [("file", row["sha256"])]
        if row["frame_count"] == 1:
            keys += [
                ("pixels", row["pixel_sha256"]),
                ("lossless", row["lossless_transform_sha256"]),
            ]
        keys += [("relationship", key) for key in row.get("relationship_keys", [])]
        for name, value in row.get("lineage", {}).items():
            if not value:
                continue
            if name == "parent_image_id":
                join(image_id, value)
            elif name == "source_parent_id":
                keys.append((source, name, value))
            else:
                keys.append(("lineage", name, value))
        canonical = row.get("canonical_image_id")
        if canonical:
            join(image_id, canonical)
        if source not in {"idphy", "tgfc", "soil"}:
            raise ValueError(f"Unsupported experiment source: {source}")
        # Adapters record source asset/export and explicit culture references in
        # relationship_keys. Do not infer biological lineage from filename ordinals,
        # taxon labels, gallery entities, or an entire acquisition collection.
        for key in keys:
            if key in seen:
                join(image_id, seen[key])
            else:
                seen[key] = image_id
    for pair in near_pairs:
        join(pair["image_id_a"], pair["image_id_b"])
    components = defaultdict(list)
    for image_id in sorted(ids):
        components[find(image_id)].append(image_id)
    return {
        image_id: "eg-" + object_hash(members)[:16]
        for members in components.values()
        for image_id in members
    }


def _taxonomy(row: dict) -> dict:
    return row.get("taxonomy") or derive_taxonomy(
        row["source_id"], row.get("source_labels", []), row["candidate_groups"]
    )


def _features(row: dict) -> list[tuple[str, str]]:
    taxonomy = _taxonomy(row)
    return [(rank, taxonomy[rank]) for rank in RANKS if taxonomy.get(rank)] + [
        ("source", row["source_id"])
    ]


def _group_features(records: list[dict], groups: dict[str, str]):
    sizes = defaultdict(Counter)
    support = defaultdict(set)
    for row in records:
        group = groups[row["image_id"]]
        for feature in _features(row):
            sizes[group][feature] += 1
            support[feature].add(group)
    totals = sum(sizes.values(), Counter())
    return sizes, support, totals


def assign_groups(records: list[dict], groups: dict[str, str], seed: int) -> dict[str, str]:
    """Jointly stratify observed groups by broad label, fine labels and source.

    All broad labels must occur in every partition. Fine labels with at least
    three groups receive coverage priority; incompatible overlapping relationships
    can make that impossible, so export reports actual support and masks their
    evaluation. Resolved labels get training-support priority before ratio fit;
    when support conflicts with required coverage, the report exposes the missing
    training labels and masks their holdout ranks. Counts never split a component.
    """
    sizes, support, totals = _group_features(records, groups)
    broad = {("group", label) for label in LABEL_MAP}
    for feature in broad:
        if len(support[feature]) < 3:
            raise ValueError(f"Fewer than three observed groups contain {feature[1]}")
    covered = {feature for feature in support if feature[0] in RANKS and len(support[feature]) >= 3}
    balanced = {
        feature for feature in support if feature[0] in {"group", "source"} or feature in covered
    }
    counts = {split: Counter() for split in SPLITS}
    remaining_support = Counter({feature: len(members) for feature, members in support.items()})
    remaining_types = Counter(frozenset(size.keys() & broad) for size in sizes.values())
    assignment = {}
    choices = ("train", "validation", "test")
    rank_counts = Counter(feature[0] for feature in balanced)
    rank_weights = {"group": 2.0, "genus": 1.0, "species": 1.0, "source": 0.5}
    ordered = sorted(
        sizes,
        key=lambda group: (
            -max(
                sizes[group][feature] / totals[feature]
                for feature in sizes[group]
                if feature in balanced
            ),
            min(len(support[feature]) for feature in sizes[group] if feature in covered),
            object_hash([seed, group]),
        ),
    )

    def feature_score(feature):
        score = 0.0
        if feature in balanced:
            score += (
                rank_weights[feature[0]]
                / rank_counts[feature[0]]
                * sum(
                    (counts[split][feature] / totals[feature] - TARGET_RATIOS[split]) ** 2
                    for split in SPLITS
                )
            )
        if feature in covered:
            score += sum(
                (8.0 if split == "train" else 4.0)
                for split in SPLITS
                if counts[split][feature] == 0
            )
        elif feature[0] in {"genus", "species"} and counts["train"][feature] == 0:
            # Training coverage outweighs the bounded aggregate ratio objective.
            score += 8.0
        return score

    def feasible(features):
        return all(
            sum(counts[split][feature] == 0 for split in SPLITS) <= remaining_support[feature]
            for feature in features
        )

    def broad_feasible():
        # Two-class shared components can need the same future group in different
        # partitions. Per-label remaining counts alone miss that conflict.
        left, right = sorted(broad)
        missing_left = {split for split in SPLITS if counts[split][left] == 0}
        missing_right = {split for split in SPLITS if counts[split][right] == 0}
        shared = remaining_types[frozenset((left, right))]
        need_left = max(0, len(missing_left) - remaining_types[frozenset((left,))])
        need_right = max(0, len(missing_right) - remaining_types[frozenset((right,))])
        needed_shared = max(
            need_left, need_right, need_left + need_right - len(missing_left & missing_right)
        )
        return needed_shared <= shared

    for group in ordered:
        features = sizes[group]
        remaining_support.subtract(features.keys())
        remaining_types[frozenset(features.keys() & broad)] -= 1
        candidates = []
        for split in choices:
            counts[split].update(features)
            if broad_feasible():
                candidates.append(
                    (
                        not feasible(covered),
                        sum(feature_score(feature) for feature in sorted(features)),
                        choices.index(split),
                        split,
                    )
                )
            counts[split].subtract(features)
        if not candidates:
            raise ValueError("Cannot retain both classes in all three grouped partitions")
        split = min(candidates)[-1]
        assignment[group] = split
        counts[split].update(features)

    # Deterministic coordinate descent uses only labels, counts and known groups.
    # Two bounded passes avoid quadratic work on image collections.
    for _ in range(2):
        changed = False
        for group in ordered:
            features = sizes[group]
            source = assignment[group]
            original = sum(feature_score(feature) for feature in sorted(features))
            counts[source].subtract(features)
            if any(counts[source][feature] <= 0 for feature in broad) or any(
                counts[source][feature] <= 0 for feature in features if feature in covered
            ):
                counts[source].update(features)
                continue
            candidates = []
            for destination in choices:
                if destination == source:
                    continue
                counts[destination].update(features)
                score = sum(feature_score(feature) for feature in sorted(features))
                counts[destination].subtract(features)
                if score < original - 1e-12:
                    candidates.append((score, choices.index(destination), destination))
            destination = min(candidates)[-1] if candidates else source
            assignment[group] = destination
            counts[destination].update(features)
            changed |= destination != source
        if not changed:
            break
    return assignment


def label_support(records: list[dict], groups: dict[str, str], assignments: dict[str, str]) -> dict:
    """Report actual image/group support, including deliberately masked fine ranks."""
    image_counts = defaultdict(lambda: Counter())
    group_sets = defaultdict(lambda: defaultdict(set))
    for row in records:
        group = groups[row["image_id"]]
        split = assignments[group]
        for taxon in _taxonomy(row).get("taxa", []):
            for rank in RANKS:
                if taxon.get(rank):
                    image_counts[rank, taxon[rank]]
        for feature in _features(row):
            image_counts[feature][split] += 1
            group_sets[feature][split].add(group)
    report = {rank: {} for rank in (*RANKS, "source")}
    for (rank, label), counts in sorted(image_counts.items()):
        partitions = group_sets[rank, label]
        observed = set().union(*partitions.values())
        reasons = []
        if rank in {"genus", "species"}:
            if len(observed) < 3:
                reasons.append("fewer_than_three_observed_groups")
            if not counts["train"]:
                reasons.append("no_training_examples")
            if any(not counts[split] for split in SPLITS):
                reasons.append("incomplete_threeway_coverage")
        report[rank][label] = {
            "images": sum(counts.values()),
            "observed_groups": len(observed),
            "image_counts": {split: counts[split] for split in SPLITS},
            "group_counts": {split: len(partitions[split]) for split in SPLITS},
            "training_supported": counts["train"] > 0,
            "evaluation_supported": not reasons,
            "evaluation_mask_reasons": reasons,
            "train_only_supervision": rank in {"genus", "species"}
            and bool(reasons)
            and counts["train"] > 0,
        }
    return report


def split_targets(taxonomy: dict, taxonomy_map: dict, support: dict, split: str) -> dict:
    result = hierarchy_targets(taxonomy, taxonomy_map)
    reasons = {}
    for rank in RANKS:
        if not result["target_mask"][rank]:
            reasons[rank] = ["unresolved_publisher_label"]
        elif split != "train" and rank in {"genus", "species"}:
            label_support = support[rank][taxonomy[rank]]
            if not label_support["evaluation_supported"]:
                result["target_mask"][rank] = False
                reasons[rank] = label_support["evaluation_mask_reasons"]
    result["target_mask_reasons"] = reasons
    return result


def _validate_inputs(catalog_dir: Path, usage: dict, raw_root: Path | None):
    catalog = read_json(catalog_dir / "catalog.json")
    if catalog.get("schema_version") != "3.0":
        raise ValueError(
            "Hierarchical splits require catalog schema 3.0; re-ingest source annotations"
        )
    if usage.get("catalog_id") != catalog["catalog_id"] or usage.get(
        "catalog_sha256"
    ) != sha256_file(catalog_dir / "catalog.json"):
        raise ValueError("Source usage does not bind this exact catalog")
    for key, filename in (
        ("images_jsonl_sha256", "images.jsonl"),
        ("sources_json_sha256", "sources.json"),
    ):
        if usage.get(key) != catalog["file_sha256"].get(filename) or usage[key] != sha256_file(
            catalog_dir / filename
        ):
            raise ValueError(f"Source usage does not bind {filename}")
    for filename in ("near_duplicate_candidates.jsonl",):
        if catalog["file_sha256"].get(filename) != sha256_file(catalog_dir / filename):
            raise ValueError(f"Catalog integrity mismatch: {filename}")
    records = read_jsonl(catalog_dir / "images.jsonl")
    for row in records:
        if row.get("taxonomy") != derive_taxonomy(
            row["source_id"], row["source_labels"], row["candidate_groups"]
        ):
            raise ValueError("Catalog taxonomy must match preserved publisher labels")
        if not isinstance(row.get("annotations"), list):
            raise ValueError("Catalog schema 3.0 requires preserved source annotations")
    by_id = {row["image_id"]: row for row in records}
    eligible = usage.get("eligible_image_ids", [])
    if not eligible or len(eligible) != len(set(eligible)) or set(eligible) - by_id.keys():
        raise ValueError("Source usage requires unique known eligible image IDs")
    source_policies = usage.get("sources", {})
    for source in {by_id[image_id]["source_id"] for image_id in eligible}:
        policy = source_policies.get(source, {})
        if not policy.get("license") or not policy.get("attribution"):
            raise ValueError(f"Missing source permission evidence: {source}")
        if raw_root is not None and sha256_file(
            safe_path(raw_root, policy["license_evidence_ref"])
        ) != policy.get("license_evidence_sha256"):
            raise ValueError(f"Source license evidence changed: {source}")
        actual = sum(by_id[image_id]["source_id"] == source for image_id in eligible)
        if policy.get("expected_images") != actual:
            raise ValueError(f"Source eligibility counts changed: {source}")
    for image_id in eligible:
        row = by_id[image_id]
        candidates = row["candidate_groups"]
        if (
            row["selection_status"] != "included"
            or row["frame_count"] != 1
            or len(candidates) != 1
            or candidates[0] not in LABEL_MAP
            or "duplicate_label_conflict" in row["import_flags"]
            or row["review_status"] != "unreviewed"
            or row["group_label"] is not None
        ):
            raise ValueError(f"Ineligible exploratory image: {image_id}")
        role = source_policies[row["source_id"]].get("required_image_role")
        if role and row["image_role"] != role:
            raise ValueError(f"Image role violates source policy: {image_id}")
    return catalog, records, set(eligible)


def build_split(
    catalog_dir: Path,
    usage_path: Path,
    output_dir: Path,
    seed: int = 42,
    *,
    raw_root: Path | None = None,
) -> dict:
    """Write an immutable split and optional image-reference folders atomically."""
    catalog_dir, usage_path, output_dir = map(Path, (catalog_dir, usage_path, output_dir))
    if raw_root is not None:
        raw_root = Path(raw_root).resolve()
        if not raw_root.is_dir() or output_dir.resolve().is_relative_to(raw_root):
            raise ValueError("Use an existing raw root and an output outside it")
    if output_dir.exists():
        raise ValueError(f"Split output already exists: {output_dir}")
    if output_dir.resolve().is_relative_to(catalog_dir.resolve()):
        raise ValueError("Split output must be outside the immutable catalog")
    usage = read_json(usage_path)
    catalog, records, eligible = _validate_inputs(catalog_dir, usage, raw_root)
    near_pairs = read_jsonl(catalog_dir / "near_duplicate_candidates.jsonl")
    groups = observed_groups(records, near_pairs)
    selected = [row for row in records if row["image_id"] in eligible]
    assignments = assign_groups(selected, groups, seed)
    taxonomy_map = build_taxonomy_map(selected)
    support = label_support(selected, groups, assignments)
    examples = []
    for row in sorted(selected, key=lambda item: item["image_id"]):
        group = groups[row["image_id"]]
        label = row["candidate_groups"][0]
        example = {
            key: row[key]
            for key in (
                "image_id",
                "source_id",
                "relative_path",
                "sha256",
                "pixel_sha256",
                "lossless_transform_sha256",
                "width_px",
                "height_px",
                "source_labels",
                "review_status",
                "import_flags",
                "lineage",
                "relationship_keys",
            )
        }
        example.update(
            **{
                key: row.get(key)
                for key in (
                    "taxonomy",
                    "annotations",
                    "source_metadata",
                    "source_url",
                    "source_split",
                    "microscopy",
                    "usage_permission_ref",
                )
            },
            **split_targets(row["taxonomy"], taxonomy_map, support, assignments[group]),
            source_leakage_group_id=row["leakage_group_id"],
            leakage_group_id=group,
            experiment_group_id=group,
            candidate_group=label,
            class_id=LABEL_MAP[label],
            label_origin="publisher_mapping",
            split=assignments[group],
            state="review_required",
        )
        examples.append(example)
    components = defaultdict(list)
    for row in records:
        components[groups[row["image_id"]]].append(row)
    group_records = [
        {
            "experiment_group_id": group,
            "split": assignments.get(group),
            "catalog_image_ids": sorted(row["image_id"] for row in members),
            "eligible_image_ids": sorted(
                row["image_id"] for row in members if row["image_id"] in eligible
            ),
            "sources": sorted({row["source_id"] for row in members}),
            "source_labels": sorted({label for row in members for label in row["source_labels"]}),
            "candidate_groups": sorted(
                {label for row in members for label in row["candidate_groups"]}
            ),
        }
        for group, members in sorted(components.items())
    ]
    counts = {}
    for split in SPLITS:
        part = [row for row in examples if row["split"] == split]
        counts[split] = {
            "images": len(part),
            "class_counts": dict(sorted(Counter(row["candidate_group"] for row in part).items())),
            "source_counts": dict(sorted(Counter(row["source_id"] for row in part).items())),
            "observed_groups": len({row["experiment_group_id"] for row in part}),
        }
    exclusions_by_id = {row["image_id"]: row for row in usage.get("excluded_images", [])}
    exclusions = [
        {
            **exclusions_by_id.get(
                row["image_id"],
                {"image_id": row["image_id"], "reason": "not_in_frozen_eligibility_allowlist"},
            ),
            "relative_path": row["relative_path"],
            "source_id": row["source_id"],
            "experiment_group_id": groups[row["image_id"]],
            "related_partition": assignments.get(groups[row["image_id"]]),
        }
        for row in sorted(records, key=lambda item: item["image_id"])
        if row["image_id"] not in eligible
    ]
    partition_groups = defaultdict(set)
    for example in examples:
        partition_groups[example["experiment_group_id"]].add(example["split"])
    overlaps = sum(len(partitions) > 1 for partitions in partition_groups.values())
    if overlaps:
        raise ValueError("Known image relationships cross partitions")
    metadata = {
        "schema_version": "3.0",
        "kind": "exploratory_hierarchical_split",
        "status": "publisher_labels_unreviewed",
        "policy_version": POLICY_VERSION,
        "experiment_kind": "exploratory_hierarchical_publisher_labels",
        "seed": seed,
        "target_ratios": TARGET_RATIOS,
        "label_map": LABEL_MAP,
        "catalog_id": catalog["catalog_id"],
        "catalog_sha256": sha256_file(catalog_dir / "catalog.json"),
        "source_usage_sha256": sha256_file(usage_path),
        "catalog_images_sha256": sha256_file(catalog_dir / "images.jsonl"),
        "near_duplicate_candidates_sha256": sha256_file(
            catalog_dir / "near_duplicate_candidates.jsonl"
        ),
        "catalog_images": len(records),
        "eligible_images": len(examples),
        "excluded_images": len(exclusions),
        "catalog_observed_groups": len(components),
        "eligible_observed_groups": len(assignments),
        "near_duplicate_pairs_grouped": len(near_pairs),
        "counts": counts,
        "cross_partition_known_relationships": overlaps,
        "verified_partition_groups": len(partition_groups),
        "rank_coverage": {
            rank: {
                "labels": len(support[rank]),
                "training_supported_labels": sum(
                    row["training_supported"] for row in support[rank].values()
                ),
                "evaluation_supported_labels": sum(
                    row["evaluation_supported"] for row in support[rank].values()
                ),
                "train_only_supervision_labels": [
                    label for label, row in support[rank].items() if row["train_only_supervision"]
                ],
                "untrained_labels": [
                    label for label, row in support[rank].items() if not row["training_supported"]
                ],
            }
            for rank in RANKS
        },
        "target_mask_policy": (
            "Unresolved ranks are masked; holdout fine ranks additionally require at least "
            "three observed groups and actual train/validation/test support. "
            "IDs and publisher labels remain preserved."
        ),
        "image_storage": "symlink_references_to_unchanged_originals"
        if raw_root
        else "manifest_references",
        "grouping": [
            "file, decoded-pixel, and lossless-transform equivalence",
            "explicit image parents, canonical IDs, biological lineage, and relationship keys",
            "every catalog near-duplicate candidate pair, including excluded bridging images",
            "source export/asset references and explicit culture/specimen relationship keys",
            "soil full-field/crop source-parent links",
            "taxon labels, species-page entities and filename ordinals do not define groups",
        ],
        "limitations": [
            "Observed image relationships and culture references do not establish "
            "independent trees or specimens.",
            "Fine labels with insufficient groups or incomplete partition coverage "
            "remain preserved but are masked for holdout evaluation.",
            "Rare fine labels prefer train; conflicting components or broad class "
            "coverage may leave them untrained and masked in holdouts.",
            "Near-duplicate candidate links can merge unrelated images "
            "and miss other relationships.",
            "Large indivisible families can substantially deviate from target split ratios.",
            "Labels remain publisher-derived and unreviewed; source appearance can predict class.",
            "Partitioning cannot undo prior use of images in model development; "
            "record previous exposure before interpreting held-out evaluation.",
            "This policy replaces partition assignment, not source eligibility "
            "or permission decisions recorded in source_usage.",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".split-", dir=output_dir.parent))
    try:
        if raw_root is not None:
            for split in SPLITS:
                for label in LABEL_MAP:
                    (staging / "images" / split / label).mkdir(parents=True)
            for row in examples:
                original = safe_path(raw_root, row["relative_path"])
                if not original.is_file() or sha256_file(original) != row["sha256"]:
                    raise ValueError(f"Source image changed: {row['image_id']}")
                relative_link = (
                    Path("images")
                    / row["split"]
                    / row["candidate_group"]
                    / (row["image_id"] + original.suffix)
                )
                link = staging / relative_link
                link.symlink_to(os.path.relpath(original, link.parent))
                row["partition_image_path"] = relative_link.as_posix()
        _write_jsonl(staging / "manifest.jsonl", examples)
        _write_csv(staging / "manifest.csv", examples)
        (staging / "partitions").mkdir()
        for split in SPLITS:
            part = [row for row in examples if row["split"] == split]
            _write_jsonl(staging / "partitions" / f"{split}.jsonl", part)
            _write_csv(staging / "partitions" / f"{split}.csv", part)
        _write_jsonl(staging / "groups.jsonl", group_records)
        _write_jsonl(staging / "exclusions.jsonl", exclusions)
        write_json(staging / "label_map.json", LABEL_MAP)
        write_json(staging / "taxonomy_map.json", taxonomy_map)
        write_json(staging / "label_support.json", support)
        shutil.copyfile(usage_path, staging / "source_usage.json")
        shutil.copyfile(catalog_dir / "sources.json", staging / "sources.json")
        metadata["file_sha256"] = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and not path.is_symlink()
        }
        metadata["manifest_sha256"] = metadata["file_sha256"]["manifest.jsonl"]
        metadata["split_version"] = "split-" + object_hash(metadata)[:16]
        write_json(staging / "split.json", metadata)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return metadata
