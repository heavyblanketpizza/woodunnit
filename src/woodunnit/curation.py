"""Deterministic image equivalence checks; visual similarity never deletes data."""

import hashlib
from collections import Counter, defaultdict

from PIL import Image

from .schema import ImageRecord


def _pixel_hash(image: Image.Image) -> str:
    digest = hashlib.sha256(f"rgb-pixels-v1:{image.width}:{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def fingerprints(image: Image.Image) -> dict[str, str]:
    """Hash actual RGB input pixels and their eight lossless square symmetries."""
    rgb = image.convert("RGB")
    pixel_hash = _pixel_hash(rgb)
    transforms = [
        Image.Transpose.FLIP_LEFT_RIGHT,
        Image.Transpose.FLIP_TOP_BOTTOM,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
        Image.Transpose.TRANSPOSE,
        Image.Transpose.TRANSVERSE,
    ]
    equivalents = [pixel_hash]
    for operation in transforms:
        variant = rgb.transpose(operation)
        equivalents.append(_pixel_hash(variant))
        variant.close()
    thumbnail = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = thumbnail.tobytes()
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    thumbnail.close()
    rgb.close()
    return {
        "pixel_sha256": pixel_hash,
        "lossless_transform_sha256": min(equivalents),
        "perceptual_dhash64": f"{bits:016x}",
    }


def curate(records: list[ImageRecord]) -> dict:
    """Select one representative of proven equivalent images, retaining every source record."""
    clusters = defaultdict(list)
    for record in records:
        # A multiframe file can share its first frame while differing in later frames.
        key = (
            ("bytes", record.sha256)
            if record.frame_count != 1
            else ("lossless_rgb", record.lossless_transform_sha256)
        )
        clusters[key].append(record)
    duplicates = []
    for members in clusters.values():
        members.sort(key=lambda r: r.relative_path)
        canonical = members[0]
        # A missing finer rank adds no contradiction; two resolved names do.
        conflict = len({tuple(r.candidate_groups) for r in members}) > 1 or any(
            len(
                {
                    value
                    for record in members
                    if (value := getattr(record.taxonomy, rank)) is not None
                }
            )
            > 1
            for rank in ("genus", "species")
        )
        for record in members:
            record.canonical_image_id = canonical.image_id
            record.selection_status = "included"
            record.exclusion_reason = None
            if conflict:
                record.import_flags = sorted(
                    set(record.import_flags) | {"duplicate_label_conflict"}
                )
            if record is canonical:
                continue
            record.selection_status = "excluded_redundant"
            if record.sha256 == canonical.sha256:
                record.exclusion_reason = "exact_file_duplicate"
            elif record.pixel_sha256 == canonical.pixel_sha256:
                record.exclusion_reason = "exact_pixel_duplicate"
            else:
                record.exclusion_reason = "lossless_transform_duplicate"
        if len(members) > 1:
            duplicates.append(
                {
                    "canonical_image_id": canonical.image_id,
                    "image_ids": [r.image_id for r in members],
                    "source_labels_conflict": conflict,
                    "excluded": [
                        {"image_id": r.image_id, "reason": r.exclusion_reason} for r in members[1:]
                    ],
                }
            )
    reasons = Counter(r.exclusion_reason for r in records if r.exclusion_reason)
    return {
        "fingerprint_version": "rgb-pixels-v1/d4-lossless-v1/dhash64-v1",
        "input_images": len(records),
        "included_images": sum(r.selection_status == "included" for r in records),
        "excluded_images": sum(r.selection_status != "included" for r in records),
        "exclusion_counts": dict(sorted(reasons.items())),
        "equivalent_image_clusters": duplicates,
        "canonical_selection": "lexicographically first relative_path",
        "policy": [
            "Byte equality, decoded RGB equality, and exact lossless flip/rotation equality "
            "may remove redundant working records; raw source images remain unchanged.",
            "Similarity hashes only propose review pairs; they are not proof of duplication.",
            "No crop, resize, brightness, or contrast variants are automatically discarded.",
            "Conflicting source labels remain flagged for review, not resolved by equivalence.",
        ],
    }


def near_duplicate_candidates(records: list[ImageRecord], max_distance: int = 2) -> list[dict]:
    """Find low dHash distances among retained records for inspection, not auto-exclusion.

    Three disjoint bit chunks guarantee retrieval of all 64-bit hash pairs with
    distance <= 2. Uninformative all-zero/all-one hashes are not compared.
    """
    if max_distance != 2:
        raise ValueError("The indexed candidate search supports distance 2 only")
    index = defaultdict(list)
    previous = {}
    pairs = []
    for record in sorted(records, key=lambda r: r.image_id):
        if record.selection_status != "included":
            continue
        bits = int(record.perceptual_dhash64, 16)
        if bits.bit_count() < 4 or bits.bit_count() > 60:
            continue
        keys = [
            (part, (bits >> shift) & ((1 << length) - 1))
            for part, (shift, length) in enumerate(((0, 22), (22, 21), (43, 21)))
        ]
        candidate_ids = {image_id for key in keys for image_id in index[key]}
        for other_id in sorted(candidate_ids):
            other, other_bits = previous[other_id]
            distance = (bits ^ other_bits).bit_count()
            if distance <= max_distance:
                pairs.append(
                    {
                        "image_id_a": other.image_id,
                        "image_id_b": record.image_id,
                        "relative_path_a": other.relative_path,
                        "relative_path_b": record.relative_path,
                        "dhash_hamming_distance": distance,
                        "status": "requires_visual_review",
                    }
                )
        previous[record.image_id] = (record, bits)
        for key in keys:
            index[key].append(record.image_id)
    return pairs
