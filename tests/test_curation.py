"""Proven image equivalence removes working copies; similarity needs human review."""

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from woodunnit.curation import curate, fingerprints, near_duplicate_candidates
from woodunnit.pipeline import group_records, image_id
from woodunnit.schema import ImageRecord
from woodunnit.taxonomy import derive_taxonomy


@pytest.fixture
def microscope_pattern():
    """An asymmetric raster avoids accidentally testing a transform-invariant solid color."""
    image = Image.new("RGB", (31, 23))
    image.putdata(
        [
            ((x * 13 + y * 7) % 256, (x * y * 3) % 256, (x + y * 11) % 256)
            for y in range(image.height)
            for x in range(image.width)
        ]
    )
    yield image
    image.close()


def make_record(
    root: Path,
    relative_path: str,
    image: Image.Image,
    *,
    source_id: str = "tgfc",
    group: str = "fungi",
    source_labels: list[str] | None = None,
    **save_options,
) -> ImageRecord:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **save_options)
    contents = path.read_bytes()
    labels = [f"Synthetic {group} taxon"] if source_labels is None else source_labels
    with Image.open(path) as saved:
        saved.load()
        return ImageRecord(
            image_id=image_id(source_id, relative_path),
            source_id=source_id,
            relative_path=relative_path,
            sha256=hashlib.sha256(contents).hexdigest(),
            **fingerprints(saved),
            bytes=len(contents),
            width_px=saved.width,
            height_px=saved.height,
            file_format=saved.format,
            image_mode=saved.mode,
            frame_count=getattr(saved, "n_frames", 1),
            source_labels=labels,
            candidate_groups=[group],
            taxonomy=derive_taxonomy(source_id, labels, [group]),
            image_role="full_field",
            leakage_group_id="pending",
            usage_permission_ref=f"sources.json#{source_id}",
            import_flags=["source_label_unreviewed"],
        )


def test_exact_file_duplicate_keeps_deterministic_representative(tmp_path, microscope_pattern):
    first = make_record(tmp_path, "tgfc/a.png", microscope_pattern)
    second = make_record(tmp_path, "tgfc/z.png", microscope_pattern)
    assert first.sha256 == second.sha256
    records = [second, first]
    result = curate(records)
    assert result["included_images"] == 1
    assert result["excluded_images"] == 1
    assert result["exclusion_counts"] == {"exact_file_duplicate": 1}
    assert first.selection_status == "included"
    assert second.selection_status == "excluded_redundant"
    assert first.canonical_image_id == second.canonical_image_id == first.image_id
    assert second.exclusion_reason == "exact_file_duplicate"
    assert curate(list(reversed(records))) == result
    assert all((tmp_path / r.relative_path).is_file() for r in records)
    assert all(r.review_status == "unreviewed" and r.group_label is None for r in records)


def test_different_encodings_of_identical_rgb_pixels_are_redundant(tmp_path, microscope_pattern):
    first = make_record(tmp_path, "tgfc/a.png", microscope_pattern, compress_level=0)
    second = make_record(tmp_path, "tgfc/b.png", microscope_pattern, compress_level=9)
    assert first.sha256 != second.sha256
    assert first.pixel_sha256 == second.pixel_sha256
    result = curate([first, second])
    assert result["exclusion_counts"] == {"exact_pixel_duplicate": 1}
    assert second.canonical_image_id == first.image_id


@pytest.mark.parametrize("operation", list(Image.Transpose))
def test_lossless_rotations_and_flips_are_redundant(tmp_path, microscope_pattern, operation):
    first = make_record(tmp_path, "tgfc/a.png", microscope_pattern)
    with microscope_pattern.transpose(operation) as transformed:
        second = make_record(tmp_path, "tgfc/b.png", transformed)
    assert first.sha256 != second.sha256
    assert first.pixel_sha256 != second.pixel_sha256
    assert first.lossless_transform_sha256 == second.lossless_transform_sha256
    result = curate([second, first])
    assert result["exclusion_counts"] == {"lossless_transform_duplicate": 1}
    assert second.canonical_image_id == first.image_id


def test_pixel_dimensions_participate_in_equivalence_fingerprint():
    with Image.new("RGB", (8, 12), "gray") as first:
        with Image.new("RGB", (6, 16), "gray") as second:
            assert first.tobytes() == second.tobytes()
            assert fingerprints(first)["pixel_sha256"] != fingerprints(second)["pixel_sha256"]
            assert (
                fingerprints(first)["lossless_transform_sha256"]
                != fingerprints(second)["lossless_transform_sha256"]
            )


def test_distinct_similar_structure_is_retained(tmp_path, microscope_pattern):
    first = make_record(tmp_path, "tgfc/a.png", microscope_pattern)
    with microscope_pattern.copy() as difficult:
        red, green, blue = difficult.getpixel((5, 5))
        difficult.putpixel((5, 5), (red ^ 1, green, blue))
        second = make_record(tmp_path, "tgfc/b.png", difficult)
    assert first.pixel_sha256 != second.pixel_sha256
    assert first.lossless_transform_sha256 != second.lossless_transform_sha256
    result = curate([first, second])
    assert result["included_images"] == 2
    assert result["excluded_images"] == 0
    assert first.canonical_image_id == first.image_id
    assert second.canonical_image_id == second.image_id


def test_multiframe_files_are_not_deduplicated_from_first_frame(tmp_path, microscope_pattern):
    with Image.new("RGB", microscope_pattern.size, "red") as second_frame:
        first = make_record(
            tmp_path,
            "tgfc/a.tif",
            microscope_pattern,
            save_all=True,
            append_images=[second_frame],
        )
        exact = make_record(
            tmp_path,
            "tgfc/a_copy.tif",
            microscope_pattern,
            save_all=True,
            append_images=[second_frame],
        )
    with Image.new("RGB", microscope_pattern.size, "blue") as second_frame:
        different = make_record(
            tmp_path,
            "tgfc/b.tif",
            microscope_pattern,
            save_all=True,
            append_images=[second_frame],
        )
    single = make_record(tmp_path, "tgfc/single.png", microscope_pattern)
    assert first.frame_count == exact.frame_count == different.frame_count == 2
    assert first.sha256 == exact.sha256 != different.sha256
    assert first.pixel_sha256 == different.pixel_sha256 == single.pixel_sha256
    records = [first, exact, different, single]
    result = curate(records)
    assert result["included_images"] == 3
    assert result["exclusion_counts"] == {"exact_file_duplicate": 1}
    assert different.selection_status == single.selection_status == "included"


def test_conflicting_group_labels_remain_flagged_and_unreviewed(tmp_path, microscope_pattern):
    fungus = make_record(tmp_path, "tgfc/reference.png", microscope_pattern)
    oomycete = make_record(
        tmp_path,
        "idphy/reference.png",
        microscope_pattern,
        source_id="idphy",
        group="oomycetes",
    )
    records = [fungus, oomycete]
    result = curate(records)
    assert result["equivalent_image_clusters"][0]["source_labels_conflict"] is True
    for record in records:
        assert "duplicate_label_conflict" in record.import_flags
        assert "source_label_unreviewed" in record.import_flags
        assert record.group_label is None
        assert record.review_status == "unreviewed"
    assert fungus.candidate_groups == ["fungi"]
    assert oomycete.candidate_groups == ["oomycetes"]


@pytest.mark.parametrize(
    ("source", "group", "labels"),
    [
        ("tgfc", "fungi", ["Colletotrichum siamense", "Olivea tectonae"]),
        ("idphy", "oomycetes", ["Phytophthora examplea", "Phytophthora exampleb"]),
    ],
)
def test_conflicting_fine_labels_flag_every_equivalent_member(
    tmp_path, microscope_pattern, source, group, labels
):
    records = [
        make_record(
            tmp_path,
            f"{source}/{index}.png",
            microscope_pattern,
            source_id=source,
            group=group,
            source_labels=[label],
        )
        for index, label in enumerate(labels)
    ]
    records.append(
        make_record(
            tmp_path,
            f"{source}/unresolved.png",
            microscope_pattern,
            source_id=source,
            group=group,
        )
    )
    originals = [(record.source_labels.copy(), record.taxonomy.model_dump()) for record in records]
    result = curate(records)
    assert result["included_images"] == 1 and result["excluded_images"] == 2
    assert result["equivalent_image_clusters"][0]["source_labels_conflict"] is True
    assert all("duplicate_label_conflict" in record.import_flags for record in records)
    assert [(record.source_labels, record.taxonomy.model_dump()) for record in records] == originals
    assert all(record.review_status == "unreviewed" for record in records)


@pytest.mark.parametrize("partial_label", ["Phytophthora sp.", "Synthetic unresolved taxon"])
def test_missing_fine_rank_does_not_conflict_with_a_compatible_species(
    tmp_path, microscope_pattern, partial_label
):
    records = [
        make_record(
            tmp_path,
            f"idphy/{index}.png",
            microscope_pattern,
            source_id="idphy",
            group="oomycetes",
            source_labels=[label],
        )
        for index, label in enumerate(["Phytophthora examplea", partial_label])
    ]
    result = curate(records)
    assert result["equivalent_image_clusters"][0]["source_labels_conflict"] is False
    assert all("duplicate_label_conflict" not in record.import_flags for record in records)
    assert result["included_images"] == 1 and result["excluded_images"] == 1


def test_equivalent_images_connect_cross_source_leakage_groups(tmp_path, microscope_pattern):
    fungus = make_record(tmp_path, "tgfc/reference.png", microscope_pattern, compress_level=0)
    oomycete = make_record(
        tmp_path,
        "idphy/reference.png",
        microscope_pattern,
        source_id="idphy",
        group="oomycetes",
        compress_level=9,
    )
    assert fungus.sha256 != oomycete.sha256
    curate([fungus, oomycete])
    group_records([fungus, oomycete])
    assert fungus.leakage_group_id == oomycete.leakage_group_id


def test_parent_links_connect_cross_source_leakage_groups(tmp_path, microscope_pattern):
    parent = make_record(tmp_path, "tgfc/reference.png", microscope_pattern)
    with microscope_pattern.crop((0, 0, 10, 10)) as cropped:
        child = make_record(tmp_path, "soil/crop.png", cropped, source_id="soil")
    child.lineage["parent_image_id"] = parent.image_id
    child.image_role = "crop"
    curate([parent, child])
    group_records([parent, child])
    assert parent.selection_status == child.selection_status == "included"
    assert parent.leakage_group_id == child.leakage_group_id


def test_near_hash_pairs_are_review_candidates_only(tmp_path, microscope_pattern):
    records = [
        make_record(tmp_path, f"tgfc/{name}.png", microscope_pattern)
        for name in ("a", "b", "c", "d")
    ]
    # Exercise indexed retrieval across two changed hash chunks and distance filtering.
    base = 0xA5A5A5A5A5A5A5A5
    bits = [base, base ^ 1, base ^ 1 ^ (1 << 30), base ^ 1 ^ (1 << 30) ^ (1 << 50)]
    for record, value in zip(records, bits, strict=True):
        record.perceptual_dhash64 = f"{value:016x}"
    before = [record.model_dump() for record in records]
    candidates = near_duplicate_candidates(records)
    expected = {
        frozenset((left.image_id, right.image_id))
        for index, left in enumerate(records)
        for right in records[index + 1 :]
        if (int(left.perceptual_dhash64, 16) ^ int(right.perceptual_dhash64, 16)).bit_count() <= 2
    }
    assert {frozenset((r["image_id_a"], r["image_id_b"])) for r in candidates} == expected
    assert all(item["status"] == "requires_visual_review" for item in candidates)
    assert before == [record.model_dump() for record in records]
    assert all(record.selection_status == "included" for record in records)


def test_near_hash_search_omits_redundant_or_uninformative_records(tmp_path, microscope_pattern):
    records = [make_record(tmp_path, f"tgfc/{index}.png", microscope_pattern) for index in range(4)]
    records[0].perceptual_dhash64 = "0000000000000000"
    records[1].perceptual_dhash64 = "ffffffffffffffff"
    records[2].perceptual_dhash64 = records[3].perceptual_dhash64 = "a5a5a5a5a5a5a5a5"
    records[2].selection_status = "excluded_redundant"
    records[2].canonical_image_id = records[3].image_id
    records[2].exclusion_reason = "exact_file_duplicate"
    assert near_duplicate_candidates(records) == []


def test_near_hash_search_rejects_unsupported_distance():
    with pytest.raises(ValueError, match="distance 2 only"):
        near_duplicate_candidates([], max_distance=3)
