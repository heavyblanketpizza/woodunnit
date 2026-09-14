import json

import pytest
from PIL import Image
from test_experiment_split import split_inputs as source_split_inputs  # noqa: F401
from test_release import make_inputs, save_assignments

from woodunnit.dataset import HierarchicalManifestDataset, ManifestDataset
from woodunnit.experiment_split import build_split
from woodunnit.io import object_hash, write_json
from woodunnit.release import ReleaseError, build_release, read_jsonl, sha256_file, write_jsonl
from woodunnit.taxonomy import derive_taxonomy


def test_reader_returns_original_sized_rgb_with_stable_class_id(tmp_path):
    args = make_inputs(tmp_path)
    build_release(*args)
    dataset = ManifestDataset(args[4], args[1], split="train")
    image, label = dataset[0]
    assert len(dataset) == 1
    assert image.mode == "RGB"
    assert image.size == (8, 6)
    assert label == 0
    assert dataset.class_map == {"fungi": 0, "oomycetes": 1}
    assert len(ManifestDataset(args[4] / "manifest.jsonl", args[1], split="test")) == 0
    with pytest.raises(IndexError):
        dataset[1]


def test_transform_is_explicit_and_receives_rgb(tmp_path):
    args = make_inputs(tmp_path)
    build_release(*args)
    dataset = ManifestDataset(args[4], args[1], transform=lambda image: (image.mode, image.size))
    value, label = dataset[0]
    assert value == ("RGB", (8, 6))
    assert label == 0


def test_catalog_is_not_a_reviewed_release(tmp_path):
    args = make_inputs(tmp_path)
    with pytest.raises(ReleaseError, match="Missing release.json"):
        ManifestDataset(args[0], args[1])


def test_modified_manifest_is_rejected_before_reading_images(tmp_path):
    args = make_inputs(tmp_path)
    build_release(*args)
    path = args[4] / "manifest.jsonl"
    path.write_text(path.read_text().replace('"class_id": 0', '"class_id": 1'))
    with pytest.raises(ReleaseError, match="Checksum mismatch"):
        ManifestDataset(args[4], args[1], verify_hash=False)


def test_raw_hash_is_checked_each_access(tmp_path):
    args = make_inputs(tmp_path)
    build_release(*args)
    dataset = ManifestDataset(args[4], args[1])
    Image.new("RGB", (8, 6), (250, 250, 250)).save(args[1] / "0.png")
    with pytest.raises(ReleaseError, match="Image checksum mismatch"):
        dataset[0]


def test_reader_detects_dimension_change_even_with_hash_check_disabled(tmp_path):
    args = make_inputs(tmp_path)
    build_release(*args)
    dataset = ManifestDataset(args[4], args[1], verify_hash=False)
    Image.new("RGB", (7, 5)).save(args[1] / "0.png")
    with pytest.raises(ReleaseError, match="Image dimensions changed"):
        dataset[0]


def test_release_status_and_split_are_checked(tmp_path):
    args = make_inputs(tmp_path)
    build_release(*args)
    with pytest.raises(ReleaseError, match="Invalid split"):
        ManifestDataset(args[4], args[1], split="val")
    path = args[4] / "release.json"
    release = json.loads(path.read_text())
    release["status"] = "unreviewed"
    path.write_text(json.dumps(release))
    with pytest.raises(ReleaseError, match="unapproved"):
        ManifestDataset(args[4], args[1])


def test_reader_validates_path_and_class_mapping_even_with_rebound_hash(tmp_path):
    args = make_inputs(tmp_path)
    build_release(*args)
    manifest = args[4] / "manifest.jsonl"
    row = json.loads(manifest.read_text())
    row["relative_path"] = "../escape.png"
    manifest.write_text(json.dumps(row) + "\n")
    metadata_path = args[4] / "release.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["file_sha256"]["manifest.jsonl"] = sha256_file(manifest)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ReleaseError, match="Unsafe relative path"):
        ManifestDataset(args[4], args[1])


def test_reader_supports_validation_partition(tmp_path):
    args = make_inputs(tmp_path)
    save_assignments(args[3], [("group-0", "validation")])
    build_release(*args)
    assert len(ManifestDataset(args[4], args[1], split="validation")) == 1
    assert len(ManifestDataset(args[4], args[1], split="train")) == 0


@pytest.mark.parametrize("split", ["tune", "calibration"])
def test_reader_rejects_retired_partitions(tmp_path, split):
    args = make_inputs(tmp_path)
    build_release(*args)
    with pytest.raises(ReleaseError, match="Invalid split"):
        ManifestDataset(args[4], args[1], split=split)


@pytest.mark.parametrize(
    "class_map",
    [
        {"fungi": 0},
        {"fungi": 1, "oomycetes": 0},
        {"fungi": 0, "oomycetes": 1, "bacteria": 2},
        {"fungi": False, "oomycetes": True},
    ],
)
def test_reader_requires_stable_binary_class_map(tmp_path, class_map):
    args = make_inputs(tmp_path)
    build_release(*args)
    path = args[4] / "release.json"
    metadata = json.loads(path.read_text())
    metadata["class_map"] = class_map
    path.write_text(json.dumps(metadata))
    with pytest.raises(ReleaseError, match="Invalid stable class map"):
        ManifestDataset(args[4], args[1])


@pytest.fixture
def hierarchical_inputs(request, tmp_path):
    catalog, usage, raw = request.getfixturevalue("source_split_inputs")
    path = catalog / "images.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for index, row in enumerate(rows):
        Image.new("RGB", (8, 6), (index * 19, 20, 100)).save(raw / row["relative_path"])
        row.update(sha256=sha256_file(raw / row["relative_path"]), width_px=8, height_px=6)
        name = (
            "Colletotrichum siamense"
            if row["source_id"] == "tgfc"
            else "Phytophthora example"
            if row["image_id"] != "o5"
            else "Phytophthora rare"
        )
        row["source_labels"] = [name]
        row["taxonomy"] = derive_taxonomy(row["source_id"], [name], row["candidate_groups"])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    descriptor = json.loads((catalog / "catalog.json").read_text())
    descriptor["file_sha256"]["images.jsonl"] = sha256_file(path)
    write_json(catalog / "catalog.json", descriptor)
    approval = json.loads(usage.read_text())
    approval.update(
        catalog_sha256=sha256_file(catalog / "catalog.json"),
        images_jsonl_sha256=sha256_file(path),
    )
    write_json(usage, approval)
    output = tmp_path / "hierarchical"
    build_split(catalog, usage, output, raw_root=raw)
    return output, raw


def test_hierarchical_reader_requires_explicit_unreviewed_opt_in(hierarchical_inputs):
    with pytest.raises(ReleaseError, match="allow_unreviewed=True"):
        HierarchicalManifestDataset(*hierarchical_inputs)
    with pytest.raises(ReleaseError, match="Missing release.json"):
        ManifestDataset(*hierarchical_inputs)


def test_hierarchical_reader_exposes_rank_ids_masks_and_original_path(hierarchical_inputs):
    output, raw = hierarchical_inputs
    dataset = HierarchicalManifestDataset(output, raw, allow_unreviewed=True)
    image, labels = dataset[0]
    assert (image.mode, image.size) == ("RGB", (8, 6))
    assert labels == {key: dataset.records[0][key] for key in ("targets", "target_mask")}
    assert labels["target_mask"]["group"] is True
    assert dataset.label_support["species"]["Phytophthora rare"]["evaluation_supported"] is False
    transformed = HierarchicalManifestDataset(
        output, raw, transform=lambda path: path.read_bytes(), allow_unreviewed=True
    )
    assert transformed[0][0] == (raw / dataset.records[0]["relative_path"]).read_bytes()
    assert len(
        HierarchicalManifestDataset(output, raw, split="test", allow_unreviewed=True)
    ) == sum(row["split"] == "test" for row in dataset.records)


@pytest.mark.parametrize("field", ["targets", "target_mask", "relative_path"])
def test_hierarchical_reader_rejects_rebound_corrupt_records(hierarchical_inputs, field):
    output, raw = hierarchical_inputs
    path = output / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if field == "targets":
        rows[0][field]["species"] = 999
    elif field == "target_mask":
        rows[0][field]["group"] = False
    else:
        rows[0][field] = "../escape.png"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metadata = json.loads((output / "split.json").read_text())
    metadata["file_sha256"]["manifest.jsonl"] = sha256_file(path)
    metadata["manifest_sha256"] = sha256_file(path)
    metadata.pop("split_version")
    metadata["split_version"] = "split-" + object_hash(metadata)[:16]
    write_json(output / "split.json", metadata)
    with pytest.raises(ReleaseError, match="targets or masks|Unsafe relative path"):
        HierarchicalManifestDataset(output, raw, allow_unreviewed=True)


def test_hierarchical_reader_detects_changed_original(hierarchical_inputs):
    output, raw = hierarchical_inputs
    dataset = HierarchicalManifestDataset(output, raw, allow_unreviewed=True)
    (raw / dataset.records[0]["relative_path"]).write_bytes(b"changed")
    with pytest.raises(ReleaseError, match="Image checksum mismatch"):
        dataset[0]


def rebind_hierarchical_files(output, *filenames):
    path = output / "split.json"
    metadata = json.loads(path.read_text())
    for filename in filenames:
        metadata["file_sha256"][filename] = sha256_file(output / filename)
    metadata["manifest_sha256"] = metadata["file_sha256"]["manifest.jsonl"]
    metadata.pop("split_version")
    metadata["split_version"] = "split-" + object_hash(metadata)[:16]
    write_json(path, metadata)


def test_hierarchical_reader_requires_bound_full_catalog_group_inventory(hierarchical_inputs):
    output, raw = hierarchical_inputs
    path = output / "split.json"
    metadata = json.loads(path.read_text())
    del metadata["file_sha256"]["groups.jsonl"]
    write_json(path, metadata)
    rebind_hierarchical_files(output)
    with pytest.raises(ReleaseError, match="Incomplete hierarchical split file inventory"):
        HierarchicalManifestDataset(output, raw, allow_unreviewed=True)


@pytest.mark.parametrize("change", ["manifest_group", "excluded_membership"])
def test_hierarchical_reader_validates_full_catalog_membership(hierarchical_inputs, change):
    output, raw = hierarchical_inputs
    filename = "manifest.jsonl" if change == "manifest_group" else "groups.jsonl"
    rows = read_jsonl(output / filename)
    if change == "manifest_group":
        rows[0]["experiment_group_id"] = rows[0]["leakage_group_id"] = "forged-group"
    else:
        group = next(row for row in rows if "excluded" in row["catalog_image_ids"])
        group["catalog_image_ids"].remove("excluded")
    write_jsonl(output / filename, rows)
    rebind_hierarchical_files(output, filename)
    with pytest.raises(ReleaseError, match="frozen group inventory|group inventory coverage"):
        HierarchicalManifestDataset(output, raw, allow_unreviewed=True)


@pytest.mark.parametrize(
    "relation",
    [
        "relationship_keys",
        "sha256",
        "pixel_sha256",
        "lossless_transform_sha256",
        "specimen_id",
        "source_parent_id",
        "parent_image_id",
        "canonical_image_id",
    ],
)
def test_hierarchical_reader_checks_visible_links_and_excluded_parents(
    hierarchical_inputs, relation
):
    output, raw = hierarchical_inputs
    rows = read_jsonl(output / "manifest.jsonl")
    left = next(row for row in rows if row["image_id"] == "f0")
    right = next(
        row
        for row in rows
        if row["source_id"] == left["source_id"] and row["split"] != left["split"]
    )
    if relation == "relationship_keys":
        right[relation] = left[relation]
    elif relation.endswith("sha256"):
        right[relation] = left[relation]
    elif relation == "canonical_image_id":
        right[relation] = "excluded"
    elif relation == "parent_image_id":
        right["lineage"][relation] = "excluded"
    else:
        left["lineage"][relation] = right["lineage"][relation] = "shared-synthetic-parent"
    write_jsonl(output / "manifest.jsonl", rows)
    rebind_hierarchical_files(output, "manifest.jsonl")
    with pytest.raises(ReleaseError, match="Known image relationships cross"):
        HierarchicalManifestDataset(output, raw, allow_unreviewed=True)


@pytest.mark.parametrize("rank", ["group", "genus", "species"])
@pytest.mark.parametrize("non_integer", [False, 0.0])
def test_hierarchical_reader_requires_integer_taxonomy_map_ids(
    hierarchical_inputs, rank, non_integer
):
    output, raw = hierarchical_inputs
    path = output / "taxonomy_map.json"
    taxonomy_map = json.loads(path.read_text())
    label = next(label for label, value in taxonomy_map[rank].items() if value == 0)
    taxonomy_map[rank][label] = non_integer
    write_json(path, taxonomy_map)
    rebind_hierarchical_files(output, "taxonomy_map.json")
    with pytest.raises(ReleaseError, match="taxonomy map requires integer IDs"):
        HierarchicalManifestDataset(output, raw, allow_unreviewed=True)


@pytest.mark.parametrize("class_id", [1, False])
def test_hierarchical_flat_class_id_matches_its_group_target(hierarchical_inputs, class_id):
    output, raw = hierarchical_inputs
    rows = read_jsonl(output / "manifest.jsonl")
    row = next(row for row in rows if row["candidate_group"] == "fungi")
    row["class_id"] = class_id
    write_jsonl(output / "manifest.jsonl", rows)
    rebind_hierarchical_files(output, "manifest.jsonl")
    with pytest.raises(ReleaseError, match="broad class ID differs from group target"):
        HierarchicalManifestDataset(output, raw, allow_unreviewed=True)


def test_hierarchical_opencv_tensors_and_masks_collate_consistently(hierarchical_inputs):
    pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from woodunnit.preprocessing import OpenCVPreprocessor, PreprocessConfig

    output, raw = hierarchical_inputs
    preprocess = OpenCVPreprocessor(PreprocessConfig())
    dataset = HierarchicalManifestDataset(output, raw, transform=preprocess, allow_unreviewed=True)
    assert torch.equal(dataset[0][0], preprocess(raw / dataset.records[0]["relative_path"]))
    images, labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    assert images.shape == (2, 3, 224, 224)
    for rank in ("group", "genus", "species"):
        assert labels["targets"][rank].shape == (2,)
        assert labels["targets"][rank].dtype == torch.int64
        assert labels["target_mask"][rank].dtype == torch.bool


def test_reader_rejects_old_release_without_rewriting_it(tmp_path):
    args = make_inputs(tmp_path)
    build_release(*args)
    path = args[4] / "release.json"
    metadata = json.loads(path.read_text())
    metadata["schema_version"] = "1.0"
    path.write_text(json.dumps(metadata))
    before = path.read_bytes()
    with pytest.raises(ReleaseError, match="Unsupported or unapproved"):
        ManifestDataset(args[4], args[1])
    assert path.read_bytes() == before
