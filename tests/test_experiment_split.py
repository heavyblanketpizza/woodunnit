import json
from collections import Counter

import pytest

from woodunnit.experiment_split import (
    assign_groups,
    build_split,
    label_support,
    observed_groups,
    read_jsonl,
)
from woodunnit.io import canonical_json, object_hash, sha256_file, write_json
from woodunnit.taxonomy import derive_taxonomy


def record(image_id, source="tgfc", taxon=None, label="fungi"):
    return {
        "image_id": image_id,
        "source_id": source,
        "source_labels": [taxon or image_id],
        "taxonomy": derive_taxonomy(source, [taxon or image_id], [label]),
        "annotations": [],
        "candidate_groups": [label],
        "source_metadata": {"entity_ids": [taxon or image_id]} if source == "idphy" else {},
        "sha256": object_hash([image_id, "file"]),
        "pixel_sha256": object_hash([image_id, "pixels"]),
        "lossless_transform_sha256": object_hash([image_id, "lossless"]),
        "frame_count": 1,
        "lineage": {},
        "relationship_keys": [],
        "canonical_image_id": image_id,
        "selection_status": "included",
        "review_status": "unreviewed",
        "group_label": None,
        "import_flags": [],
        "image_role": "full_field",
        "relative_path": f"{source}/{image_id}.png",
        "width_px": 1,
        "height_px": 1,
        "leakage_group_id": "whole-source-" + source,
    }


def save_jsonl(path, rows):
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))


@pytest.fixture
def split_inputs(tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    rows = [record(f"f{i}") for i in range(6)] + [
        record(f"o{i}", "idphy", label="oomycetes") for i in range(6)
    ]
    excluded = record("excluded", taxon="f0")
    rows[0]["relationship_keys"] = ["observed-fixture-family"]
    excluded["relationship_keys"] = ["observed-fixture-family"]
    rows.append(excluded)
    for row in rows:
        original = raw_root / row["relative_path"]
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(row["image_id"].encode())
        row["sha256"] = sha256_file(original)
    save_jsonl(catalog / "images.jsonl", rows)
    write_json(catalog / "sources.json", {})
    save_jsonl(catalog / "near_duplicate_candidates.jsonl", [])
    descriptor = {
        "schema_version": "3.0",
        "catalog_id": "fixture-catalog",
        "file_sha256": {
            name: sha256_file(catalog / name)
            for name in ("images.jsonl", "sources.json", "near_duplicate_candidates.jsonl")
        },
    }
    write_json(catalog / "catalog.json", descriptor)
    policies = {}
    for source in ("tgfc", "idphy"):
        evidence = raw_root / f"{source}-license.json"
        evidence.write_text("fixture license evidence")
        policies[source] = {
            "license": "fixture license",
            "attribution": "fixture attribution",
            "expected_images": 6,
            "split": "train",
            "license_evidence_ref": evidence.name,
            "license_evidence_sha256": sha256_file(evidence),
        }
    usage = {
        "catalog_id": "fixture-catalog",
        "catalog_sha256": sha256_file(catalog / "catalog.json"),
        "images_jsonl_sha256": descriptor["file_sha256"]["images.jsonl"],
        "sources_json_sha256": descriptor["file_sha256"]["sources.json"],
        "eligible_image_ids": [row["image_id"] for row in rows[:-1]],
        "sources": policies,
        "excluded_images": [{"image_id": "excluded", "reason": "deferred fixture"}],
    }
    usage_path = tmp_path / "usage.json"
    write_json(usage_path, usage)
    return catalog, usage_path, raw_root


def test_threeway_is_deterministic_and_preserves_eligibility(split_inputs, tmp_path):
    catalog, usage, _ = split_inputs
    original_catalog_hash = sha256_file(catalog / "images.jsonl")
    first, second = tmp_path / "first", tmp_path / "second"
    result = build_split(catalog, usage, first)
    assert result == build_split(catalog, usage, second)
    assert result["eligible_images"] == 12
    assert result["excluded_images"] == 1
    assert result["catalog_observed_groups"] == 12
    assert result["catalog_observed_groups"] == result["eligible_observed_groups"]
    assert sha256_file(catalog / "images.jsonl") == original_catalog_hash
    rows = read_jsonl(first / "manifest.jsonl")
    assert {row["image_id"] for row in rows} == set(
        json.loads(usage.read_text())["eligible_image_ids"]
    )
    for split in ("train", "validation", "test"):
        part = read_jsonl(first / "partitions" / f"{split}.jsonl")
        assert {row["class_id"] for row in part} == {0, 1}
        assert all(row["split"] == split for row in part)
        assert all(row["review_status"] == "unreviewed" for row in part)
        assert all(row["state"] == "review_required" for row in part)
    groups = {row["experiment_group_id"]: row for row in read_jsonl(first / "groups.jsonl")}
    excluded = read_jsonl(first / "exclusions.jsonl")[0]
    assert excluded["reason"] == "deferred fixture"
    assert excluded["related_partition"] == groups[excluded["experiment_group_id"]]["split"]
    assert all(row["source_leakage_group_id"].startswith("whole-source-") for row in rows)
    assert all(row["experiment_group_id"] == row["leakage_group_id"] for row in rows)
    for name, digest in result["file_sha256"].items():
        assert sha256_file(first / name) == digest


def test_image_folders_link_unchanged_originals(split_inputs, tmp_path):
    catalog, usage, raw_root = split_inputs
    output = tmp_path / "split"
    result = build_split(catalog, usage, output, raw_root=raw_root)
    assert result["image_storage"] == "symlink_references_to_unchanged_originals"
    for row in read_jsonl(output / "manifest.jsonl"):
        link = output / row["partition_image_path"]
        assert link.is_symlink()
        assert link.resolve() == (raw_root / row["relative_path"]).resolve()
        assert sha256_file(link) == row["sha256"]
        assert link.parts[-3:-1] == (row["split"], row["candidate_group"])


def test_immutable_output_is_not_overwritten(split_inputs, tmp_path):
    catalog, usage, _ = split_inputs
    output = tmp_path / "split"
    build_split(catalog, usage, output)
    with pytest.raises(ValueError, match="already exists"):
        build_split(catalog, usage, output)


def test_split_cli_creates_partitions_with_original_image_links(split_inputs, tmp_path, capsys):
    from woodunnit.cli import main

    catalog, usage, raw_root = split_inputs
    config = tmp_path / "ingestion.toml"
    (raw_root / "image_inventory.csv").write_text(
        "collection,path,bytes,sha256,width,height,format\n"
    )
    config.write_text('raw_root = "raw"\noutput_root = "derived"\n')
    output = tmp_path / "split"
    assert (
        main(
            [
                "split",
                "--config",
                str(config),
                "--catalog",
                str(catalog),
                "--usage",
                str(usage),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["eligible_images"] == 12
    for row in read_jsonl(output / "manifest.jsonl"):
        assert (output / row["partition_image_path"]).resolve() == raw_root / row["relative_path"]


@pytest.mark.parametrize(
    "changed_file", ["images.jsonl", "sources.json", "near_duplicate_candidates.jsonl"]
)
def test_catalog_tampering_fails(split_inputs, tmp_path, changed_file):
    catalog, usage, _ = split_inputs
    with (catalog / changed_file).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="bind|integrity"):
        build_split(catalog, usage, tmp_path / "split")


def test_unknown_eligibility_fails(split_inputs, tmp_path):
    catalog, usage, _ = split_inputs
    data = json.loads(usage.read_text())
    data["eligible_image_ids"].append("unknown")
    write_json(usage, data)
    with pytest.raises(ValueError, match="unique known"):
        build_split(catalog, usage, tmp_path / "split")


def test_changed_license_evidence_fails(split_inputs, tmp_path):
    catalog, usage, raw_root = split_inputs
    (raw_root / "tgfc-license.json").write_text("changed")
    with pytest.raises(ValueError, match="license evidence changed"):
        build_split(catalog, usage, tmp_path / "split", raw_root=raw_root)


def test_changed_original_cleans_partial_output(split_inputs, tmp_path):
    catalog, usage, raw_root = split_inputs
    (raw_root / "tgfc/f0.png").write_text("changed")
    with pytest.raises(ValueError, match="Source image changed"):
        build_split(catalog, usage, tmp_path / "split", raw_root=raw_root)
    assert not (tmp_path / "split").exists()
    assert not list(tmp_path.glob(".split-*"))


def test_source_coarse_ids_are_not_mistaken_for_observed_family():
    rows = [
        record("a", taxon="taxon-a"),
        record("b", taxon="taxon-b"),
        record("c", taxon="taxon-a"),
    ]
    groups = observed_groups(rows, [])
    assert len(set(groups.values())) == 3


def test_excluded_records_keep_relationship_bridges():
    rows = [record("a"), record("bridge"), record("b")]
    rows[0]["relationship_keys"] = ["shared-observation"]
    rows[1]["relationship_keys"] = ["shared-observation"]
    rows[1]["selection_status"] = "excluded_redundant"
    groups = observed_groups(rows, [{"image_id_a": "bridge", "image_id_b": "b"}])
    assert len(set(groups.values())) == 1


@pytest.mark.parametrize("field", ["sha256", "pixel_sha256", "lossless_transform_sha256"])
def test_equivalence_links_different_source_families(field):
    rows = [record("a"), record("b", "soil")]
    rows[1][field] = rows[0][field]
    assert len(set(observed_groups(rows, []).values())) == 1


def test_multiframe_first_frame_equality_does_not_link_different_files():
    rows = [record("a"), record("b")]
    for row in rows:
        row.update(frame_count=2, pixel_sha256="same", lossless_transform_sha256="same")
    assert len(set(observed_groups(rows, []).values())) == 2


def test_species_page_entities_and_aliases_are_not_biological_relationships():
    rows = [record("a", "idphy"), record("alias", "idphy"), record("b", "idphy")]
    rows[0]["source_metadata"]["entity_ids"] = ["entity-a"]
    rows[1]["source_metadata"]["entity_ids"] = ["entity-a", "entity-b"]
    rows[2]["source_metadata"]["entity_ids"] = ["entity-b"]
    assert len(set(observed_groups(rows, []).values())) == 3


def test_parent_and_biological_ids_are_joined():
    rows = [record("parent"), record("crop"), record("other", "soil")]
    rows[1]["lineage"] = {"parent_image_id": "parent", "specimen_id": "shared-specimen"}
    rows[2]["lineage"] = {"specimen_id": "shared-specimen"}
    assert len(set(observed_groups(rows, []).values())) == 1


def test_unknown_near_pair_endpoint_fails():
    with pytest.raises(ValueError, match="unknown image"):
        observed_groups([record("a")], [{"image_id_a": "a", "image_id_b": "missing"}])


def test_too_few_class_groups_is_not_fixed_by_splitting_relatives():
    rows = [record("a"), record("b")] + [
        record(f"o{i}", "idphy", label="oomycetes") for i in range(3)
    ]
    with pytest.raises(ValueError, match="Fewer than three"):
        assign_groups(rows, observed_groups(rows, []), 42)


def test_large_recorded_families_are_kept_intact_with_deterministic_assignment():
    rows = []
    for group, size in enumerate((70, 15, 10, 5)):
        for index in range(size):
            row = record(f"f{group}-{index}", taxon="Colletotrichum siamense")
            row["relationship_keys"] = [f"observed-fungal-family-{group}"]
            rows.append(row)
    rows.extend(
        record(f"o{i}", "idphy", taxon="Phytophthora cinnamomi", label="oomycetes")
        for i in range(60)
    )
    groups = observed_groups(rows, [])
    assignments = assign_groups(rows, groups, 42)
    assert assignments == assign_groups(list(reversed(rows)), groups, 42)
    assert groups == observed_groups(list(reversed(rows)), [])
    counts = Counter(
        (assignments[groups[row["image_id"]]], row["candidate_groups"][0]) for row in rows
    )
    assert all(
        counts[split, label] > 0
        for split in ("train", "validation", "test")
        for label in ("fungi", "oomycetes")
    )
    assert max(counts[split, "fungi"] for split in ("train", "validation", "test")) >= 70
    assert len({groups[row["image_id"]] for row in rows if row["source_id"] == "tgfc"}) == 4


def test_supported_fine_labels_are_represented_in_every_partition():
    rows = [
        record(f"{taxon}-{index}", source, taxon, label)
        for source, label, taxon in (
            ("tgfc", "fungi", "Colletotrichum siamense"),
            ("tgfc", "fungi", "Olivea tectonae"),
            ("tgfc", "fungi", "Neopestalotiopsis sp."),
            ("idphy", "oomycetes", "Phytophthora cinnamomi"),
            ("idphy", "oomycetes", "Phytophthora ramorum"),
        )
        for index in range(9)
    ]
    groups = observed_groups(rows, [])
    assignment = assign_groups(rows, groups, 13)
    support = label_support(rows, groups, assignment)
    assert len(support["species"]) == 4
    for rank in ("group", "genus", "species"):
        assert all(item["evaluation_supported"] for item in support[rank].values())
        assert all(all(item["group_counts"].values()) for item in support[rank].values())


def test_rare_fine_labels_do_not_force_a_whole_class_into_train():
    # All source species have one observed family. Broad/group labels can still
    # be evaluated, while none of these species can support three-way evaluation.
    rows = [record(f"f{i}", taxon="Colletotrichum siamense") for i in range(9)]
    rows += [
        record(f"o{i}", "idphy", f"Phytophthora species{chr(97 + i)}", "oomycetes")
        for i in range(9)
    ]
    groups = observed_groups(rows, [])
    assignments = assign_groups(rows, groups, 42)
    support = label_support(rows, groups, assignments)
    assert all(support["group"]["oomycetes"]["image_counts"].values())
    rare = [item for label, item in support["species"].items() if label.startswith("Phytophthora")]
    assert len(rare) == 9
    assert all(not item["evaluation_supported"] for item in rare)
    assert sum(item["training_supported"] for item in rare) >= 5
    assert all(
        "fewer_than_three_observed_groups" in item["evaluation_mask_reasons"] for item in rare
    )


def test_explicit_culture_references_bridge_taxa_and_excluded_images():
    rows = [
        record("a", "idphy", "Phytophthora ramorum", "oomycetes"),
        record("bridge", "idphy", "Phytophthora ramorum", "oomycetes"),
        record("b", "idphy", "Phytophthora capsici", "oomycetes"),
    ]
    rows[0]["relationship_keys"] = ["idphy:culture:CPHST BL 55G"]
    rows[1]["relationship_keys"] = ["idphy:culture:CPHST BL 55G", "idphy:culture:CPHST BL 33"]
    rows[1]["selection_status"] = "excluded_redundant"
    rows[2]["relationship_keys"] = ["idphy:culture:CPHST BL 33"]
    assert len(set(observed_groups(rows, []).values())) == 1


def test_legacy_catalog_requires_reingestion(split_inputs, tmp_path):
    catalog, usage, _ = split_inputs
    descriptor = json.loads((catalog / "catalog.json").read_text())
    descriptor["schema_version"] = "2.0"
    write_json(catalog / "catalog.json", descriptor)
    with pytest.raises(ValueError, match="re-ingest"):
        build_split(catalog, usage, tmp_path / "split")


def test_manifest_preserves_annotations_and_masks_unsupported_holdout_ranks(split_inputs, tmp_path):
    catalog, usage, _ = split_inputs
    records = read_jsonl(catalog / "images.jsonl")
    for row in records:
        name = (
            "Colletotrichum siamense"
            if row["source_id"] == "tgfc"
            else "Phytophthora taxon" + chr(97 + records.index(row))
        )
        row["source_labels"] = [name]
        row["taxonomy"] = derive_taxonomy(row["source_id"], [name], row["candidate_groups"])
        row["annotations"] = [{"label": name, "coordinates": [0.5, 0.5, 0.2, 0.2]}]
    save_jsonl(catalog / "images.jsonl", records)
    descriptor = json.loads((catalog / "catalog.json").read_text())
    descriptor["file_sha256"]["images.jsonl"] = sha256_file(catalog / "images.jsonl")
    write_json(catalog / "catalog.json", descriptor)
    approval = json.loads(usage.read_text())
    approval["images_jsonl_sha256"] = descriptor["file_sha256"]["images.jsonl"]
    approval["catalog_sha256"] = sha256_file(catalog / "catalog.json")
    write_json(usage, approval)
    output = tmp_path / "split"
    result = build_split(catalog, usage, output)
    assert result["schema_version"] == "3.0"
    assert result["kind"] == "exploratory_hierarchical_split"
    assert result["cross_partition_known_relationships"] == 0
    assert result["verified_partition_groups"] == result["eligible_observed_groups"]
    assert {"taxonomy_map.json", "label_support.json"} <= result["file_sha256"].keys()
    originals = {row["image_id"]: row for row in records}
    for row in read_jsonl(output / "manifest.jsonl"):
        assert row["annotations"] == originals[row["image_id"]]["annotations"]
        assert row["source_metadata"] == originals[row["image_id"]]["source_metadata"]
        assert row["taxonomy"] == originals[row["image_id"]]["taxonomy"]
        assert row["targets"]["group"] == row["class_id"]
        assert row["target_mask"]["genus"]
        if row["source_id"] == "idphy" and row["split"] != "train":
            assert row["targets"]["species"] >= 0
            assert not row["target_mask"]["species"]
            assert "fewer_than_three_observed_groups" in row["target_mask_reasons"]["species"]
        else:
            assert row["target_mask"]["species"]


@pytest.mark.parametrize("destination", ["raw", "catalog"])
def test_output_cannot_mutate_source_or_catalog(split_inputs, destination):
    catalog, usage, raw_root = split_inputs
    root = raw_root if destination == "raw" else catalog
    with pytest.raises(ValueError, match="outside"):
        build_split(catalog, usage, root / "split", raw_root=raw_root)


def test_shared_cross_class_components_do_not_exhaust_incompatible_holdouts():
    rows = [
        record("f-only", taxon="Colletotrichum siamense"),
        record("o-only", "idphy", "Phytophthora ramorum", "oomycetes"),
    ]
    for index in range(2):
        fungus = record(f"f{index}", taxon="Colletotrichum siamense")
        oomycete = record(f"o{index}", "idphy", "Phytophthora ramorum", "oomycetes")
        fungus["relationship_keys"] = oomycete["relationship_keys"] = [f"shared-{index}"]
        rows += [fungus, oomycete]
    groups = observed_groups(rows, [])
    for seed in range(10):
        assignment = assign_groups(rows, groups, seed)
        support = label_support(rows, groups, assignment)
        assert assignment[groups["f-only"]] == assignment[groups["o-only"]]
        assert all(all(item["group_counts"].values()) for item in support["group"].values())


def test_fabricated_taxonomy_is_rejected_even_if_catalog_hashes_are_rebound(split_inputs, tmp_path):
    catalog, usage, _ = split_inputs
    rows = read_jsonl(catalog / "images.jsonl")
    rows[0]["taxonomy"]["species"] = "Colletotrichum fabricated"
    save_jsonl(catalog / "images.jsonl", rows)
    descriptor = json.loads((catalog / "catalog.json").read_text())
    descriptor["file_sha256"]["images.jsonl"] = sha256_file(catalog / "images.jsonl")
    write_json(catalog / "catalog.json", descriptor)
    approval = json.loads(usage.read_text())
    approval["images_jsonl_sha256"] = descriptor["file_sha256"]["images.jsonl"]
    approval["catalog_sha256"] = sha256_file(catalog / "catalog.json")
    write_json(usage, approval)
    with pytest.raises(ValueError, match="taxonomy must match"):
        build_split(catalog, usage, tmp_path / "split")


def test_rare_species_retain_training_examples_when_other_families_cover_holdouts():
    rows = [record(f"f{i}", taxon="Colletotrichum siamense") for i in range(12)]
    rows += [record(f"common{i}", "idphy", "Phytophthora ramorum", "oomycetes") for i in range(12)]
    rows += [
        record(f"rare{i}", "idphy", f"Phytophthora rare{chr(97 + i)}", "oomycetes")
        for i in range(6)
    ]
    groups = observed_groups(rows, [])
    assignment = assign_groups(rows, groups, 42)
    support = label_support(rows, groups, assignment)
    for index in range(6):
        assert assignment[groups[f"rare{index}"]] == "train"
    assert all(item["training_supported"] for item in support["species"].values())
    assert support["species"]["Phytophthora ramorum"]["evaluation_supported"]
