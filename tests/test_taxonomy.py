"""Synthetic labels exercise partial, ambiguous, and hierarchical supervision."""

import copy

import pytest
from pydantic import ValidationError

from woodunnit.schema import ImageRecord, SourceTaxon, Taxonomy
from woodunnit.taxonomy import build_taxonomy_map, derive_taxonomy, hierarchy_targets


@pytest.mark.parametrize(
    ("label", "genus", "species"),
    [
        ("Colletotrichum siamense", "Colletotrichum", "Colletotrichum siamense"),
        ("Olivea tectonae", "Olivea", "Olivea tectonae"),
        ("Neopestalotiopsis sp.", "Neopestalotiopsis", None),
    ],
)
def test_tgfc_recovers_only_the_supplied_ranks(label, genus, species):
    result = derive_taxonomy("tgfc", [label], ["fungi"])
    assert result["group"] == "fungi"
    assert result["genus"] == genus
    assert result["species"] == species
    assert result["status"] == "single_taxon"
    assert result["taxa"][0]["source_label"] == label
    assert Taxonomy.model_validate(result).model_dump() == result


def test_mixedclass_does_not_invent_per_image_taxa():
    result = derive_taxonomy("tgfc", ["MixedClass"], ["fungi"])
    assert (result["group"], result["genus"], result["species"]) == ("fungi", None, None)
    assert result["status"] == "group_only"
    assert result["taxa"][0]["source_label"] == "MixedClass"
    assert "unavailable" in result["taxa"][0]["mapping_note"]


def test_multiple_tgfc_labels_keep_all_source_taxa_without_a_single_fine_target():
    labels = ["Olivea tectonae", "Colletotrichum siamense"]
    result = derive_taxonomy("tgfc", labels, ["fungi"])
    assert result["status"] == "multiple_taxa"
    assert result["group"] == "fungi"
    assert result["genus"] is None and result["species"] is None
    assert [taxon["source_label"] for taxon in result["taxa"]] == sorted(labels)
    assert [taxon["species"] for taxon in result["taxa"]] == sorted(labels)


def test_multiple_species_can_supply_a_shared_genus_target():
    labels = ["Phytophthora examplea", "Phytophthora exampleb"]
    result = derive_taxonomy("idphy", labels, ["oomycetes"])
    assert result["status"] == "multiple_taxa"
    assert result["genus"] == "Phytophthora"
    assert result["species"] is None
    label_map = build_taxonomy_map([{"taxonomy": result}])
    assert label_map["species"] == {name: index for index, name in enumerate(labels)}
    assert hierarchy_targets(result, label_map) == {
        "targets": {"group": 1, "genus": 0, "species": -1},
        "target_mask": {"group": True, "genus": True, "species": False},
    }


@pytest.mark.parametrize("label", ["Phytophtora", "Phytophthora"])
def test_soil_spelling_mapping_preserves_original_and_does_not_invent_species(label):
    result = derive_taxonomy("soil", [label], ["oomycetes"])
    assert result["genus"] == "Phytophthora"
    assert result["species"] is None
    taxon = result["taxa"][0]
    assert taxon["source_label"] == label
    if label == "Phytophtora":
        assert "Phytophtora mapped explicitly to Phytophthora" in taxon["mapping_note"]
    else:
        assert taxon["mapping_note"] is None


@pytest.mark.parametrize("label", ["Fusarium", "Trichoderma", "Verticillium"])
def test_soil_only_supplies_genus_labels(label):
    result = derive_taxonomy("soil", [label], ["fungi"])
    assert result["genus"] == label
    assert result["species"] is None
    assert result["taxa"][0]["rank"] == "genus"


def test_idphy_editorial_suffix_preserves_exact_label_and_mapping_note():
    label = "Phytophthora examplea (in progress, description incomplete)"
    result = derive_taxonomy("idphy", [label], ["oomycetes"])
    assert result["species"] == "Phytophthora examplea"
    assert result["taxa"][0]["source_label"] == label
    assert "Editorial in-progress suffix" in result["taxa"][0]["mapping_note"]


@pytest.mark.parametrize(
    "label",
    [
        "Phytophthora examplea complex",
        "Phytophthora examplea (species complex)",
        "Phytophthora cf. examplea",
        "Phytophthora aff. examplea",
        "Phytophthora sp. example",
        "Phytophthora × examplea",
        "Phytophthora examplea x exampleb",
        "Phytophthora examplea (uncertain)",
        "Phytophthora examplea isolate 10",
        "Phytophthora hybrid",
        "Phytophthora sp",
    ],
)
def test_ambiguous_or_extended_idphy_labels_do_not_supply_species(label):
    result = derive_taxonomy("idphy", [label], ["oomycetes"])
    assert result["genus"] == "Phytophthora"
    assert result["species"] is None
    assert result["taxa"][0]["source_label"] == label
    assert "unresolved" in result["taxa"][0]["mapping_note"]


def test_unmapped_source_label_remains_unresolved():
    result = derive_taxonomy("soil", ["Unknown"], [])
    assert result["status"] == "unresolved"
    assert all(result[rank] is None for rank in ("group", "genus", "species"))
    label_map = build_taxonomy_map([{"taxonomy": result}])
    assert hierarchy_targets(result, label_map)["target_mask"] == {
        "group": False,
        "genus": False,
        "species": False,
    }


def test_maps_are_order_independent_and_group_ids_are_fixed():
    records = [
        {"taxonomy": derive_taxonomy("tgfc", ["Olivea tectonae"], ["fungi"])},
        {"taxonomy": derive_taxonomy("soil", ["Fusarium"], ["fungi"])},
        {"taxonomy": derive_taxonomy("idphy", ["Phytophthora examplea"], ["oomycetes"])},
    ]
    label_map = build_taxonomy_map(records)
    assert label_map == build_taxonomy_map(list(reversed(records)))
    assert label_map == {
        "group": {"fungi": 0, "oomycetes": 1},
        "genus": {"Fusarium": 0, "Olivea": 1, "Phytophthora": 2},
        "species": {"Olivea tectonae": 0, "Phytophthora examplea": 1},
        "genus_to_group": {"Fusarium": "fungi", "Olivea": "fungi", "Phytophthora": "oomycetes"},
        "species_to_genus": {"Olivea tectonae": "Olivea", "Phytophthora examplea": "Phytophthora"},
    }
    assert hierarchy_targets(records[1]["taxonomy"], label_map)["targets"] == {
        "group": 0,
        "genus": 0,
        "species": -1,
    }


def test_unmapped_frozen_target_fails_instead_of_silently_becoming_unresolved():
    result = derive_taxonomy("soil", ["Fusarium"], ["fungi"])
    with pytest.raises(ValueError, match="absent from frozen label map"):
        hierarchy_targets(result, build_taxonomy_map([]))


def test_inconsistent_parent_maps_are_rejected():
    result = derive_taxonomy("soil", ["Fusarium"], ["fungi"])
    label_map = build_taxonomy_map([{"taxonomy": result}])
    label_map["genus_to_group"]["Fusarium"] = "oomycetes"
    with pytest.raises(ValueError, match="genus and group disagree"):
        hierarchy_targets(result, label_map)
    conflicting = copy.deepcopy(result)
    conflicting["taxa"][0]["group"] = "oomycetes"
    with pytest.raises(ValueError, match="Conflicting parent groups"):
        build_taxonomy_map([{"taxonomy": result}, {"taxonomy": conflicting}])


def test_model_rejects_fabricated_consensus_for_a_multitaxon_image():
    result = derive_taxonomy(
        "idphy", ["Phytophthora examplea", "Phytophthora exampleb"], ["oomycetes"]
    )
    result["species"] = "Phytophthora examplea"
    with pytest.raises(ValidationError, match="consensus"):
        Taxonomy.model_validate(result)


def test_taxon_model_rejects_orphaned_species_and_incorrect_rank():
    taxon = derive_taxonomy("tgfc", ["Olivea tectonae"], ["fungi"])["taxa"][0]
    with pytest.raises(ValidationError, match="parent genus"):
        SourceTaxon.model_validate({**taxon, "genus": "Different"})
    with pytest.raises(ValidationError, match="deepest supplied label"):
        SourceTaxon.model_validate({**taxon, "rank": "genus"})


def test_catalog_model_requires_taxonomy_and_rejects_invented_species():
    record = {
        "source_id": "soil",
        "image_id": "synthetic",
        "relative_path": "images/example.png",
        "sha256": "a" * 64,
        "pixel_sha256": "b" * 64,
        "lossless_transform_sha256": "c" * 64,
        "perceptual_dhash64": "d" * 16,
        "bytes": 12,
        "width_px": 3,
        "height_px": 4,
        "file_format": "PNG",
        "source_labels": ["Fusarium"],
        "candidate_groups": ["fungi"],
        "image_role": "full_field",
        "leakage_group_id": "synthetic_group",
        "usage_permission_ref": "sources.json#soil",
    }
    with pytest.raises(ValidationError, match="taxonomy"):
        ImageRecord.model_validate(record)
    record["taxonomy"] = derive_taxonomy("soil", ["Fusarium"], ["fungi"])
    assert ImageRecord.model_validate(record).schema_version == "3.0"
    record["taxonomy"]["species"] = "Fusarium examplea"
    record["taxonomy"]["taxa"][0].update(species="Fusarium examplea", rank="species")
    with pytest.raises(ValidationError, match="supported mapping"):
        ImageRecord.model_validate(record)
