"""Conservative, publisher-derived ranks; these are not reviewed identifications."""

from __future__ import annotations

import re
from typing import Any

GROUP_IDS = {"fungi": 0, "oomycetes": 1}
RANKS = ("group", "genus", "species")
_TGFC = {
    "Colletotrichum siamense": ("Colletotrichum", "Colletotrichum siamense"),
    "Olivea tectonae": ("Olivea", "Olivea tectonae"),
    "Neopestalotiopsis sp.": ("Neopestalotiopsis", None),
}
_SOIL = {
    "Fusarium": ("fungi", "Fusarium"),
    "Trichoderma": ("fungi", "Trichoderma"),
    "Verticillium": ("fungi", "Verticillium"),
    "Phytophtora": ("oomycetes", "Phytophthora"),
    "Phytophthora": ("oomycetes", "Phytophthora"),
}


def _taxon(source_id: str, label: str, fallback_group: str | None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "source_label": label,
        "group": fallback_group,
        "genus": None,
        "species": None,
        "rank": "group" if fallback_group else "unresolved",
        "mapping_note": None,
    }
    if source_id == "tgfc" and label in _TGFC:
        item["group"] = "fungi"
        item["genus"], item["species"] = _TGFC[label]
    elif source_id == "tgfc" and label == "MixedClass":
        item["group"] = "fungi"
        item["mapping_note"] = "Mixed fungal subset; individual image taxa are unavailable."
    elif source_id == "soil" and label in _SOIL:
        item["group"], item["genus"] = _SOIL[label]
        if label == "Phytophtora":
            item["mapping_note"] = (
                "Publisher spelling Phytophtora mapped explicitly to Phytophthora; "
                "source_label is unchanged."
            )
    elif source_id == "idphy" and re.match(r"^Phytophthora(?:\s|$)", label):
        item["group"], item["genus"] = "oomycetes", "Phytophthora"
        # Only the source's full binomial, optionally followed by an editorial
        # work-in-progress note, supplies a species. Never parse image filenames.
        match = re.fullmatch(r"(Phytophthora ([a-z][a-z-]+))(?:\s+(\(in progress[^()]*\)))?", label)
        if match and match.group(2) not in {"sp", "spp", "cf", "aff", "complex", "hybrid"}:
            item["species"] = match.group(1)
            if match.group(3):
                item["mapping_note"] = (
                    "Editorial in-progress suffix omitted from normalized binomial; "
                    "source_label is unchanged."
                )
        else:
            item["mapping_note"] = (
                "Source label does not supply an unqualified binomial; species is unresolved."
            )
    else:
        item["mapping_note"] = "No finer source-label mapping; source_label is unchanged."
    item["rank"] = next((rank for rank in reversed(RANKS) if item[rank]), "unresolved")
    return item


def taxonomy_consensus(taxa: list[dict[str, Any]], empty_group: str | None = None) -> dict:
    """Only a rank shared by every supplied taxon becomes one image-level target."""
    result: dict[str, Any] = {}
    for rank in RANKS:
        values = {item[rank] for item in taxa}
        result[rank] = next(iter(values)) if len(values) == 1 and None not in values else None
    if not taxa:
        result["group"] = empty_group
    if result["group"] is None:
        result["genus"] = None
    if result["genus"] is None:
        result["species"] = None
    identities = {(item["group"], item["genus"], item["species"]) for item in taxa}
    if result["group"] is None:
        status = "multiple_taxa" if len(identities) > 1 else "unresolved"
    elif len(identities) > 1:
        status = "multiple_taxa"
    elif result["genus"]:
        status = "single_taxon"
    else:
        status = "group_only"
    return {**result, "taxa": taxa, "status": status}


def derive_taxonomy(
    source_id: str, source_labels: list[str], candidate_groups: list[str]
) -> dict[str, Any]:
    """Recover supported ranks from exact source labels, retaining partial labels."""
    if source_id not in {"tgfc", "idphy", "soil"}:
        raise ValueError("Unsupported taxonomy source")
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("Repeated taxonomy source labels")
    if not all(isinstance(label, str) and label for label in source_labels):
        raise ValueError("Taxonomy source labels must be nonempty strings")
    if (
        len(set(candidate_groups)) != len(candidate_groups)
        or set(candidate_groups) - GROUP_IDS.keys()
    ):
        raise ValueError("Invalid candidate taxonomy groups")
    group = candidate_groups[0] if len(candidate_groups) == 1 else None
    taxa = [_taxon(source_id, label, group) for label in sorted(source_labels)]
    if candidate_groups and any(
        item["group"] is not None and item["group"] not in candidate_groups for item in taxa
    ):
        raise ValueError("Source taxonomy conflicts with candidate group")
    return taxonomy_consensus(taxa, group)


def build_taxonomy_map(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze deterministic rank IDs and parent links from retained source taxa."""
    genus_to_group: dict[str, str] = {}
    species_to_genus: dict[str, str] = {}
    for record in records:
        taxonomy = record["taxonomy"]
        if hasattr(taxonomy, "model_dump"):
            taxonomy = taxonomy.model_dump()
        for taxon in taxonomy["taxa"]:
            group, genus, species = (taxon[rank] for rank in RANKS)
            if genus:
                if group not in GROUP_IDS:
                    raise ValueError("Genus requires a supported parent group")
                if genus in genus_to_group and genus_to_group[genus] != group:
                    raise ValueError(f"Conflicting parent groups for genus {genus}")
                genus_to_group[genus] = group
            if species:
                if not genus or not species.startswith(genus + " "):
                    raise ValueError("Species requires its corresponding parent genus")
                if species in species_to_genus and species_to_genus[species] != genus:
                    raise ValueError(f"Conflicting parent genera for species {species}")
                species_to_genus[species] = genus
    return {
        "group": GROUP_IDS.copy(),
        "genus": {name: index for index, name in enumerate(sorted(genus_to_group))},
        "species": {name: index for index, name in enumerate(sorted(species_to_genus))},
        "genus_to_group": dict(sorted(genus_to_group.items())),
        "species_to_genus": dict(sorted(species_to_genus.items())),
    }


def hierarchy_targets(taxonomy: dict[str, Any], label_map: dict[str, Any]) -> dict[str, Any]:
    """Return rank IDs plus masks; missing or ambiguous ranks never get a class ID."""
    if label_map["group"] != GROUP_IDS:
        raise ValueError("Broad group IDs must remain fungi=0 and oomycetes=1")
    targets = {rank: -1 for rank in RANKS}
    for rank in RANKS:
        name = taxonomy[rank]
        if name is not None:
            if name not in label_map[rank]:
                raise ValueError(f"Taxonomy {rank} is absent from frozen label map: {name}")
            targets[rank] = label_map[rank][name]
    if (
        taxonomy["genus"]
        and label_map["genus_to_group"].get(taxonomy["genus"]) != taxonomy["group"]
    ):
        raise ValueError("Taxonomy genus and group disagree with frozen label map")
    if (
        taxonomy["species"]
        and label_map["species_to_genus"].get(taxonomy["species"]) != taxonomy["genus"]
    ):
        raise ValueError("Taxonomy species and genus disagree with frozen label map")
    return {"targets": targets, "target_mask": {rank: targets[rank] != -1 for rank in RANKS}}
