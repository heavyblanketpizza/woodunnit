"""Versioned catalog contract. Publisher labels never become reviewed targets here."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from woodunnit.taxonomy import RANKS, derive_taxonomy, taxonomy_consensus

SCHEMA_VERSION = "3.0"
PROJECT_GROUPS = ("fungi", "oomycetes")
CANDIDATE_GROUPS = PROJECT_GROUPS
SourceId = Literal["tgfc", "idphy", "soil"]
CandidateGroup = Literal["fungi", "oomycetes"]


class SourceTaxon(BaseModel):
    """One unchanged publisher label with only supported normalized ranks."""

    model_config = ConfigDict(extra="forbid")

    source_label: str = Field(min_length=1)
    group: CandidateGroup | None
    genus: str | None
    species: str | None
    rank: Literal["group", "genus", "species", "unresolved"]
    mapping_note: str | None = None

    @model_validator(mode="after")
    def ranks_have_parents(self) -> "SourceTaxon":
        if self.genus is not None and (not self.genus or self.group is None):
            raise ValueError("Genus requires a parent group")
        if self.species is not None and (
            self.genus is None or not self.species.startswith(self.genus + " ")
        ):
            raise ValueError("Species requires its corresponding parent genus")
        rank = next((rank for rank in reversed(RANKS) if getattr(self, rank)), "unresolved")
        if self.rank != rank:
            raise ValueError("Taxon rank must match its deepest supplied label")
        return self


class Taxonomy(BaseModel):
    """Single-image consensus plus all source taxa; imported evidence stays unreviewed."""

    model_config = ConfigDict(extra="forbid")

    group: CandidateGroup | None
    genus: str | None
    species: str | None
    taxa: list[SourceTaxon]
    status: Literal["single_taxon", "multiple_taxa", "group_only", "unresolved"]

    @model_validator(mode="after")
    def consensus_matches_source_taxa(self) -> "Taxonomy":
        labels = [taxon.source_label for taxon in self.taxa]
        if len(labels) != len(set(labels)):
            raise ValueError("Repeated taxonomy source labels")
        expected = taxonomy_consensus([taxon.model_dump() for taxon in self.taxa], self.group)
        if self.model_dump() != expected:
            raise ValueError("Image taxonomy must match the consensus of all source taxa")
        return self


class ImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["3.0"] = SCHEMA_VERSION
    image_id: str
    source_id: SourceId
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pixel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lossless_transform_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    perceptual_dhash64: str = Field(pattern=r"^[0-9a-f]{16}$")
    canonical_image_id: str | None = None
    selection_status: Literal["included", "excluded_redundant"] = "included"
    exclusion_reason: (
        Literal["exact_file_duplicate", "exact_pixel_duplicate", "lossless_transform_duplicate"]
        | None
    ) = None
    bytes: int = Field(gt=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    file_format: str
    image_mode: str | None = None
    frame_count: int | None = Field(default=None, ge=1)
    source_labels: list[str]
    candidate_groups: list[CandidateGroup]
    taxonomy: Taxonomy
    source_split: str | None = None
    source_url: str | None = None
    image_role: Literal["unknown", "full_field", "crop", "composite", "video_frame"]
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    microscopy: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    relationship_keys: list[str] = Field(default_factory=list)
    leakage_group_id: str
    grouping_basis: Literal["conservative_source_and_equivalent_images"] = (
        "conservative_source_and_equivalent_images"
    )
    import_flags: list[str] = Field(default_factory=list)
    usage_permission_ref: str
    review_status: Literal["unreviewed"] = "unreviewed"
    disposition: None = None
    group_label: None = None
    split: None = None

    @model_validator(mode="after")
    def taxonomy_matches_source_labels(self) -> "ImageRecord":
        expected = derive_taxonomy(self.source_id, self.source_labels, self.candidate_groups)
        if self.taxonomy.model_dump() != expected:
            raise ValueError("Taxonomy must preserve the supported mapping of source labels")
        return self

    @field_validator("relative_path")
    @classmethod
    def relative_posix_path(cls, value: str) -> str:
        from pathlib import PurePosixPath

        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("Image paths must be relative POSIX paths without traversal")
        if path.as_posix() != value:
            raise ValueError("Image paths must be normalized")
        return value

    @field_validator("candidate_groups", "source_labels", "relationship_keys", "import_flags")
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Repeated values are not allowed")
        return value
