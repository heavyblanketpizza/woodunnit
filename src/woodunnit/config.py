"""Small, portable TOML configuration; no data paths are compiled into the package."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schema import PROJECT_GROUPS


class IngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_root: Path
    output_root: Path
    verify_images: Literal[True] = True
    task_groups: list[str] = Field(default_factory=lambda: list(PROJECT_GROUPS))

    @field_validator("raw_root", "output_root")
    @classmethod
    def normalize_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def validate_paths_and_scope(self):
        if self.output_root == self.raw_root or self.output_root.is_relative_to(self.raw_root):
            raise ValueError("output_root must be outside the original dataset tree")
        if self.task_groups != list(PROJECT_GROUPS):
            raise ValueError(
                "task_groups must be ['fungi', 'oomycetes'] in that order for schema 2.0"
            )
        return self


def load_config(path: Path) -> IngestionConfig:
    path = path.resolve()
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    for field in ("raw_root", "output_root"):
        if field not in data:
            raise ValueError(f"Configuration is missing {field}")
        data[field] = (path.parent / Path(data[field]).expanduser()).resolve()
    config = IngestionConfig.model_validate(data)
    if not config.raw_root.is_dir():
        raise ValueError(
            "Dataset root is unavailable; connect your storage or update raw_root: "
            f"{config.raw_root}"
        )
    if not (config.raw_root / "image_inventory.csv").is_file():
        raise ValueError(f"Acquisition image_inventory.csv is missing from {config.raw_root}")
    return config
