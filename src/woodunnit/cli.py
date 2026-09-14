"""Command-line entry points for catalog ingestion, validation, and reviewed exports."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .io import read_json, sha256_file


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Woodunnit local microscopy data pipeline")
    root.add_argument("--version", action="version", version=f"woodunnit {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Verify sources and build a unified JSONL catalog")
    ingest.add_argument("--config", type=Path, default=Path("configs/ingestion.toml"))

    validate = commands.add_parser(
        "validate", help="Validate catalog records, hashes, and grouping"
    )
    validate.add_argument("--catalog", type=Path, required=True)
    validate.add_argument("--config", type=Path, default=Path("configs/ingestion.toml"))
    validate.add_argument(
        "--metadata-only",
        action="store_true",
        help="Check catalog/metadata integrity without decoding source images",
    )

    report = commands.add_parser("report", help="Show catalog counts and readiness limits")
    report.add_argument("--catalog", type=Path, required=True)

    release = commands.add_parser("release", help="Export an explicitly reviewed, grouped dataset")
    release.add_argument("--catalog", type=Path, required=True)
    release.add_argument("--config", type=Path, default=Path("configs/ingestion.toml"))
    release.add_argument("--reviews", type=Path, required=True)
    release.add_argument("--assignments", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--release-version", required=True)

    split = commands.add_parser("split", help="Freeze related image families into three partitions")
    split.add_argument("--config", type=Path, default=Path("configs/ingestion.toml"))
    split.add_argument("--catalog", type=Path, required=True)
    split.add_argument("--usage", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)

    preprocess = commands.add_parser(
        "preprocess", help="Prepare one image as a normalized RGB tensor"
    )
    preprocess.add_argument("--image", type=Path, required=True)
    preprocess.add_argument("--output", type=Path, required=True)
    preprocess.add_argument("--image-size", type=int, default=224)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "ingest":
            from .pipeline import ingest

            result = ingest(load_config(args.config), progress=lambda s: print(s, flush=True))
            print(json.dumps({"catalog_dir": result["catalog_dir"], **result["summary"]}, indent=2))
        elif args.command == "validate":
            from .pipeline import validate_catalog

            result = validate_catalog(
                args.catalog,
                load_config(args.config).raw_root,
                verify_images=not args.metadata_only,
            )
            print(f"Validated {result['catalog_id']}: {result['summary']['images']:,} records")
        elif args.command == "report":
            print(json.dumps(read_json(args.catalog / "catalog.json")["summary"], indent=2))
        elif args.command == "release":
            from .pipeline import validate_catalog
            from .release import build_release

            config = load_config(args.config)
            validate_catalog(args.catalog, config.raw_root, verify_images=True)
            result = build_release(
                args.catalog,
                config.raw_root,
                args.reviews,
                args.assignments,
                args.output,
                args.release_version,
            )
            print(json.dumps(result, indent=2))
        elif args.command == "split":
            from .experiment_split import build_split

            result = build_split(
                args.catalog,
                args.usage,
                args.output,
                args.seed,
                raw_root=load_config(args.config).raw_root,
            )
            print(json.dumps(result, indent=2))
        elif args.command == "preprocess":
            import numpy as np

            from .preprocessing import OpenCVPreprocessor, PreprocessConfig

            output = args.output
            sidecar = output.with_suffix(".json")
            if output.suffix.lower() != ".npz":
                raise ValueError("Preprocessing output must have the .npz extension")
            if output.exists() or sidecar.exists() or output.is_symlink() or sidecar.is_symlink():
                raise ValueError("Preprocessing outputs already exist")
            prepare = OpenCVPreprocessor(PreprocessConfig(image_size=args.image_size))
            checksum = sha256_file(args.image)
            tensor = prepare(args.image).numpy()
            if sha256_file(args.image) != checksum:
                raise ValueError("Source image changed during preprocessing")
            metadata = {
                "source_sha256": checksum,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "preprocessing": prepare.description(),
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                np.savez_compressed(stream, image=tensor)
            with sidecar.open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
                stream.write("\n")
            print(json.dumps(metadata, indent=2))
        return 0
    except ImportError as exc:
        print(
            f"woodunnit: {exc}. For image tensors, run uv sync --locked --extra preprocessing",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"woodunnit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
