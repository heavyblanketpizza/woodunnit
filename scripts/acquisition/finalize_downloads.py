#!/usr/bin/env python3
"""Inventory the three retained fungi/oomycete collections; preserve provenance."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
COLLECTIONS = {
    "01_tgfc": {
        "expected_images": 5266,
        "title": "TgFC: teak-associated fungal spores",
        "source": "https://figshare.com/articles/dataset/TgFC_i_Tectona_grandis_i_Fungal_Community_Dataset/28855910",
        "rights": (
            "CC BY 4.0. Creator and version details are preserved in "
            "metadata/repository_record.json."
        ),
        "notes": (
            "Original ZIPs and extracted images/YOLO labels are retained. The main release "
            "contains 5,236 images; MixedClass adds 30 images. All source labels are fungi; "
            "the mixed set concerns multiple fungal taxa. No source splits were changed."
        ),
    },
    "02_idphy_microscopy": {
        "expected_images": 726,
        "title": "IDphy: selected Phytophthora microscopy references",
        "source": "https://idtools.org/phytophthora/index.cfm?pageID=1547",
        "rights": (
            "Public domain unless otherwise indicated. Retain individual credits and exceptions. "
            "Policy: https://idtools.org/phytophthora/index.cfm?pageID=1543 . Source captions "
            "and species pages were checked for textual exceptions; embedded image markings "
            "have not all been visually inspected."
        ),
        "notes": (
            "Selected using all 19 site Morphological Structure filter options across 12 pages. "
            "Exact image URLs were deduplicated; six composites with colony panels were "
            "excluded. Files are the original assets served by the gallery, not necessarily "
            "raw camera photographs. Captions, credits, species/entity links and ambiguity "
            "flags are in metadata/selection_manifest.json. Three URL-reuse groups have "
            "multiple species labels and need expert resolution before label use. Images "
            "remain unedited."
        ),
    },
    "03_soil_fungi_phytophthora": {
        "expected_images": 174,
        "title": "Soil fungi and Phytophthora: public microscopy sample",
        "source": "https://zenodo.org/records/7965200",
        "rights": (
            "CC BY 4.0. Attribution: Karol Struniawski; DOI 10.5281/zenodo.7965200. "
            "Original creator/license metadata preserved."
        ),
        "notes": (
            "174 PNG files: 20 full fields and 154 crops. Some crops reference parents absent "
            "from this sample. Original labels and filenames, including the source spelling "
            "Phytophtora, are preserved. This is the public sample, not the larger collection "
            "described in the paper. Parent IDs are recorded in download_manifest.json."
        ),
    },
}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preserve_idphy_provenance(metadata_root, cache_root=None):
    """Use saved provenance first; an explicit cache only fills missing records."""
    selection = metadata_root / "selection_manifest.json"
    if not selection.is_file():
        candidates = [metadata_root / "idphy_download_manifest.json"]
        if cache_root is not None:
            candidates.append(cache_root / "idphy_download_manifest.json")
        fallback = next((path for path in candidates if path.is_file()), None)
        if fallback is None:
            raise RuntimeError("Missing IDphy selection provenance in storage and cache")
        metadata_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback, selection)
    names = [
        "IDPHY_SELECTION.md",
        "idphy_download_manifest.json",
        "discover_idphy.py",
        "finalize_idphy.py",
    ]
    for name in names:
        target = metadata_root / name
        cached = cache_root / name if cache_root is not None else None
        if not target.exists() and cached is not None and cached.is_file():
            shutil.copy2(cached, target)
    saved_html = metadata_root / "idphy_metadata"
    cached_html = cache_root / "idphy_metadata" if cache_root is not None else None
    if cached_html is not None and cached_html.is_dir():
        for cached in sorted(cached_html.rglob("*")):
            if cached.is_file():
                target = saved_html / cached.relative_to(cached_html)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cached, target)
    return {
        "selection_manifest": str(selection.relative_to(metadata_root)),
        "source_html_cache_present": saved_html.is_dir(),
        "optional_provenance_missing": [
            name for name in names if not (metadata_root / name).is_file()
        ],
    }


def existing_directory(value, label):
    directory = Path(value).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"{label} must be an existing directory")
    if directory.parent == directory or directory == Path.home().resolve():
        raise ValueError(f"{label} must be a dedicated directory, not a filesystem or home root")
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, required=True, help="Existing dataset directory"
    )
    parser.add_argument(
        "--metadata-cache", type=Path, help="Optional saved IDphy provenance directory"
    )
    parser.add_argument(
        "--report-dir", type=Path, help="Optional existing directory for compact reports"
    )
    args = parser.parse_args(argv)
    base = existing_directory(args.destination, "Destination")
    cache = (
        existing_directory(args.metadata_cache, "Metadata cache")
        if args.metadata_cache is not None
        else None
    )
    report_dir = (
        existing_directory(args.report_dir, "Report directory")
        if args.report_dir is not None
        else None
    )
    if report_dir is not None and report_dir.is_relative_to(base):
        raise ValueError("Report directory must be outside the dataset directory")
    for slug in COLLECTIONS:
        manifest = json.loads((base / slug / "download_manifest.json").read_text())
        if manifest.get("errors"):
            raise RuntimeError(f"Unresolved download failures: {slug}")
        if manifest.get("expected_files", len(manifest["downloads"])) != len(manifest["downloads"]):
            raise RuntimeError(f"Incomplete collection: {slug}")
    idphy_meta = base / "02_idphy_microscopy" / "metadata"
    idphy_provenance = preserve_idphy_provenance(idphy_meta, cache)
    idphy_selection = json.loads((idphy_meta / "selection_manifest.json").read_text())
    with (base / "02_idphy_microscopy" / "image_credits.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "species",
                "caption",
                "photographer",
                "source_page",
                "image_url",
                "rights_status",
                "source_record_count",
            ],
        )
        writer.writeheader()
        for entry in idphy_selection["images"]:
            records = entry.get("gallery_records", [entry])
            row = {key: entry.get(key, "") for key in writer.fieldnames}
            row["species"] = " | ".join(sorted({r.get("species", "") for r in records}))
            row["source_record_count"] = len(records)
            writer.writerow(row)
    (base / "acquisition_scripts").mkdir(exist_ok=True)
    for name in ["download_collections.py", "finalize_downloads.py"]:
        source = Path(__file__).resolve().parent / name
        target = base / "acquisition_scripts" / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

    rows, summaries, failures = [], [], []
    for slug, info in COLLECTIONS.items():
        root = base / slug
        image_root = root / ("extracted" if (root / "extracted").exists() else "images")
        hashes, sizes, groups = defaultdict(list), Counter(), Counter()
        files = sorted(
            p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT
        )
        if len(files) != info["expected_images"]:
            raise RuntimeError(
                f"Unexpected retained image count for {slug}: "
                f"{len(files)} instead of {info['expected_images']}"
            )
        for p in files:
            relative = str(p.relative_to(base))
            try:
                with Image.open(p) as im:
                    width, height = im.size
                    im.load()
                    format_name = im.format
                digest = sha256(p)
                hashes[digest].append(relative)
                sizes[(width, height)] += 1
                groups[p.parent.name] += 1
                rows.append(
                    {
                        "collection": slug,
                        "path": relative,
                        "bytes": p.stat().st_size,
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "format": format_name,
                    }
                )
            except Exception as exc:
                failures.append({"path": relative, "error": str(exc)})
        summary = {
            "collection": slug,
            "title": info["title"],
            "images": len(files),
            "image_bytes": sum(p.stat().st_size for p in files),
            "total_folder_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
            "source": info["source"],
            "rights": info["rights"],
            "notes": info["notes"],
            "subfolder_counts": dict(groups),
            "exact_duplicate_groups": [paths for paths in hashes.values() if len(paths) > 1],
        }
        summaries.append(summary)
        text = (
            f"# {info['title']}\n\nWoodunnit source collection.\n\n"
            f"Source: {info['source']}\n\n## Rights and attribution\n\n{info['rights']}\n\n"
            f"## Contents\n\n{info['notes']}\n\n"
            f"Verified inventory: **{len(files):,} image files**. Checksums, source URLs and "
            "repository versions are in `download_manifest.json`. The root "
            "`image_inventory.csv` records image dimensions and SHA-256 hashes. Source files "
            "are preserved; this acquisition inventory does not assign model-training splits "
            "or adjudicate tree diagnoses.\n"
        )
        (root / "README.md").write_text(text)
        print(f"Audited {slug}: {len(files)} image files", flush=True)
    with (base / "image_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["collection", "path", "bytes", "sha256", "width", "height", "format"]
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "verified_at": datetime.now(UTC).isoformat(),
        "destination": ".",
        "total_image_files": sum(s["images"] for s in summaries),
        "collections": summaries,
        "image_decode_failures": failures,
        "idphy_provenance": idphy_provenance,
        "checks": (
            "Publisher MD5/size for repository downloads; ZIP CRC while extracting; "
            "first-frame raster decode and SHA-256 for every inventoried image. "
            "These checks do not validate biological labels."
        ),
    }
    save(base / "acquisition_summary.json", report)
    table = "\n".join(
        f"| [{s['title']}]({s['collection']}/) | {s['images']:,} |" for s in summaries
    )
    (base / "README.md").write_text(
        "# Woodunnit microscope datasets\n\n"
        f"| Collection | Image files |\n| --- | ---: |\n{table}\n\n"
        f"Total: **{report['total_image_files']:,} image files**. Crops, montages and mixed-taxa "
        "images are included in these file counts; they are not counts of independent "
        "specimens.\n\nOriginal ZIPs are retained under `archives/`, with browseable copies "
        "under `extracted/`. IDphy and soil images are under their `images/` folders. Each "
        "collection has a README, source/license metadata and download manifest. "
        "`image_inventory.csv` lists paths, dimensions and SHA-256 hashes; "
        "`acquisition_summary.json` records checks and duplicate groups.\n\n"
        "IDphy is a selected morphology gallery, not a published train/test dataset. Its "
        "per-image captions, credits and unresolved labels are preserved. This acquisition "
        "script does not edit images or train models.\n"
    )
    if report_dir is not None:
        save(
            report_dir / "ACQUISITION_SUMMARY.json",
            {k: v for k, v in report.items() if k != "collections"}
            | {
                "collections": [
                    {
                        k: v
                        for k, v in s.items()
                        if k not in {"exact_duplicate_groups", "subfolder_counts"}
                    }
                    for s in summaries
                ]
            },
        )
    if failures:
        raise RuntimeError(f"{len(failures)} image decode failures; see acquisition_summary.json")
    project_table = "\n".join(
        f"| [{s['title']}]({s['source']}) | {s['images']:,} | `{s['collection']}` |"
        for s in summaries
    )
    inventory_roots = [base / slug for slug in COLLECTIONS] + [base / "acquisition_scripts"]
    total_bytes = sum(
        p.stat().st_size for root in inventory_roots for p in root.rglob("*") if p.is_file()
    ) + sum(p.stat().st_size for p in base.iterdir() if p.is_file())
    project_note = (
        "# Downloaded microscope collections\n\n"
        f"Current inventory checked {report['verified_at']}. "
        "Image paths are relative to the configured dataset directory.\n\n"
        "| Collection | Downloaded image files | Folder |\n| --- | ---: | --- |\n"
        f"{project_table}\n\n"
        "Retained scope: fungi and oomycetes from three source collections. "
        f"Total: **{report['total_image_files']:,} image files**; "
        f"**{total_bytes / 1_000_000_000:.2f} GB** on disk including retained ZIPs, "
        "extracted files, annotations and metadata. These are file counts, not independent "
        "specimens.\n\n"
        "Repository size and MD5 checks passed during acquisition. Retained ZIPs passed "
        "CRC checks during extraction. Every listed image passed raster decoding of its "
        "first frame and has a recorded SHA-256 hash. Exact duplicate groups are recorded "
        "without deleting original source files. Biological labels and transfer to the "
        "doctor's microscope remain unvalidated.\n\n"
        "IDphy contains 726 selected unique gallery URLs across 112 named entities. Six "
        "composites with colony panels were excluded; three URL-reuse groups have multiple "
        "species labels and are flagged in the source manifest. Its images are the files "
        "served by the gallery, not necessarily raw camera originals. Per-image "
        "captions/credits, selection criteria and rights evidence are preserved. Available "
        "source HTML and optional discovery records remain in metadata; "
        "acquisition_summary.json records any missing optional provenance files.\n\n"
        "The soil sample contains 20 full-field PNGs and 154 crops, including crops whose "
        "parent fields are not in the release. TgFC contains 5,236 main images plus "
        "30 mixed-taxa images.\n\n"
        "Browse the external folder's README.md, image_inventory.csv and "
        "acquisition_summary.json for file paths, dimensions, checksums and duplicate groups. "
        "Each collection's README and download_manifest.json retain license, attribution "
        "and source/version information. A compact machine-readable record is in "
        "[ACQUISITION_SUMMARY.json](ACQUISITION_SUMMARY.json).\n\n"
        "Earlier source memos describe pre-download screening. This acquisition record "
        "supersedes their statements that archives had not been downloaded. This script "
        "inventories source files; it does not create reviewed training labels or models. "
        "Use the separate ingestion pipeline to create model manifests.\n"
    )
    if report_dir is not None:
        (report_dir / "DATA_DOWNLOADS.md").write_text(project_note)
    print(
        json.dumps(
            {"total_image_files": report["total_image_files"], "decode_failures": len(failures)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
