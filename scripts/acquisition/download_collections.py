#!/usr/bin/env python3
"""Download the three retained fungi/oomycete collections; no model training.

Uses curl for HTTPS, preserves repository records, verifies publisher checksums,
and inventories archive contents without executing any downloaded material.
"""

import argparse
import hashlib
import http.client
import json
import shutil
import ssl
import subprocess
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlparse

ALLOWED = {
    "api.figshare.com",
    "ndownloader.figshare.com",
    "api.zenodo.org",
    "zenodo.org",
    "idtools.org",
    "www.idtools.org",
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
CONNECTIONS = threading.local()
FIGSHARE_COLLECTIONS = {"01_tgfc": 28855910}


def idphy_transfer(url, target):
    """Reuse one validated TLS connection per download worker."""
    parsed = urlparse(url)
    path = quote(parsed.path, safe="/%:@!$&'()*+,;=-._~")
    if parsed.query:
        path += "?" + parsed.query
    for attempt in range(3):
        conn = getattr(CONNECTIONS, "connection", None)
        try:
            if conn is None:
                conn = http.client.HTTPSConnection(
                    parsed.hostname, timeout=90, context=ssl.create_default_context()
                )
                CONNECTIONS.connection = conn
            conn.request("GET", path, headers={"User-Agent": "Woodunnit dataset acquisition/1.0"})
            response = conn.getresponse()
            if response.status != 200:
                status = response.status
                response.read()
                conn.close()
                CONNECTIONS.connection = None
                if status in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise ValueError(f"IDphy HTTP {status}: {url}")
            with target.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            if response.will_close:
                conn.close()
                CONNECTIONS.connection = None
            return
        except (OSError, http.client.HTTPException):
            if conn:
                conn.close()
            CONNECTIONS.connection = None
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def now():
    return datetime.now(UTC).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, target, expected_size=None, checksum=None):
    if urlparse(url).scheme != "https" or urlparse(url).hostname not in ALLOWED:
        raise ValueError(f"Unapproved download endpoint: {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    algorithm, expected_hash = checksum.split(":", 1) if checksum else (None, None)
    if target.exists():
        if expected_size is not None and target.stat().st_size != expected_size:
            raise ValueError(f"Existing file has unexpected size; preserved: {target}")
        if expected_hash and digest(target, algorithm) != expected_hash:
            raise ValueError(f"Existing file has unexpected checksum; preserved: {target}")
        if not expected_hash:
            raise ValueError(f"Existing unverified file preserved: {target}")
    else:
        partial = target.with_name(target.name + ".part")
        if urlparse(url).hostname == "idtools.org":
            idphy_transfer(url, partial)
        else:
            result = subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "5",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "1800",
                    "--output",
                    str(partial),
                    url,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(f"Download failed for {url}: {result.stderr.strip()}")
        if expected_size is not None and partial.stat().st_size != expected_size:
            raise ValueError(f"Size mismatch: {target.name}")
        if expected_hash and digest(partial, algorithm) != expected_hash:
            raise ValueError(f"Checksum mismatch: {target.name}")
        partial.replace(target)
    return {
        "filename": target.name,
        "bytes": target.stat().st_size,
        "sha256": digest(target, "sha256"),
        "source_url": url,
        "publisher_checksum": checksum,
        "verified_at": now(),
    }


def metadata(url, path):
    if path.exists():
        return json.loads(path.read_text())
    fetch(url, path)
    return json.loads(path.read_text())


def extract_and_inventory(archive, directory):
    report_path = directory.with_name(directory.name + "_inventory.json")
    if directory.exists():
        if report_path.exists():
            return json.loads(report_path.read_text())
        raise ValueError(f"Existing extraction without inventory preserved: {directory}")
    staging = directory.with_name(directory.name + ".extracting")
    staging.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > 10 * 1024**3:
            raise ValueError("Unexpected archive expansion exceeds 10 GiB")
        for item in z.infolist():
            destination = (staging / item.filename).resolve()
            if not destination.is_relative_to(staging.resolve()):
                raise ValueError(f"Unsafe archive path: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink rejected: {item.filename}")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            rows.append(
                {
                    "path": item.filename,
                    "bytes": item.file_size,
                    "crc32": f"{item.CRC:08x}",
                    "kind": "image" if destination.suffix.lower() in IMAGE_EXT else "other",
                }
            )
    report = {
        "archive": archive.name,
        "files": len(rows),
        "images": sum(r["kind"] == "image" for r in rows),
        "uncompressed_bytes": sum(r["bytes"] for r in rows),
        "zip_crc_verified_by_full_read": True,
        "entries": rows,
    }
    save_json(report_path, report)
    staging.replace(directory)
    return report


def figshare(base, slug, article_id):
    if slug not in FIGSHARE_COLLECTIONS or FIGSHARE_COLLECTIONS[slug] != article_id:
        raise ValueError("Figshare collection is outside the retained fungi/oomycete scope")
    root = base / slug
    record = metadata(
        f"https://api.figshare.com/v2/articles/{article_id}",
        root / "metadata" / "repository_record.json",
    )
    print(f"{slug}: record v{record.get('version')}, {len(record['files'])} file(s)", flush=True)
    downloads, inventories = [], []
    for entry in record["files"]:
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError("Unexpected repository filename")
        archive = root / "archives" / name
        md5 = entry.get("computed_md5") or entry.get("supplied_md5")
        downloads.append(
            fetch(entry["download_url"], archive, entry.get("size"), f"md5:{md5}" if md5 else None)
        )
        if archive.suffix.lower() == ".zip":
            inventories.append(extract_and_inventory(archive, root / "extracted" / archive.stem))
        print(f"{slug}: verified {name}", flush=True)
    result = {
        "collection": slug,
        "downloaded_at": now(),
        "title": record.get("title"),
        "source_url": record.get("url_public_html"),
        "doi": record.get("doi"),
        "version": record.get("version"),
        "license": record.get("license"),
        "authors": record.get("authors"),
        "downloads": downloads,
        "extractions": [{k: v for k, v in i.items() if k != "entries"} for i in inventories],
    }
    save_json(root / "download_manifest.json", result)
    return result


def soil(base):
    root = base / "03_soil_fungi_phytophthora"
    record = metadata(
        "https://zenodo.org/api/records/7965200", root / "metadata" / "repository_record.json"
    )
    entries = record["files"]
    print(f"soil: {len(entries)} files, downloading with two connections", flush=True)

    def one(entry):
        name = entry["key"]
        if Path(name).name != name:
            raise ValueError("Unexpected repository filename")
        result = fetch(
            entry["links"]["self"], root / "images" / name, entry.get("size"), entry.get("checksum")
        )
        fields = Path(name).stem.split("_")
        result.update(
            {
                "source_label": fields[0],
                "parent_image_id": "_".join(fields[:2]),
                "image_role": "crop" if len(fields) > 2 else "full_field",
            }
        )
        time.sleep(0.25)
        return result

    downloads, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = {executor.submit(one, entry): entry["key"] for entry in entries}
        for job in as_completed(jobs):
            try:
                downloads.append(job.result())
            except Exception as exc:
                errors.append({"file": jobs[job], "error": str(exc)})
            if (len(downloads) + len(errors)) % 25 == 0:
                print(f"soil: {len(downloads)} verified, {len(errors)} errors", flush=True)
    downloads.sort(key=lambda r: r["filename"])
    result = {
        "collection": "soil_fungi_phytophthora",
        "downloaded_at": now(),
        "source_url": "https://zenodo.org/records/7965200",
        "doi": record.get("doi"),
        "license": record["metadata"].get("license"),
        "authors": record["metadata"].get("creators"),
        "expected_files": len(entries),
        "full_fields": sum(r["image_role"] == "full_field" for r in downloads),
        "crops": sum(r["image_role"] == "crop" for r in downloads),
        "downloads": downloads,
        "errors": errors,
    }
    save_json(root / "download_manifest.json", result)
    if errors:
        raise RuntimeError(f"soil: {len(errors)} failed downloads; see manifest")
    return result


def idphy(base, manifest_path):
    root = base / "02_idphy_microscopy"
    selection_path = root / "metadata" / "selection_manifest.json"
    manifest_path = Path(manifest_path) if manifest_path is not None else selection_path
    if not manifest_path.is_file():
        raise ValueError("IDphy requires an existing selection manifest or --idphy-manifest")
    source = json.loads(manifest_path.read_text())
    entries = source if isinstance(source, list) else source["images"]
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != source:
            raise ValueError(
                "IDphy selection differs from saved provenance; existing selection preserved"
            )
    else:
        save_json(selection_path, source)
    print(f"IDphy: {len(entries)} selected images", flush=True)

    def one(entry):
        url = entry.get("image_url") or entry.get("url")
        name = entry.get("filename") or Path(urlparse(url).path).name
        if Path(name).name != name:
            raise ValueError("Unexpected IDphy filename")
        target = root / "images" / name
        previous = root / "metadata" / "files" / (name + ".json")
        expected = (
            "sha256:" + json.loads(previous.read_text())["sha256"] if previous.exists() else None
        )
        result = fetch(url, target, checksum=expected)
        result.update({"source_metadata": entry})
        save_json(previous, result)
        time.sleep(0.35)
        return result

    downloads, errors = [], []
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = {executor.submit(one, entry): entry for entry in entries}
        for job in as_completed(jobs):
            try:
                downloads.append(job.result())
            except Exception as exc:
                errors.append({"entry": jobs[job], "error": str(exc)})
            if (len(downloads) + len(errors)) % 50 == 0:
                print(f"IDphy: {len(downloads)} verified, {len(errors)} errors", flush=True)
    result = {
        "collection": "idphy_microscopy",
        "downloaded_at": now(),
        "source_url": "https://idtools.org/phytophthora/index.cfm?pageID=1547",
        "license": "Public domain unless otherwise indicated; retain per-image credits/exceptions",
        "expected_files": len(entries),
        "downloads": sorted(downloads, key=lambda r: r["filename"]),
        "errors": errors,
    }
    save_json(root / "download_manifest.json", result)
    if errors:
        raise RuntimeError(f"IDphy: {len(errors)} failures; see manifest")
    return result


def existing_destination(value):
    """Require an explicit data directory without silently creating a substitute."""
    base = Path(value).expanduser().resolve()
    if not base.is_dir():
        raise ValueError("Destination must be an existing directory")
    if base.parent == base or base == Path.home().resolve():
        raise ValueError("Choose a dedicated data directory, not a filesystem or home root")
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, required=True, help="Existing dataset directory"
    )
    parser.add_argument("--collection", choices=["repositories", "idphy"], required=True)
    parser.add_argument("--idphy-manifest", type=Path)
    args = parser.parse_args(argv)
    base = existing_destination(args.destination)
    tasks = (
        [("TgFC", lambda: figshare(base, "01_tgfc", 28855910)), ("soil", lambda: soil(base))]
        if args.collection == "repositories"
        else [("IDphy", lambda: idphy(base, args.idphy_manifest))]
    )
    failed = []
    for name, task in tasks:
        try:
            result = task()
            print(f"COMPLETE {name}: {len(result['downloads'])} downloaded file(s)", flush=True)
        except Exception as exc:
            failed.append({"collection": name, "error": str(exc)})
            print(f"FAILED {name}: {exc}", flush=True)
    save_json(
        base / f"acquisition_{args.collection}_status.json",
        {"finished_at": now(), "failures": failed},
    )
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
