"""Check Git's publication candidate files without exposing matched content."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

MAX_BYTES = 1024 * 1024
# Only this reviewed composition may be published; source artwork stays private.
PUBLIC_ASSETS = {
    "assets/woodunnit-banner.webp": (
        "8d324cae347c38a97253d427d277b6db2a1450afca300d0891129e2519d071a3"
    ),
}
ARTIFACT_EXTENSIONS = {
    ".arrow",
    ".avi",
    ".bin",
    ".bmp",
    ".bz2",
    ".ckpt",
    ".csv",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".heic",
    ".jpeg",
    ".joblib",
    ".jpg",
    ".jsonl",
    ".keras",
    ".m4v",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pb",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tflite",
    ".tif",
    ".tiff",
    ".tsv",
    ".webp",
    ".xz",
    ".zip",
    ".7z",
}
TEXT_RULES = {
    "personal-machine-path": re.compile(
        r"/(?:Users|home)/[A-Za-z0-9._-]+(?:/|\b)"
        r"|/(?:Volumes)/[A-Za-z0-9][^\r\n\"'`<>]*"
        r"|[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\s\\/\"']+",
        re.IGNORECASE,
    ),
    "email-address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "authenticated-url": re.compile(r"https?://[^/\s@]+@", re.IGNORECASE),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential-token": re.compile(
        r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        r"|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
    ),
    "credential-assignment": re.compile(
        r"\b(?:password|passwd|api[_-]?key|secret|access[_-]?token|auth[_-]?token)"
        r"\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']",
        re.IGNORECASE,
    ),
}


def repository_command(root: Path) -> list[str] | None:
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == root:
        return ["git", "-C", str(root), "-c", "core.excludesFile=/dev/null"]
    return None


def publication_files(root: Path) -> list[Path]:
    """Include tracked files even when ignored, plus nonignored untracked files.

    Before Git initialization, use an isolated temporary index to apply the
    project's ignore rules. Never create Git metadata inside the workspace.
    """
    root = root.resolve()
    command = repository_command(root)
    with tempfile.TemporaryDirectory(prefix="woodunnit-publication-") as temporary:
        if command is None:
            git_dir = Path(temporary) / "git"
            subprocess.run(
                ["git", "init", "--bare", "--quiet", str(git_dir)],
                capture_output=True,
                check=True,
            )
            command = ["git", f"--git-dir={git_dir}", f"--work-tree={root}"]
        result = subprocess.run(
            command
            + [
                "-c",
                "core.excludesFile=/dev/null",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            check=True,
        )
    return [
        Path(name.decode("utf-8", errors="surrogateescape"))
        for name in sorted(set(result.stdout.split(b"\0")) - {b""})
    ]


def artifact_rules(path: Path) -> list[str]:
    if any(suffix.lower() in ARTIFACT_EXTENSIONS for suffix in path.suffixes):
        return ["dataset-or-model-artifact"]
    return []


def check_content(content: bytes) -> list[str]:
    problems = []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return ["non-UTF8-file"]
    if b"\0" in content:
        problems.append("binary-file")
    problems.extend(name for name, pattern in TEXT_RULES.items() if pattern.search(text))
    return problems


def check_payload(relative: Path, content: bytes) -> list[str]:
    expected = PUBLIC_ASSETS.get(relative.as_posix())
    if expected is not None:
        if hashlib.sha256(content).hexdigest() != expected:
            return ["public-asset-sha256-mismatch"]
        return []
    return artifact_rules(relative) + check_content(content)


def check_file(root: Path, relative: Path) -> list[str]:
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            return ["symlink"]
    if not path.exists():
        return []  # The index is checked separately, including unstaged deletions.
    if not path.is_file():
        return ["non-regular-file"]
    if path.stat().st_size > MAX_BYTES:
        return artifact_rules(relative) + ["over-1-MiB"]
    return check_payload(relative, path.read_bytes())


def check_index(root: Path) -> list[tuple[Path, str]]:
    """Check staged blobs, which can differ from clean or deleted working files."""
    command = repository_command(root)
    if command is None:
        return []
    result = subprocess.run(
        command + ["ls-files", "--stage", "-z"], capture_output=True, check=True
    )
    entries = []
    for entry in result.stdout.split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, object_id, _stage = metadata.split()
            entries.append((mode, object_id, name))
    if not entries:
        return []
    ignored = subprocess.run(
        command + ["check-ignore", "--no-index", "--stdin", "-z"],
        input=b"".join(name + b"\0" for _, _, name in entries),
        capture_output=True,
        check=False,
    )
    if ignored.returncode not in (0, 1):
        raise subprocess.CalledProcessError(ignored.returncode, "git check-ignore")
    findings = {
        (Path(name.decode("utf-8", errors="surrogateescape")), "ignored-tracked-file")
        for name in ignored.stdout.split(b"\0")
        if name
    }
    for mode, object_id, name in entries:
        path = Path(name.decode("utf-8", errors="surrogateescape"))
        rules = []
        if mode == b"120000":
            rules.extend(artifact_rules(path) + ["symlink"])
        elif mode not in (b"100644", b"100755"):
            rules.extend(artifact_rules(path) + ["non-regular-file"])
        else:
            object_name = object_id.decode("ascii")
            size = subprocess.run(
                command + ["cat-file", "-s", object_name], capture_output=True, check=True
            )
            if int(size.stdout) > MAX_BYTES:
                rules.extend(artifact_rules(path) + ["over-1-MiB"])
            else:
                blob = subprocess.run(
                    command + ["cat-file", "blob", object_name], capture_output=True, check=True
                )
                rules.extend(check_payload(path, blob.stdout))
        findings.update((path, f"staged-{rule}") for rule in rules)
    return sorted(findings)


def audit(root: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    root = root.resolve()
    files = publication_files(root)
    findings = [(path, rule) for path in files for rule in check_file(root, path)]
    return files, sorted(set(findings + check_index(root)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        files, findings = audit(args.root)
    except (OSError, subprocess.SubprocessError):
        print("Publication check could not inspect files or Git state.")
        return 2
    for path, rule in findings:
        print(f"{json.dumps(path.as_posix())}: {rule}")
    if findings:
        print(f"Publication check failed: {len(findings)} findings in {len(files)} files.")
        return 1
    print(f"Publication check passed: {len(files)} files; matched content is never printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
