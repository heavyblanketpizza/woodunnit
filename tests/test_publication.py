"""Publication checks use synthetic sensitive strings and temporary repositories."""

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_publication.py"
SPEC = importlib.util.spec_from_file_location("check_publication", SCRIPT)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


@pytest.fixture
def public_banner(tmp_path, monkeypatch):
    relative = Path("assets/woodunnit-banner.webp")
    content = b"RIFF\xff\x00synthetic banner"
    monkeypatch.setitem(
        publication.PUBLIC_ASSETS, relative.as_posix(), hashlib.sha256(content).hexdigest()
    )
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(content)
    return relative, content


def test_reviewed_banner_digest_passes_working_and_index_checks(tmp_path, public_banner):
    relative, _ = public_banner
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", relative.as_posix()], check=True)
    assert publication.audit(tmp_path) == ([relative], [])


def test_same_banner_bytes_at_another_path_are_not_exempt(tmp_path, public_banner):
    _, content = public_banner
    relative = Path("assets/copy.webp")
    (tmp_path / relative).write_bytes(content)
    _, findings = publication.audit(tmp_path)
    assert (relative, "dataset-or-model-artifact") in findings
    assert (relative, "non-UTF8-file") in findings


@pytest.mark.parametrize("tamper_staged", [False, True], ids=["working", "staged"])
def test_banner_digest_checks_working_and_index_independently(
    tmp_path, public_banner, tamper_staged
):
    relative, approved = public_banner
    path = tmp_path / relative
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    path.write_bytes(approved + b"changed" if tamper_staged else approved)
    subprocess.run(["git", "-C", str(tmp_path), "add", relative.as_posix()], check=True)
    path.write_bytes(approved if tamper_staged else approved + b"changed")
    _, findings = publication.audit(tmp_path)
    prefix = "staged-" if tamper_staged else ""
    assert findings == [(relative, prefix + "public-asset-sha256-mismatch")]


def test_banner_digest_is_checked_when_working_file_is_deleted(tmp_path, public_banner):
    relative, approved = public_banner
    path = tmp_path / relative
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    path.write_bytes(approved + b"changed")
    subprocess.run(["git", "-C", str(tmp_path), "add", relative.as_posix()], check=True)
    path.unlink()
    assert publication.audit(tmp_path)[1] == [(relative, "staged-public-asset-sha256-mismatch")]


def test_banner_exception_does_not_bypass_size_limit(tmp_path, public_banner, monkeypatch):
    relative, _ = public_banner
    content = b"x" * (publication.MAX_BYTES + 1)
    monkeypatch.setitem(
        publication.PUBLIC_ASSETS, relative.as_posix(), hashlib.sha256(content).hexdigest()
    )
    (tmp_path / relative).write_bytes(content)
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", relative.as_posix()], check=True)
    _, findings = publication.audit(tmp_path)
    assert (relative, "over-1-MiB") in findings
    assert (relative, "staged-over-1-MiB") in findings


def test_banner_exception_does_not_bypass_symlink_checks(tmp_path, public_banner):
    relative, approved = public_banner
    target = tmp_path / "original.txt"
    target.write_bytes(approved)
    path = tmp_path / relative
    path.unlink()
    path.symlink_to(target)
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", relative.as_posix()], check=True)
    _, findings = publication.audit(tmp_path)
    assert (relative, "symlink") in findings
    assert (relative, "staged-symlink") in findings


def test_banner_exception_does_not_bypass_git_allowlist(tmp_path, public_banner):
    relative, _ = public_banner
    (tmp_path / ".gitignore").write_text("assets/\n")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", relative.as_posix()], check=True)
    _, findings = publication.audit(tmp_path)
    assert findings == [(relative, "ignored-tracked-file")]


def test_uninitialized_project_respects_ignore_without_creating_git(tmp_path):
    (tmp_path / ".gitignore").write_text("private/\n*.pt\n")
    (tmp_path / "code.py").write_text("value = 1\n")
    private = tmp_path / "private"
    private.mkdir()
    (private / "notes.txt").write_text("local notes")
    (tmp_path / "model.pt").write_bytes(b"weights")
    files, findings = publication.audit(tmp_path)
    assert set(files) == {Path(".gitignore"), Path("code.py")}
    assert findings == []
    assert not (tmp_path / ".git").exists()


def test_tracked_ignored_artifact_is_still_checked(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / "model.pt").write_bytes(b"weights")
    subprocess.run(["git", "-C", str(tmp_path), "add", "model.pt"], check=True)
    (tmp_path / ".gitignore").write_text("*.pt\n")
    files, findings = publication.audit(tmp_path)
    assert Path("model.pt") in files
    assert (Path("model.pt"), "dataset-or-model-artifact") in findings


def test_staged_sensitive_content_is_checked_after_working_file_is_clean(tmp_path, capsys):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    secret = "ghp_" + "a" * 30
    path = tmp_path / "code.py"
    path.write_text(secret)
    subprocess.run(["git", "-C", str(tmp_path), "add", "code.py"], check=True)
    path.write_text("value = 1\n")
    assert publication.main(["--root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "staged-credential-token" in output
    assert secret not in output


def test_staged_artifact_is_checked_after_working_file_is_deleted(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    subprocess.run(["git", "-C", str(tmp_path), "add", "model.pt"], check=True)
    path.unlink()
    _, findings = publication.audit(tmp_path)
    assert (Path("model.pt"), "staged-dataset-or-model-artifact") in findings


def test_force_added_private_config_cannot_bypass_ignore_allowlist(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("*.toml\n")
    (tmp_path / "local.toml").write_text("value = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", "local.toml"], check=True)
    _, findings = publication.audit(tmp_path)
    assert (Path("local.toml"), "ignored-tracked-file") in findings


@pytest.mark.parametrize(
    ("content", "rule"),
    [
        ("/" + "Users" + "/fixture/project", "personal-machine-path"),
        ("/" + "home" + "/fixture/project", "personal-machine-path"),
        ("/" + "Volumes" + "/Example Drive/data", "personal-machine-path"),
        ("C:" + "\\Users\\fixture\\project", "personal-machine-path"),
        ("fixture" + "@" + "example.org", "email-address"),
        ("https://" + "fixture:credential" + "@" + "example.org", "authenticated-url"),
        ("https://" + "synthetic-token" + "@" + "example.org", "authenticated-url"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private-key"),
        ("ghp_" + "a" * 30, "credential-token"),
        ("password" + " = " + repr("synthetic-value"), "credential-assignment"),
    ],
    ids=[
        "mac-home",
        "linux-home",
        "volume",
        "windows-home",
        "email",
        "auth-url",
        "token-url",
        "private-key",
        "token",
        "assignment",
    ],
)
def test_sensitive_text_is_detected_without_printing_values(tmp_path, capsys, content, rule):
    (tmp_path / "sample.txt").write_text(content)
    assert publication.main(["--root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert rule in output
    assert "sample.txt" in output
    assert content not in output


@pytest.mark.parametrize(
    ("name", "content", "rule"),
    [
        ("records.jsonl", b"{}\n", "dataset-or-model-artifact"),
        ("large.txt", b"x" * (publication.MAX_BYTES + 1), "over-1-MiB"),
        ("binary.txt", b"\xff", "non-UTF8-file"),
        ("null.txt", b"\0", "binary-file"),
    ],
    ids=["dataset", "oversized", "non-utf8", "null-byte"],
)
def test_artifacts_and_nontext_files_fail(tmp_path, name, content, rule):
    (tmp_path / name).write_bytes(content)
    _, findings = publication.audit(tmp_path)
    assert (Path(name), rule) in findings


def test_symlinks_are_rejected_without_reading_target(tmp_path):
    (tmp_path / "link.txt").symlink_to(tmp_path / "missing-target")
    _, findings = publication.audit(tmp_path)
    assert findings == [(Path("link.txt"), "symlink")]


def test_scanner_and_synthetic_tests_do_not_trigger_own_rules():
    assert publication.check_file(SCRIPT.parent, Path(SCRIPT.name)) == []
    own_path = Path(__file__).resolve()
    assert publication.check_file(own_path.parent, Path(own_path.name)) == []
