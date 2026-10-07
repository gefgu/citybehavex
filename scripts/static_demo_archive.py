#!/usr/bin/env python3
"""Package, fetch, and validate Pages data using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def read_json(path: Path, *, wrapped: bool = False):
    value = json.loads(path.read_text(encoding="utf-8"))
    if wrapped:
        if not isinstance(value, dict) or "data" not in value:
            raise ValueError(f"{path}: expected a JSON object with a data field")
        return value["data"]
    return value


def child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"Invalid dataset path component: {name!r}")
    return root / name


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(root: Path) -> None:
    manifest = read_json(root / "manifest.json")
    experiments = read_json(root / "experiments.json", wrapped=True)
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments.json must contain a nonempty data array")
    expected = {(entry["id"], entry["run_id"]) for entry in manifest["experiments"]}
    actual = set()
    ids = set()
    for experiment in experiments:
        exp_id = experiment["id"]
        if exp_id in ids:
            raise ValueError(f"Duplicate experiment: {exp_id}")
        ids.add(exp_id)
        detail = read_json(child(root / "experiments", exp_id + ".json"), wrapped=True)
        if detail != experiment:
            raise ValueError(f"Experiment details disagree with index: {exp_id}")
        if not experiment["runs"]:
            raise ValueError(f"No runs for {exp_id}")
        for run in experiment["runs"]:
            run_id = run["run_id"]
            actual.add((exp_id, run_id))
            run_root = child(child(root, exp_id), run_id)
            for filename in ("charts/base.json", "metrics-export.json", "network-validation.json",
                             "home-work/all.json", "timeline/meta.json"):
                read_json(run_root / filename, wrapped=True)
            sections = list((run_root / "charts/sections").glob("*/*.json"))
            if not sections:
                raise ValueError(f"No chart sections for {exp_id}/{run_id}")
            chunks = read_json(run_root / "timeline/chunks.json", wrapped=True)["chunks"]
            for chunk in chunks:
                payload = read_json(child(run_root / "timeline/legs", chunk["file"]), wrapped=True)
                if payload["run_id"] != run_id:
                    raise ValueError(f"Timeline chunk has wrong run: {chunk['file']}")
            for agent in (run_root / "timeline/agents").glob("*"):
                for filename in ("profile.json", "crp.json", "social.json"):
                    read_json(agent / filename, wrapped=True)
            if manifest.get("export_agent_details", True):
                for uid in range(1, int(manifest["timeline_max_agents"]) + 1):
                    for filename in ("profile.json", "crp.json", "social.json"):
                        read_json(run_root / "timeline/agents" / str(uid) / filename, wrapped=True)
    if actual != expected:
        raise ValueError("Experiment/run index disagrees with export manifest")
    for path in root.rglob("*.json"):
        read_json(path, wrapped=path.name != "manifest.json")
    print(f"Validated {len(experiments)} experiments in {root}")


def pack(args) -> None:
    root = args.root.resolve()
    if root.name != "demo-data":
        raise ValueError("Export directory must be named demo-data")
    validate(root)
    archive = args.archive.resolve()
    if archive.is_relative_to(root):
        raise ValueError("Archive must be outside the dataset directory")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Dataset must not contain symlinks")
    with tarfile.open(archive, "w:gz") as output:
        output.add(root, arcname="demo-data")
    pin = {"release_tag": args.release_tag, "asset": archive.name, "sha256": digest(archive)}
    args.config.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    print(f"Packaged {archive}; release pin written to {args.config}")


def fetch(args) -> None:
    pin = read_json(args.config)
    tag, asset, checksum = pin.get("release_tag"), pin.get("asset"), pin.get("sha256")
    if not tag or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Demo release is not configured. Export and package data following web/README.md, "
                         "publish the archive, and commit the generated web/demo_release.json.")
    child(Path("."), asset)
    if not args.repository:
        raise ValueError("--repository OWNER/REPO is required")
    destination = args.public / "demo-data"
    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination}; remove old exported data first")
    args.public.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        subprocess.run(["gh", "release", "download", tag, "--repo", args.repository,
                        "--pattern", asset, "--dir", temporary], check=True)
        archive = staging / asset
        if digest(archive) != checksum:
            raise ValueError("Demo archive SHA-256 mismatch; check the release pin and uploaded asset")
        extracted = staging / "extracted"
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                parts = Path(member.name).parts
                if (not parts or parts[0] != "demo-data" or ".." in parts
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f"Unsafe archive member: {member.name}")
            source.extractall(extracted, filter="data")
        validate(extracted / "demo-data")
        shutil.copytree(extracted / "demo-data", destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate")
    check.add_argument("root", type=Path)
    package = commands.add_parser("pack")
    package.add_argument("--root", type=Path, default=Path("web/frontend/public/demo-data"))
    package.add_argument("--archive", type=Path, default=Path("demo-data.tar.gz"))
    package.add_argument("--config", type=Path, default=Path("web/demo_release.json"))
    package.add_argument("--release-tag", required=True)
    download = commands.add_parser("fetch")
    download.add_argument("--config", type=Path, default=Path("web/demo_release.json"))
    download.add_argument("--public", type=Path, default=Path("web/frontend/public"))
    download.add_argument("--repository", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            validate(args.root)
        elif args.command == "pack":
            pack(args)
        else:
            fetch(args)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Static demo data error: {error}\n")


if __name__ == "__main__":
    main()
