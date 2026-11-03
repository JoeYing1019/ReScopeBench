"""Install checksum-verified cases from the dataset revision pinned in config/."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from artifact_utils import safe_member, sha256


ROOT = Path(__file__).resolve().parents[1]


def inventory(manifest_path, config):
    if sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Dataset manifest checksum differs from the pinned revision")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "rescopebench-cases/v1" or not manifest.get("files"):
        raise ValueError("Invalid cases manifest")
    selected = {}
    for name, record in manifest["files"].items():
        relative = safe_member(name)
        if len(relative.parts) < 2 or relative.parts[0] != "cases":
            raise ValueError("Dataset member is outside cases/: " + name)
        if (type(record.get("bytes")) is not int or record["bytes"] < 0
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])):
            raise ValueError("Invalid dataset checksum record: " + name)
        if not any(fnmatchcase(name, pattern) for pattern in config.get("exclude", [])):
            selected[name] = record
    if not selected:
        raise ValueError("The cases selection is empty")
    return selected


def destination(root, name):
    path = root / safe_member(name)
    for member in (path, *path.parents):
        if member == root:
            break
        if member.is_symlink():
            raise ValueError("Refusing a symlink in the cases destination: " + name)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Cases destination escapes the package")
    return path


def matches(path, record):
    return (path.is_file() and path.stat().st_size == record["bytes"]
            and sha256(path) == record["sha256"])


def install(root, files, fetch, *, verify_only=False, workers=4):
    """Refuse modified files; interrupted downloads never become installed files."""
    root = Path(root).resolve()
    missing = []
    for name, record in files.items():
        path = destination(root, name)
        if path.exists():
            if not matches(path, record):
                raise ValueError("Local case differs; preserve your edits before reinstalling: " + name)
        else:
            missing.append(name)
    case_root = root / "cases"
    if case_root.exists():
        for directory, dirs, names in os.walk(case_root, followlinks=False):
            for child in [*dirs, *names]:
                path = Path(directory) / child
                name = path.relative_to(root).as_posix()
                destination(root, name)
                if child.startswith("."):
                    raise ValueError("Hidden file or directory in cases/: " + name)
                if path.is_file() and name not in files:
                    raise ValueError("Unlisted file in cases/; review it before installing: " + name)
    if verify_only and missing:
        raise ValueError(f"Missing {len(missing)} case files; run without --verify-only to download them")
    if missing:
        required = sum(files[name]["bytes"] for name in missing)
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < 2 * required + 1024 ** 3:
            raise ValueError("Insufficient disk space for cases and the download cache")

    def download(name):
        record = files[name]
        source = Path(fetch(name))
        if not matches(source, record):
            raise ValueError("Downloaded case checksum differs: " + name)
        target = destination(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix="case-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                with source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, stream, 1024 * 1024)
            if not matches(temporary, record):
                raise ValueError("Case copy checksum differs: " + name)
            temporary.chmod(0o644)
            # link() publishes a complete file without overwriting a racing writer.
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.is_symlink() or not matches(target, record):
                    raise ValueError("Concurrent case installation differs: " + name)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    if missing:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for count, _ in enumerate(pool.map(download, missing), 1):
                if count % 100 == 0 or count == len(missing):
                    print(f"Installed {count}/{len(missing)} missing files", flush=True)
    return {"status": "verified", "files": len(files), "downloaded": len(missing),
            "bytes": sum(row["bytes"] for row in files.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 17), metavar="1-16")
    parser.add_argument("--cache-dir", type=Path, help="HF cache outside this repository")
    parser.add_argument("--local-files-only", action="store_true", help="Use the local HF cache without network access")
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        parser.error("Install the setup dependency: python -m pip install huggingface_hub")
    cache = (args.cache_dir or Path(HF_HUB_CACHE)).expanduser().resolve()
    if cache.is_relative_to(ROOT):
        parser.error("Keep the HF cache outside the repository")
    config = json.loads((ROOT / "config/cases.json").read_text())

    def fetch(name):
        return hf_hub_download(repo_id=config["repo_id"], repo_type="dataset", filename=name,
                               revision=config["revision"], cache_dir=cache,
                               local_files_only=args.local_files_only)

    manifest = Path(fetch(config["manifest"]))
    result = install(ROOT / "rescopebench", inventory(manifest, config), fetch,
                     verify_only=args.verify_only, workers=args.workers)
    print(json.dumps({**result, "revision": config["revision"]}, indent=2))


if __name__ == "__main__":
    main()
