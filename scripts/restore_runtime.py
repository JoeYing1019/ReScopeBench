"""Reassemble the pinned runtime from small Git-compatible parts; no downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from artifact_utils import safe_member, sha256


def restore(root):
    root = Path(root).resolve()
    vendor = root / "rescopebench/vendor"
    inventory = json.loads((vendor / "runtime-parts.json").read_text())
    image = json.loads((vendor / "image-manifest.json").read_text())
    if (inventory["archive"] != image["archive"] or inventory["bytes"] != image["archive_bytes"]
            or inventory["sha256"] != image["archive_sha256"]):
        raise ValueError("Runtime parts do not identify the pinned image")

    def member(name):
        relative = safe_member(name)
        path = vendor / relative
        if len(relative.parts) != 1 or path.is_symlink() or not path.resolve().is_relative_to(vendor):
            raise ValueError("Unsafe runtime member")
        return path

    target = member(inventory["archive"])
    if target.exists():
        if (not target.is_file() or target.stat().st_size != inventory["bytes"]
                or sha256(target) != inventory["sha256"]):
            raise ValueError("Existing runtime archive differs; move it aside before restoring")
        return {"status": "already_verified", "archive": str(target), "sha256": inventory["sha256"]}
    parts = inventory["parts"]
    if (not parts or len({r["name"] for r in parts}) != len(parts)
            or sum(r["bytes"] for r in parts) != inventory["bytes"]):
        raise ValueError("Incomplete runtime parts inventory")
    temporary = None
    try:
        total_hash, written = hashlib.sha256(), 0
        with tempfile.NamedTemporaryFile(mode="wb", dir=vendor, prefix="runtime-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            for record in parts:
                source = member(record["name"])
                if not source.is_file() or source.stat().st_size != record["bytes"]:
                    raise ValueError("Missing runtime part: " + record["name"])
                part_hash = hashlib.sha256()
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        part_hash.update(block)
                        total_hash.update(block)
                        written += len(block)
                        output.write(block)
                if part_hash.hexdigest() != record["sha256"]:
                    raise ValueError("Runtime part checksum differs: " + record["name"])
        if written != inventory["bytes"] or total_hash.hexdigest() != inventory["sha256"]:
            raise ValueError("Reconstructed runtime checksum differs")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"status": "restored", "archive": str(target), "bytes": written, "sha256": total_hash.hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(restore(Path(__file__).resolve().parents[1]), indent=2))
