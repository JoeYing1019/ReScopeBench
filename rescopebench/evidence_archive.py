"""Keep exact model/worker evidence compactly before removing working copies."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import uuid


ARCHIVE = "retained-model-evidence.tar.gz"
INDEX = "evidence-index.json"
PATHS = (
    "provider/wire", "provider/requests", "provider/responses",
    "episode/hermes-wire", "episode/provider-admitted", "episode/provider-returned",
    "episode/exchange", "episode/events", "episode/history-events",
    "episode/sources-before.json", "episode/sources-after.json",
)


def _files(attempt):
    for relative in PATHS:
        path = attempt / relative
        if not path.exists():
            continue
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        for item in files:
            if item.is_symlink():
                raise ValueError("cannot archive symlinked model evidence")
            if item.is_file():
                yield item


def verify_archive(path):
    with tarfile.open(path, "r:gz") as bundle:
        index = json.load(bundle.extractfile(INDEX))
        for name, record in index["files"].items():
            stream = bundle.extractfile(name)
            digest, size = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            if digest.hexdigest() != record["sha256"] or size != record["bytes"]:
                raise ValueError(f"archive evidence mismatch: {name}")
    return index


def archive_attempt(attempt):
    attempt = Path(attempt).resolve()
    destination = attempt / ARCHIVE
    if destination.exists():
        index = verify_archive(destination)
        # A completed attempt is immutable; do not silently ignore new evidence.
        for path in _files(attempt):
            name = path.relative_to(attempt).as_posix()
            if name not in index["files"] or hashlib.sha256(path.read_bytes()).hexdigest() != index["files"][name]["sha256"]:
                raise ValueError(f"evidence changed after archiving: {name}")
        return {"status": "archived", "path": ARCHIVE, "bytes": destination.stat().st_size,
                "files": len(index["files"]), "logical_bytes": index["logical_bytes"]}
    files = list(_files(attempt))
    if not files:
        return {"status": "no_raw_evidence_available", "path": None, "files": 0}
    index = {"schema": "rescopebench-exact-model-evidence/v1", "files": {}, "logical_bytes": 0}
    temporary = destination.with_name(f".{ARCHIVE}.{uuid.uuid4().hex}.tmp")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=3) as bundle:
            for path in files:
                data = path.read_bytes()
                name = path.relative_to(attempt).as_posix()
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                entry.mode = 0o600
                bundle.addfile(entry, io.BytesIO(data))
                index["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                index["logical_bytes"] += len(data)
            payload = json.dumps(index, sort_keys=True).encode()
            entry = tarfile.TarInfo(INDEX)
            entry.size = len(payload)
            entry.mode = 0o600
            bundle.addfile(entry, io.BytesIO(payload))
        verify_archive(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": "archived", "path": ARCHIVE, "bytes": destination.stat().st_size,
            "files": len(files), "logical_bytes": index["logical_bytes"]}


def read_json(attempt, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence path must stay within an attempt")
    attempt = Path(attempt)
    if (attempt / path).is_file():
        return json.loads((attempt / path).read_text())
    archive = attempt / ARCHIVE
    if not archive.is_file():
        raise FileNotFoundError(attempt / path)
    with tarfile.open(archive, "r:gz") as bundle:
        try:
            return json.load(bundle.extractfile(path.as_posix()))
        except KeyError as error:
            raise FileNotFoundError(attempt / path) from error
