"""Path and checksum checks shared by the two setup commands."""

import hashlib
from pathlib import PurePosixPath


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_member(name):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("Unsafe artifact member")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts
            or str(relative) != name or not relative.parts):
        raise ValueError("Unsafe artifact member")
    if any(part.startswith(".") for part in relative.parts):
        raise ValueError("Hidden artifact member is not allowed: " + name)
    return relative
