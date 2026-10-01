"""Hashing, canonical JSON and atomic file writes shared by every module."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class DiffLabError(Exception):
    exit_code = 1


class ConfigError(DiffLabError):
    exit_code = 2


class PolicyError(DiffLabError):
    exit_code = 3


class IntegrityError(DiffLabError):
    exit_code = 4


class ExecutionError(DiffLabError):
    exit_code = 5


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode())


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def commit_generation(base: str | Path, kind: str, tmp_dir: str | Path, keep: int = 2) -> Path:
    """Move a fully written tmp dir to an immutable `<kind>-NNNNN` generation, then atomically repoint `<kind>.json`.

    A crash at any point leaves the previous pointer and its generation intact; older generations are
    pruned only after the new pointer is durable.
    """
    base = Path(base)
    gens = sorted(p for p in base.glob(f"{kind}-[0-9]*") if p.is_dir())
    seq = int(gens[-1].name.rsplit("-", 1)[1]) + 1 if gens else 1
    final = base / f"{kind}-{seq:05d}"
    os.replace(tmp_dir, final)
    files = {p.relative_to(final).as_posix(): sha256_file(p) for p in sorted(final.rglob("*")) if p.is_file()}
    atomic_write_json(base / f"{kind}.json", {"generation": final.name, "files_sha256": files})
    for old in gens[: max(len(gens) + 1 - keep, 0)]:
        _rmtree(old)
    return final


def resolve_generation(base: str | Path, kind: str, verify: bool = True) -> Path | None:
    base = Path(base)
    ptr = base / f"{kind}.json"
    if not ptr.exists():
        return None
    meta = read_json(ptr)
    path = base / meta["generation"]
    if verify:
        for rel, digest in meta["files_sha256"].items():
            f = path / rel
            if not f.is_file() or sha256_file(f) != digest:
                raise IntegrityError(f"checkpoint {path.name}/{rel} missing or corrupted")
    return path


def _rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path)


def tree_digest(root: str | Path, patterns: tuple[str, ...]) -> str:
    """sha256 over (relative path, file sha256) of every file matching the glob patterns under root."""
    root = Path(root)
    files = sorted({p for pat in patterns for p in root.glob(pat) if p.is_file() and "__pycache__" not in p.parts})
    return sha256_json([[p.relative_to(root).as_posix(), sha256_file(p)] for p in files])
