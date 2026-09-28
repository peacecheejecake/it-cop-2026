from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class BenchError(ValueError):
    """A failed data contract or unsafe experiment operation."""


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc(value: str | int | float) -> datetime:
    if isinstance(value, (float, int)):
        return datetime.fromtimestamp(value, timezone.utc)
    value = str(value).strip()
    if value.replace(".", "", 1).isdigit():
        return datetime.fromtimestamp(float(value), timezone.utc)
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise BenchError(f"Timezone required: {value!r}")
    return d.astimezone(timezone.utc)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise BenchError(f"Invalid JSON at line {i}") from e
                if not isinstance(row, dict):
                    raise BenchError(f"JSON object required at line {i}")
                rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical(row) + "\n")
    tmp.replace(path)


def new_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise BenchError(f"Refusing to overwrite nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def unique(rows: list[dict], key: str = "id") -> dict[str, dict]:
    out = {}
    for r in rows:
        if not isinstance(r.get(key), str) or not r[key]:
            raise BenchError(f"Nonempty string {key} required")
        if r[key] in out:
            raise BenchError(f"Duplicate {key}: {r[key]}")
        out[r[key]] = r
    return out


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scikit-learn", "torch", "transformers", "httpx", "safetensors"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def assert_binary(labels: list[dict], both: bool = False) -> list[int]:
    ys = []
    for r in labels:
        y = r.get("label")
        if type(y) is not int or y not in (0, 1):
            raise BenchError("Labels must be integer 0/1; unresolved cases must be excluded explicitly")
        ys.append(y)
    if both and set(ys) != {0, 1}:
        raise BenchError("Both classes are required in public train and validation")
    return ys


def check_score(score: Any) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise BenchError("Finite numeric score required")
    return float(score)


def seal_files(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: file_hash(directory / name) for name in names}


def verify_files(directory: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        p = (directory / name).resolve()
        if not p.is_relative_to(directory.resolve()) or not p.is_file() or file_hash(p) != expected:
            raise BenchError(f"Artifact integrity check failed: {name}")


def fingerprint_record(row: dict) -> dict:
    return {"id": row["id"], "patch_hash": row["patch_hash"], "group_id": row["group_id"]}


def guard_overlap(train_fingerprints: list[dict], target: list[dict]) -> None:
    for key in ("id", "patch_hash", "group_id"):
        seen = {r[key] for r in train_fingerprints}
        if seen.intersection(r[key] for r in target):
            raise BenchError(f"Train/target overlap in {key}; do not silently exclude target rows")
