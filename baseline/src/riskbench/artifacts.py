from __future__ import annotations

from pathlib import Path
from .common import BenchError, file_hash, now, read_json, verify_files, write_json, write_jsonl


def seal_predictions(path: str, predictions: list[dict], metadata: dict) -> None:
    p = Path(path)
    if p.exists() or Path(path + ".meta.json").exists():
        raise BenchError("Refusing to overwrite predictions; use a new run name")
    write_jsonl(p, predictions)
    write_json(path + ".meta.json", dict(metadata, predictions_sha256=file_hash(p), created_at=now()))


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    raise BenchError(f"Missing lock target: {path}")


def freeze(paths: list[str], out: str) -> dict:
    lock = Path(out).resolve()
    if lock.exists():
        raise BenchError("Experiment lock already exists; do not overwrite after internal evaluation")
    base = lock.parent; hashes = {}; roots = []
    for name in paths:
        root = Path(name).resolve()
        if not root.is_relative_to(base):
            raise BenchError("Freeze targets must be under the lock's parent for portable offline transfer")
        roots.append(str(root.relative_to(base)))
        for p in _files(root):
            if p.resolve() == lock or not p.resolve().is_relative_to(base):
                raise BenchError("Lock cannot include itself or external symlinks")
            hashes[str(p.relative_to(base))] = file_hash(p)
    if not hashes:
        raise BenchError("No artifacts to freeze")
    report = {"schema": "riskbench-lock-v1", "created_at": now(), "roots": roots, "files": hashes,
              "implementation_files": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
              "note": "Integrity guard, not access control or proof that internal results have never been inspected"}
    write_json(lock, report)
    return report


def verify_lock(path: str | None, required_paths: list[str], internal: bool) -> str | None:
    if not path:
        if internal:
            raise BenchError("Internal inference requires --lock created after public-only selection")
        return None
    lock = Path(path).resolve(); base = lock.parent; m = read_json(lock)
    verify_files(base, m["files"])
    implementation = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    if implementation != m.get("implementation_files"):
        raise BenchError("Package code changed after experiment freeze")
    current = set()
    for name in m["roots"]:
        current.update(str(p.relative_to(base)) for p in _files(base / name))
    if current != set(m["files"]):
        raise BenchError("Files added/removed under frozen artifact roots")
    for name in required_paths:
        for p in _files(Path(name).resolve()):
            if not p.is_relative_to(base) or str(p.relative_to(base)) not in m["files"]:
                raise BenchError("Model/config/examples were not included in experiment lock")
    return file_hash(lock)
