"""Download a complete offline wheel set for Linux x86_64 / CPython 3.11 from the project lock.

torch is replaced by the CUDA 12.6 build of the same version (2.14.1+cu126, driver >= 525 via CUDA 12 minor-version
compatibility) instead of PyPI's CUDA 13 build (driver >= 580); its nvidia/triton dependencies come from its own metadata.
Markers are evaluated for the target, not this machine, and every wheel is fetched with --no-deps so nothing resolves
against macOS.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement

PROJECT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TORCH = "torch==2.14.1+cu126"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
ENV = {"sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64", "os_name": "posix",
       "python_version": "3.11", "python_full_version": "3.11.13", "implementation_name": "cpython",
       "platform_python_implementation": "CPython", "extra": ""}
PLATFORM = ["--platform", "manylinux_2_28_x86_64", "--platform", "manylinux_2_27_x86_64", "--platform", "manylinux_2_17_x86_64",
            "--platform", "manylinux2014_x86_64", "--platform", "linux_x86_64", "--python-version", "3.11",
            "--implementation", "cp", "--abi", "cp311", "--abi", "abi3", "--abi", "none", "--only-binary=:all:", "--no-deps"]
CUDA13 = ("torch", "triton", "cuda-", "nvidia-")


def pip_download(spec: str, index: str | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", "download", spec, "-d", str(OUT), *PLATFORM]
    if index:
        cmd += ["--index-url", index, "--extra-index-url", "https://pypi.org/simple"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


lock = subprocess.run(["uv", "export", "--frozen", "--extra", "neural", "--no-dev", "--no-emit-project", "--no-hashes",
                       "--format", "requirements-txt"], cwd=PROJECT, check=True, capture_output=True, text=True).stdout
OUT.mkdir(parents=True, exist_ok=True)
pins = []
for line in lock.splitlines():
    if not line or line.startswith(("#", " ")):
        continue
    req = Requirement(line)
    if req.marker and not req.marker.evaluate(ENV):
        continue
    if req.name.startswith(CUDA13):
        continue
    pins.append(f"{req.name}{req.specifier}")
pip_download(TORCH, TORCH_INDEX)
whl = next(OUT.glob("torch-2.14.1+cu126-*.whl"))
meta = zipfile.ZipFile(whl).read(next(n for n in zipfile.ZipFile(whl).namelist() if n.endswith(".dist-info/METADATA"))).decode()
for line in meta.splitlines():
    if line.startswith("Requires-Dist:"):
        req = Requirement(line.split(":", 1)[1].strip())
        if (req.marker is None or req.marker.evaluate(ENV)) and req.name.startswith(("nvidia-", "triton", "cuda-")):
            pins.append(str(req).split(";")[0].strip())
for p in pins:
    pip_download(p, TORCH_INDEX if p.startswith(("nvidia-", "triton")) else None)
    print("ok", p, flush=True)
(OUT / "requirements-offline.txt").write_text("\n".join(sorted([TORCH, *pins])) + "\n")
print(len(list(OUT.glob("*.whl"))), "wheels")
