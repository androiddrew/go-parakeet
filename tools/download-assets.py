"""Provision and verify pinned model assets; Python stdlib only, development use."""

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request


REVISION = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"
BASE = f"https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/{REVISION}/"
CHECKSUMS = Path(__file__).resolve().parents[1] / "SHA256SUMS"


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runtime-assets"))
    parser.add_argument("--precision", choices=("int8", "fp32", "all"), default="fp32")
    args = parser.parse_args()
    for line in CHECKSUMS.read_text().splitlines():
        expected, relative = line.split()
        name = Path(relative).name
        if "model" in name:
            precision = "int8" if ".int8." in name else "fp32"
            if args.precision not in (precision, "all"):
                continue
        destination = args.root / relative
        if destination.exists():
            if sha256(destination) != expected:
                raise RuntimeError(f"checksum mismatch: {destination}; move it aside before retrying")
            print(f"verified {destination}", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            print(f"downloading {name}", flush=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(BASE + name + "?download=true", timeout=120) as response:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            actual = sha256(temporary)
            if actual != expected:
                raise RuntimeError(f"download checksum mismatch: {name} ({temporary.stat().st_size} bytes): expected {expected}, got {actual}")
            temporary.replace(destination)
            print(f"verified {destination}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
