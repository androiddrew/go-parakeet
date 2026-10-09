"""Freeze the recorded synthetic corpus and precision-specific reference stages."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

NAMES = ("clean-us", "clean-gb", "repetition-us", "repetition-gb")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("testdata/native"))
    args = parser.parse_args()

    def write(relative, data):
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f"refusing to change frozen fixture {path}")
        path.write_bytes(data)

    provenance = {"license": "Apache-2.0 (original prompts and fixture data)",
                  "model_revision": "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
                  "reference_runtime": "onnxruntime 1.22.0 CPU, one thread",
                  "reference_algorithm": "onnx-asr 675f0e68c24d846ee1775743e92d9b4ed452380e",
                  "sources": {}}
    for precision in ("fp32", "int8"):
        source = args.artifacts / f"reference-{precision}"
        for name in NAMES:
            reference = json.loads((source / f"{name}.json").read_text())
            for stage in reference["stages"].values():
                raw = (source / stage["file"]).read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != stage["sha256"] or len(raw) != math.prod(stage["shape"]) * 4:
                    raise RuntimeError(f"invalid stage: {stage['file']}")
                blob = f"blobs/{digest}.f32.gz"
                write(blob, gzip.compress(raw, mtime=0))
                stage["file"] = "../" + blob
            write(f"{precision}/{name}.json", (json.dumps(reference, indent=2, ensure_ascii=False) + "\n").encode())
            provenance["sources"][name] = json.loads((args.artifacts / "fixtures" / f"{name}.json").read_text())
    write("PROVENANCE.json", (json.dumps(provenance, indent=2) + "\n").encode())
    checksums = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.output)}")
    write("SHA256SUMS", ("\n".join(checksums) + "\n").encode())
    print(f"frozen {len(checksums)} files, {sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())} bytes")


if __name__ == "__main__":
    main()
