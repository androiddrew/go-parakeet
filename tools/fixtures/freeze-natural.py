"""Freeze Python-reference text/token outputs separately from human transcripts."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
parser.add_argument("--output", type=Path, default=Path("testdata/natural"))
args = parser.parse_args()
manifest = json.loads((args.output / "manifest.json").read_text())
for precision in ("fp32", "int8"):
    result = {}
    for entry in manifest["entries"]:
        path = args.artifacts / f"natural-reference-{precision}" / f"{entry['name']}.json"
        ref = json.loads(path.read_text())
        if ref["stages"]["samples"]["sha256"] != entry["sha256"]:
            raise RuntimeError(f"reference input differs: {entry['name']}")
        result[entry["name"]] = {k: ref[k] for k in ("runtime", "precision", "text", "tokens")}
    target = args.output / f"{precision}.json"
    data = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if target.exists() and target.read_text() != data:
        raise RuntimeError(f"refusing to overwrite {target}")
    target.write_text(data)
sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(args.output.iterdir()) if p.name != "SHA256SUMS"]
(args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
