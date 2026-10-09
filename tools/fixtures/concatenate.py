"""Build a reproducible long PCM fixture from the saved short reference inputs."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    audio = args.output / "recording.samples.f32"
    manifest = args.output / "recording.json"
    if audio.exists() or manifest.exists():
        raise RuntimeError("fixture already exists; use a new output directory")
    blocks, texts, sources = [], [], []
    for name in ("clean-us", "clean-gb", "repetition-us", "repetition-gb"):
        ref = json.loads((args.references / f"{name}.json").read_text())
        source = ref["stages"]["samples"]
        raw = (args.references / source["file"]).read_bytes()
        if source["file"].endswith(".gz"):
            raw = gzip.decompress(raw)
        if len(raw) % 4 or hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise RuntimeError(f"invalid/checksum-mismatched PCM: {source['file']}")
        blocks.extend((raw, bytes(2560 * 4)))  # 160 ms gap at 16 kHz
        texts.append(ref["text"])
        sources.append(source)
    digest = hashlib.sha256()
    with audio.open("xb") as output:
        for _ in range(args.repeat):
            for block in blocks:
                output.write(block)
                digest.update(block)
    sample_count = sum(map(len, blocks)) // 4 * args.repeat
    manifest.write_text(json.dumps({
        "sample_rate": 16000,
        "format": "little-endian float32 mono PCM",
        "samples": sample_count,
        "seconds": sample_count / 16000,
        "sha256": digest.hexdigest(),
        "repeat": args.repeat,
        "gap_samples": 2560,
        "sources": sources,
        "reference_text": (" ".join(texts) + " ") * args.repeat,
    }, indent=2) + "\n")
    print(f"{audio}: {sample_count / 16000:.3f} seconds, sha256 {digest.hexdigest()}")


if __name__ == "__main__":
    main()
