"""Record the first three WAV members per language from pinned FLEURS test archives.

No selection by recognition quality. Audio is converted to 16 kHz mono PCM16
for the existing Python reference exporter. FLEURS is CC-BY-4.0 (Google, 2022).
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request
import wave
from array import array
import sys

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data"
LANGUAGES = ("en_us", "fr_fr", "de_de", "es_419", "ru_ru", "el_gr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/natural"))
    parser.add_argument("--freeze", type=Path, default=Path("testdata/natural"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.freeze.mkdir(parents=True, exist_ok=True)
    manifest = args.freeze / "manifest.json"
    if manifest.exists():
        raise RuntimeError("natural corpus already frozen; use a new directory to regenerate")
    entries = []
    ffmpeg = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    for language in LANGUAGES:
        tsv_url = f"{BASE}/{language}/test.tsv"
        with urllib.request.urlopen(tsv_url, timeout=120) as response:
            tsv = response.read()
        rows = {row[1]: row for row in csv.reader(io.StringIO(tsv.decode()), delimiter="\t")}
        archive_url = f"{BASE}/{language}/audio/test.tar.gz"
        count = 0
        with urllib.request.urlopen(archive_url, timeout=120) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                basename = Path(member.name).name
                if not member.isfile() or basename not in rows:
                    continue
                original = archive.extractfile(member).read()
                row = rows[basename]
                name = f"{language}-{count}"
                wav = args.output / f"{name}.wav"
                subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                                "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                               input=original, check=True)
                with wave.open(str(wav)) as audio:
                    pcm = array("h", audio.readframes(audio.getnframes()))
                if sys.byteorder != "little":
                    pcm.byteswap()
                floats = array("f", (x / 32768 for x in pcm))
                if sys.byteorder != "little":
                    floats.byteswap()
                raw = floats.tobytes()
                target = args.freeze / f"{name}.f32.gz"
                target.write_bytes(gzip.compress(raw, mtime=0))
                entries.append({"name": name, "language": language, "id": row[0], "file": target.name,
                                "samples": len(pcm), "sha256": hashlib.sha256(raw).hexdigest(),
                                "transcript": row[2], "normalized_transcript": row[3],
                                "archive": archive_url, "member": member.name,
                                "source_sha256": hashlib.sha256(original).hexdigest(),
                                "tsv_sha256": hashlib.sha256(tsv).hexdigest()})
                print(f"{name} {len(pcm)/16000:.2f}s: {row[2]}", flush=True)
                count += 1
                if count == 3:
                    break
        if count != 3:
            raise RuntimeError(f"missing examples for {language}")
    manifest.write_text(json.dumps({"dataset": "google/fleurs", "revision": REVISION,
                                   "license": "CC-BY-4.0", "attribution": "Google FLEURS, Conneau et al., 2022",
                                   "selection": "first three WAV archive members in each selected test split",
                                   "conversion": "FFmpeg mono 16 kHz PCM16, then float32 divided by 32768",
                                   "ffmpeg": ffmpeg, "entries": entries}, ensure_ascii=False, indent=2) + "\n")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(args.freeze.iterdir()) if p.name != "SHA256SUMS"]
    (args.freeze / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
