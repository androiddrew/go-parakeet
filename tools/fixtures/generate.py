"""Record a controlled English corpus from a Kokoro API (explicit developer action)."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


CASES = [
    ("clean-us", "af_heart", "The little bird landed on the windowsill. Please open the blue door."),
    ("clean-gb", "bf_emma", "Our meeting starts at half past ten. Please bring a notebook and a pencil."),
    ("repetition-us", "am_adam", "This is very, very important. We need to check it again, and again, before we leave."),
    ("repetition-gb", "bm_george", "One more time. One more time. The red boat passed the green boat near the bridge."),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8880")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    for name, voice, text in CASES:
        paths = {ext: args.output / f"{name}.{ext}" for ext in ("source.wav", "wav", "json")}
        if any(p.exists() for p in paths.values()):
            raise RuntimeError(f"fixture {name} already exists; use a new output directory")
        request = {"model": "kokoro-v1.0", "voice": voice, "input": text, "speed": 1, "response_format": "wav"}
        http = urllib.request.Request(args.base_url.rstrip("/") + "/v1/audio/speech",
                                      data=json.dumps(request).encode(),
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=180) as response:
            original = response.read()
        # Save a seekable WAV so its sample count is present in the RIFF header.
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                   "-i", "pipe:0", "-af", "aresample=16000:resampler=soxr:precision=28",
                   "-ac", "1", "-c:a", "pcm_s16le", str(paths["wav"])]
        subprocess.run(command, input=original, check=True)
        paths["source.wav"].write_bytes(original)
        manifest = {
            "request": request,
            "expected_spoken_text": text,
            "source": args.base_url,
            "ffmpeg": ffmpeg,
            "conversion": command,
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for ext, p in paths.items() if ext != "json"},
        }
        paths["json"].write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"recorded {name}", flush=True)


if __name__ == "__main__":
    main()
