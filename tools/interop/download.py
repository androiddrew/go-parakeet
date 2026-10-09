"""Download pinned Kokoro assets for the optional same-process validation tool."""
import argparse
import hashlib
import tempfile
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("runtime-assets"))
args = parser.parse_args()
for name, checksum in {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}.items():
    path = args.root / "kokoro" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as output:
                temporary = Path(output.name)
                response = urllib.request.urlopen(f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/{name}", timeout=120)
                with response:
                    while block := response.read(1024*1024):
                        output.write(block)
            with temporary.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != checksum:
                    raise RuntimeError(f"checksum mismatch: {name}")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != checksum:
            raise RuntimeError(f"checksum mismatch: {path}")
    print(f"verified {path}")
