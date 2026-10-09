"""Build a pinned ORT 1.22.2 for release validation without installing it globally.

Requires Linux, git, patch, CMake >=3.28, Ninja, a C/C++ compiler, and Python >=3.11. CUDA
additionally requires the caller's CUDA toolkit and cuDNN development files.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

COMMIT = "5630b081cd25e4eccc7516a652ff956e51676794"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--cudnn-home", type=Path)
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or bool(args.cuda_home) != bool(args.cudnn_home):
        parser.error("positive --jobs and both CUDA/cuDNN homes are required")
    args.source, args.build = args.source.resolve(), args.build.resolve()
    args.build.mkdir(parents=True, exist_ok=True)
    if args.background:
        command = [sys.executable, str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != "--background"]]
        with (args.build / "build.log").open("ab") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"build pid {child.pid}; log: {args.build / 'build.log'}")
        return
    report = {"source_commit": COMMIT, "pid": os.getpid(), "started": time.time(), "status": "building"}
    report_path = args.build / "build-report.json"
    try:
        if not args.source.exists():
            subprocess.run(["git", "clone", "--depth", "1", "--branch", "v1.22.2",
                            "https://github.com/microsoft/onnxruntime.git", str(args.source)], check=True)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip()
        if actual != COMMIT:
            raise RuntimeError(f"expected ORT {COMMIT}, found {actual}")
        if subprocess.check_output(["git", "diff", "HEAD", "--stat"], cwd=args.source):
            raise RuntimeError("ORT source has local modifications")
        command = [sys.executable, str(args.source / "tools/ci_build/build.py"),
                   "--build_dir", str(args.build), "--config", "Release", "--update", "--build",
                   "--build_shared_lib", "--skip_tests", "--parallel", str(args.jobs),
                   "--cmake_generator", "Ninja", "--compile_no_warning_as_error"]
        defines = ["onnxruntime_BUILD_UNIT_TESTS=OFF", f"CMAKE_C_COMPILER={args.cc}", f"CMAKE_CXX_COMPILER={args.cxx}"]
        if args.cuda_home:
            command += ["--use_cuda", "--cuda_home", str(args.cuda_home.resolve()),
                        "--cudnn_home", str(args.cudnn_home.resolve()), "--nvcc_threads", "1"]
            defines += [f"CMAKE_CUDA_ARCHITECTURES={args.cuda_arch}", f"CMAKE_CUDA_HOST_COMPILER={args.cxx}"]
        command += ["--cmake_extra_defines", *defines]
        report["command"] = command
        report["compiler"] = subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0]
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        subprocess.run(command, cwd=args.source, check=True)
        library = args.build / "Release/libonnxruntime.so.1.22.2"
        with library.open("rb") as stream:
            report["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        report.update(status="built", library=str(library))
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["finished"] = time.time()
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
