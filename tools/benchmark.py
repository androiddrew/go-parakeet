"""Run inference-bench with sampled per-process GPU memory (Linux developer tool)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--nvml-library", type=Path, help="optional matching NVML library, preloaded only into the sampler")
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    parser.error("a benchmark command is required after --")
if args.output.exists():
    raise RuntimeError("report already exists")
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
sampler_env = os.environ.copy()
if args.nvml_library:
    sampler_env["LD_PRELOAD"] = str(args.nvml_library.resolve())
peak_mib = 0
errors = []
stopped = threading.Event()


def sample():
    global peak_mib
    while not stopped.wait(0.1):
        if not args.gpu:
            continue
        try:
            query = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                             "--format=csv,noheader,nounits"], text=True, timeout=5, env=sampler_env)
            for row in query.splitlines():
                pid, memory = row.split(",")
                if int(pid.strip()) == process.pid:
                    peak_mib = max(peak_mib, int(memory.strip()))
        except Exception as error:
            errors.append(str(error))
            return


monitor = threading.Thread(target=sample)
monitor.start()
start = time.monotonic()
stdout, stderr = process.communicate()
stopped.set()
monitor.join()
if process.returncode:
    raise RuntimeError(stderr)
result = json.loads(stdout)
result.update(command=command, wall_including_loading_seconds=time.monotonic()-start,
              gpu_sampled_peak_mib=peak_mib if args.gpu else None,
              gpu_sampling_interval_seconds=0.1 if args.gpu else None,
              gpu_sampler_nvml_library=str(args.nvml_library) if args.nvml_library else None,
              gpu_sampling_errors=errors)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
if args.gpu and (errors or peak_mib == 0):
    raise RuntimeError(f"GPU memory measurement failed (partial report saved): {errors}")
print(args.output)
