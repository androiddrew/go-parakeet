# Release procedure

The v0.1.0 local acceptance matrix is recorded in [VALIDATION.md](VALIDATION.md).
The hosted workflow is configured in `.github/workflows/ci.yml`; its result must
be checked on the candidate commit before publishing a tag. This repository is
a Go module. Its native runtime and model assets are provisioned independently.

## Prepare native runtime and assets

Install a C API 22-compatible runtime, or build the tested ORT 1.22.2 revision:

```bash
python3 tools/build-runtime.py --source /path/to/ort-source \
    --build /path/to/ort-cpu --jobs 2 --cc /usr/bin/gcc-12 --cxx /usr/bin/g++-12

# With a CUDA toolkit and cuDNN development files, add:
# --cuda-home /path/to/cuda --cudnn-home /path/to/cudnn --cuda-arch 89
# Use a separate build directory for CUDA. Choose the architecture for your GPU.

python3 tools/download-assets.py --root runtime-assets --precision all
```

The Linux build helper requires Python >=3.11, CMake >=3.28, Ninja, git, patch,
and a C/C++ compiler. On Ubuntu 24.04, install the CPU build prerequisites with
`sudo apt-get install g++-12 cmake ninja-build git patch python3`. The shared-library
build uses Python's standard library; the fixture-generation Python packages are
not required. It verifies upstream commit
`5630b081cd25e4eccc7516a652ff956e51676794`, builds into the requested directory,
and records its command/compiler/library SHA in `build-report.json`. It does
not install globally. `--background` writes a build log and reports its PID.
The local CUDA validation used nvcc 12.0, GCC 12 and cuDNN 9.8 on an RTX 4090;
details and tested binary hashes are in `validation/runtime-builds.json`.

## Acceptance

```bash
GOTOOLCHAIN=go1.25.0 go test -race ./...
go vet ./...
go build ./cmd/inference-probe
go build ./cmd/inference-bench

GOTOOLCHAIN=go1.25.0 \
PARAKEET_TEST_ORT=/path/to/ort-cpu/Release/libonnxruntime.so.1.22.2 \
PARAKEET_TEST_EXPECT_ORT=1.22.2 PARAKEET_TEST_ASSETS=runtime-assets/parakeet \
PARAKEET_TEST_LONG=1 go test -race -count=1 -timeout=15m ./...

# Repeat CPU acceptance with PARAKEET_TEST_PRECISION=int8.
PARAKEET_TEST_CUDA_ORT=/path/to/ort-cuda/Release/libonnxruntime.so.1.22.2 \
PARAKEET_TEST_ASSETS=runtime-assets/parakeet PARAKEET_TEST_LONG=1 \
    go test -race -tags=cuda -count=1 -timeout=15m -run '^TestCUDA' .
```

Tests default to the frozen, checksummed fixtures in `testdata/`; a TTS service
or Python environment is unnecessary. `PARAKEET_TEST_REFERENCE` can override the
synthetic reference directory for development. Once native testing is enabled,
incomplete configuration fails instead of silently skipping. Inspect CUDA node
placement with the probe's `--provider cuda --verbose` flag.

For an export/runtime change, run the expensive graph-boundary check separately
with `PARAKEET_TEST_LIMIT=1` and `-run '^TestNativeEncoderLimit$'` (or CUDA's
`TestCUDAEncoderLimit`). It bypasses the public 300-second cap and requires
substantial memory. A successful run at 5,001 encoder frames does not establish
valid positional attention; the fixed table covers 5,000 frames. See VALIDATION.md.

Run `tools/interop` against both runtime builds as its README describes. Inspect
fixture attribution, checksums, and dependency versions before changing any goldens.

## Performance record

```bash
python3 tools/fixtures/concatenate.py --references testdata/native/fp32 \
    --output artifacts/bench-source
python3 tools/benchmark.py --output artifacts/release/bench-cpu-fp32.json -- \
    ./inference-bench --assets runtime-assets/parakeet --ort /path/to/ort.so \
    --input artifacts/bench-source/recording.samples.f32 --provider cpu --precision fp32
# Repeat as bench-cpu-int8.json with --precision int8.
# Repeat as bench-cuda-fp32.json with wrapper --gpu and benchmark --provider cuda.
python3 tools/summarize-benchmarks.py --gpu-name 'your measured GPU'
```

The benchmark defaults to 5, 30 and 308 seconds, one first request, one warmup
and three warm measured requests, with 30-second windows and 4-second overlap.
Run providers sequentially. Report warm median/range, first-chunk latency, RSS
(which includes native allocations), Go heap, and sampled GPU memory. First
request timings exclude loading; initialization is recorded separately. GPU
sampling can optionally use `--nvml-library /path/to/libnvidia-ml.so` if a matching
library is needed; that preload applies only to the sampler. Failed sampling
saves a partial report and exits nonzero.

## Publish the module

1. Review the candidate diff, dependency resolution, frozen data changes and notices.
2. Confirm local acceptance. Rename the CHANGELOG's Unreleased entry to `0.1.0`
   with the release date and add a fresh Unreleased heading. Update the README
   release status.
3. Commit the release preparation and push the candidate branch for hosted CI.
4. Confirm hosted CI passes on the exact commit to be tagged, including any final
   documentation edits. If fixes are needed, repeat local checks and hosted CI.
5. Create and push the annotated `v0.1.0` tag on that tested commit. Go consumers
   then install `github.com/androiddrew/go-parakeet@v0.1.0`.
6. Publish release notes from the changelog, linking ASSETS.md and VALIDATION.md.

Keep model/runtime binaries out of the module. Preserve the retained notices and
the frozen fixtures' original licensing in any source archive.
