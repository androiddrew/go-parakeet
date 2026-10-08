"""Record CPU ONNX stage outputs and TDT decisions for Go parity checks.

Decoding adapts istupakov/onnx-asr at 675f0e68c24d846ee1775743e92d9b4ed452380e
(src/onnx_asr/asr.py and models/nemo.py), MIT; see THIRD_PARTY.md. Changes:
single-item execution, explicit tensor types, stage dumps, and fixture metadata.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("audio", type=Path, nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    suffix = ".int8" if args.precision == "int8" else ""
    def session(filename):
        return ort.InferenceSession(str(args.assets / filename), sess_options=options,
                                    providers=["CPUExecutionProvider"])
    pre = session("nemo128.onnx")
    encoder = session(f"encoder-model{suffix}.onnx")
    decoder = session(f"decoder_joint-model{suffix}.onnx")
    vocab = {int(i): t for t, i in (line.rsplit(" ", 1) for line in
                                    (args.assets / "vocab.txt").read_text().splitlines())}
    blank = next(i for i, t in vocab.items() if t == "<blk>")
    for audio in args.audio:
        out = args.output / audio.stem
        if out.with_suffix(".json").exists():
            raise RuntimeError(f"reference already exists: {out}")
        with wave.open(str(audio)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        features, lens = pre.run(None, {"waveforms": samples[None], "waveforms_lens": np.array([samples.size], dtype=np.int64)})
        enc, enc_lens = encoder.run(None, {"audio_signal": features, "length": lens})
        state = [np.zeros((2, 1, 640), dtype=np.float32) for _ in range(2)]
        previous, frame, emitted = blank, 0, 0
        tokens = []
        while frame < enc_lens[0]:
            logits, s1, s2 = decoder.run(["outputs", "output_states_1", "output_states_2"], {
                "encoder_outputs": enc[:, :, frame:frame+1],
                "targets": np.array([[previous]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": state[0], "input_states_2": state[1],
            })
            logits = logits.reshape(-1)
            token, advance = int(logits[:len(vocab)].argmax()), int(logits[len(vocab):].argmax())
            if token != blank:
                state, previous = [s1, s2], token
                tokens.append({"id": token, "piece": vocab[token], "frame": frame,
                               "start_seconds": frame * 0.08, "advance": advance})
                emitted += 1
            if advance > 0:
                frame += advance
                emitted = 0
            elif token == blank or emitted == 10:
                frame += 1
                emitted = 0
        text = "".join(t["piece"].replace("▁", " ") for t in tokens)
        text = re.sub(r"\A\s|\s\B|(\s)\b", lambda m: " " if m.group(1) else "", text)
        files = {}
        for name, values in (("samples", samples), ("features", features), ("encoder", enc)):
            path = out.with_suffix(f".{name}.f32")
            raw = values.astype("<f4").tobytes()
            path.write_bytes(raw)
            files[name] = {"file": path.name, "shape": list(values.shape),
                           "sha256": hashlib.sha256(raw).hexdigest()}
        result = {"runtime": ort.__version__, "precision": args.precision,
                  "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                  "feature_length": int(lens[0]), "encoded_length": int(enc_lens[0]),
                  "text": text, "tokens": tokens, "stages": files}
        out.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"{audio.stem}: {text}", flush=True)


if __name__ == "__main__":
    main()
