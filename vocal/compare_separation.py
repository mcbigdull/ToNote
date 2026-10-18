"""Compare the mix, existing HTDemucs vocals and BS-RoFormer with Basic Pitch.

Run with Python 3.11. Heavy dependencies stay in separate subprocess environments.
All paths default relative to this file, independent of the working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

VOCAL = Path(__file__).resolve().parent
ROOT = VOCAL.parent
MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
SOURCES = ("original_mix", "demucs", "bs_roformer")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def invoke(python, worker, args, *extra):
    if not python.is_file():
        raise FileNotFoundError(f"Python environment missing: {python}. See HEATMAP_COMPARISON.md")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="2",
               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               TF_NUM_INTRAOP_THREADS=str(args.threads), TF_NUM_INTEROP_THREADS="1")
    subprocess.run([str(python), str(Path(__file__).resolve()), str(args.audio),
                    "--output", str(args.output), "--worker", worker,
                    "--threads", str(args.threads), "--segment-seconds", str(args.segment_seconds),
                    *map(str, extra)], env=env, check=True)


def prepare(args):
    import shutil
    import numpy as np
    import soundfile as sf
    from vocal_transcriber.audio import decode_audio, find_ffmpeg
    from vocal_transcriber.separation import separate_vocals, choose_device

    original = args.output / "original_mix" / "audio.wav"
    original_meta = original.parent / "audio.json"
    if original.exists() and original_meta.exists():
        if digest(original) != read_json(original_meta)["sha256"]:
            raise ValueError("Cached original mix has changed; use a new output directory")
    if not original.exists():
        decode_audio(args.audio, original, find_ffmpeg())
    data, sr = sf.read(original, dtype="float32", always_2d=True)
    if not np.isfinite(data).all() or np.max(np.abs(data)) < 1e-7:
        raise ValueError("Original audio is empty or non-finite")
    write_json(original.parent / "audio.json", dict(sample_rate=sr, frames=len(data),
               duration_sec=len(data) / sr, channels=data.shape[1], sha256=digest(original)))
    target = args.output / "demucs" / "audio.wav"
    metadata_path = target.parent / "separation.json"
    if target.exists() and metadata_path.exists():
        if digest(target) != read_json(metadata_path)["audio_sha256"]:
            raise ValueError("Cached Demucs audio has changed")
        return
    candidate = args.demucs_vocals
    if candidate is None:
        proposed = VOCAL / "output" / "revised" / args.audio.stem / "vocals.wav"
        report = proposed.parent / "run.json"
        if proposed.is_file() and report.is_file():
            old_input = Path(read_json(report).get("input", ""))
            if old_input.resolve() == args.audio.resolve():
                candidate = proposed
    if candidate:
        audio, rate = sf.read(candidate, dtype="float32", always_2d=True)
        if rate != sr or audio.shape != data.shape or not np.isfinite(audio).all():
            raise ValueError("Existing Demucs vocals must match mix sample rate, length and channels")
        shutil.copyfile(candidate, target)
        provenance = dict(mode="reused", source=str(candidate.resolve()),
                          historical_run=read_json(candidate.parent / "run.json")
                          if (candidate.parent / "run.json").exists() else None,
                          note="Historical source path/duration checked; historical input hash was not recorded.")
    else:
        os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
        import torch
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        generated = separate_vocals(original, target.parent, choose_device("auto"))
        generated.replace(target)
        provenance = dict(mode="new", shifts=1, overlap=0.25, seed=0)
    write_json(metadata_path, dict(model="htdemucs", audio_sha256=digest(target), **provenance))


def export_listening_mp3(source):
    import imageio_ffmpeg
    source = Path(source)
    target = source.with_name("vocals.mp3")
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    temporary = target.with_name("vocals.tmp.mp3")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                    "-y", "-i", str(source), "-vn", "-c:a", "libmp3lame",
                    "-b:a", "320k", str(temporary)], check=True)
    temporary.replace(target)
    print(f"MP3 listening copy: {target}", flush=True)
    return target


def separate_bs(args):
    import importlib.metadata as packages
    import shutil
    import numpy as np
    import soundfile as sf
    import torch
    import imageio_ffmpeg
    from audio_separator.separator import Separator

    folder = args.output / "bs_roformer"
    input_hash = digest(args.output / "original_mix" / "audio.wav")
    target = folder / "audio.wav"
    meta = folder / "separation.json"
    if target.exists() and meta.exists():
        saved = read_json(meta)
        if (saved["model"] != MODEL or saved["audio_sha256"] != digest(target)
                or saved.get("input_sha256") != input_hash):
            raise ValueError("Cached BS-RoFormer model/audio mismatch")
        print("Reusing BS-RoFormer audio", flush=True)
        export_listening_mp3(target)
        return
    # audio-separator probes the command 'ffmpeg'; use the project's bundled binary.
    tools_dir = VOCAL / ".cache" / "bin"
    tools_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = tools_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not ffmpeg.exists():
        shutil.copyfile(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
    os.environ["PATH"] = str(tools_dir) + os.pathsep + os.environ.get("PATH", "")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model_dir = VOCAL / ".cache" / "audio-separator"
    start = time.monotonic()
    separator = Separator(
        model_file_dir=str(model_dir), output_dir=str(folder), output_format="WAV",
        output_single_stem="Vocals", use_soundfile=True,
        normalization_threshold=1.0, amplification_threshold=0.0,
        mdxc_params={"segment_size": 801, "override_model_segment_size": False,
                     "batch_size": 1, "overlap": 2, "pitch_shift": 0})
    separator.load_model(model_filename=MODEL)
    files = separator.separate(str(args.output / "original_mix" / "audio.wav"))
    if len(files) != 1:
        raise RuntimeError(f"Expected a single Vocals file, got {files}")
    path = Path(files[0])
    if not path.is_absolute():
        path = folder / path
    if not path.is_file():
        raise FileNotFoundError(path)
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    original = sf.info(args.output / "original_mix" / "audio.wav")
    if sr != original.samplerate or len(audio) != original.frames or not np.isfinite(audio).all():
        raise ValueError("BS-RoFormer changed the audio length/rate or produced invalid samples")
    path.replace(target)
    configs = {}
    for config_path in model_dir.glob("*.yaml"):
        if "317" in config_path.name:
            shutil.copyfile(config_path, folder / config_path.name)
            configs[config_path.name] = digest(config_path)
    write_json(meta, dict(model=MODEL, audio_separator_version=packages.version("audio-separator"),
                         input_sha256=input_hash,
                         package_versions={name: packages.version(name) for name in
                                           ("numpy", "librosa", "soundfile", "einops", "rotary-embedding-torch")},
                         torch_version=torch.__version__, device=str(separator.torch_device),
                         overlap=2, batch_size=1, segment_size="model config", seed=0,
                         cpu_threads=args.threads,
                         normalization_threshold=1.0, amplification_threshold=0.0,
                         elapsed_sec=time.monotonic() - start, audio_sha256=digest(target),
                         model_sha256=digest(model_dir / MODEL), config_sha256=configs,
                         sample_rate=sr, frames=len(audio), peak=float(np.max(np.abs(audio)))))
    export_listening_mp3(target)


def heatmap(args):
    import importlib.metadata as packages
    import numpy as np
    import soundfile as sf
    import basic_pitch
    from basic_pitch.inference import Model, get_audio_input
    from basic_pitch.note_creation import model_frames_to_time, model_output_to_notes
    from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP

    model = None
    for name in SOURCES:
        folder = args.output / name
        source = folder / "audio.wav"
        if not source.exists():
            continue  # Allows the first two plots to be computed while installing BS-RoFormer.
        fingerprint = dict(audio_sha256=digest(source), basic_pitch=packages.version("basic-pitch"),
                           onset_threshold=0.5, frame_threshold=0.3, minimum_note_length_ms=127.7,
                           multiple_pitch_bends=True, melodia_trick=True, pitch_filter=None,
                           gain_normalization=False, format_version=2,
                           time_mapping="actual overlapped audio windows; preserve tail")
        meta = folder / "analysis.json"
        if (meta.exists() and (folder / "model_outputs.npz").exists()
                and (folder / "notes_raw.json").exists()
                and read_json(meta)["fingerprint"] == fingerprint):
            print(f"Reusing {name} Basic Pitch outputs", flush=True)
            continue
        print(f"Basic Pitch: {name}", flush=True)
        start = time.monotonic()
        if model is None:
            model = Model(basic_pitch.ICASSP_2022_MODEL_PATH)
        # Same model/windows/cropping as Basic Pitch 0.4.0 run_inference.
        # Its unwrap_output trims using integer FPS=86, which loses about 2 s
        # on a 5 min track. Keep all real windows and map frames to audio samples.
        parts = {key: [] for key in ("note", "contour", "onset")}
        time_parts = []
        overlap = 30 * FFT_HOP
        for window, window_time, original_samples in get_audio_input(source, overlap, AUDIO_N_SAMPLES - overlap):
            prediction = model.predict(window)
            retained = prediction["note"].shape[1] - 30
            frame_times = window_time["start"] + np.arange(retained) * FFT_HOP / AUDIO_SAMPLE_RATE
            valid = frame_times < original_samples / AUDIO_SAMPLE_RATE
            time_parts.append(frame_times[valid])
            for key in parts:
                parts[key].append(prediction[key][0, 15:-15][valid])
        arrays = {key: np.concatenate(value) for key, value in parts.items()}
        times = np.concatenate(time_parts)
        if len(times) < 2 or not np.all(np.diff(times) > 0):
            raise ValueError("Invalid audio-window frame mapping")
        upstream_times = model_frames_to_time(len(times))
        midi, events = model_output_to_notes(
            arrays, onset_thresh=0.5, frame_thresh=0.3,
            min_note_len=int(np.round(127.7 / 1000 * AUDIO_SAMPLE_RATE / FFT_HOP)),
            min_freq=None, max_freq=None, multiple_pitch_bends=True, melodia_trick=True)
        for key in ("note", "contour", "onset"):
            if len(arrays[key]) != len(times) or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid Basic Pitch {key} output")
        info = sf.info(source)
        np.savez_compressed(folder / "model_outputs.npz", **arrays, frame_times_sec=times,
                            upstream_frame_times_sec=upstream_times,
                            note_midi=21 + np.arange(arrays["note"].shape[1]),
                            contour_midi=21 + np.arange(arrays["contour"].shape[1]) / 3,
                            audio_duration_sec=info.duration)
        notes = [dict(start=float(np.interp(s, upstream_times, times)),
                      end=float(np.interp(e, upstream_times, times)),
                      upstream_start=float(s), upstream_end=float(e), pitch=int(p), score=float(a),
                      bends=[int(b) for b in bends] if bends is not None else None)
                 for s, e, p, a, bends in events]
        write_json(folder / "notes_raw.json", notes)
        midi.write(str(folder / "basic_pitch_raw.mid"))
        write_json(meta, dict(fingerprint=fingerprint, runtime=model.model_type.name,
                             elapsed_sec=time.monotonic() - start,
                             note_count=len(notes), frames=len(times), duration_sec=info.duration,
                             last_frame_time=float(times[-1]), tail_gap_sec=info.duration-float(times[-1]),
                             shapes={k: list(v.shape) for k, v in arrays.items()},
                             interpretation="Unfiltered Basic Pitch model activations; not a separation accuracy score."))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="?", default=ROOT / "input" / "莫非定律乐团 - Lidocaine.mp3")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--demucs-vocals", type=Path, help="Reuse existing HTDemucs vocals of the SAME full song")
    parser.add_argument("--resume", action="store_true", help="Continue a matching existing comparison")
    parser.add_argument("--stage", choices=("all", "prepare", "bs", "analyze", "render"), default="all")
    parser.add_argument("--segment-seconds", type=float, default=15)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--vocal-python", type=Path, default=ROOT / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--pitch-python", type=Path, default=ROOT / "violin" / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--separator-python", type=Path, default=VOCAL / ".venv-separator" / "Scripts" / "python.exe")
    parser.add_argument("--worker", choices=("prepare", "bs", "heatmap", "render"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.audio = args.audio.resolve()
    args.output = (args.output or VOCAL / "output" / (args.audio.stem + "_separation_compare")).resolve()
    if not args.audio.is_file():
        parser.error(f"Audio not found: {args.audio}")
    if not math.isfinite(args.segment_seconds) or args.segment_seconds < 1 or args.threads < 1:
        parser.error("Require segment-seconds >= 1 and threads >= 1")
    if args.worker:
        if args.worker == "render":
            from vocal_transcriber.heatmap_comparison import render
            render(args.output, args.segment_seconds)
        else:
            {"prepare": prepare, "bs": separate_bs, "heatmap": heatmap}[args.worker](args)
        return
    manifest = args.output / "comparison.json"
    identity = dict(input=str(args.audio), input_sha256=digest(args.audio), bs_model=MODEL,
                    demucs_vocals=str(args.demucs_vocals.resolve()) if args.demucs_vocals else "auto",
                    schema_version=1)
    if manifest.exists():
        if not args.resume:
            parser.error("Output already exists. Use --resume, or choose another --output directory.")
        report = read_json(manifest)
        if report["identity"] != identity:
            parser.error("Existing comparison belongs to different input/settings. Choose another output.")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Output is nonempty without comparison.json; refusing to overwrite")
        args.output.mkdir(parents=True, exist_ok=True)
        report = dict(identity=identity, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    for name in SOURCES:
        (args.output / name).mkdir(exist_ok=True)
    report.update(status="running", stage=args.stage, cpu_threads=args.threads)
    write_json(manifest, report)
    try:
        if args.stage in ("all", "prepare"):
            extra = ("--demucs-vocals", args.demucs_vocals) if args.demucs_vocals else ()
            invoke(args.vocal_python, "prepare", args, *extra)
        if args.stage in ("all", "bs"):
            invoke(args.separator_python, "bs", args)
        if args.stage in ("all", "analyze"):
            invoke(args.pitch_python, "heatmap", args)
        if args.stage in ("all", "render"):
            invoke(args.pitch_python, "render", args)
        report.update(status="success" if args.stage in ("all", "render") else "partial",
                      completed_stage=args.stage, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        report.pop("error", None)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(manifest, report)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
