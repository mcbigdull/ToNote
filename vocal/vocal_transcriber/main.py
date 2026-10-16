import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .config import Config, TranscriptionError, note_name, note_number


def parser():
    result = argparse.ArgumentParser(description="本地歌曲人声自动扒谱：音频 → Demucs → F0 → MIDI / MusicXML")
    result.add_argument("audio", type=Path)
    result.add_argument("--output", type=Path, default=Path("output"))
    result.add_argument("--skip-separation", action="store_true", help="输入已是纯人声时跳过 Demucs")
    result.add_argument("--vocals-file", type=Path, help="复用已分离的人声 WAV；节拍仍使用原歌曲")
    result.add_argument("--octave-check", action=argparse.BooleanOptionalAction, default=True,
                        help="默认用第二种模型和频谱检查八度，保存疑点记录")
    result.add_argument("--keep-intermediate", action=argparse.BooleanOptionalAction, default=True,
                        help="默认保留中间数据；--no-keep-intermediate 在成功后仅移除 WAV")
    result.add_argument("--min-note", default="C2")
    result.add_argument("--max-note", default="C7")
    result.add_argument("--min-note-duration", type=float, default=0.08)
    result.add_argument("--pitch-tolerance-cents", type=float, default=50)
    result.add_argument("--min-confidence", type=float, default=0.5)
    result.add_argument("--smooth-ms", type=float, default=70)
    result.add_argument("--stable-ms", type=float, default=100)
    result.add_argument("--max-gap-ms", type=float, default=30)
    result.add_argument("--silence-db", type=float, default=-55)
    result.add_argument("--quantization-strength", type=float, default=1.0)
    result.add_argument("--time-signature", default="4/4")
    result.add_argument("--key", help="手动调号，例如 G、F#m、Bb")
    result.add_argument("--bpm", type=float, help="手动四分音符 BPM，覆盖自动检测")
    result.add_argument("--beat-offset", type=float, default=0.0, help="手动 BPM 时的节拍相位（秒）")
    result.add_argument("--pitch-backend", choices=("pyin", "crepe"), default="pyin")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--verbose", action="store_true")
    return result


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def save_pitch(path, track):
    write_csv(path, ["time", "frequency", "confidence", "midi_pitch"],
              zip(track.time, track.frequency, track.confidence, track.midi_pitch))


def run(args, config, output):
    from .audio import decode_audio, find_ffmpeg, load_mono
    from .beat import analyze_beats
    from .midi_export import export_midi
    from .musicxml_export import export_musicxml, parse_key
    from .pitch import extract_pitch, smooth_pitch
    from .quantization import quantize_notes
    from .segmentation import segment_notes
    from .separation import choose_device, separate_vocals
    from .visualization import plot_pitch

    if config.key:
        parse_key(config.key)
    device = choose_device(config.device)
    logging.info("设备：%s；F0：%s（pYIN 始终使用 CPU）", device, config.backend)
    ffmpeg = find_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="decode_", dir=output) as temporary:
        decoded = Path(temporary) / "input.wav"
        logging.info("解码音频：%s", args.audio)
        decode_audio(args.audio.resolve(), decoded, ffmpeg)
        original, sr = load_mono(decoded)
        if args.vocals_file:
            vocals = output / "vocals.wav"
            decode_audio(args.vocals_file.resolve(), vocals, ffmpeg)
        elif args.skip_separation:
            vocals = output / "vocals.wav"
            shutil.copyfile(decoded, vocals)
        else:
            logging.info("分离人声；首次运行需要下载预训练模型。")
            vocals = separate_vocals(decoded, output, device)
    voice, sr = load_mono(vocals)
    if abs(len(voice) - len(original)) > sr * .1:
        raise TranscriptionError("复用的人声与原歌曲长度不一致，无法对齐。")
    raw = extract_pitch(voice, sr, config, device)
    save_pitch(output / "pitch_raw.csv", raw)
    corrected = raw
    if args.octave_check:
        from dataclasses import replace
        from .octaves import reconcile_octaves
        # CREPE cannot represent C7; unsupported frames remain uncorrected.
        reference_config = replace(config, backend="crepe" if config.backend == "pyin" else "pyin",
                                   min_pitch=max(32, config.min_pitch), max_pitch=min(95, config.max_pitch))
        if reference_config.min_pitch >= reference_config.max_pitch:
            raise TranscriptionError("音域超出交叉检查支持范围，请使用 --no-octave-check。")
        logging.info("使用 %s 交叉检查八度。", reference_config.backend)
        reference = extract_pitch(voice, sr, reference_config, device)
        save_pitch(output / "pitch_reference.csv", reference)
        corrected, audit = reconcile_octaves(raw, reference, voice, sr, config)
        save_pitch(output / "pitch_corrected.csv", corrected)
        (output / "octave_review.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("八度校验：修正 %d 段，保留待复核 %d 段。", sum(r['action']=='corrected' for r in audit), sum(r['action']=='review' for r in audit))
    smoothed = smooth_pitch(corrected, config)
    save_pitch(output / "pitch_smoothed.csv", smoothed)
    notes = segment_notes(smoothed, config)
    write_csv(output / "notes_raw.csv", ["pitch", "note_name", "start_sec", "end_sec", "duration_sec", "confidence"],
              ((n.pitch, note_name(n.pitch), n.start, n.end, n.end - n.start, n.confidence) for n in notes))
    if not notes:
        plot_pitch(raw, smoothed, [], output / "debug_pitch.png")
        raise TranscriptionError("F0 未检出足够稳定的有效音符。请检查 vocals.wav 和 pitch_raw.csv，调整音域、置信度或稳定时长。未生成空 MIDI。")
    logging.info("已提取 %d 个音符，使用原始歌曲分析节拍。", len(notes))
    grid = analyze_beats(original, sr, config)
    (output / "beats.json").write_text(json.dumps({"bpm": grid.bpm, "source": grid.source,
        "beat_times": grid.times.tolist(), "beat_positions": grid.beats.tolist()}, indent=2), encoding="utf-8")
    quantized = quantize_notes(notes, grid, config)
    write_csv(output / "notes_quantized.csv", ["pitch", "note_name", "start_sec", "end_sec", "duration_sec", "start_beat", "duration_beats", "confidence"],
              ((n.pitch, note_name(n.pitch), n.start_sec, n.end_sec, n.end_sec - n.start_sec,
                n.start_beat, n.duration_beats, n.confidence) for n in quantized))
    export_midi(quantized, grid, config, output / "melody.mid")
    plot_pitch(raw, smoothed, quantized, output / "debug_pitch.png")
    try:
        key = export_musicxml(quantized, grid, config, output / "melody.musicxml")
    except Exception as exc:
        raise TranscriptionError(f"MIDI 已保存，但 MusicXML 导出失败：{exc}。CSV 和调试图已保留。") from exc
    if not args.keep_intermediate:
        for name in ("vocals.wav", "accompaniment.wav"):
            (output / name).unlink(missing_ok=True)
    return {"device": device, "bpm": grid.bpm, "key": key, "notes": len(quantized), "status": "success"}


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    output = None
    metadata = {}
    try:
        config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__
                           if hasattr(args, name)}, min_pitch=note_number(args.min_note),
                        max_pitch=note_number(args.max_note), backend=args.pitch_backend)
        config.validate()
        if args.beat_offset and config.bpm is None:
            raise ValueError("--beat-offset 需要同时指定 --bpm")
        if not args.audio.is_file():
            raise ValueError(f"输入文件不存在：{args.audio}")
        if sys.version_info[:2] not in ((3, 10), (3, 11)):
            raise TranscriptionError("请使用 Python 3.10 或 3.11 创建虚拟环境；当前版本未经此依赖组合支持。详见 README.md。")
        base = args.output.resolve()
        base.mkdir(parents=True, exist_ok=True)
        output = base / args.audio.stem
        try:
            output.mkdir()
        except FileExistsError:
            output = Path(tempfile.mkdtemp(prefix=args.audio.stem + "_", dir=base))
        # Keep model and compiler caches local to the project.
        cache = Path(__file__).resolve().parents[2] / ".cache"
        os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
        os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
        metadata = {"input": str(args.audio.resolve()), "config": asdict(config),
                    "started": datetime.now().isoformat(), "output": str(output),
                    "octave_check": args.octave_check,
                    "vocals_file": str(args.vocals_file.resolve()) if args.vocals_file else None}
        metadata.update(run(args, config, output))
        logging.info("完成：%s", output)
        return 0
    except KeyboardInterrupt:
        metadata.update(status="interrupted", error="用户中断")
        logging.error("已中断，现有中间文件保留在 %s", output)
        return 130
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        if isinstance(exc, ImportError):
            logging.error("依赖缺失或不兼容，请在 Python 3.11 环境运行 python -m pip install -r requirements.txt。%s", exc)
        else:
            logging.error("%s", exc, exc_info=args.verbose)
        if output:
            logging.error("本次结果及诊断文件：%s", output)
        return 1
    finally:
        if output:
            try:
                (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as exc:
                logging.error("无法写入 run.json：%s", exc)
