"""Actual pretrained-model integration, including a synthetic double stop.

Run from violin/: .venv/Scripts/python.exe -m tests.smoke_basic_pitch
The synthetic fixture and results remain in output/validation for inspection.
"""
from pathlib import Path
import json
import logging
import numpy as np
import soundfile as sf

from violin_transcriber.main import main
from violin_transcriber.midi import inspect_file


def run():
    root = Path("output") / "validation"
    root.mkdir(parents=True, exist_ok=True)
    sr = 22050
    duration = 6.0
    t = np.arange(round(duration * sr)) / sr
    audio = np.zeros_like(t)
    # Each tone has its own vibrato, harmonics and finite attack/release.
    specifications = [(69, .5, 2.0, 5.0, .18), (55, 3.0, 5.0, 5.5, .15), (62, 3.05, 4.95, 6.2, .17)]
    for pitch, start, end, rate, depth in specifications:
        local = t - start
        hz = 440 * 2 ** ((pitch - 69 + depth * np.sin(2 * np.pi * rate * local)) / 12)
        phase = 2 * np.pi * np.cumsum(hz) / sr
        envelope = np.clip(local / .04, 0, 1) * np.clip((end - t) / .06, 0, 1)
        audio += .17 * envelope * (np.sin(phase) + .3 * np.sin(2 * phase) + .12 * np.sin(3 * phase))
    source = root / "中文 双音 vibrato.wav"
    sf.write(str(source), np.column_stack([audio, audio]), sr)
    result_root = root / "synthetic"
    existing = set(root.iterdir())
    assert main([str(source), "--output", str(result_root)]) == 0
    new_dirs = [p for p in root.iterdir() if p.is_dir() and p not in existing]
    if len(new_dirs) == 1:
        result_root = new_dirs[0]
    expected_files = ("violin_raw.mid", "violin_plain.mid", "violin_expressive.mid", "notes_raw.csv",
                      "notes_filtered.csv", "suspected_errors.csv", "pitch_bends.json", "model_outputs.npz",
                      "piano_roll.png", "pitch_debug.png", "report.txt", "run.json")
    assert all((result_root / name).stat().st_size > 0 for name in expected_files)
    plain = inspect_file(result_root / "violin_plain.mid")
    expressive = inspect_file(result_root / "violin_expressive.mid")
    bends_document = json.loads((result_root / "pitch_bends.json").read_text(encoding="utf-8"))
    from basic_pitch.note_creation import note_events_to_midi
    events = [(n["start_sec"], n["end_sec"], n["pitch_midi"], n["amplitude_or_score"], n["raw_bend_bins"])
              for n in bends_document["notes"]]
    upstream_multiple = note_events_to_midi(events, multiple_pitch_bends=True)
    upstream_default = note_events_to_midi(events, multiple_pitch_bends=False)
    multiple_count = sum(len(i.pitch_bends) for i in upstream_multiple.instruments)
    default_count = sum(len(i.pitch_bends) for i in upstream_default.instruments)
    pair = [(a, b) for a in plain["notes"] for b in plain["notes"]
            if a["pitch"] == 55 and b["pitch"] == 62 and
            min(a["end"], b["end"], 5) - max(a["start"], b["start"], 3) > .5]
    summary = dict(expected_double_stop=[55, 62], detected_double_stop=bool(pair),
                   detected_pitches=sorted({n["pitch"] for n in plain["notes"]}),
                   note_count=len(plain["notes"]), bends=len(expressive["bends"]),
                   upstream_true_bend_count=multiple_count, upstream_false_bend_count=default_count,
                   reference="Synthesized tones, not a real violin accuracy benchmark")
    (result_root / "synthetic_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert not plain["bends"]
    assert len(plain["notes"]) == len(expressive["notes"])
    assert pair, "Model missed the synthetic double stop; inspect saved baseline"
    assert expressive["bends"]
    assert multiple_count > default_count, "multiple_pitch_bends did not preserve additional overlap bends"
    print(json.dumps(summary, indent=2))
    # Empty audio must not be reported as successful; diagnostics must survive.
    silent = root / "silence.wav"
    sf.write(str(silent), np.zeros(sr), sr)
    with np.errstate(divide="ignore", invalid="ignore"):
        assert main([str(silent), "--output", str(root / "silence")]) == 1
    print("PASS: real model, stereo Chinese/space path, double stop, bends, all artifacts, silence failure")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
