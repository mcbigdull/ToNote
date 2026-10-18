"""Matched-scale Basic Pitch visual diagnostics, without pitch filtering."""
import html
import json
import math
from pathlib import Path

import numpy as np

LABELS = {"original_mix": "Original mix", "demucs": "HTDemucs vocals",
          "bs_roformer": "BS-RoFormer Viperx 1297 vocals"}


def pooled(times, matrix, start, end, max_columns=4500):
    indices = np.flatnonzero((times >= start) & (times < end))
    if len(indices) < 2:
        raise ValueError("Too few frames in requested display interval")
    values = matrix[indices]
    selected = times[indices]
    step = max(1, math.ceil(len(indices) / max_columns))
    starts = np.arange(0, len(indices), step)
    rows = np.maximum.reduceat(values, starts, axis=0)
    centers = np.array([selected[i:i + step].mean() for i in starts])
    return centers, rows, step


def render(output, segment_seconds=15, preview_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output = Path(output)
    manifest = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    # Use an ASCII plot title to avoid missing CJK fonts; the HTML retains full paths.
    song_label = "Separation comparison"
    arrays, notes = {}, {}
    for name in LABELS:
        if preview_only and not (output / name / "model_outputs.npz").exists():
            continue
        with np.load(output / name / "model_outputs.npz") as archive:
            arrays[name] = {key: archive[key] for key in archive.files}
        notes[name] = json.loads((output / name / "notes_raw.json").read_text(encoding="utf-8"))
    reference = arrays["original_mix"]
    end = float(reference["audio_duration_sec"])
    frame_times = reference["frame_times_sec"]
    if (len(frame_times) < 2 or frame_times[0] != 0 or not np.all(np.diff(frame_times) > 0)
            or not 0 <= end - frame_times[-1] <= 0.03):
        raise ValueError("Frame timestamps must cover the original audio through its final 30 ms")
    for name, data in arrays.items():
        if (data["note"].shape != reference["note"].shape
                or data["contour"].shape != reference["contour"].shape
                or not np.array_equal(data["frame_times_sec"], reference["frame_times_sec"])
                or abs(float(data["audio_duration_sec"]) - end) > 1e-6):
            raise ValueError(f"Unaligned comparison: {name}")
    ticks = [21, 24, 36, 48, 60, 72, 84, 96, 108]
    tick_labels = ["A0 / 21"] + [f"C{n // 12 - 1} / {n}" for n in ticks[1:]]

    def decorate(axis, start, stop):
        axis.set(xlim=(start, stop), ylim=(20.5, 109), ylabel="Pitch / MIDI")
        axis.set_yticks(ticks, tick_labels)
        axis.grid(axis="y", color="white", alpha=0.1, linewidth=0.5)

    def heat(axis, name, key, start, stop):
        data = arrays[name]
        centers, values, step = pooled(data["frame_times_sec"], data[key], start, stop)
        factor = 1 if key == "note" else 3
        image = axis.pcolormesh(centers, 21 + np.arange(values.shape[1]) / factor,
                                values.T, shading="nearest", vmin=0, vmax=1, cmap="magma",
                                rasterized=True)
        decorate(axis, start, stop)
        axis.set_title(f"{LABELS[name]} | {key} | display max-pooling: {step} frame(s)", fontsize=10)
        return image

    def compare(start, stop, target):
        fig, axes = plt.subplots(3, 2, figsize=(24, 13), sharex=True, sharey=True,
                                 layout="constrained")
        for row, name in enumerate(LABELS):
            for column, key in enumerate(("note", "contour")):
                image = heat(axes[row, column], name, key, start, stop)
        for axis in axes[-1]:
            axis.set_xlabel("Original audio time (s)")
        fig.colorbar(image, ax=axes, shrink=0.85, label="Model activation (0-1), not calibrated probability")
        fig.suptitle(f"{song_label} | {start:.1f}-{stop:.1f} s\n"
                     "Same Basic Pitch model; no pitch filtering, gain matching or time shifting", fontsize=13)
        fig.savefig(target, dpi=150)
        plt.close(fig)

    for name in arrays:
        fig, axes = plt.subplots(3, 1, figsize=(24, 12), sharex=True, layout="constrained")
        for axis, key in zip(axes[:2], ("note", "contour")):
            image = heat(axis, name, key, 0, end)
            fig.colorbar(image, ax=axis, pad=0.01, label="Activation (0-1)")
        for n in notes[name]:
            axes[2].add_patch(Rectangle((n["start"], n["pitch"] - 0.35),
                                       n["end"] - n["start"], 0.7, color="#1677b8", linewidth=0))
            if n["bends"]:
                axes[2].plot(np.linspace(n["start"], n["end"], len(n["bends"])),
                             n["pitch"] + np.asarray(n["bends"]) / 3, color="#d95f02", linewidth=0.5)
        decorate(axes[2], 0, end)
        axes[2].set_title("Unfiltered Basic Pitch notes + raw contour bends (orange, 1/3 semitone bins)")
        axes[2].set_xlabel("Original audio time (s)")
        fig.suptitle(LABELS[name] + " | model activations, not a ground-truth score")
        fig.savefig(output / name / "pitch_debug.png", dpi=150)
        plt.close(fig)
    if preview_only:
        return
    compare(0, end, output / "comparison_full.png")
    from .heatmap_player import write_player
    write_player(output, arrays)
    validation = dict(status="passed", aligned_frame_times=True, full_pitch_range=[20.5, 109],
                      color_limits=[0, 1], duration_sec=end, frames=len(reference["frame_times_sec"]),
                      note_bins=reference["note"].shape[1], contour_bins=reference["contour"].shape[1],
                      last_frame_time=float(frame_times[-1]), tail_gap_sec=end-float(frame_times[-1]),
                      viewer="continuous_audio_synced", segment_count=0,
                      sources=list(LABELS), accuracy_metrics=None)
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
