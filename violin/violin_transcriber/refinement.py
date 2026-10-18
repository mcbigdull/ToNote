"""Conservative per-pitch merging with independent audio attack evidence.

Never changes the cache/baseline, quantizes time, removes harmonic candidates,
or makes polyphonic events compete for one melody slot.
"""
from dataclasses import asdict, replace
import logging
import os
from pathlib import Path

import numpy as np

from .config import Config, RefinementConfig, refinement_parser
from .main import reserve_output, sha256, write_json
from .midi import bend_samples, export, validate_export
from .notes import filter_range, from_events, overlap_pairs, write_notes


def audio_onsets(path, config):
    import librosa
    y, sr = librosa.load(str(path), sr=config.audio_sample_rate, mono=True)
    # Frequency-axis maximum filtering helps suppress vibrato-driven flux.
    strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.audio_hop_length,
                                          max_size=config.vibrato_max_size)
    frames = librosa.onset.onset_detect(onset_envelope=strength, sr=sr,
        hop_length=config.audio_hop_length, units="frames", normalize=True,
        delta=config.audio_onset_delta,
        wait=round(config.audio_onset_wait_ms / 1000 * sr / config.audio_hop_length))
    times = librosa.frames_to_time(np.arange(len(strength)), sr=sr, hop_length=config.audio_hop_length)
    scale = float(np.ptp(strength))
    normalized = (strength - strength.min()) / scale if scale > 0 else np.zeros_like(strength)
    return times[frames], times, normalized, len(y) / sr


def merge_fragments(notes, arrays, times, attacks, config):
    """Only adjacent non-overlapping same-pitch events can be joined.

    Intervening same-pitch overlaps are barriers, preserving distinct unisons.
    A real audio attack or a strong model onset vetoes a merge independently.
    """
    groups, decisions, output, provenance = {}, [], [], {}
    for note in notes:
        groups.setdefault(note.pitch, []).append(note)
    tolerance = config.onset_tolerance_ms / 1000
    for pitch, selected in sorted(groups.items()):
        selected.sort(key=lambda n: (n.start, n.end, n.note_id))
        current, ids = selected[0], [selected[0].note_id]
        occupied_end = current.end
        for nxt in selected[1:]:
            gap = nxt.start - current.end
            overlaps_unison = nxt.start < occupied_end
            near = (times >= nxt.start - tolerance) & (times <= nxt.start + tolerance)
            model_peak = float(arrays["onset"][near, pitch - 21].max()) if np.any(near) else 1.0
            distance = float(np.min(np.abs(attacks - nxt.start))) if len(attacks) else None
            boundary = (times >= min(current.end, nxt.start) - config.overlap_tolerance_sec) & (times <= max(current.end, nxt.start) + config.overlap_tolerance_sec)
            if not np.any(boundary):
                boundary[np.argmin(np.abs(times - nxt.start))] = True
            active_fraction = float(np.mean(arrays["note"][boundary, pitch - 21] >= config.frame_threshold))
            reason = ("overlapping_unison" if overlaps_unison else
                      "gap_too_long" if gap > config.merge_gap_ms / 1000 else
                      "audio_attack" if distance is not None and distance <= tolerance else
                      "strong_model_onset" if model_peak >= config.strong_model_onset else
                      "activation_gap" if active_fraction < config.bridge_active_fraction else
                      "continuous_same_pitch")
            merged = reason == "continuous_same_pitch"
            decisions.append(dict(left_source_ids=list(ids), right_source_id=nxt.note_id,
                pitch_midi=pitch, boundary_sec=nxt.start, gap_sec=gap,
                model_onset_peak=model_peak, nearest_audio_onset_distance_sec=distance,
                bridge_active_fraction=active_fraction, merged=merged, reason=reason))
            if merged:
                ta, _ = bend_samples(current)
                tb, _ = bend_samples(nxt)
                duration = (current.end - current.start) + (nxt.end - nxt.start)
                score = (current.score * (current.end - current.start) + nxt.score * (nxt.end - nxt.start)) / duration
                current = replace(current, end=nxt.end, score=score,
                    bends=tuple(current.bends or ()) + tuple(nxt.bends or ()),
                    bend_times=tuple(ta) + tuple(tb))
                ids.append(nxt.note_id)
            else:
                output.append(current)
                provenance[current.note_id] = list(ids)
                current, ids = nxt, [nxt.note_id]
            occupied_end = max(occupied_end, nxt.end)
        output.append(current)
        provenance[current.note_id] = list(ids)
    output.sort(key=lambda n: (n.start, n.pitch, n.end))
    return output, decisions, provenance


def decode(arrays, config):
    from basic_pitch.note_creation import model_output_to_notes
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
    _, events = model_output_to_notes(arrays, onset_thresh=config.onset_threshold,
        frame_thresh=config.frame_threshold,
        min_note_len=round(config.minimum_note_length / 1000 * AUDIO_SAMPLE_RATE / FFT_HOP),
        min_freq=None, max_freq=None, include_pitch_bends=True,
        multiple_pitch_bends=True, melodia_trick=config.melodia_trick, midi_tempo=config.midi_tempo)
    return filter_range(from_events(events), config)


def plot_comparison(output, baseline, decoded, refined, arrays, times, attacks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.font_manager import FontProperties
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    panels = [(baseline, "原始默认解码 · 起音阈值0.5 / 持续阈值0.3 / Melodia开启"),
              (decoded, "调整参数后的解码 · 尚未合并碎片"),
              (refined, "保守同音合并后 · 绿色短线为音频起音候选")]
    tail = ("refinement_110_end.png", (110, float(times[-1]))) if times[-1] > 110 else (
        "refinement_detail.png", (0, float(times[-1])))
    for filename, limits in [("refinement_comparison.png", (0, float(times[-1]))), tail]:
        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(18, 9))
        for ax, (notes, title) in zip(axes, panels):
            step = max(1, len(times) // 3200)
            ax.pcolormesh(times[::step], 21 + np.arange(arrays["note"].shape[1]),
                          arrays["note"][::step].T, shading="nearest", cmap="magma", vmin=0, vmax=1)
            ax.add_collection(LineCollection([[(n.start, n.pitch), (n.end, n.pitch)] for n in notes],
                                              colors="cyan", linewidths=1.8))
            ax.plot([n.start for n in notes], [n.pitch for n in notes], "|", color="white", markersize=5)
            ax.set(title=f"{title}：{len(notes)}个音符", ylabel="MIDI 音高", ylim=(53, 103), xlim=limits)
        axes[-1].vlines(attacks, 54, 55, color="lime", linewidth=.6)
        axes[-1].set_xlabel("原音频时间（秒，未量化）· 背景为同一原始激活；青色线为音符，白色短线为切分起点")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def run(audio, cache, output, config):
    from .audio import readable_audio
    from .review import detect, write_review
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ("note", "onset", "contour")}
        times = saved["frame_times_sec"]
    sibling_run = cache.parent / "run.json"
    # An unrelated audio/cache pair must not silently produce plausible MIDI.
    input_hash = sha256(audio)
    if sibling_run.is_file():
        import json
        expected = json.loads(sibling_run.read_text(encoding="utf-8")).get("input_sha256")
        if expected and expected != input_hash:
            raise ValueError("Audio SHA256 does not match the cached model output")
    baseline = decode(arrays, Config())
    decoded = decode(arrays, config)
    with readable_audio(audio, output) as (read_path, _):
        attacks, audio_times, strength, duration = audio_onsets(read_path, config)
    refined, decisions, provenance = merge_fragments(decoded, arrays, times, attacks, config)
    if not refined:
        raise ValueError("No notes remain; inspect decoder settings")
    write_notes(output / "notes_baseline.csv", baseline)
    write_notes(output / "notes_decoded.csv", decoded)
    write_notes(output / "notes_refined.csv", refined)
    validation = {}
    for stage, notes in (("decoded", decoded), ("refined", refined)):
        for expressive in (False, True):
            label = "expressive" if expressive else "plain"
            path = output / f"violin_{stage}_{label}.mid"
            assignment = export(notes, path, config, expressive)
            validation[f"{stage}_{label}"] = validate_export(path, notes, config, expressive, assignment)
    write_json(output / "merge_decisions.json", decisions)
    write_json(output / "pitch_bends_refined.json", dict(
        source_id_mapping="source_ids refer to notes_decoded.csv, not baseline/raw IDs",
        time_origin="Original decoded audio seconds; no quantization",
        bend_timing="All adjusted-decoder bend samples retained at their original export times; no stretching across merged notes",
        notes=[dict(note_id=n.note_id, source_ids=provenance[n.note_id], start_sec=n.start, end_sec=n.end,
                    pitch_midi=n.pitch, bend_times_sec=bend_samples(n)[0].tolist(),
                    raw_bend_bins=list(n.bends or ())) for n in refined]))
    np.savez_compressed(output / "audio_onsets.npz", times_sec=audio_times,
                        normalized_onset_strength=strength, onset_times_sec=attacks)
    write_review(output / "suspected_errors_refined.csv", detect(refined, arrays, times, config))
    plot_comparison(output, baseline, decoded, refined, arrays, times, attacks)
    parameter_counts = {}
    for threshold in (.5, config.onset_threshold):
        for melodia in (False, True):
            candidate = replace(config, onset_threshold=threshold, melodia_trick=melodia)
            parameter_counts[f"onset={threshold},melodia={melodia}"] = len(decode(arrays, candidate))
    report = dict(config=asdict(config), input=str(audio), model_output=str(cache),
        input_sha256=input_hash, model_output_sha256=sha256(cache),
        audio_duration_sec=duration, baseline_notes=len(baseline), decoded_notes=len(decoded),
        refined_notes=len(refined), merges=sum(d["merged"] for d in decisions),
        detected_audio_onsets=len(attacks), parameter_comparison_counts=parameter_counts,
        decoded_bend_samples=sum(len(n.bends or ()) for n in decoded),
        refined_bend_samples=sum(len(n.bends or ()) for n in refined),
        overlapping_pairs_before=len(list(overlap_pairs(decoded))),
        overlapping_pairs_after=len(list(overlap_pairs(refined))), validation=validation,
        limitations=["Counts are not accuracy; listen and inspect real repeated bow attacks.",
            "Decoder parameter changes can lose real notes; baseline is retained separately.",
            "Audio onsets are broadband veto evidence, not per-note proof; another voice can veto merging.",
            "Harmonic candidates are flagged, never automatically deleted.",
            "Model uses its existing frame timing; possible model/audio timing differences remain.",
            "No beat detection, rhythm quantization, MusicXML, or inferred bowing labels."])
    write_json(output / "refinement_report.json", report)
    (output / "report.txt").write_text(
        f"Baseline {len(baseline)} -> adjusted decoder {len(decoded)} -> refined {len(refined)} notes\n"
        f"Merged {report['merges']} boundaries; audio onsets {len(attacks)}\n"
        "All original files unchanged. See merge_decisions.json for every boundary decision.\n"
        "Bends retain adjusted-decoder source timestamps. No rhythm quantization.\n"
        "Fewer notes does not establish better accuracy; compare with recording.\n", encoding="utf-8")
    return report


def main(argv=None):
    args = refinement_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(__file__).resolve().parents[1]
    for key, value in (("NUMBA_CACHE_DIR", root / ".cache/numba"),
                       ("MPLCONFIGDIR", root / ".cache/matplotlib"), ("TF_CPP_MIN_LOG_LEVEL", "2")):
        os.environ.setdefault(key, str(value))
    try:
        config = RefinementConfig(**{name: getattr(args, name) for name in RefinementConfig.__dataclass_fields__ if hasattr(args, name)})
        config.validate()
        audio, cache = args.audio.resolve(), args.model_output.resolve()
        if not audio.is_file() or not cache.is_file():
            raise ValueError("Audio and cached model_outputs.npz must both exist")
        output = reserve_output(args.output)
        result = run(audio, cache, output, config)
        print(f"Notes: {result['baseline_notes']} -> {result['decoded_notes']} -> {result['refined_notes']}")
        print(output)
        return 0
    except Exception as error:
        logging.error("%s", error, exc_info=args.verbose)
        return 1
