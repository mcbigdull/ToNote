import hashlib
import inspect
import json
import logging
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from importlib.metadata import version
from pathlib import Path
import sys
import secrets
import time

from .config import Config, parser


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def reserve_output(base):
    base = base.resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        base.mkdir()
    except FileExistsError:
        if not base.is_dir():
            raise ValueError(f"Output is not a directory: {base}")
        if any(base.iterdir()):
            # A result directory is persistent user data, not a private OS temp
            # directory. Normal mkdir also inherits ordinary Windows ACLs.
            while True:
                candidate = base.with_name(base.name + "_" + secrets.token_hex(4))
                try:
                    candidate.mkdir()
                    base = candidate
                    break
                except FileExistsError:
                    continue
    return base


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def bends_document(raw, filtered, times, assignment):
    import numpy as np
    from .midi import bend_samples
    kept = {n.note_id for n in filtered}
    records = []
    for n in raw:
        export_times, values = bend_samples(n)
        index = int(np.searchsorted(times, n.start))
        records.append(dict(note_id=n.note_id, start_sec=n.start, end_sec=n.end,
                            pitch_midi=n.pitch, amplitude_or_score=n.score, velocity=n.velocity,
                            kept_in_filtered=n.note_id in kept,
                            raw_bend_bins=list(n.bends) if n.bends is not None else None,
                            raw_bend_semitones=[b / 3 for b in n.bends] if n.bends is not None else None,
                            source_frame_indices=list(range(index, index + len(n.bends or ()))),
                            source_frame_times_sec=times[index:index + len(n.bends or ())].tolist(),
                            basic_pitch_export_times_sec=export_times.tolist(),
                            exported_midi_pitchwheel=values.tolist(),
                            expressive_route=assignment.get(n.note_id)))
    return dict(schema_version=1, time_origin="Original decoded audio sample zero; no trimming or quantization",
                raw_bend_unit="1/3 semitone (Basic Pitch contour argmax, not continuous F0)",
                midi_pitch_bend_range_semitones=2,
                export_time_mapping="Upstream uses linspace(start, end, number_of_bends), including endpoint",
                source_frame_time_mapping="Basic Pitch model_frames_to_time; source frames are end-exclusive",
                source_bend_clipping="Raw bins retained; upstream MIDI conversion saturates at [-8192,8191]",
                notes=records)


def run(source, output, config, metadata):
    import numpy as np
    import basic_pitch
    from basic_pitch.inference import Model, predict
    from basic_pitch.note_creation import model_frames_to_time
    from .audio import readable_audio
    from .midi import export, inspect_file, validate_export
    from .notes import filter_range, from_events, overlap_pairs, polyphony_segments, write_notes
    from .plots import save_plots
    from .review import detect, write_review

    signature = inspect.signature(predict)
    required = {"multiple_pitch_bends", "minimum_note_length", "minimum_frequency", "maximum_frequency"}
    if not required.issubset(signature.parameters):
        raise RuntimeError(f"Unsupported Basic Pitch predict signature: {signature}")
    model_path = basic_pitch.ICASSP_2022_MODEL_PATH
    logging.info("Basic Pitch %s: loading %s", version("basic-pitch"), model_path)
    model = Model(model_path)
    metadata.update(basic_pitch_version=version("basic-pitch"), predict_signature=str(signature),
                    model_path=str(model_path), model_runtime=model.model_type.name,
                    input_sha256=sha256(source),
                    package_versions={name: version(name) for name in
                                      ("tensorflow", "numpy", "librosa", "pretty-midi", "mido", "matplotlib")})
    with readable_audio(source, output) as (read_path, audio_info):
        metadata["audio"] = audio_info
        # Raw baseline: run ONCE, no violin range constraints and no second
        # decode with a different model. All subsequent filtering is on events.
        arrays, raw_midi, events = predict(
            read_path, model_or_model_path=model,
            onset_threshold=config.onset_threshold, frame_threshold=config.frame_threshold,
            minimum_note_length=config.minimum_note_length,
            minimum_frequency=None, maximum_frequency=None,
            multiple_pitch_bends=True, melodia_trick=config.melodia_trick, midi_tempo=config.midi_tempo)
    raw_midi.write(str(output / "violin_raw.mid"))  # untouched upstream programs and bends
    times = model_frames_to_time(arrays["note"].shape[0])
    np.savez_compressed(output / "model_outputs.npz", **arrays, frame_times_sec=times,
                        note_midi=21 + np.arange(arrays["note"].shape[1]),
                        contour_midi=21 + np.arange(arrays["contour"].shape[1]) / 3,
                        audio_duration_sec=np.array(metadata["audio"]["duration_sec"]))
    raw = from_events(events)
    filtered = filter_range(raw, config)
    write_notes(output / "notes_raw.csv", raw)
    write_notes(output / "notes_filtered.csv", filtered)
    review = detect(raw, arrays, times, config)
    write_review(output / "suspected_errors.csv", review)
    assignment = export(filtered, output / "violin_expressive.mid", config, expressive=True)
    plain_assignment = export(filtered, output / "violin_plain.mid", config, expressive=False)
    write_json(output / "pitch_bends.json", bends_document(raw, filtered, times, assignment))
    save_plots(output, raw, filtered, arrays, times, config)
    validation = dict(
        plain=validate_export(output / "violin_plain.mid", filtered, config, False, plain_assignment),
        expressive=validate_export(output / "violin_expressive.mid", filtered, config, True, assignment))
    raw_inspection = inspect_file(output / "violin_raw.mid")
    # Detect overlapping tracks that share a physical MIDI port/channel in raw.
    raw_routes = []
    for ni, instrument in enumerate(raw_midi.instruments):
        channel = tuple(c for c in range(16) if c != 9)[ni % 15]
        for note in instrument.notes:
            raw_routes.append((note.start, note.end, channel, ni, note.pitch))
    active, collision_pairs = [], 0
    for event in sorted(raw_routes):
        start, end, channel, instrument, pitch = event
        active = [n for n in active if n[1] > start]
        collision_pairs += sum(n[2] == channel and n[3] != instrument for n in active)
        active.append(event)
    kinds = Counter(r["kind"] for r in review)
    overlaps = list(overlap_pairs(filtered))
    segments = list(polyphony_segments(filtered))
    overlap_ids = {n.note_id for a, b, _ in overlaps for n in (a, b)}
    metadata.update(
        raw_note_count=len(raw), filtered_note_count=len(filtered),
        overlapping_note_count=len(overlap_ids), overlapping_pair_count=len(overlaps),
        max_simultaneous_notes=max((len(ids) for _, _, ids in segments), default=0),
        double_note_duration_sec=sum(end - start for start, end, ids in segments if len(ids) == 2),
        three_plus_note_duration_sec=sum(end - start for start, end, ids in segments if len(ids) >= 3),
        suspected_octave_pair_count=kinds["octave_overlap"],
        suspected_harmonic_pair_count=kinds["harmonic_overlap"] + kinds["octave_overlap"],
        short_note_count=kinds["short_note"], out_of_range_count=kinds["out_of_range"],
        same_pitch_overlap_pair_count=kinds["same_pitch_overlap"], review_counts=dict(kinds),
        raw_midi_duration_sec=raw_inspection["duration_sec"],
        raw_instrument_count=len(raw_midi.instruments),
        raw_cross_track_channel_collision_pairs=collision_pairs,
        original_bend_sample_count=sum(len(n.bends or ()) for n in raw),
        filtered_bend_sample_count=sum(len(n.bends or ()) for n in filtered),
        bends_saturated_sample_count=sum(abs(b) >= 6 for n in raw for b in (n.bends or ())),
        last_filtered_note_end_sec=max((n.end for n in filtered), default=0),
        notes_extending_past_audio=[n.note_id for n in raw if n.end > metadata["audio"]["duration_sec"]],
        effective_midi_pitch_range=list(config.pitch_bounds),
        validation=validation,
        accuracy_metrics=None,
        status="success" if filtered else "no_notes")
    if not filtered:
        raise ValueError("No notes in requested range. Raw baseline and diagnostics retained; no successful transcription.")
    return metadata


def write_report(output, metadata):
    lines = ["Basic Pitch violin transcription / validation report",
             f"Status: {metadata.get('status')}", f"Input: {metadata.get('input')}",
             f"Started (Asia/Shanghai): {metadata.get('started')}",
             f"Output: {output}", "",
             "Parameters (minimum_note_length is in milliseconds):",
             json.dumps(metadata.get("config", {}), ensure_ascii=False, indent=2), ""]
    for key in ("basic_pitch_version", "model_runtime", "raw_note_count", "filtered_note_count",
                "overlapping_note_count", "overlapping_pair_count", "max_simultaneous_notes",
                "double_note_duration_sec", "three_plus_note_duration_sec", "suspected_octave_pair_count",
                "suspected_harmonic_pair_count", "short_note_count", "out_of_range_count",
                "same_pitch_overlap_pair_count", "raw_instrument_count", "raw_cross_track_channel_collision_pairs",
                "original_bend_sample_count", "filtered_bend_sample_count", "bends_saturated_sample_count",
                "raw_midi_duration_sec", "last_filtered_note_end_sec", "notes_extending_past_audio", "elapsed_sec", "error"):
        if key in metadata:
            lines.append(f"{key}: {metadata[key]}")
    lines.extend(["", "Audio: " + json.dumps(metadata.get("audio", {})),
                  "MIDI round-trip verification: " + json.dumps(metadata.get("validation", {})), "",
                  "Interpretation / limits:",
                  "- Raw = upstream predict(..., multiple_pitch_bends=True), no violin range constraint.",
                  "- Filtered = inclusive nearest MIDI centers for configured Hz range; no timing, pitch or polyphony edits.",
                  "- Raw MIDI keeps upstream Electric Piano program, instrument layout and bend events.",
                  "- Plain/expressive use Violin program 40 (zero-based); instruments/channels are one violin, not separated instruments.",
                  "- Plain has no bends. Expressive isolates overlapping notes by channel and preserves every upstream bend sample.",
                  "- Expressive adds +/-2 semitone RPN and zero resets, avoiding raw per-pitch channel cycling conflicts.",
                  "- More than 15 concurrent expressive notes require MIDI ports; some players ignore ports. See validation.ports.",
                  "- Reviews are candidates only; real octave double stops and high harmonics are possible. No auto-deletion.",
                  "- Harmonic checks cannot establish a lone note's true fundamental or determine upward vs downward octave error.",
                  "- Activations/amplitudes are model scores, not calibrated probabilities.",
                  "- Contour bends use 3 bins per semitone (~33.3 cents); frame hop is 256/22050 s (~11.61 ms).",
                  "- Upstream exported bend times use linspace over each note and MIDI saturates at +/-2 semitones.",
                  "- pitch_bends.json retains raw bins, source frame times and exported times/values, including out-of-range notes.",
                  "- No fine-resolution F0, technique labels, onset refinement, rhythm quantization or VioPTT.",
                  "- MIDI tempo 120 is only a seconds-to-ticks encoding; no detected musical tempo is claimed.",
                  "- MIDI ends at last event; trailing silence need not match audio duration. No origin shift or trimming.",
                  "- No reference score yet: precision/recall/F1 and an accuracy percentage are NOT computed.",
                  "- MIDI parsed and channel/timing/bend preservation checked; GUI import in MuseScore is a separate check.",
                  "- Next: compare sustained octave pairs / short repeated notes against annotated audio or reference MIDI."])
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    for library in ("numba", "matplotlib", "tensorflow"):
        logging.getLogger(library).setLevel(logging.WARNING)
    output = None
    started = time.monotonic()
    metadata = {}
    try:
        config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__ if hasattr(args, name)})
        config.validate()
        source = args.audio.resolve()
        if not source.is_file():
            raise ValueError(f"Audio file not found: {source}")
        output = reserve_output(args.output)
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
        os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        metadata = dict(status="running", input=str(source), output=str(output), config=asdict(config),
                        python=sys.version, started=datetime.now(timezone(timedelta(hours=8))).isoformat())
        write_json(output / "run.json", metadata)
        run(source, output, config, metadata)
        logging.info("Complete: %s", output)
        logging.info("Notes raw/filtered: %d/%d; overlap pairs: %d; octave candidates: %d",
                     metadata["raw_note_count"], metadata["filtered_note_count"],
                     metadata["overlapping_pair_count"], metadata["suspected_octave_pair_count"])
        return 0
    except KeyboardInterrupt:
        metadata.update(status="interrupted", error="User interrupted")
        return 130
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        logging.error("%s", exc, exc_info=args.verbose)
        return 1
    finally:
        if output:
            metadata["elapsed_sec"] = round(time.monotonic() - started, 3)
            try:
                write_json(output / "run.json", metadata)
                write_report(output, metadata)
            except OSError as exc:
                logging.error("Cannot save report in %s: %s", output, exc)
                return 1
