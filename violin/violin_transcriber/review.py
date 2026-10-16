"""Review candidates only. No note, bend, or model activation is modified."""
import csv
import math
import numpy as np
from .notes import overlap_pairs, polyphony_segments


FIELDS = ["kind", "start_sec", "end_sec", "note_ids", "pitches", "overlap_ratio",
          "harmonic_number", "harmonic_error_cents", "score_ratio_high_low",
          "activation_mean_low", "activation_mean_high", "reason"]


def activation_mean(note, output, times):
    index = note.pitch - 21
    if not 0 <= index < output["note"].shape[1]:
        return None
    selected = (times >= note.start) & (times < note.end)
    return float(np.mean(output["note"][selected, index])) if selected.any() else None


def detect(notes, output, times, config):
    rows = []

    def add(kind, selected, reason, start=None, end=None, **evidence):
        rows.append(dict(kind=kind, start_sec=min(n.start for n in selected) if start is None else start,
                         end_sec=max(n.end for n in selected) if end is None else end,
                         note_ids=";".join(n.note_id for n in selected),
                         pitches=";".join(str(n.pitch) for n in selected), reason=reason, **evidence))

    low, high = config.pitch_bounds
    for n in notes:
        if not low <= n.pitch <= high:
            add("out_of_range", [n], "Outside configured inclusive MIDI centers; retained in raw baseline")
        if n.end - n.start < config.short_note_ms / 1000:
            add("short_note", [n], "Short event; could be a valid fast note, not automatically removed")
        if n.velocity == 0:
            add("zero_velocity", [n], "Upstream amplitude rounds to zero MIDI velocity; inspect baseline")
    for a, b, overlap in overlap_pairs(notes):
        if a.pitch == b.pitch:
            add("same_pitch_overlap", [a, b], "Same-pitch overlap; raw MIDI may share bends/note-offs")
            continue
        a, b = sorted((a, b), key=lambda n: n.pitch)
        ratio = 2 ** ((b.pitch - a.pitch) / 12)
        harmonic = round(ratio)
        if not 2 <= harmonic <= config.max_harmonic:
            continue
        error = 1200 * math.log2(ratio / harmonic)
        coverage = overlap / min(a.end - a.start, b.end - b.start)
        if abs(error) > config.harmonic_tolerance_cents or coverage < config.harmonic_overlap_ratio:
            continue
        if abs(a.start - b.start) > config.coincident_onset_sec or abs(a.end - b.end) > config.coincident_end_sec:
            continue
        add("octave_overlap" if harmonic in (2, 4, 8) else "harmonic_overlap", [a, b],
            "Coincident integer-frequency pair: possible harmonic OR real double stop; no correction",
            overlap_ratio=coverage, harmonic_number=harmonic, harmonic_error_cents=error,
            score_ratio_high_low=b.score / max(a.score, 1e-12),
            activation_mean_low=activation_mean(a, output, times),
            activation_mean_high=activation_mean(b, output, times))
    by_id = {n.note_id: n for n in notes}
    for start, end, ids in polyphony_segments(notes):
        if len(ids) >= 3:
            add("polyphony_ge_3", [by_id[i] for i in ids],
                "Three or more simultaneous events; review chord, bow transition, reverb or false notes",
                start=start, end=end)
    return rows


def write_review(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
