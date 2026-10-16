import numpy as np

from .config import NoteEvent
from .pitch import valid_runs
from .slides import stable_target


def segment_notes(track, config):
    result = []
    stable = max(2, int(np.ceil(config.stable_ms / 1000 / track.hop)))
    tolerance = config.pitch_tolerance_cents / 100
    for begin, end in valid_runs(np.isfinite(track.midi_pitch)):
        values = track.midi_pitch
        anchor = None
        note_start = begin
        center = None
        i = begin
        while i + stable <= end:
            target = stable_target(values[i:i + stable], tolerance)
            if target is not None:
                if anchor is None:
                    anchor = target
                    center = target
                elif abs(target - anchor) > tolerance and round(target) != round(center):
                    if (i - note_start) * track.hop >= config.min_note_duration:
                        result.append(NoteEvent(int(round(center)), float(track.time[note_start]),
                                                float(track.time[i]), float(np.mean(track.confidence[note_start:i]))))
                        note_start, anchor, center = i, target, target
                elif abs(target - anchor) <= tolerance:
                    center = float(np.median(values[note_start:i + stable]))
            i += 1
        if center is not None and (end - note_start) * track.hop >= config.min_note_duration:
            result.append(NoteEvent(int(round(center)), float(track.time[note_start]),
                                    float(track.time[end - 1] + track.hop), float(np.mean(track.confidence[note_start:end]))))
    merged = []
    for note in result:
        if merged and note.pitch == merged[-1].pitch and note.start - merged[-1].end <= config.max_gap_ms / 1000 + 1e-9:
            previous = merged[-1]
            d1, d2 = previous.end - previous.start, note.end - note.start
            previous.confidence = (previous.confidence * d1 + note.confidence * d2) / (d1 + d2)
            previous.end = note.end
        else:
            merged.append(note)
    return merged
