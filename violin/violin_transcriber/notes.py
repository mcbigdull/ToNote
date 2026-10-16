from dataclasses import dataclass
import csv
import math
import numpy as np


@dataclass(frozen=True)
class Note:
    note_id: str
    start: float
    end: float
    pitch: int
    score: float
    bends: tuple[int, ...] | None

    @property
    def velocity(self):
        return int(np.round(127 * self.score))  # matches Basic Pitch

    @property
    def name(self):
        return ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[self.pitch % 12] + str(self.pitch // 12 - 1)


def from_events(events):
    notes = []
    for index, (start, end, pitch, score, bends) in enumerate(events):
        if not all(math.isfinite(float(v)) for v in (start, end, pitch, score)):
            raise ValueError("Basic Pitch returned a non-finite note")
        if start < 0 or end <= start or not 0 <= pitch <= 127 or not 0 <= score <= 1:
            raise ValueError("Basic Pitch returned an invalid note; baseline retained")
        notes.append(Note(f"n{index + 1:06d}", float(start), float(end), int(pitch), float(score),
                          tuple(int(b) for b in bends) if bends is not None else None))
    return notes


def filter_range(notes, config):
    low, high = config.pitch_bounds
    return [n for n in notes if low <= n.pitch <= high]


def write_notes(path, notes):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["note_id", "start_sec", "end_sec", "pitch_midi", "note_name", "velocity", "amplitude_or_score"])
        writer.writerows((n.note_id, repr(n.start), repr(n.end), n.pitch, n.name, n.velocity, repr(n.score)) for n in notes)


def overlap_pairs(notes):
    active = []
    for note in sorted(notes, key=lambda n: (n.start, n.end, n.note_id)):
        active = [other for other in active if other.end > note.start]
        for other in active:
            yield other, note, min(other.end, note.end) - note.start
        active.append(note)


def polyphony_segments(notes):
    events = {}
    for n in notes:
        events.setdefault(n.start, [set(), set()])[1].add(n.note_id)
        events.setdefault(n.end, [set(), set()])[0].add(n.note_id)
    active, previous = set(), None
    for time, (ends, starts) in sorted(events.items()):
        if previous is not None and time > previous and active:
            yield previous, time, sorted(active)
        active.difference_update(ends)
        active.update(starts)
        previous = time
