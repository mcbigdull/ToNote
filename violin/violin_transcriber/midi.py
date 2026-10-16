"""Preserve upstream bend samples, assigning independent channels while notes overlap.

pretty_midi cycles its pitch instruments through 15 melodic channels. Long pieces
can therefore have bend collisions even with multiple_pitch_bends=True. Raw MIDI
is left untouched; these derived MIDI files allocate channels by actual overlap.
"""
from collections import defaultdict
import numpy as np
import mido

CHANNELS = tuple(c for c in range(16) if c != 9)


def bend_samples(note):
    if not note.bends:
        return np.array([]), np.array([], dtype=int)
    # Exact Basic Pitch 0.4.0 conversion: +/-2 semitones, 3 bins/semitone.
    times = np.linspace(note.start, note.end, len(note.bends))
    values = np.clip(np.round(np.asarray(note.bends) * 4096 / 3), -8192, 8191).astype(int)
    return times, values


def allocate(notes, expressive):
    slots = []
    assignment = {}
    for note in sorted(notes, key=lambda n: (n.start, n.end, n.note_id)):
        selected = None
        for index, active in enumerate(slots):
            active[:] = [n for n in active if n.end > note.start]
            if not active or (not expressive and all(n.pitch != note.pitch for n in active)):
                selected = index
                break
        if selected is None:
            selected = len(slots)
            slots.append([])
        slots[selected].append(note)
        assignment[note.note_id] = {"port": selected // len(CHANNELS),
                                    "channel": CHANNELS[selected % len(CHANNELS)]}
    return assignment


def export(notes, path, config, expressive):
    assignment = allocate(notes, expressive)
    midi = mido.MidiFile(type=1, ticks_per_beat=config.midi_ticks_per_beat)
    tempo = mido.bpm2tempo(config.midi_tempo)
    conductor = mido.MidiTrack()
    midi.tracks.append(conductor)
    conductor.append(mido.MetaMessage("track_name", name="Violin / absolute audio time"))
    conductor.append(mido.MetaMessage("set_tempo", tempo=tempo))
    # No guessed time signature, beat offset or quantization.
    conductor.append(mido.MetaMessage("end_of_track"))
    groups = defaultdict(list)
    for note in notes:
        route = assignment[note.note_id]
        groups[route["port"], route["channel"]].append(note)
    for (port, channel), selected in sorted(groups.items()):
        if port > 127:
            raise ValueError("MIDI port limit exceeded")
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("midi_port", port=port))
        track.append(mido.MetaMessage("track_name", name=f"Violin port {port} channel {channel + 1}"))
        track.append(mido.Message("program_change", program=config.violin_program, channel=channel))
        if expressive:
            # Explicit RPN pitch bend sensitivity; independent of synthesizer defaults.
            for control, value in ((101, 0), (100, 0), (6, config.bend_range_semitones),
                                   (38, 0), (101, 127), (100, 127)):
                track.append(mido.Message("control_change", control=control, value=value, channel=channel))
        events = []
        for n in selected:
            events.extend([(n.end, 0, mido.Message("note_off", note=n.pitch, velocity=0, channel=channel)),
                           (n.start, 4, mido.Message("note_on", note=n.pitch, velocity=n.velocity, channel=channel))])
            if expressive:
                times, values = bend_samples(n)
                for index, (time, value) in enumerate(zip(times, values)):
                    # At a reused boundary: old off, old endpoint bend, reset,
                    # new initial bend, new note on. All source bend events remain.
                    events.append((float(time), 3 if index == 0 else 1,
                                   mido.Message("pitchwheel", pitch=int(value), channel=channel)))
                if not len(times):
                    events.append((n.start, 3, mido.Message("pitchwheel", pitch=0, channel=channel)))
                events.append((n.end, 2, mido.Message("pitchwheel", pitch=0, channel=channel)))
        timed = [(round(mido.second2tick(time, config.midi_ticks_per_beat, tempo)), priority, message)
                 for time, priority, message in events]
        previous = 0
        for tick, _, message in sorted(timed, key=lambda e: (e[0], e[1])):
            track.append(message.copy(time=tick - previous))
            previous = tick
        track.append(mido.MetaMessage("end_of_track"))
    midi.save(str(path))
    return assignment


def inspect_file(path):
    """Independently decode SMF channel events for round-trip checks."""
    midi = mido.MidiFile(str(path))
    tempo_events = []
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo_events.append((tick, msg.tempo))
    tempo_events.sort(key=lambda e: e[0])

    def seconds(target):
        seconds_so_far, prev, tempo = 0.0, 0, 500000
        for tick, value in tempo_events:
            if tick > target:
                break
            seconds_so_far += mido.tick2second(tick - prev, midi.ticks_per_beat, tempo)
            prev, tempo = tick, value
        return seconds_so_far + mido.tick2second(target - prev, midi.ticks_per_beat, tempo)

    notes, bends, programs = [], [], []
    # Raw Basic Pitch can have multiple tracks on the same channel. Parse per
    # track for baseline auditing; cross-track conflicts are reported separately.
    for ti, track in enumerate(midi.tracks):
        tick, port, active = 0, 0, defaultdict(list)
        for msg in track:
            tick += msg.time
            time = seconds(tick)
            if msg.type == "midi_port":
                port = msg.port
            elif msg.type == "program_change":
                programs.append(msg.program)
            elif msg.type == "pitchwheel":
                bends.append(dict(time=time, value=msg.pitch, port=port, channel=msg.channel, track=ti))
            elif msg.type == "note_on" and msg.velocity:
                active[(msg.channel, msg.note)].append((time, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and not msg.velocity):
                queue = active[(msg.channel, msg.note)]
                if queue:
                    start, velocity = queue.pop(0)
                    notes.append(dict(start=start, end=time, pitch=msg.note, velocity=velocity,
                                      port=port, channel=msg.channel, track=ti))
        if any(active.values()):
            raise ValueError(f"Unclosed note in {path}")
    return dict(notes=notes, bends=bends, programs=programs, duration_sec=midi.length,
                ticks_per_beat=midi.ticks_per_beat)


def validate_export(path, notes, config, expressive, assignment):
    actual = inspect_file(path)
    expected = sorted(notes, key=lambda n: (n.pitch, n.start, n.end, n.velocity))
    observed = sorted(actual["notes"], key=lambda n: (n["pitch"], n["start"], n["end"], n["velocity"]))
    if len(expected) != len(observed):
        raise ValueError(f"MIDI note count mismatch: {path}")
    errors = []
    tolerance = 1.1 * 60 / config.midi_tempo / config.midi_ticks_per_beat
    for n, event in zip(expected, observed):
        error = max(abs(n.start - event["start"]), abs(n.end - event["end"]))
        errors.append(error)
        if n.pitch != event["pitch"] or n.velocity != event["velocity"] or error > tolerance:
            raise ValueError(f"MIDI note round-trip failed: {n.note_id}")
    if not expressive and actual["bends"]:
        raise ValueError("Plain MIDI must not contain bends")
    preserved = 0
    if expressive:
        # Multiset match verifies every original bend value/time, including
        # endpoint events; extra resets are allowed. Quantize to file ticks.
        from collections import Counter
        scale = config.midi_ticks_per_beat * config.midi_tempo / 60
        pool = Counter((b["port"], b["channel"], round(b["time"] * scale), b["value"]) for b in actual["bends"])
        for n in notes:
            route = assignment[n.note_id]
            times, values = bend_samples(n)
            for time, value in zip(times, values):
                key = (route["port"], route["channel"], round(float(time) * scale), int(value))
                if pool[key] <= 0:
                    raise ValueError(f"Lost pitch bend: {n.note_id}")
                pool[key] -= 1
                preserved += 1
        from .notes import overlap_pairs
        for a, b, _ in overlap_pairs(notes):
            if assignment[a.note_id] == assignment[b.note_id]:
                raise ValueError("Overlapping expressive notes share a bend channel")
    return dict(notes=len(expected), pitchwheel_events=len(actual["bends"]),
                source_bends_verified=preserved, max_time_error_sec=max(errors, default=0),
                midi_duration_sec=actual["duration_sec"],
                ports=1 + max((r["port"] for r in assignment.values()), default=0))
