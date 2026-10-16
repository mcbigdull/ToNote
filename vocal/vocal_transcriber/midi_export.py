def export_midi(notes, grid, config, path):
    import mido
    from .config import TranscriptionError
    if not notes:
        raise TranscriptionError("没有有效音符，无法导出 MIDI。")
    ticks_per_beat = 960
    midi = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    numerator, denominator = map(int, config.time_signature.split("/"))
    events = [(0, 0, mido.MetaMessage("track_name", name="Vocal Melody")),
              (0, 0, mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator)),
              (0, 0, mido.Message("program_change", program=53, channel=0))]
    # Shift the extrapolated grid origin so MIDI time zero is audio time zero.
    origin = float(grid.seconds_to_beats(0))
    for beat, bpm in grid.tempo_events():
        tick = max(0, round((beat - origin) * ticks_per_beat))
        events.append((tick, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
    for note in notes:
        start, end = grid.seconds_to_beats([note.start_sec, note.end_sec]) - origin
        start_tick = max(0, round(float(start) * ticks_per_beat))
        end_tick = max(start_tick + 1, round(float(end) * ticks_per_beat))
        events.append((start_tick, 2, mido.Message("note_on", note=note.pitch, velocity=85, channel=0)))
        events.append((end_tick, 1, mido.Message("note_off", note=note.pitch, velocity=0, channel=0)))
    previous = 0
    for tick, priority, message in sorted(events, key=lambda event: (event[0], event[1])):
        track.append(message.copy(time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track"))
    midi.save(str(path))
