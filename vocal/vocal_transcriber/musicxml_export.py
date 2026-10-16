import logging


def parse_key(value):
    import re
    from music21 import key
    if not re.fullmatch(r"[A-Ga-g][#b]?m?", value):
        raise ValueError("无效调号，示例：G、F#m、Bb")
    minor = value.endswith("m")
    tonic = value[:-1] if minor else value
    return key.Key(tonic[0].upper() + tonic[1:].replace("b", "-"), "minor" if minor else "major")


def export_musicxml(notes, grid, config, path):
    from music21 import clef, instrument, meter, metadata, note, stream, tempo
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Vocal Melody")
    part = stream.Part(id="vocal")
    part.insert(0, instrument.Vocalist())
    part.insert(0, clef.TrebleClef())
    part.insert(0, meter.TimeSignature(config.time_signature))
    part.insert(0, tempo.MetronomeMark(number=grid.bpm))
    cursor = 0.0
    for event in notes:
        if event.start_beat > cursor:
            part.insert(cursor, note.Rest(quarterLength=event.start_beat - cursor))
        # Numeric MIDI constructors attach explicit naturals even to white keys.
        # Start with a spelling; let notation infer necessary accidentals in context.
        spelling = note.Note(event.pitch).nameWithOctave
        item = note.Note(spelling, quarterLength=event.duration_beats)
        part.insert(event.start_beat, item)
        cursor = event.start_beat + event.duration_beats
    if config.key:
        selected_key = parse_key(config.key)
    else:
        selected_key = part.analyze("key")
        logging.info("调号估计：%s（仅用于记谱，未更改音高）。", selected_key)
    part.insert(0, selected_key)
    # Prefer spellings consistent with the selected key, preserving sounding pitch.
    if selected_key.sharps < 0:
        for item in part.notes:
            if item.pitch.accidental and item.pitch.accidental.alter == 1:
                item.pitch = item.pitch.getEnharmonic()
    part.makeMeasures(inPlace=True)
    # makeNotation computes accidentals BEFORE splitting ties. Pre-splitting ties
    # makes music21 believe accidental display states have already been computed.
    part.makeNotation(inPlace=True)
    score.insert(0, part)
    score.write("musicxml", fp=str(path))
    return str(selected_key)
