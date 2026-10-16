import logging
import numpy as np
from .config import QuantizedNote, TranscriptionError


def quantize_notes(notes, grid, config):
    result = []
    dropped = 0
    strength = config.quantization_strength
    for note in notes:
        raw_start, raw_end = grid.seconds_to_beats([note.start, note.end])
        start = max(0.0, float(np.floor(raw_start * 4 + 0.5) / 4))
        end = max(start + 0.25, float(np.floor(raw_end * 4 + 0.5) / 4))
        # Quantized collisions must not create polyphony or zero-length notes.
        if result and start < result[-1].start_beat + result[-1].duration_beats:
            previous = result[-1]
            if start <= previous.start_beat:
                dropped += 1
                if note.confidence <= previous.confidence:
                    continue
                result.pop()
            else:
                previous.duration_beats = start - previous.start_beat
        target_start, target_end = grid.beats_to_seconds([start, end])
        performed_start = (1 - strength) * note.start + strength * float(target_start)
        performed_end = (1 - strength) * note.end + strength * float(target_end)
        result.append(QuantizedNote(note.pitch, max(0., performed_start), performed_end,
                                    start, end - start, note.confidence))
    for i, note in enumerate(result):
        if i + 1 < len(result):
            note.end_sec = min(note.end_sec, result[i + 1].start_sec)
        if note.end_sec <= note.start_sec:
            raise TranscriptionError("量化产生无效时长，请降低量化强度或检查节拍网格。")
    if not result:
        raise TranscriptionError("量化后无有效音符，未生成空 MIDI。")
    if dropped:
        logging.warning("有 %d 个音符落在同一十六分音符位置，已保留置信度较高者。", dropped)
    return result
