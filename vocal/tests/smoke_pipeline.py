"""Real pYIN/FFmpeg/exports, no model downloads. Run from repository root."""
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
from music21 import converter
import mido

from vocal_transcriber.main import main


def run():
    with tempfile.TemporaryDirectory(prefix="人声 smoke ") as temporary:
        root = Path(temporary)
        sr = 22050
        pieces = [np.zeros(sr // 4)]
        for pitch in [60, 64, 67, 69]:
            t = np.arange(sr // 2) / sr
            frequency = 440 * 2 ** ((pitch - 69 + .2 * np.sin(2 * np.pi * 5 * t)) / 12)
            phase = 2 * np.pi * np.cumsum(frequency) / sr
            envelope = np.minimum(1, np.minimum(t / .02, (.5 - t) / .02))
            pieces.extend([.3 * np.sin(phase) * envelope, np.zeros(sr // 8)])
        audio = np.concatenate(pieces).astype("float32")
        source = root / "中文 纯人声.wav"
        sf.write(source, np.column_stack([audio, audio]), sr)
        arguments = [str(source), "--skip-separation", "--bpm", "120", "--output", str(root / "results")]
        assert main(arguments) == 0
        output = root / "results" / source.stem
        pitches = [m.note for m in mido.MidiFile(output / "melody.mid").tracks[0] if m.type == "note_on"]
        assert pitches == [60, 64, 67, 69], pitches
        assert converter.parse(output / "melody.musicxml").parts
        assert json.loads((output / "run.json").read_text(encoding="utf-8"))["status"] == "success"
        assert (output / "debug_pitch.png").stat().st_size > 1000
        assert (output / "pitch_raw.csv").exists()
        # Exercise the optional neural F0 path and successful WAV cleanup too.
        assert main(arguments + ["--pitch-backend", "crepe", "--max-note", "B6", "--no-keep-intermediate"]) == 0
        crepe_outputs = [p for p in (root / "results").iterdir() if p != output]
        assert len(crepe_outputs) == 1
        neural = crepe_outputs[0]
        neural_pitches = [m.note for m in mido.MidiFile(neural / "melody.mid").tracks[0] if m.type == "note_on"]
        assert neural_pitches == [60, 64, 67, 69], neural_pitches
        assert not (neural / "vocals.wav").exists()
        assert (neural / "pitch_raw.csv").exists()
        # Empty vocals and invalid files must fail explicitly, not emit empty MIDI.
        silent = root / "空白.wav"
        sf.write(silent, np.zeros(sr), sr)
        assert main([str(silent), "--skip-separation", "--bpm", "120", "--output", str(root / "results")]) == 1
        assert not (root / "results" / silent.stem / "melody.mid").exists()
        print("PASS: real FFmpeg + stereo/22050Hz + pYIN/CREPE + MIDI/XML/PNG + Chinese paths + silence failure")


if __name__ == "__main__":
    run()
