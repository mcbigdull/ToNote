import tempfile
import unittest
from pathlib import Path

import numpy as np

from vocal_transcriber.beat import BeatGrid
from vocal_transcriber.config import Config, NoteEvent, note_number
from vocal_transcriber.pitch import PitchTrack, smooth_pitch
from vocal_transcriber.segmentation import segment_notes
from vocal_transcriber.quantization import quantize_notes
from vocal_transcriber.midi_export import export_midi
from vocal_transcriber.musicxml_export import export_musicxml


def track(pitches, confidence=None):
    pitches = np.asarray(pitches, dtype=float)
    return PitchTrack(np.arange(len(pitches)) * .01, 440 * 2 ** ((pitches - 69) / 12),
                      np.ones(len(pitches)) if confidence is None else confidence, pitches, .01)


class CoreTests(unittest.TestCase):
    def segment(self, pitches, **overrides):
        config = Config(**overrides)
        return segment_notes(smooth_pitch(track(pitches), config), config)

    def test_vibrato_and_two_frame_octave_outlier(self):
        values = 69 + .3 * np.sin(2 * np.pi * 6 * np.arange(150) * .01)
        values[40:42] = 84
        notes = self.segment(values)
        self.assertEqual([n.pitch for n in notes], [69])
        self.assertAlmostEqual(notes[0].end, 1.5)

    def test_slide_has_only_stable_destinations(self):
        notes = self.segment(np.r_[np.full(40, 60), np.linspace(60, 67, 15), np.full(40, 67)])
        self.assertEqual([n.pitch for n in notes], [60, 67])

    def test_semitone_change_survives(self):
        notes = self.segment(np.r_[np.full(40, 69.15), np.full(40, 70.1)])
        self.assertEqual([n.pitch for n in notes], [69, 70])

    def test_breath_preserves_repeated_notes(self):
        notes = self.segment(np.r_[np.full(40, 69), np.full(10, np.nan), np.full(40, 69)])
        self.assertEqual(len(notes), 2)
        self.assertGreater(notes[1].start - notes[0].end, .08)

    def test_low_confidence_and_short_noise_rejected(self):
        config = Config()
        silent = track(np.full(30, 69), np.zeros(30))
        self.assertEqual(segment_notes(smooth_pitch(silent, config), config), [])
        self.assertEqual(self.segment([69, 69, 69]), [])

    def test_variable_tempo_mapping_round_trip(self):
        grid = BeatGrid(120, np.array([-.2, .3, .85, 1.3]), np.arange(4.), "test")
        seconds = np.array([0., .25, 1., 3.])
        np.testing.assert_allclose(grid.beats_to_seconds(grid.seconds_to_beats(seconds)), seconds)

    def test_midi_preserves_absolute_timing_and_program(self):
        import mido
        grid = BeatGrid(120, np.array([-.2, .3, .85, 1.3]), np.arange(4.), "test")
        config = Config(quantization_strength=0)
        notes = quantize_notes([NoteEvent(69, .4, .9, .9), NoteEvent(72, 1.1, 1.6, .9)], grid, config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "人声 with spaces.mid"
            export_midi(notes, grid, config, path)
            midi = mido.MidiFile(path)
            self.assertEqual(midi.type, 0)
            elapsed, starts, ends = 0., [], []
            for message in midi:
                elapsed += message.time
                if message.type == "note_on":
                    starts.append(elapsed)
                elif message.type == "note_off":
                    ends.append(elapsed)
                elif message.type == "program_change":
                    self.assertEqual(message.program, 53)
            np.testing.assert_allclose(starts, [.4, 1.1], atol=.001)
            np.testing.assert_allclose(ends, [.9, 1.6], atol=.001)

    def test_quantization_collision_remains_monophonic(self):
        grid = BeatGrid(120, np.array([0., .5]), np.array([0., 1.]), "test")
        notes = quantize_notes([NoteEvent(60, .01, .06, .7), NoteEvent(62, .07, .14, .9),
                                NoteEvent(64, .15, .3, .8)], grid, Config())
        for previous, current in zip(notes, notes[1:]):
            self.assertLessEqual(previous.end_sec, current.start_sec)
            self.assertLessEqual(previous.start_beat + previous.duration_beats, current.start_beat)

    def test_musicxml_rests_ties_meter_key_and_total_duration(self):
        from music21 import converter
        grid = BeatGrid(120, np.array([0., .5]), np.array([0., 1.]), "test")
        config = Config(key="F#m", time_signature="3/4")
        notes = quantize_notes([NoteEvent(66, .5, 4, .9)], grid, config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "melody.musicxml"
            export_musicxml(notes, grid, config, path)
            score = converter.parse(path)
            part = score.parts[0]
            self.assertGreaterEqual(len(part.getElementsByClass("Measure")), 3)
            self.assertTrue(any(n.tie for n in part.recurse().notes))
            self.assertEqual({n.pitch.midi for n in part.recurse().notes}, {66})
            self.assertAlmostEqual(sum(n.quarterLength for n in part.recurse().notes), 7)

    def test_invalid_config(self):
        for config in (Config(min_confidence=2), Config(quantization_strength=float("nan")),
                       Config(min_note_duration=-1), Config(time_signature="4/3")):
            with self.assertRaises(ValueError):
                config.validate()
        self.assertEqual(note_number("C2"), 36)
        self.assertEqual(note_number("Bb4"), 70)


if __name__ == "__main__":
    unittest.main()
