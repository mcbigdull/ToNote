import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from violin_transcriber.config import Config, parser
from violin_transcriber.main import bends_document, reserve_output
from violin_transcriber.midi import allocate, bend_samples, export, inspect_file, validate_export
from violin_transcriber.notes import Note, filter_range, overlap_pairs
from violin_transcriber.review import detect


class TranscriptionTests(unittest.TestCase):
    def test_default_bends_and_inclusive_range(self):
        config = Config()
        config.validate()
        self.assertTrue(parser().parse_args(["file.wav"]).multiple_pitch_bends)
        self.assertEqual(config.pitch_bounds, (55, 100))
        notes = [Note(str(p), 1, 2, p, .8, (0,)) for p in (54, 55, 69, 100, 101)]
        self.assertEqual([n.pitch for n in filter_range(notes, config)], [55, 69, 100])

    def test_invalid_parameters(self):
        for config in (Config(minimum_note_length=0), Config(frame_threshold=float("nan")),
                       Config(minimum_frequency=3000), Config(multiple_pitch_bends=False),
                       Config(onset_threshold=0)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_legitimate_fifth_not_flagged_as_harmonic(self):
        notes = [Note("low", 1, 1.8, 55, .8, (0, 1)), Note("high", 1.05, 1.75, 62, .7, (0, -1))]
        rows = detect(notes, {"note": np.ones((200, 88))}, np.arange(200) / 100, Config())
        self.assertEqual(rows, [])
        self.assertEqual(len(list(overlap_pairs(notes))), 1)

    def test_harmonics_flagged_but_never_deleted(self):
        notes = [Note("a", 1, 2, 69, .8, (0,)), Note("b", 1.01, 2.01, 81, .4, (0,)),
                 Note("c", 1.02, 2.02, 88, .3, (0,))]
        before = list(notes)
        rows = detect(notes, {"note": np.ones((300, 88))}, np.arange(300) / 100, Config())
        self.assertIn("octave_overlap", {r["kind"] for r in rows})
        self.assertIn("harmonic_overlap", {r["kind"] for r in rows})
        self.assertIn("polyphony_ge_3", {r["kind"] for r in rows})
        self.assertEqual(notes, before)

    def test_double_stop_and_same_pitch_bends_round_trip(self):
        notes = [Note("g", 1, 1.8, 55, .8, (0, 1, 0, -1)),
                 Note("d", 1.05, 1.75, 62, .7, (0, -1, 0, 1)),
                 Note("g2", 1.5, 2, 55, .6, (1, 0, -1)),
                 Note("next", 2, 3, 69, .7, (-1, 0, 1, 0))]
        with tempfile.TemporaryDirectory(prefix="中文 MIDI ") as folder:
            for expressive in (True, False):
                path = Path(folder) / f"violin {expressive}.mid"
                assignment = export(notes, path, Config(), expressive)
                result = validate_export(path, notes, Config(), expressive, assignment)
                self.assertEqual(result["notes"], 4)
                self.assertEqual(result["source_bends_verified"], 15 if expressive else 0)
                actual = inspect_file(path)
                self.assertTrue(all(program == 40 for program in actual["programs"]))
                self.assertEqual(min(n["start"] for n in actual["notes"]), 1)
                self.assertNotEqual(assignment["g"], assignment["g2"])

    def test_more_than_fifteen_pitches_and_midi_ports(self):
        # A long piece with >15 pitches needs only as many channels as concurrent
        # notes. A true 16-note chord needs a second port, never dropped notes.
        serial = [Note(str(i), i, i + .9, 55 + i, .8, (0, 1)) for i in range(20)]
        self.assertEqual(len({tuple(r.values()) for r in allocate(serial, True).values()}), 1)
        simultaneous = [Note(str(i), 1, 2, 55 + i, .8, (0, 1)) for i in range(16)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ports.mid"
            assignment = export(simultaneous, path, Config(), True)
            result = validate_export(path, simultaneous, Config(), True, assignment)
            self.assertEqual(result["ports"], 2)
            self.assertEqual(result["notes"], 16)

    def test_upstream_bend_values_saturation_and_frame_times(self):
        n = Note("note", 1, 1.04, 69, .8, (-9, -1, 0, 1, 9))
        times, values = bend_samples(n)
        self.assertEqual(values.tolist(), [-8192, -1365, 0, 1365, 8191])
        document = bends_document([n], [n], np.arange(201) / 100, {"note": {"port": 0, "channel": 0}})
        restored = json.loads(json.dumps(document))["notes"][0]
        self.assertEqual(restored["raw_bend_bins"], [-9, -1, 0, 1, 9])
        self.assertEqual(restored["source_frame_indices"], [100, 101, 102, 103, 104])
        self.assertEqual(restored["basic_pitch_export_times_sec"], times.tolist())

    def test_output_does_not_overwrite_prior_results(self):
        with tempfile.TemporaryDirectory() as folder:
            first = reserve_output(Path(folder) / "output")
            (first / "report.txt").write_text("old", encoding="utf-8")
            second = reserve_output(first)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "report.txt").read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main()
