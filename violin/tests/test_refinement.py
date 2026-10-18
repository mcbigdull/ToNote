from pathlib import Path
import tempfile
import unittest

import numpy as np

from violin_transcriber.config import RefinementConfig, refinement_parser
from violin_transcriber.midi import bend_samples, export, validate_export
from violin_transcriber.notes import Note
from violin_transcriber.refinement import merge_fragments


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.config = RefinementConfig()
        self.times = np.arange(501) / 100
        self.arrays = {"note": np.ones((501, 88)), "onset": np.full((501, 88), .72)}

    def merge(self, notes, attacks=()):
        return merge_fragments(notes, self.arrays, self.times, np.array(attacks), self.config)

    def test_merge_vibrato_fragments_but_keep_reattack_and_double_stop(self):
        notes = [Note("a", 1, 1.5, 69, .8, (0, 1)),
                 Note("b", 1.5, 2, 69, .7, (-1, 0)),
                 Note("c", 2, 2.5, 69, .8, (1, 0)),
                 Note("double", 1.1, 2.4, 62, .7, (0, -1))]
        refined, decisions, provenance = self.merge(notes, [2])
        a = next(n for n in refined if n.note_id == "a")
        self.assertEqual((a.start, a.end), (1, 2))
        self.assertEqual(provenance["a"], ["a", "b"])
        self.assertEqual(next(n for n in refined if n.note_id == "double"), notes[3])
        self.assertIn("audio_attack", [d["reason"] for d in decisions])
        np.testing.assert_array_equal(bend_samples(a)[0], [1, 1.5, 1.5, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "merged.mid"
            route = export(refined, path, self.config, True)
            result = validate_export(path, refined, self.config, True, route)
            self.assertEqual(result["source_bends_verified"], 8)
            self.assertNotEqual(route["a"], route["double"])

    def test_model_attack_alone_protects_repeat(self):
        self.arrays["onset"][150, 69 - 21] = .95
        notes = [Note("a", 1, 1.5, 69, .8, None), Note("b", 1.5, 2, 69, .8, None)]
        refined, decisions, _ = self.merge(notes)
        self.assertEqual(len(refined), 2)
        self.assertEqual(decisions[0]["reason"], "strong_model_onset")

    def test_audible_gap_and_long_gap_are_not_filled(self):
        notes = [Note("a", 1, 1.5, 69, .8, None), Note("b", 1.52, 2, 69, .8, None)]
        self.arrays["note"][150:153, 69 - 21] = 0
        refined, decisions, _ = self.merge(notes)
        self.assertEqual(len(refined), 2)
        self.assertEqual(decisions[0]["reason"], "activation_gap")
        notes[1] = Note("b", 1.7, 2, 69, .8, None)
        self.assertEqual(self.merge(notes)[1][0]["reason"], "gap_too_long")

    def test_overlapping_unisons_and_contained_fragments_are_barriers(self):
        notes = [Note("a", 1, 3, 69, .8, (0,)), Note("b", 2, 2.5, 69, .8, (0,)),
                 Note("c", 2.5, 3.5, 69, .8, (0,))]
        refined, decisions, _ = self.merge(notes)
        self.assertEqual(refined, notes)
        self.assertTrue(all(d["reason"] == "overlapping_unison" for d in decisions))

    def test_half_step_pitch_change_is_not_merged(self):
        notes = [Note("a", 1, 1.5, 69, .8, None), Note("b", 1.5, 2, 70, .8, None)]
        self.assertEqual(len(self.merge(notes)[0]), 2)

    def test_parameters(self):
        args = refinement_parser().parse_args(["a.mp3", "--no-melodia-trick"])
        self.assertEqual(args.onset_threshold, .7)
        self.assertFalse(args.melodia_trick)
        with self.assertRaises(ValueError):
            RefinementConfig(merge_gap_ms=-1).validate()


if __name__ == "__main__":
    unittest.main()
