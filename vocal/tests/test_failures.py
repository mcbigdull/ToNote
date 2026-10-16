from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vocal_transcriber.audio import find_ffmpeg
from vocal_transcriber.beat import analyze_beats
from vocal_transcriber.config import Config, TranscriptionError
from vocal_transcriber.main import main
from vocal_transcriber.pitch import extract_pitch
from vocal_transcriber.separation import choose_device, separate_vocals


class FailureTests(unittest.TestCase):
    def test_missing_ffmpeg_is_actionable(self):
        with patch("shutil.which", return_value=None), patch("imageio_ffmpeg.get_ffmpeg_exe", side_effect=RuntimeError("missing")):
            with self.assertRaisesRegex(TranscriptionError, "FFmpeg"):
                find_ffmpeg()

    def test_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(choose_device("auto"), "cpu")
            self.assertEqual(choose_device("cuda"), "cpu")

    def test_model_download_error(self):
        with patch("demucs.pretrained.get_model", side_effect=OSError("offline")):
            with self.assertRaisesRegex(TranscriptionError, "模型加载/下载失败"):
                separate_vocals(Path("unused.wav"), Path("unused"))

    def test_no_beats_requires_manual_bpm(self):
        with patch("librosa.beat.beat_track", return_value=(0, np.array([]))):
            with self.assertRaisesRegex(TranscriptionError, "--bpm"):
                analyze_beats(np.zeros(16000), 16000, Config())

    def test_crepe_range_rejected_before_inference(self):
        with self.assertRaisesRegex(TranscriptionError, "2006"):
            extract_pitch(np.zeros(16000), 16000, Config(backend="crepe"), "cpu")

    def test_failed_run_records_error_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "中文 song.wav"
            source.touch()
            existing = root / "out" / source.stem
            existing.mkdir(parents=True)
            marker = existing / "melody.mid"
            marker.write_bytes(b"previous result")
            with patch("vocal_transcriber.main.run", side_effect=TranscriptionError("test error")):
                self.assertEqual(main([str(source), "--output", str(root / "out")]), 1)
            self.assertEqual(marker.read_bytes(), b"previous result")
            reports = list((root / "out").glob("*/run.json"))
            self.assertEqual(len(reports), 1)
            self.assertIn("test error", reports[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
