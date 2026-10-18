from dataclasses import asdict, dataclass
import argparse
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length: float = 127.7  # Basic Pitch API: milliseconds
    minimum_frequency: float = 196.0
    maximum_frequency: float = 2637.0
    multiple_pitch_bends: bool = True
    melodia_trick: bool = True
    short_note_ms: float = 150.0
    harmonic_tolerance_cents: float = 35.0
    harmonic_overlap_ratio: float = 0.8
    coincident_onset_sec: float = 0.08
    coincident_end_sec: float = 0.15
    max_harmonic: int = 8
    midi_ticks_per_beat: int = 9600
    midi_tempo: int = 120  # time encoding only; no beat detection/quantization
    violin_program: int = 40  # zero-based General MIDI
    bend_range_semitones: int = 2
    refine_notes: bool = False

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.onset_threshold <= 1 or not 0 < self.frame_threshold <= 1:
            raise ValueError("onset/frame thresholds must be in (0, 1]")
        if self.minimum_note_length <= 0 or self.short_note_ms <= 0:
            raise ValueError("Note lengths must be positive (milliseconds)")
        if not 0 < self.minimum_frequency < self.maximum_frequency:
            raise ValueError("Require 0 < minimum_frequency < maximum_frequency")
        if not self.multiple_pitch_bends:
            raise ValueError("This project requires multiple_pitch_bends=True")
        low, high = self.pitch_bounds
        if low > high or high < 21 or low > 108:
            raise ValueError("Frequency range contains no supported Basic Pitch notes (A0-C8)")

    @property
    def pitch_bounds(self):
        # Inclusive nearest MIDI centers: 196 Hz includes G3 (195.9977 Hz),
        # 2637 Hz includes E7 (2637.0205 Hz). Filtering never edits bends.
        return tuple(round(69 + 12 * math.log2(hz / 440))
                     for hz in (self.minimum_frequency, self.maximum_frequency))


def parser():
    result = argparse.ArgumentParser(description="Basic Pitch violin audio to polyphonic MIDI")
    result.add_argument("audio", type=Path)
    result.add_argument("--output", type=Path, default=Path("output"),
                        help="Output directory; creates a sibling with a suffix if occupied")
    for flag in ("onset-threshold", "frame-threshold", "minimum-note-length", "short-note-ms"):
        name = flag.replace("-", "_")
        result.add_argument("--" + flag, type=float, default=getattr(Config(), name))
    result.add_argument("--min-frequency", "--minimum-frequency", dest="minimum_frequency",
                        type=float, default=Config.minimum_frequency)
    result.add_argument("--max-frequency", "--maximum-frequency", dest="maximum_frequency",
                        type=float, default=Config.maximum_frequency)
    result.add_argument("--multiple-pitch-bends", action="store_true", default=True,
                        help="Always enabled, including when this flag is omitted")
    result.add_argument("--verbose", action="store_true")
    result.add_argument("--refine", dest="refine_notes", action="store_true",
                        help="Also create an independent refined/ result after baseline transcription")
    result.add_argument("--melodia-trick", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable Basic Pitch's residual-note recovery")
    return result


@dataclass(frozen=True)
class RefinementConfig(Config):
    onset_threshold: float = 0.7
    merge_gap_ms: float = 40.0
    onset_tolerance_ms: float = 60.0
    strong_model_onset: float = 0.9
    bridge_active_fraction: float = 0.8
    audio_onset_delta: float = 0.12
    audio_onset_wait_ms: float = 80.0
    audio_sample_rate: int = 22050
    audio_hop_length: int = 256
    vibrato_max_size: int = 3
    overlap_tolerance_sec: float = 1e-6

    def validate(self):
        super().validate()
        if self.merge_gap_ms < 0 or self.onset_tolerance_ms < 0 or self.audio_onset_wait_ms < 0:
            raise ValueError("Refinement timing tolerances must be nonnegative")
        for name in ("strong_model_onset", "bridge_active_fraction", "audio_onset_delta"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.audio_sample_rate <= 0 or self.audio_hop_length <= 0 or self.vibrato_max_size < 1:
            raise ValueError("Invalid onset analysis settings")


def refinement_parser():
    result = parser()
    root = Path(__file__).resolve().parents[1]
    result.description = "Refine saved Basic Pitch activations without rerunning the model"
    result.set_defaults(onset_threshold=RefinementConfig.onset_threshold, output=root / "output_refined")
    result.add_argument("--model-output", type=Path, default=root / "output/model_outputs.npz")
    for name in ("merge_gap_ms", "onset_tolerance_ms", "strong_model_onset", "audio_onset_delta"):
        result.add_argument("--" + name.replace("_", "-"), type=float,
                            default=getattr(RefinementConfig(), name))
    return result
