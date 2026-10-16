from dataclasses import dataclass
import math
import re


def note_number(name: str) -> int:
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", name)
    if not match:
        raise ValueError(f"无效音名 {name!r}，示例：C2、F#4、Bb3")
    letter, accidental, octave = match.groups()
    value = 12 * (int(octave) + 1) + dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)[letter.upper()]
    value += {"": 0, "#": 1, "b": -1}[accidental]
    if not 0 <= value <= 127:
        raise ValueError("音域必须在 MIDI 0–127 内")
    return value


def note_name(pitch: int) -> str:
    return ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[pitch % 12] + str(pitch // 12 - 1)


@dataclass
class Config:
    min_pitch: int = 36
    max_pitch: int = 96
    min_confidence: float = 0.5
    min_note_duration: float = 0.08
    pitch_tolerance_cents: float = 50
    smooth_ms: float = 70
    stable_ms: float = 100
    max_gap_ms: float = 30
    silence_db: float = -55
    quantization_strength: float = 1.0
    time_signature: str = "4/4"
    key: str | None = None
    bpm: float | None = None
    beat_offset: float = 0.0
    backend: str = "pyin"
    device: str = "auto"
    hop_seconds: float = 0.01

    def validate(self):
        for name in ("min_confidence", "min_note_duration", "pitch_tolerance_cents", "smooth_ms", "stable_ms", "max_gap_ms", "silence_db", "quantization_strength", "beat_offset"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} 必须是有限数值")
        if self.min_pitch >= self.max_pitch:
            raise ValueError("最低音必须低于最高音")
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.quantization_strength <= 1:
            raise ValueError("置信度与量化强度必须在 0–1 之间")
        if min(self.min_note_duration, self.pitch_tolerance_cents, self.smooth_ms, self.stable_ms) <= 0 or self.max_gap_ms < 0:
            raise ValueError("时长、平滑及容差参数必须为正，间隙不能为负")
        if self.bpm is not None and (not math.isfinite(self.bpm) or not 10 <= self.bpm <= 400):
            raise ValueError("BPM 必须在 10–400 之间")
        if self.beat_offset < 0:
            raise ValueError("beat-offset 不能为负")
        if not re.fullmatch(r"([1-9]|[12]\d|3[0-2])/(1|2|4|8|16)", self.time_signature):
            raise ValueError("无效拍号，示例：4/4、3/4、6/8")


@dataclass
class NoteEvent:
    pitch: int
    start: float
    end: float
    confidence: float


@dataclass
class QuantizedNote:
    pitch: int
    start_sec: float
    end_sec: float
    start_beat: float
    duration_beats: float
    confidence: float


class TranscriptionError(RuntimeError):
    pass
