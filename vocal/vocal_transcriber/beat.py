from dataclasses import dataclass
import logging
import numpy as np

from .config import TranscriptionError


@dataclass
class BeatGrid:
    bpm: float
    times: np.ndarray
    beats: np.ndarray
    source: str

    @staticmethod
    def _interpolate(values, x, y):
        values = np.asarray(values, dtype=float)
        result = np.interp(values, x, y)
        result = np.where(values < x[0], y[0] + (values - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), result)
        return np.where(values > x[-1], y[-1] + (values - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]), result)

    def seconds_to_beats(self, seconds):
        return self._interpolate(seconds, self.times, self.beats)

    def beats_to_seconds(self, beats):
        return self._interpolate(beats, self.beats, self.times)

    def tempo_events(self):
        # Export the same piecewise tempo map used for quantization.
        events = []
        for i, interval in enumerate(np.diff(self.times)):
            beat = 0.0 if i == 0 else float(self.beats[i])
            bpm = 60 * (self.beats[i + 1] - self.beats[i]) / interval
            events.append((beat, float(bpm)))
        return events


def analyze_beats(audio, sr, config):
    import librosa
    duration = len(audio) / sr
    if config.bpm is not None:
        interval = 60 / config.bpm
        # Beat zero is an extrapolated beat at or before audio time zero.
        origin = config.beat_offset - np.ceil(config.beat_offset / interval) * interval
        times = np.array([origin, origin + interval])
        return BeatGrid(config.bpm, times, np.array([0., 1.]), "manual")
    tempo, positions = librosa.beat.beat_track(y=audio, sr=sr, hop_length=256, units="time", trim=False)
    times = np.asarray(positions, dtype=float)
    times = times[(times >= 0) & (times <= duration)]
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise TranscriptionError("无法可靠检测节拍。请提供 --bpm 120（替换为实际 BPM），必要时加 --beat-offset 秒数。")
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    interval = float(np.median(np.diff(times)))
    prefix = int(np.ceil(times[0] / interval))
    if prefix:
        times = np.r_[times[0] - np.arange(prefix, 0, -1) * interval, times]
    logging.info("节拍估计 %.2f BPM；自动节拍不等于强拍检测，默认从首个网格拍按指定拍号分小节。", bpm)
    return BeatGrid(bpm, times, np.arange(len(times), dtype=float), "automatic")
