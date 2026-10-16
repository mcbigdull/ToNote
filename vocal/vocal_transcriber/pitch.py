from dataclasses import dataclass
import logging
import numpy as np
from scipy.ndimage import median_filter

from .config import Config, TranscriptionError


@dataclass
class PitchTrack:
    time: np.ndarray
    frequency: np.ndarray
    confidence: np.ndarray
    midi_pitch: np.ndarray
    hop: float


def extract_pitch(audio, sr: int, config: Config, device: str) -> PitchTrack:
    import librosa
    hop = round(sr * config.hop_seconds)
    fmin, fmax = librosa.midi_to_hz([config.min_pitch, config.max_pitch])
    if config.backend == "crepe" and (fmin < 50 or fmax > 2006):
        raise TranscriptionError("CREPE 支持约 50–2006 Hz；请使用 --max-note B6 --min-note C2，或选择 --pitch-backend pyin（默认，支持 C7）。")
    # Overlapping chunks bound pYIN/Viterbi memory; retain only core frames.
    chunk = sr * 20
    context = sr
    frequencies, confidences, times = [], [], []
    for start in range(0, len(audio), chunk):
        left, right = max(0, start - context), min(len(audio), start + chunk + context)
        segment = audio[left:right]
        logging.info("F0: %.1f / %.1f 秒", start / sr, len(audio) / sr)
        if config.backend == "pyin":
            f0, voiced, confidence = librosa.pyin(segment, sr=sr, fmin=float(fmin), fmax=float(fmax),
                                                  frame_length=2048, hop_length=hop, fill_na=np.nan)
            confidence = np.where(voiced, confidence, 0.0)
        else:
            import torch
            import torchcrepe
            try:
                with torch.inference_mode():
                    f0, confidence = torchcrepe.predict(torch.from_numpy(segment.copy())[None], sr, hop,
                                                        float(fmin), float(fmax), "full", batch_size=512,
                                                        device=device, return_periodicity=True)
                f0, confidence = f0[0].cpu().numpy(), confidence[0].cpu().numpy()
            except torch.cuda.OutOfMemoryError as exc:
                raise TranscriptionError("CREPE 显存不足，请用 --device cpu 重试。") from exc
        rms = librosa.feature.rms(y=segment, frame_length=2048, hop_length=hop)[0]
        count = min(len(f0), len(rms))
        confidence = np.nan_to_num(confidence[:count])
        confidence[rms[:count] < 10 ** (config.silence_db / 20)] = 0
        t = (left + np.arange(count) * hop) / sr
        keep = (t >= start / sr) & (t < min(start + chunk, len(audio)) / sr)
        frequencies.extend(f0[:count][keep])
        confidences.extend(confidence[keep])
        times.extend(t[keep])
    frequency = np.asarray(frequencies)
    midi = np.full(len(frequency), np.nan)
    valid = np.isfinite(frequency) & (frequency > 0)
    midi[valid] = librosa.hz_to_midi(frequency[valid])
    return PitchTrack(np.asarray(times), frequency, np.asarray(confidences), midi, hop / sr)


def valid_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def smooth_pitch(track: PitchTrack, config: Config) -> PitchTrack:
    pitch = track.midi_pitch.copy()
    good = np.isfinite(pitch) & (track.confidence >= config.min_confidence)
    good &= (pitch >= config.min_pitch) & (pitch <= config.max_pitch)
    pitch[~good] = np.nan
    # Never interpolate across unvoiced gaps: breaths and repeated notes survive.
    window = max(1, round(config.smooth_ms / 1000 / track.hop)) | 1
    for start, end in valid_runs(good):
        pitch[start:end] = median_filter(pitch[start:end], size=window, mode="nearest")
    frequency = 440 * 2 ** ((pitch - 69) / 12)
    return PitchTrack(track.time.copy(), frequency, track.confidence.copy(), pitch, track.hop)
