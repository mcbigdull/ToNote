"""Use Basic Pitch's reader normally; reuse existing FFmpeg decode on fallback."""
from contextlib import contextmanager
from pathlib import Path
import logging
import sys
import tempfile


@contextmanager
def readable_audio(source, output):
    import soundfile as sf
    try:
        info = sf.info(str(source))
        if info.frames <= 0:
            raise ValueError("Audio has no samples")
    except sf.LibsndfileError:
        # Do not change existing vocal code or use its monophonic pitch pipeline.
        vocal_root = str(Path(__file__).resolve().parents[2] / "vocal")
        sys.path.insert(0, vocal_root)
        try:
            from vocal_transcriber.audio import find_ffmpeg, decode_audio
        finally:
            sys.path.remove(vocal_root)
        with tempfile.TemporaryDirectory(prefix="decode_", dir=output) as temporary:
            decoded = Path(temporary) / "input.wav"
            decode_audio(source, decoded, find_ffmpeg())
            info = sf.info(str(decoded))
            yield decoded, dict(duration_sec=info.duration, sample_rate=info.samplerate,
                                channels=info.channels, decoder="Existing vocal FFmpeg fallback, no trimming")
    else:
        logging.info("Audio: %.3fs, %dHz, %d channels", info.duration, info.samplerate, info.channels)
        yield source, dict(duration_sec=info.duration, sample_rate=info.samplerate,
                           channels=info.channels, decoder="Basic Pitch/librosa (original file)")
