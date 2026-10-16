from pathlib import Path
import shutil
import subprocess

from .config import TranscriptionError


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise TranscriptionError("未找到 FFmpeg。请安装 requirements.txt 中的 imageio-ffmpeg，或将 ffmpeg.exe 加入 PATH。") from exc
    try:
        subprocess.run([executable, "-version"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TranscriptionError(f"FFmpeg 无法执行：{executable}") from exc
    return executable


def decode_audio(source: Path, target: Path, ffmpeg: str):
    result = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source),
                             "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", str(target)],
                            capture_output=True)
    if result.returncode:
        raise TranscriptionError("音频解码失败：" + result.stderr.decode("utf-8", errors="replace")[-2000:])


def load_mono(path: Path, sr: int = 16000):
    import librosa
    import numpy as np
    import soundfile as sf
    audio, original_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if len(audio) < original_sr * 0.08 or not np.isfinite(audio).all():
        raise TranscriptionError("音频过短、为空或包含无效采样值。")
    mono = audio.mean(axis=1)
    # Opposite-phase stereo should not become artificial silence.
    if np.sqrt(np.mean(mono ** 2)) < 0.1 * np.sqrt(np.mean(audio ** 2)):
        mono = audio[:, np.argmax(np.mean(audio ** 2, axis=0))]
    if np.max(np.abs(mono)) < 1e-6:
        raise TranscriptionError("人声/输入音频为空白，无法提取旋律。")
    return librosa.resample(mono, orig_sr=original_sr, target_sr=sr), sr
