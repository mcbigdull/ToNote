import logging
from pathlib import Path

from .config import TranscriptionError


def choose_device(requested: str) -> str:
    import torch
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        logging.warning("CUDA 不可用，改用 CPU。GPU 需要 CUDA 版 PyTorch 和兼容驱动。")
    return "cuda" if requested != "cpu" and available else "cpu"


def separate_vocals(audio_path: Path, output_dir: Path, device: str = "cpu") -> Path:
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    try:
        model = get_model("htdemucs")
        model.eval()
    except Exception as exc:
        raise TranscriptionError("Demucs 预训练模型加载/下载失败。首次运行需联网访问 dl.fbaipublicfiles.com；请检查网络及模型缓存后重试。" + str(exc)) from exc
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if sr != model.samplerate or audio.shape[1] != model.audio_channels:
        raise TranscriptionError("Demucs 输入必须是解码后的 44100 Hz 双声道音频。")
    wav = torch.from_numpy(audio.T.copy())
    ref = wav.mean(0)
    scale = ref.std()
    if not torch.isfinite(scale) or scale < 1e-7:
        # Handles anti-phase channels without dividing by zero.
        scale = wav.std()
    if not torch.isfinite(scale) or scale < 1e-7:
        raise TranscriptionError("输入音频为空白，无法分离人声。")
    mean = ref.mean()
    normalized = (wav - mean) / scale

    def infer(target):
        with torch.inference_mode():
            return apply_model(model, normalized[None], device=target, shifts=1,
                               split=True, overlap=0.25, progress=True)[0].cpu()

    try:
        try:
            sources = infer(device)
        except torch.cuda.OutOfMemoryError:
            logging.warning("显存不足，使用 CPU 重试人声分离。")
            model.cpu()
            torch.cuda.empty_cache()
            sources = infer("cpu")
        sources = (sources * scale + mean).numpy()
    except Exception as exc:
        raise TranscriptionError(f"Demucs 分离失败，可尝试 --device cpu。详细信息：{exc}") from exc
    index = model.sources.index("vocals")
    vocals_path = output_dir / "vocals.wav"
    sf.write(str(vocals_path), sources[index].T, sr, subtype="FLOAT")
    sf.write(str(output_dir / "accompaniment.wav"), np.delete(sources, index, axis=0).sum(axis=0).T, sr, subtype="FLOAT")
    return vocals_path
