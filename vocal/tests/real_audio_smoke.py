"""Bounded real-song validation; preserves its clip and all pipeline artifacts."""
import argparse
from pathlib import Path
import subprocess

from vocal_transcriber.audio import find_ffmpeg
from vocal_transcriber.main import main


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--start", type=float, default=45)
    parser.add_argument("--duration", type=float, default=20)
    args = parser.parse_args()
    folder = Path("output") / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    clip = folder / f"{args.audio.stem} 验证片段.wav"
    subprocess.run([find_ffmpeg(), "-nostdin", "-v", "error", "-y", "-ss", str(args.start),
                    "-i", str(args.audio.resolve()), "-t", str(args.duration), "-ac", "2", "-ar", "44100", str(clip)], check=True)
    return main([str(clip), "--output", str(folder / "results")])


if __name__ == "__main__":
    raise SystemExit(run())
