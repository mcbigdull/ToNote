"""Create ordinary-Git 160 kbps MP3 copies for a comparison page."""
import argparse
from pathlib import Path
import subprocess

SOURCES = ('original_mix', 'demucs', 'bs_roformer')


def export(output):
    output = Path(output).resolve()
    ffmpeg = Path(__file__).resolve().parent / '.cache' / 'bin' / 'ffmpeg.exe'
    if not ffmpeg.is_file():
        import imageio_ffmpeg
        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    page = output / 'index.html'
    document = page.read_text(encoding='utf-8')
    for name in SOURCES:
        if not (output / name / 'audio.wav').is_file():
            raise FileNotFoundError(output / name / 'audio.wav')
    folder = output / 'web_audio'
    folder.mkdir(exist_ok=True)
    (folder / '.gitattributes').write_text('*.mp3 -filter -diff -merge -text\n', encoding='utf-8')
    for name in SOURCES:
        target = folder / f'{name}.mp3'
        temporary = folder / f'{name}.tmp.mp3'
        subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(output / name / 'audio.wav'), '-map', '0:a:0', '-vn',
                        '-c:a', 'libmp3lame', '-b:a', '160k', str(temporary)], check=True)
        temporary.replace(target)
        for old in (f'{name}/audio.wav', f'{name}/vocals.mp3'):
            document = document.replace(f'"{old}"', f'"web_audio/{name}.mp3"')
    page.write_text(document, encoding='utf-8')
    print(f'Exported web audio: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='+')
    args = parser.parse_args()
    for directory in args.output:
        export(directory)
