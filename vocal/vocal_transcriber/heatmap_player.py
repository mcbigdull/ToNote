"""Offline, audio-clock-driven viewer of cached Basic Pitch activations."""
import html
import json
from pathlib import Path

import numpy as np


def write_player(output, arrays=None):
    from matplotlib import colormaps
    from PIL import Image

    output = Path(output)
    labels = {'original_mix': '原曲', 'demucs': 'HTDemucs 人声',
              'bs_roformer': 'BS-RoFormer 人声'}
    if arrays is None:
        arrays = {}
        for name in labels:
            with np.load(output / name / 'model_outputs.npz') as archive:
                arrays[name] = {key: archive[key] for key in archive.files}
    reference = arrays['original_mix']
    times = reference['frame_times_sec']
    duration = float(reference['audio_duration_sec'])
    # Uniform image pixels map directly to audio seconds, including the final frame.
    width = len(times)
    centers = (np.arange(width) + .5) * duration / width
    right = np.clip(np.searchsorted(times, centers), 1, len(times) - 1)
    indices = np.where(centers - times[right - 1] <= times[right] - centers, right - 1, right)
    assets = output / 'player_assets'
    assets.mkdir(exist_ok=True)
    players, rows, axes = [], [], []
    for name, label in labels.items():
        data = arrays[name]
        if not np.array_equal(times, data['frame_times_sec']) or abs(float(data['audio_duration_sec']) - duration) > 1e-6:
            raise ValueError('Player requires aligned audio timelines')
        audio = 'vocals.mp3' if (output / name / 'vocals.mp3').exists() else 'audio.wav'
        audio_path = f'web_audio/{name}.mp3' if (output / 'web_audio' / f'{name}.mp3').is_file() else f'{name}/{audio}'
        players.append(f'<section><strong>{label}</strong><audio controls preload="metadata" data-label="{label}" src="{audio_path}"></audio><a href="{audio_path}" download>下载音频</a></section>')
    for key in ('note', 'contour'):
        for name, label in labels.items():
            data = arrays[name]
            values = np.clip(data[key][indices].T[::-1], 0, 1)
            Image.fromarray(colormaps['magma'](values, bytes=True)[:, :, :3]).save(assets / f'{name}_{key}.png')
            rows.append(f'<div class="row" data-kind="{key}"><img draggable="false" src="player_assets/{name}_{key}.png" alt="{label} {key}"></div>')
            # Image spans bin edges; contour uses three bins per semitone.
            low, high = (20.5, 108.5) if key == 'note' else (21 - 1/6, 21 + 263/3 + 1/6)
            ticks = ''.join(f'<span style="top:{100*(high-p)/(high-low):.4f}%">C{p//12-1}</span>' for p in (24, 36, 48, 60, 72, 84, 96, 108))
            axes.append(f'<div class="axis" data-kind="{key}"><b>{label}<br>{key}</b>{ticks}</div>')
    template = Path(__file__).with_name('heatmap_player.html').read_text(encoding='utf-8')
    title = output.name.removesuffix('_separation_compare')
    page = (template.replace('__TITLE__', html.escape(title)).replace('__PLAYERS__', ''.join(players))
            .replace('__ROWS__', ''.join(rows)).replace('__AXES__', ''.join(axes))
            .replace('__DURATION__', json.dumps(duration)))
    (output / 'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    import sys
    write_player(sys.argv[1])
