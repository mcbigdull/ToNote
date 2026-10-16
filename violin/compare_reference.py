"""Plot the independently transcribed score and existing Basic Pitch activations.

No model inference, note filtering, or MIDI changes. Score timing is estimated
from the original audio's chroma, not copied from Basic Pitch note events.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache' / 'numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))

import numpy as np


@dataclass(frozen=True)
class AlignmentConfig:
    sample_rate: int = 22050
    hop_length: int = 1024
    score_frames_per_quarter: int = 12
    chroma_octaves: int = 6
    audio_start_activity_db_below_peak: float = 26.0
    audio_activity_db_below_peak: float = 38.0
    normalized_alignment_band: float = 0.18
    minimum_plot_pitch: int = 53
    maximum_plot_pitch: int = 103


def pitch_number(name):
    match = re.fullmatch(r'([A-G])([#b]?)(\d+)', name)
    if not match:
        raise ValueError(f'Invalid score pitch: {name}')
    letter, accidental, octave = match.groups()
    return 12 * (int(octave) + 1) + {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[letter] + {'': 0, '#': 1, 'b': -1}[accidental]


def read_score(path):
    score = json.loads(path.read_text(encoding='utf-8'))
    source_image = path.parent / score['source_image']
    if hashlib.sha256(source_image.read_bytes()).hexdigest() != score['source_sha256']:
        raise ValueError('Archived score image differs from the manual transcription source')
    notes, symbols = [], []
    current_tie = None
    for expected, measure in enumerate(score['measures'], 1):
        if measure['number'] != expected:
            raise ValueError('Measures must be consecutive')
        offset = 0.0
        for index, event in enumerate(measure['events'], 1):
            name, duration, *tie = event
            duration = float(duration)
            if duration <= 0:
                raise ValueError('Invalid score duration')
            begin = (expected - 1) * 6 + offset
            pitch = None if name == 'rest' else pitch_number(name)
            tie_type = tie[0] if tie else None
            symbol = dict(symbol_id=f'm{expected:02d}s{index:02d}', measure=expected,
                          start_quarter=begin, end_quarter=begin + duration,
                          duration_quarter=duration, pitch_midi=pitch, note_name=name, tie=tie_type)
            symbols.append(symbol)
            if pitch is not None:
                if tie_type in ('continue', 'stop'):
                    if current_tie is None or current_tie['pitch_midi'] != pitch or abs(current_tie['end_quarter'] - begin) > 1e-8:
                        raise ValueError(f'Invalid tie at {symbol["symbol_id"]}')
                    current_tie['end_quarter'] = begin + duration
                    current_tie['symbol_ids'].append(symbol['symbol_id'])
                    current_tie['last_measure'] = expected
                    if tie_type == 'stop':
                        current_tie = None
                else:
                    if current_tie is not None:
                        raise ValueError('Unfinished tie')
                    note = dict(note_id=f'r{len(notes) + 1:04d}', pitch_midi=pitch, note_name=name,
                                first_measure=expected, last_measure=expected,
                                start_quarter=begin, end_quarter=begin + duration,
                                symbol_ids=[symbol['symbol_id']])
                    notes.append(note)
                    if tie_type == 'start':
                        current_tie = note
            elif current_tie is not None:
                raise ValueError('Rest inside an unfinished tie')
            offset += duration
        if abs(offset - 6) > 1e-8:
            raise ValueError(f'Measure {expected} sums to {offset}, expected 6')
    if current_tie is not None:
        raise ValueError('Unfinished final tie')
    return score, notes, symbols


def align_to_audio(notes, audio_path, config, output):
    import librosa
    import soundfile as sf
    audio, sr = librosa.load(str(audio_path), sr=config.sample_rate, mono=True)
    duration = len(audio) / sr
    logging.info('Computing audio chroma, no Basic Pitch inference')
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=config.hop_length,
                                      fmin=librosa.note_to_hz('C2'), n_octaves=config.chroma_octaves,
                                      norm=None)
    rms = librosa.feature.rms(y=audio, frame_length=4096, hop_length=config.hop_length)[0]
    count = min(chroma.shape[1], len(rms))
    chroma, rms = chroma[:, :count], rms[:count]
    audio_times = librosa.frames_to_time(np.arange(count), sr=sr, hop_length=config.hop_length)
    threshold = max(float(np.max(rms)) * 10 ** (-config.audio_activity_db_below_peak / 20), 1e-6)
    active = np.flatnonzero(rms > threshold)
    if not len(active):
        raise ValueError('Audio is silent')
    strong_start = np.flatnonzero(rms > max(float(np.max(rms)) * 10 ** (-config.audio_start_activity_db_below_peak / 20), 1e-6))
    first = max(0, int(strong_start[0] if len(strong_start) else active[0]) - 1)
    last = min(count - 1, int(active[-1]) + 1)
    audio_times = audio_times[first:last + 1]
    chroma = np.sqrt(np.maximum(chroma[:, first:last + 1], 0))
    chroma /= np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-8)

    start_beat, end_beat = notes[0]['start_quarter'], notes[-1]['end_quarter']
    score_beats = np.linspace(start_beat, end_beat, round((end_beat - start_beat) * config.score_frames_per_quarter) + 1)
    score_chroma = np.zeros((12, len(score_beats)), dtype=np.float32)
    score_active = np.zeros(len(score_beats), dtype=bool)
    for n in notes:
        indices = (score_beats >= n['start_quarter']) & (score_beats < n['end_quarter'])
        score_chroma[n['pitch_midi'] % 12, indices] = 1
        score_active[indices] = True
    # Include the last sounding endpoint in the alignment feature sequence.
    score_chroma[:, -1] = score_chroma[:, -2]
    score_active[-1] = score_active[-2]
    cost = 1 - score_chroma.T @ chroma
    # Notated rests should prefer low energy, while still permitting audible reverb.
    local_energy = rms[first:last + 1]
    relative_energy = np.clip(local_energy / max(float(np.percentile(local_energy, 75)), 1e-8), 0, 1)
    cost[~score_active] = relative_energy[None, :] * .7
    score_progress = (score_beats - start_beat) / (end_beat - start_beat)
    audio_progress = (audio_times - audio_times[0]) / (audio_times[-1] - audio_times[0])
    displacement = np.abs(score_progress[:, None] - audio_progress[None, :])
    cost += .04 * displacement
    cost[displacement > config.normalized_alignment_band] = np.inf
    logging.info('Aligning %d score frames to %d audio frames', len(score_beats), len(audio_times))
    _, path = librosa.sequence.dtw(C=cost,
                                  step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1], [1, 3]]),
                                  weights_add=np.array([0.0, .025, .025, .07]))
    path = path[::-1]
    matched_time = np.full(len(score_beats), np.nan)
    for index in np.unique(path[:, 0]):
        matched_time[index] = np.median(audio_times[path[path[:, 0] == index, 1]])
    known = np.isfinite(matched_time)
    matched_time = np.interp(score_beats, score_beats[known], matched_time[known])
    matched_time = np.maximum.accumulate(matched_time)

    def beat_to_time(beats):
        return np.interp(beats, np.r_[0.0, score_beats, 168.0], np.r_[0.0, matched_time, duration])

    for n in notes:
        n['start_sec'], n['end_sec'] = map(float, beat_to_time([n['start_quarter'], n['end_quarter']]))
        if n['end_sec'] <= n['start_sec']:
            raise ValueError(f'Alignment collapsed note {n["note_id"]}')
    mapping = np.c_[score_beats, matched_time]
    np.savetxt(output / 'alignment_path.csv', mapping, delimiter=',', header='score_quarter,audio_sec', comments='')
    np.savez_compressed(output / 'audio_alignment_features.npz', audio_chroma=chroma,
                        audio_times_sec=audio_times, score_beats=score_beats,
                        score_chroma=score_chroma, aligned_audio_times_sec=matched_time)
    return duration, beat_to_time, dict(
        method='Constrained DTW: manually transcribed score chroma vs original-audio CQT chroma',
        uses_basic_pitch_for_alignment=False, independently_annotated_audio_onsets=False,
        approximate_audio_active_start_sec=float(audio_times[0]),
        approximate_audio_active_end_sec=float(audio_times[-1]),
        container_reported_duration_sec=float(sf.info(str(audio_path)).duration),
        decoded_audio_duration_sec=duration,
        mean_feature_cost=float(np.mean(cost[path[:, 0], path[:, 1]])),
        warning='Estimated score-to-recording timing for visual comparison. Not onset/offset ground truth; matching chroma can hide octave ambiguity in time alignment.')


def write_csv(path, records, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for n in records:
            writer.writerow({k: ';'.join(n[k]) if isinstance(n.get(k), list) else n.get(k) for k in fields})


def heatmap(notes, times, pitches):
    result = np.zeros((len(times), len(pitches)), dtype=np.float32)
    for n in notes:
        active = (times >= n['start_sec']) & (times < n['end_sec'])
        band = np.exp(-.5 * ((pitches - n['pitch_midi']) / .21) ** 2)
        result[active] = np.maximum(result[active], band)
    return result


def plots(output, notes, times, arrays, duration, beat_to_time, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import MultipleLocator
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        plt.rcParams['font.family'] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.facecolor'] = '#ffffff'
    contour_pitches = 21 + np.arange(arrays['contour'].shape[1]) / 3
    reference = heatmap(notes, times, contour_pitches)
    # Identical pixel columns / seconds for all three panels. Max pooling is
    # display-only; raw matrices remain untouched in the source NPZ.
    step = max(1, int(np.ceil(len(times) / 4200)))
    starts = np.arange(0, len(times), step)
    display_times = np.array([np.mean(times[i:i + step]) for i in starts])
    pooled = {k: np.maximum.reduceat(v, starts, axis=0) for k, v in dict(
        reference=reference, note=arrays['note'], contour=arrays['contour']).items()}
    panels = [('reference', contour_pitches, '参考谱 · 人工录入音高与时值，按原音频自动估计时间位置', '谱面有音'),
              ('note', 21 + np.arange(arrays['note'].shape[1]), 'Basic Pitch 原始音符激活 · 未按 threshold 删除，未清理杂音', '模型激活'),
              ('contour', contour_pitches, 'Basic Pitch 原始 contour 激活 · 保留揉弦附近的音高波动', '模型激活')]

    def draw(filename, lower, upper, only_reference=False):
        selected_panels = panels[:1] if only_reference else panels
        fig = plt.figure(figsize=(22, 3.5 if only_reference else 10.2))
        grid = fig.add_gridspec(len(selected_panels), 2, width_ratios=[1, .012], hspace=.28, wspace=.025)
        first_axis = None
        for row, (key, pitch_axis, title, label) in enumerate(selected_panels):
            ax = fig.add_subplot(grid[row, 0], sharex=first_axis, sharey=first_axis)
            if first_axis is None:
                first_axis = ax
            ax.set_facecolor('black')
            cax = fig.add_subplot(grid[row, 1])
            selection = (display_times >= lower - .1) & (display_times <= upper + .1)
            image = ax.pcolormesh(display_times[selection], pitch_axis, pooled[key][selection].T,
                                  cmap='magma', shading='nearest', vmin=0, vmax=1, rasterized=True)
            ax.set(xlim=(lower, upper), ylim=(config.minimum_plot_pitch, config.maximum_plot_pitch),
                   ylabel='MIDI 音高', title=title)
            ax.set_yticks(np.arange(55, 104, 5))
            ax.xaxis.set_major_locator(MultipleLocator(10 if upper - lower > 60 else 5))
            ax.grid(axis='x', color='white', alpha=.12)
            bar = fig.colorbar(image, cax=cax)
            bar.set_label(label)
            bar.set_ticks([0, .5, 1])
            if row == len(selected_panels) - 1:
                ax.set_xlabel('原音频时间（秒）· 各行横轴与纵轴完全相同')
            else:
                ax.tick_params(labelbottom=False)
            if key == 'reference':
                for measure in range(2, 28):
                    second = float(beat_to_time((measure - 1) * 6))
                    if lower <= second <= upper:
                        ax.axvline(second, color='white', linewidth=.4, alpha=.25)
                        if upper - lower < 60 or measure % 2 == 0:
                            ax.text(second + .12, 101.3, f'm{measure}', color='0.9', fontsize=7)
        fig.suptitle('《天鹅》参考谱与模型热图对照', fontsize=16, y=.995)
        fig.text(.06, .018 if not only_reference else .01,
                 '参考谱是音符占用图，不是频谱；无虚构的谐波、音色或揉弦。时间为基于原音频色度特征的估计对齐，不能用于证明起止准确率。',
                 fontsize=9, color='#555555')
        fig.subplots_adjust(left=.055, right=.95, top=.80 if only_reference else .935,
                            bottom=.19 if only_reference else .09)
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)

    draw('reference_vs_model_heatmap.png', 0, duration)
    draw('reference_heatmap.png', 0, duration, True)
    draw('comparison_000_040.png', 0, 40)
    draw('comparison_040_080.png', 40, 80)
    draw('comparison_080_110.png', 80, 110)
    draw('comparison_110_end.png', 110, duration)
    np.savez_compressed(output / 'reference_heatmap.npz', frame_times_sec=times,
                        pitch_midi=contour_pitches, score_note_occupancy=reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', type=Path, default=ROOT / 'reference/the_swan_score.json')
    parser.add_argument('--audio', type=Path, default=ROOT / 'input/天鹅-小提琴.mp3')
    parser.add_argument('--model-output', type=Path, default=ROOT / 'output/model_outputs.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/reference_comparison')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    for name in ('numba', 'matplotlib'):
        logging.getLogger(name).setLevel(logging.WARNING)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = AlignmentConfig()
    score, notes, symbols = read_score(args.score)
    logging.info('%d measures, %d written noteheads, %d sounding events after ties',
                 len(score['measures']), sum(s['pitch_midi'] is not None for s in symbols), len(notes))
    duration, beat_to_time, alignment = align_to_audio(notes, args.audio, config, output)
    with np.load(args.model_output, allow_pickle=False) as data:
        arrays = {k: data[k] for k in ('note', 'onset', 'contour')}
        times = data['frame_times_sec']
    write_csv(output / 'reference_notes_aligned.csv', notes,
              ['note_id', 'first_measure', 'last_measure', 'note_name', 'pitch_midi',
               'start_quarter', 'end_quarter', 'start_sec', 'end_sec', 'symbol_ids'])
    write_csv(output / 'reference_symbols.csv', symbols,
              ['symbol_id', 'measure', 'note_name', 'pitch_midi', 'start_quarter', 'end_quarter', 'duration_quarter', 'tie'])
    report = dict(source_score=str(args.score.resolve()), source_image_sha256=score['source_sha256'],
                  score_json_sha256=hashlib.sha256(args.score.read_bytes()).hexdigest(),
                  audio=str(args.audio.resolve()), audio_duration_sec=duration,
                  model_output=str(args.model_output.resolve()),
                  model_output_sha256=hashlib.sha256(args.model_output.read_bytes()).hexdigest(),
                  original_model_frame_time_range_sec=[float(times[0]), float(times[-1])],
                  manually_entered_measures=len(score['measures']),
                  written_noteheads=sum(s['pitch_midi'] is not None for s in symbols),
                  sounding_score_events_after_ties=len(notes),
                  score_pitch_range=[min(n['pitch_midi'] for n in notes), max(n['pitch_midi'] for n in notes)],
                  alignment=alignment, config=asdict(config),
                  limitations=['Manual raster-score transcription may contain reading errors; inspect source image.',
                               'Same time axis is an estimated alignment, not independent onset/offset annotation.',
                               'No inference rerun, no baseline or MIDI changes, no model notes deleted.',
                               'Reference intensity is nominal note occupancy, not acoustic energy or calibrated probability.'])
    (output / 'alignment_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    plots(output, notes, times, arrays, duration, beat_to_time, config)
    print(output)


if __name__ == '__main__':
    main()
