"""Rebuild notes/score from a saved pair of F0 tracks without model inference."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
from vocal_transcriber.audio import load_mono
from vocal_transcriber.beat import BeatGrid
from vocal_transcriber.config import Config, note_name
from vocal_transcriber.main import save_pitch, write_csv
from vocal_transcriber.pitch import PitchTrack, smooth_pitch
from vocal_transcriber.octaves import reconcile_octaves
from vocal_transcriber.segmentation import segment_notes
from vocal_transcriber.quantization import quantize_notes
from vocal_transcriber.midi_export import export_midi
from vocal_transcriber.musicxml_export import export_musicxml
from vocal_transcriber.visualization import plot_pitch


def load_track(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:
        rows=list(csv.DictReader(handle))
    arrays=[np.array([float(row[field]) for row in rows]) for field in ['time','frequency','confidence','midi_pitch']]
    if len(rows)<2:
        raise ValueError("Not enough pitch frames")
    # Avoid 0.00999999999999 turning a 100ms stability window into 11 frames.
    return PitchTrack(*arrays,round(float(np.median(np.diff(arrays[0]))),9))


def rebuild(source, output):
    source=source.resolve()
    previous=json.loads((source/'run.json').read_text(encoding='utf-8'))
    config=Config(**previous['config'])
    config.validate()
    voice,sr=load_mono(source/'vocals.wav')
    raw=load_track(source/'pitch_raw.csv')
    reference=load_track(source/'pitch_reference.csv')
    corrected,audit=reconcile_octaves(raw,reference,voice,sr,config)
    smoothed=smooth_pitch(corrected,config)
    notes=segment_notes(smoothed,config)
    if not notes:
        raise ValueError("No valid notes")
    beats=json.loads((source/'beats.json').read_text(encoding='utf-8'))
    grid=BeatGrid(beats['bpm'],np.array(beats['beat_times']),np.array(beats['beat_positions']),beats['source'])
    quantized=quantize_notes(notes,grid,config)
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','source_run':str(source),'input':previous.get('input'),
            'config':asdict(config),'octave_check':True}
    try:
        for filename in ['pitch_raw.csv','pitch_reference.csv','beats.json']:
            shutil.copyfile(source/filename,output/filename)
        save_pitch(output/'pitch_corrected.csv',corrected)
        save_pitch(output/'pitch_smoothed.csv',smoothed)
        (output/'octave_review.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        write_csv(output/'notes_raw.csv',['pitch','note_name','start_sec','end_sec','duration_sec','confidence'],
                  ((n.pitch,note_name(n.pitch),n.start,n.end,n.end-n.start,n.confidence) for n in notes))
        write_csv(output/'notes_quantized.csv',['pitch','note_name','start_sec','end_sec','duration_sec','start_beat','duration_beats','confidence'],
                  ((n.pitch,note_name(n.pitch),n.start_sec,n.end_sec,n.end_sec-n.start_sec,n.start_beat,n.duration_beats,n.confidence) for n in quantized))
        export_midi(quantized,grid,config,output/'melody.mid')
        selected_key=export_musicxml(quantized,grid,config,output/'melody.musicxml')
        plot_pitch(raw,smoothed,quantized,output/'debug_pitch.png')
        report.update(status='success',key=selected_key,bpm=grid.bpm,notes=len(quantized),
                      corrected_regions=sum(r['action']=='corrected' for r in audit),
                      review_regions=sum(r['action']=='review' for r in audit),
                      vocals_file=str(source/'vocals.wav'))
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        raise
    finally:
        (output/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rebuild(args.source,args.output)
