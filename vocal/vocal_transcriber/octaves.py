"""Conservative octave reconciliation; preserve disagreements in an audit log."""
import numpy as np
from .pitch import PitchTrack, smooth_pitch
from .segmentation import segment_notes


def spectral_support(audio, sr, time, base, target):
    radius = round(sr * .064)
    center = round(time * sr)
    clip = audio[max(0, center - radius):min(len(audio), center + radius)]
    if len(clip) < 128:
        return 0., 0.
    spectrum = np.abs(np.fft.rfft(clip * np.hanning(len(clip)), n=8192))
    frequencies = np.fft.rfftfreq(8192, 1 / sr)

    def energy(hz):
        selected = np.abs(frequencies - hz) < max(8., hz * .035)
        return float(np.max(spectrum[selected])) if selected.any() else 0.

    fbase, ftarget = 440 * 2 ** ((np.array([base, target]) - 69) / 12)
    root = energy(ftarget)
    ratio = root / max(energy(fbase), 1e-9)
    harmonic = energy(2 * ftarget) / max(root, 1e-9)
    return ratio, harmonic


def reconcile_octaves(primary, reference, audio, sr, config):
    """Only shift existing voiced F0 by exact octaves, never manufacture notes.

    Aggregate evidence over preliminary stable notes, so short confidence dips
    do not cut a sustained note into alternating octaves. Ambiguous notes remain
    unchanged and are logged. These heuristics cannot guarantee lead selection.
    """
    if len(primary.time) != len(reference.time) or not np.allclose(primary.time, reference.time, atol=1e-5):
        raise ValueError("音高交叉检查的时间网格不一致")
    original = primary.midi_pitch
    other = reference.midi_pitch
    good = (np.isfinite(original) & np.isfinite(other)
            & (primary.confidence >= config.min_confidence) & (reference.confidence >= .5))
    good &= (original >= max(32, config.min_pitch)) & (original <= min(95, config.max_pitch))
    difference = np.where(good, other - original, 0.)
    corrected = original.copy()
    audit = []
    agreement = good & (np.abs(difference) < .65)
    for note in segment_notes(smooth_pitch(primary, config), config):
        left, right = np.searchsorted(primary.time, [note.start-1e-7, note.end-1e-7])
        indices = np.arange(left, right)
        valid = indices[good[left:right]]
        if not len(valid):
            continue
        delta = float(np.median(difference[valid]))
        if abs(delta) < 1.5:
            continue
        shift = int(round(delta / 12)) * 12
        base = float(np.median(original[valid]))
        ref = float(np.median(other[valid]))
        row = {"start_sec": note.start, "end_sec": note.end,
               "primary_midi": base, "reference_midi": ref, "action": "review"}
        if shift not in (-24,-12,12,24) or abs(delta-shift) > .65:
            audit.append(dict(row, shift_semitones=0, reason="non_octave_disagreement"))
            continue
        votes = valid[np.abs(difference[valid]-shift) <= .65]
        confidence = float(np.median(reference.confidence[votes])) if len(votes) else 0.
        coverage = len(valid) / len(indices)
        support = len(votes) / len(valid)
        probes = valid[np.linspace(0,len(valid)-1,min(9,len(valid))).astype(int)]
        evidence = [spectral_support(audio,sr,primary.time[i],original[i],original[i]+shift) for i in probes]
        ratio, harmonic = np.median(evidence, axis=0)
        margin = round(1 / primary.hop)
        neighbors = np.r_[np.arange(max(0,left-margin),left),np.arange(right,min(len(good),right+margin))]
        anchored = bool(np.any(agreement[neighbors] & (np.abs(other[neighbors]-(base+shift))<=5)))
        # Overwhelming spectral evidence can compensate for moderate model
        # confidence, but still needs agreement across most of the note.
        strong = harmonic >= .08 and ((confidence >= .65 and ratio >= 3)
                                      or (confidence >= .6 and ratio >= 6 and support >= .85))
        accepted = (coverage >= .5 and support >= .65 and ratio >= 1.5 and harmonic >= .04
                    and (confidence >= .7 or strong)
                    and (anchored or note.end-note.start >= .3 or strong)
                    and config.min_pitch <= note.pitch+shift <= config.max_pitch)
        if accepted:
            corrected[left:right] += shift
        audit.append(dict(row, shift_semitones=shift, reference_confidence=confidence,
                          reference_coverage=coverage, octave_support=support,
                          spectral_ratio=float(ratio),harmonic_ratio=float(harmonic),
                          context_agreement=anchored,action="corrected" if accepted else "review"))
    return PitchTrack(primary.time.copy(), 440 * 2 ** ((corrected-69)/12),
                      primary.confidence.copy(), corrected, primary.hop), sorted(audit, key=lambda row: row['start_sec'])
