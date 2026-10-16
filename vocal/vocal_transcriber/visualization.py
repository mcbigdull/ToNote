def plot_pitch(raw, smoothed, notes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axis = plt.subplots(figsize=(18, 6))
    axis.plot(raw.time, raw.midi_pitch, color="0.65", alpha=0.5, linewidth=0.5, label="Raw F0")
    axis.plot(smoothed.time, smoothed.midi_pitch, color="#1677b8", linewidth=0.8, label="Smoothed F0")
    for i, note in enumerate(notes):
        axis.add_patch(Rectangle((note.start_sec, note.pitch - 0.3), note.end_sec - note.start_sec,
                                0.6, facecolor="#e69f00", alpha=0.65, label="Notes (MIDI timing)" if i == 0 else None))
    axis.set(xlabel="Time (seconds)", ylabel="MIDI pitch", title="Vocal transcription")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
