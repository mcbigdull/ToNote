"""Polyphonic plots; no single-F0 reduction."""
import numpy as np


def save_plots(output, raw, filtered, arrays, times, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from .midi import bend_samples

    def rectangles(axis, notes, color, alpha=0.7):
        for n in notes:
            axis.add_patch(Rectangle((n.start, n.pitch - 0.35), n.end - n.start, 0.7,
                                     facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.3))

    end = max(float(times[-1]) if len(times) else 0, max((n.end for n in raw), default=0), 1)
    low, high = config.pitch_bounds
    pitches = [n.pitch for n in raw]
    ymin, ymax = min([low] + pitches) - 2, max([high] + pitches) + 2
    fig, axis = plt.subplots(figsize=(20, 7))
    rectangles(axis, raw, "0.75", 0.55)
    rectangles(axis, filtered, "#1677b8")
    axis.set(xlim=(0, end), ylim=(ymin, ymax), xlabel="Original audio time (s)",
             ylabel="MIDI pitch", title="Violin piano roll: blue = retained; gray = raw / excluded")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "piano_roll.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(20, 13), sharex=True)
    for axis, key, factor in ((axes[0], "note", 1), (axes[1], "contour", 3)):
        matrix = arrays[key]
        # Display-only max pooling; all full-resolution matrices are saved in NPZ.
        step = max(1, int(np.ceil(len(times) / 4500)))
        rows = [np.max(matrix[i:i + step], axis=0) for i in range(0, len(times), step)]
        display_times = np.array([np.mean(times[i:i + step]) for i in range(0, len(times), step)])
        image = axis.pcolormesh(display_times, 21 + np.arange(matrix.shape[1]) / factor,
                                np.asarray(rows).T, shading="nearest", vmin=0, vmax=1,
                                cmap="magma", rasterized=True)
        axis.set(ylim=(ymin, ymax), ylabel="MIDI pitch",
                 title=f"Raw {key} activation (score, not calibrated probability); display pooling {step} frames")
        fig.colorbar(image, ax=axis, pad=0.01, label="Activation")
    for n in filtered:
        axes[0].plot([n.start, n.end], [n.pitch, n.pitch], color="#45e0ef", linewidth=0.6, alpha=0.8)
    rectangles(axes[2], raw, "0.75", 0.5)
    rectangles(axes[2], filtered, "#1677b8", 0.6)
    for n in filtered:
        bend_times, values = bend_samples(n)
        if len(bend_times):
            axes[2].plot(bend_times, n.pitch + values / 4096, color="#d95f02", linewidth=0.7)
    axes[2].set(xlim=(0, end), ylim=(ymin, ymax), xlabel="Original audio time (s)",
                ylabel="MIDI pitch", title="Filtered notes + upstream exported bends (orange; not high-resolution F0)")
    fig.tight_layout()
    fig.savefig(output / "pitch_debug.png", dpi=150)
    plt.close(fig)
