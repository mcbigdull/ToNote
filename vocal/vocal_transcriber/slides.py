"""Stable destinations of pitch movement, independent of segmentation/export."""
import numpy as np


def stable_target(values, tolerance: float):
    center = float(np.median(values))
    # A sustained glide has too much spread to count as a new stable note.
    if np.percentile(values, 90) - np.percentile(values, 10) > tolerance:
        return None
    return center
