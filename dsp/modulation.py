"""One explicit BPSK convention: bit 0 -> -A, bit 1 -> +A."""
import numpy as np


def bpsk_modulate(bits: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    bits = np.asarray(bits)
    if bits.ndim != 1 or not np.all(np.isin(bits, [0, 1])):
        raise ValueError("Expected a one-dimensional binary array")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be positive and finite")
    # Convert before arithmetic so unsigned bits cannot wrap 0 to +255.
    return amplitude * (2 * bits.astype(float) - 1)
