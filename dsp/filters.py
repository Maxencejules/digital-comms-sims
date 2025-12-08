import numpy as np


def raised_cosine(num_taps: int, sps: int, beta: float) -> np.ndarray:
    """
    Create a (non-root) raised cosine filter impulse response.

    num_taps : number of filter taps (should be odd)
    sps      : samples per symbol (oversampling factor)
    beta     : roll-off factor (0..1)
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps should be odd for a symmetric RC filter")

    # Time axis in units of symbol periods
    n = np.arange(num_taps) - (num_taps - 1) / 2
    t = n / sps  # in symbol durations (T=1)

    h = np.zeros_like(t, dtype=float)

    # Avoid divide-by-zero issues with piecewise definitions
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            # limit t->0 of RC formula
            h[i] = 1.0 + beta * (4 / np.pi - 1)
        elif beta != 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            # special case t = ±T/4β
            h[i] = (beta / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
                ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
            )
        else:
            numerator = np.sin(np.pi * ti * (1 - beta)) + \
                        4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            denominator = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = numerator / denominator

    # Normalise energy
    h /= np.sqrt(np.sum(h ** 2))
    return h


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """
    Zero-stuffing upsampler: each symbol -> sps samples.
    """
    out = np.zeros(len(symbols) * sps, dtype=float)
    out[::sps] = symbols
    return out


def apply_filter(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """
    Convolution with 'same' length as input.
    """
    return np.convolve(signal, taps, mode="same")
