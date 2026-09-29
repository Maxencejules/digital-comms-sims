"""Finite, unit-energy root-raised-cosine (RRC) pulses."""
import numpy as np


def root_raised_cosine(num_taps: int, sps: int, beta: float) -> np.ndarray:
    """RRC impulse response, including t=0 and t=±1/(4β) limits.

    Time is in symbols. A matched pair approximates a raised-cosine response;
    finite truncation leaves small residual ISI. Each pulse has sum(h²)=1.
    """
    if not isinstance(num_taps, int) or num_taps < 3 or num_taps % 2 != 1:
        raise ValueError("num_taps must be an odd integer of at least three")
    if not isinstance(sps, int) or sps < 1:
        raise ValueError("sps must be a positive integer")
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be between zero and one")
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    h = np.empty(num_taps)
    for i, ti in enumerate(t):
        if beta == 0:
            h[i] = np.sinc(ti)
        elif np.isclose(ti, 0):
            h[i] = 1 + beta * (4 / np.pi - 1)
        elif np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = beta / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta))
                    + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (
                        np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.linalg.norm(h)


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    if not isinstance(sps, int) or sps < 1:
        raise ValueError("sps must be a positive integer")
    out = np.zeros(len(symbols) * sps)
    out[::sps] = symbols
    return out


def apply_filter(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Full linear convolution: retain both tails and account for group delay."""
    return np.convolve(signal, taps, mode="full")
