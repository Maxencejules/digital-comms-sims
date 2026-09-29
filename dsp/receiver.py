"""Hard BPSK decisions and acquisition from known training symbols only."""
from dataclasses import dataclass
import numpy as np


def bpsk_demodulate(samples: np.ndarray) -> np.ndarray:
    """Map negative samples to 0, positive samples to 1 (zero ties map to 0)."""
    return (np.asarray(samples) > 0).astype(int)


def bit_error_count(bits_tx: np.ndarray, bits_rx: np.ndarray) -> int:
    bits_tx, bits_rx = np.asarray(bits_tx), np.asarray(bits_rx)
    if np.shape(bits_tx) != np.shape(bits_rx) or np.ndim(bits_tx) != 1 or len(bits_tx) == 0:
        raise ValueError("BER requires equal-length, nonempty one-dimensional bit arrays")
    if not np.all(np.isin(bits_tx, [0, 1])) or not np.all(np.isin(bits_rx, [0, 1])):
        raise ValueError("BER requires binary arrays")
    return int(np.count_nonzero(bits_tx != bits_rx))


def bit_error_rate(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    return bit_error_count(bits_tx, bits_rx) / len(bits_tx)


class TimingAcquisitionError(ValueError):
    """The known preamble did not pass the configured acquisition threshold."""


@dataclass(frozen=True)
class TimingEstimate:
    phase: int
    correlation: float


def estimate_timing(mf_output: np.ndarray, preamble_symbols: np.ndarray,
                    sps: int, preamble_start: int, threshold: float = 0.45) -> TimingEstimate:
    """Choose phase 0..sps-1 by normalized correlation with a known preamble.

    Coarse frame position, pulse delay, symbol rate, and carrier polarity are
    known. Only an integer offset within one symbol is unknown. Payload bits
    are neither accepted nor used. The caller passes interior training symbols
    to avoid filter tails from unknown neighboring payload affecting the score.
    """
    if not isinstance(sps, int) or sps < 1 or preamble_start < 0:
        raise ValueError("Invalid sampling geometry")
    preamble_symbols = np.asarray(preamble_symbols)
    if preamble_symbols.ndim != 1 or len(preamble_symbols) < 2:
        raise ValueError("At least two known training symbols are required")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    indices = preamble_start + np.arange(len(preamble_symbols)) * sps
    if indices[-1] + sps - 1 >= len(mf_output):
        raise ValueError("Incomplete preamble; no payload decoding attempted")
    scores = []
    for phase in range(sps):
        samples = mf_output[indices + phase]
        norm = np.linalg.norm(samples) * np.linalg.norm(preamble_symbols)
        scores.append(float(np.dot(samples, preamble_symbols) / norm) if norm else 0.0)
    phase = int(np.argmax(scores))
    score = scores[phase]
    if not np.isfinite(score) or score < threshold:
        raise TimingAcquisitionError(f"Preamble correlation {score:.3f} is below {threshold:.3f}")
    return TimingEstimate(phase, score)


def sample_bpsk_from_waveform(mf_output: np.ndarray, sps: int, delay: int,
                              n_symbols: int) -> np.ndarray:
    indices = delay + np.arange(n_symbols) * sps
    if sps < 1 or n_symbols < 1 or delay < 0 or indices[-1] >= len(mf_output):
        raise ValueError("The waveform must contain every requested payload symbol")
    return bpsk_demodulate(mf_output[indices])
