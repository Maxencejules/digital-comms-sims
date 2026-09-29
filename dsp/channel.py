"""Real baseband AWGN with an explicit energy-per-bit convention."""
import numpy as np


def noise_variance(ebn0_db: float, bit_energy: float = 1.0) -> float:
    """Return N0/2 = Eb/(2 * 10**(Eb/N0_dB/10)) per real sample.

    Sample time is normalized to one. For BPSK amplitude A and a pulse h,
    nominal Eb = A**2 * sum(h**2), independent of samples per symbol.
    This is not waveform-average sample-power SNR.
    """
    if not np.isfinite(ebn0_db) or not np.isfinite(bit_energy) or bit_energy <= 0:
        raise ValueError("Eb/N0 must be finite and bit energy positive and finite")
    return bit_energy / (2 * 10 ** (ebn0_db / 10))


def awgn(signal: np.ndarray, ebn0_db: float, *, rng: np.random.Generator,
         bit_energy: float = 1.0) -> np.ndarray:
    """Add independent real Gaussian samples; caller owns the random stream."""
    signal = np.asarray(signal)
    if signal.ndim != 1 or not np.isrealobj(signal) or not np.all(np.isfinite(signal)):
        raise ValueError("Expected a finite one-dimensional real baseband signal")
    return signal + rng.normal(0, np.sqrt(noise_variance(ebn0_db, bit_energy)), signal.shape)
