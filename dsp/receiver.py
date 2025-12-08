import numpy as np

def bpsk_demodulate(samples: np.ndarray) -> np.ndarray:
    """
    Decision rule for BPSK: positive -> 1, negative -> 0.
    """
    return (samples > 0).astype(int)

def bit_error_rate(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    errors = np.sum(bits_tx != bits_rx)
    return errors / len(bits_tx)

def sample_bpsk_from_waveform(
    mf_output: np.ndarray,
    sps: int,
    delay: int,
    n_symbols: int,
) -> np.ndarray:
    """
    Take the matched-filter output (mf_output) and sample every 'sps' samples
    starting at 'delay', then do hard BPSK decisions.

    Returns the detected bit array (length <= n_symbols).
    """
    start = delay
    indices = start + np.arange(n_symbols) * sps
    indices = indices[indices < len(mf_output)]

    samples = mf_output[indices]
    return bpsk_demodulate(samples)