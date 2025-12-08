import numpy as np

def bpsk_demodulate(samples: np.ndarray) -> np.ndarray:
    """
    Decision rule for BPSK: positive -> 1, negative -> 0.
    """
    return (samples > 0).astype(int)

def bit_error_rate(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    errors = np.sum(bits_tx != bits_rx)
    return errors / len(bits_tx)
