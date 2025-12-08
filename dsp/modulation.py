import numpy as np



def bpsk_modulate(bits: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    # 0 -> -A, 1 -> +A
    return amplitude * (2 * bits - 1)
