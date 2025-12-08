import numpy as np



def generate_bits(n_bits: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits, dtype=int)
