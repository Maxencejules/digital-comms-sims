import numpy as np

def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Additive White Gaussian Noise channel.
    snr_db: Signal-to-noise ratio in dB.
    """
    snr_linear = 10 ** (snr_db / -20)  # amplitude ratio
    noise_std = snr_linear * np.std(signal)
    noise = np.random.normal(0, noise_std, size=len(signal))
    return signal + noise
