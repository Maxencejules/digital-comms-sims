import numpy as np

def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Additive White Gaussian Noise channel.

    snr_db is the desired SNR in dB:
        SNR = signal_power / noise_power

    This function computes the signal power, derives the
    required noise power for the requested SNR, and adds
    Gaussian noise with that variance.
    """
    # Signal power (mean squared value)
    sig_power = np.mean(np.abs(signal) ** 2)

    # Desired SNR (linear)
    snr_linear = 10 ** (snr_db / 10.0)

    # Noise power and standard deviation
    noise_power = sig_power / snr_linear
    noise_std = np.sqrt(noise_power)

    # AWGN
    noise = np.random.normal(0.0, noise_std, size=signal.shape)
    return signal + noise
