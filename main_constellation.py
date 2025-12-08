import numpy as np
import matplotlib.pyplot as plt

from dsp.prbs import generate_bits
from dsp.modulation import bpsk_modulate
from dsp.filters import raised_cosine, upsample, apply_filter
from dsp.channel import awgn


def main():
    n_bits = 5000
    sps = 8
    beta = 0.35
    num_taps = 81

    bits = generate_bits(n_bits, seed=321)
    symbols = bpsk_modulate(bits, amplitude=1.0)

    tx_rc = raised_cosine(num_taps=num_taps, sps=sps, beta=beta)
    up = upsample(symbols, sps)
    tx_waveform = apply_filter(up, tx_rc)

    snr_db = 6.0
    rx_waveform = awgn(tx_waveform, snr_db)

    rx_rc = tx_rc[::-1]
    mf_output = apply_filter(rx_waveform, rx_rc)

    # Sample at symbol centers (account for filter delay)
    delay = (num_taps - 1)
    start = delay
    sample_indices = np.arange(start, start + n_bits * sps, sps)
    sample_indices = sample_indices[sample_indices < len(mf_output)]

    samples = mf_output[sample_indices]

    # For BPSK, constellation is purely real; treat imaginary part as 0
    I = samples
    Q = np.zeros_like(samples)

    plt.figure(figsize=(5, 5))
    plt.scatter(I, Q, s=5, alpha=0.4)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.title("Constellation Diagram (BPSK)")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
