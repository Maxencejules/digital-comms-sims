import numpy as np
import matplotlib.pyplot as plt

from dsp.prbs import generate_bits
from dsp.modulation import bpsk_modulate
from dsp.channel import awgn
from dsp.receiver import bpsk_demodulate, bit_error_rate


def main():
    n_bits = 50000
    sps = 1  # (simple model: 1 sample per symbol)

    snr_values = np.arange(-2, 10, 2)  # SNR range (dB)
    ber_values = []

    for snr in snr_values:
        bits = generate_bits(n_bits)
        symbols = bpsk_modulate(bits)

        # Channel
        rx_samples = awgn(symbols, snr)

        # Receiver
        bits_rx = bpsk_demodulate(rx_samples)

        # BER
        ber = bit_error_rate(bits, bits_rx)
        ber_values.append(ber)

        print(f"SNR={snr} dB → BER={ber:.6f}")

    # Plot BER curve
    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_values, ber_values, marker='o')
    plt.grid(True, which='both')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BPSK BER vs SNR (AWGN Channel)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
