import numpy as np
import matplotlib.pyplot as plt

from dsp.prbs import generate_bits
from dsp.modulation import bpsk_modulate
from dsp.filters import raised_cosine, upsample, apply_filter
from dsp.channel import awgn
from dsp.receiver import sample_bpsk_from_waveform, bit_error_rate

"""
Work-in-progress: BER of full pulse-shaped chain with crude timing recovery.

The simpler BER experiment (main_ber.py) already matches BPSK-in-AWGN theory.
This script is exploratory and not relied on for final results.
"""
def simulate_ber_full_chain(
    n_bits: int,
    sps: int,
    beta: float,
    num_taps: int,
    snr_db: float,
) -> float:
    """
    Full BPSK chain with pulse shaping, AWGN, matched filter,
    symbol sampling, and hard decisions.

    Uses a simple timing recovery trick:
    try all sampling phases (0..sps-1) and take the best BER.
    """
    # --- Transmitter ---
    bits_tx = generate_bits(n_bits)
    symbols = bpsk_modulate(bits_tx, amplitude=1.0)

    tx_rc = raised_cosine(num_taps=num_taps, sps=sps, beta=beta)
    up = upsample(symbols, sps)
    tx_waveform = apply_filter(up, tx_rc)

    # --- Channel ---
    rx_waveform = awgn(tx_waveform, snr_db)

    # --- Receiver: matched filter ---
    rx_rc = tx_rc[::-1]  # matched filter
    mf_output = apply_filter(rx_waveform, rx_rc)

    # Base delay from tx + rx filters ('same' conv)
    base_delay = (num_taps - 1)

    best_ber = 1.0

    # Try every sampling phase inside one symbol period
    for phase in range(sps):
        delay = base_delay + phase

        bits_rx = sample_bpsk_from_waveform(
            mf_output=mf_output,
            sps=sps,
            delay=delay,
            n_symbols=n_bits,
        )

        n_valid = min(len(bits_tx), len(bits_rx))
        if n_valid == 0:
            continue

        ber = bit_error_rate(bits_tx[:n_valid], bits_rx[:n_valid])

        if ber < best_ber:
            best_ber = ber

    return best_ber

def main():
    n_bits = 50000
    sps = 8
    beta = 0.35
    num_taps = 81

    snr_values = np.arange(0, 14, 2)  # 0,2,...,12 dB
    ber_values = []

    for snr in snr_values:
        ber = simulate_ber_full_chain(
            n_bits=n_bits,
            sps=sps,
            beta=beta,
            num_taps=num_taps,
            snr_db=snr,
        )
        ber_values.append(ber)
        print(f"SNR = {snr:2d} dB -> BER = {ber:.6e}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_values, ber_values, marker="o")
    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BPSK BER vs SNR (Full Pulse-Shaped Chain)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
