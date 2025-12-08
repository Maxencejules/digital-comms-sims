import numpy as np
import matplotlib.pyplot as plt

from dsp.prbs import generate_bits
from dsp.modulation import bpsk_modulate
from dsp.filters import raised_cosine, upsample, apply_filter
from dsp.channel import awgn


def main():
    n_bits = 5000
    sps = 8           # samples per symbol (oversampling)
    beta = 0.35       # roll-off factor
    num_taps = 81     # RC filter length (odd)

    # --- Transmitter ---
    bits = generate_bits(n_bits, seed=123)
    symbols = bpsk_modulate(bits, amplitude=1.0)

    tx_rc = raised_cosine(num_taps=num_taps, sps=sps, beta=beta)

    # Upsample and pulse-shape
    up = upsample(symbols, sps)
    tx_waveform = apply_filter(up, tx_rc)

    # --- Channel (optional AWGN) ---
    snr_db = 8.0
    rx_waveform = awgn(tx_waveform, snr_db)

    # --- Receiver matched filter (same RC) ---
    rx_rc = tx_rc[::-1]  # matched filter (time-reversed)
    mf_output = apply_filter(rx_waveform, rx_rc)

    # Ignore filter start-up transient, look at middle region
    start = num_taps
    mf_valid = mf_output[start:-start]

    # --- Build eye diagram ---
    eye_span = 2  # 2 symbol periods
    window = eye_span * sps

    # Align so that "0" on x axis is symbol center-ish
    num_traces = 200  # # of eye traces to plot
    traces = []

    # choose a region away from edges
    center_offset = sps // 2
    for i in range(num_traces):
        idx = start + i * sps
        if idx - window // 2 < 0 or idx + window // 2 >= len(mf_output):
            continue
        segment = mf_output[idx - window // 2: idx + window // 2]
        traces.append(segment)

    traces = np.array(traces)
    t = (np.arange(window) - window / 2) / sps  # in symbol durations

    plt.figure(figsize=(8, 5))
    for seg in traces:
        plt.plot(t, seg, color="C0", alpha=0.2)

    plt.title("Eye Diagram (BPSK, RC pulse shaping)")
    plt.xlabel("Time (symbol periods)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/eye_diagram.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
