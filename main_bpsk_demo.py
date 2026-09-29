import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dsp.prbs import generate_bits
from dsp.modulation import bpsk_modulate



def main():
    n_bits = 50
    bits = generate_bits(n_bits, seed=42)
    symbols = bpsk_modulate(bits, amplitude=1.0)

    # simple waveform: hold each symbol for N samples
    sps = 10  # samples per symbol
    waveform = np.repeat(symbols, sps)
    t = np.arange(len(waveform))

    plt.figure(figsize=(10, 4))
    plt.step(t, waveform, where="post")
    plt.title("BPSK Baseband Waveform (No Pulse Shaping)")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    Path("images").mkdir(exist_ok=True)
    plt.savefig("images/bpsk_waveform.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
