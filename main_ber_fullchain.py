"""Pulse-shaped BPSK using only a known preamble for timing acquisition."""
from dsp.experiments import run_single_ber

if __name__ == "__main__":
    run_single_ber("fullchain")
