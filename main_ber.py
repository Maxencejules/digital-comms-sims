"""Symbol-rate BPSK baseline; --help lists reproducible CLI options."""
from dsp.experiments import run_single_ber

if __name__ == "__main__":
    run_single_ber("symbol")
