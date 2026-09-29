"""Save deterministic eye and real-baseband constellation figures."""
from pathlib import Path
from dsp.experiments import waveform_figures

if __name__ == "__main__":
    waveform_figures(Path("."), 2026)
