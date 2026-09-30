"""Generate both BER experiments, CSV data, figures, and a seed/config manifest."""
import argparse
import json
from pathlib import Path
import platform
import matplotlib
import numpy as np
from dsp.experiments import EBN0_VALUES, ber_experiment, waveform_figures
from dsp.simulation import PREAMBLE_BITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    for model in ("symbol", "fullchain"):
        ber_experiment(model, args.bits, args.seed, args.output_dir)
    waveform_figures(args.output_dir, args.seed)
    manifest = {"payload_bits_per_point": args.bits, "base_seed": args.seed,
                "point_seed": "base_seed + point index", "ebn0_db": list(EBN0_VALUES),
                "sps": 8, "rrc_beta": 0.35, "rrc_num_taps": 81, "preamble_bits": PREAMBLE_BITS,
                "frame_payload_bits": 50000, "timing_offsets": "random integer 0..7 per frame",
                "preamble_seed": 271828, "acquisition_threshold": 0.45,
                "bit_energy": 1, "real_noise_variance": "1 / (2 * 10**(EbN0_dB/10))",
                "confidence": 0.95, "python": platform.python_version(),
                "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    (args.output_dir / "results" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
