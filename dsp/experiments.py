"""Reproducible CSV measurements and Matplotlib scientific figures."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .simulation import receive_frame, simulate_full_chain, simulate_symbol_rate
from .statistics import bpsk_theory

EBN0_VALUES = (-2, 0, 2, 4, 6, 8, 10)


def ber_experiment(model: str, bits: int, seed: int, output_dir: Path):
    simulator = simulate_symbol_rate if model == "symbol" else simulate_full_chain
    measurements = [simulator(bits, ebn0, seed + i) for i, ebn0 in enumerate(EBN0_VALUES)]
    records = [measurement.record() for measurement in measurements]
    results = output_dir / "results"
    images = output_dir / "images"
    results.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    with (results / f"ber_{model}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_theory = np.linspace(EBN0_VALUES[0], EBN0_VALUES[-1], 200)
    ax.semilogy(x_theory, [bpsk_theory(x) for x in x_theory], color="0.25", label="Coherent BPSK theory")
    nonzero = [row for row in records if row["errors"]]
    if nonzero:
        x = [r["ebn0_db"] for r in nonzero]
        y = [r["ber"] for r in nonzero]
        ax.errorbar(x, y, yerr=[[r["ber"] - r["lower_95"] for r in nonzero],
                               [r["upper_95"] - r["ber"] for r in nonzero]],
                    fmt="o", capsize=4, label="Measured BER; nominal 95% Wilson interval")
    zero = [row for row in records if not row["errors"]]
    if zero:
        ax.scatter([r["ebn0_db"] for r in zero], [r["upper_95"] for r in zero],
                   marker="v", color="C1", label="Zero errors: 95% upper bound")
    ax.set(xlabel="Eb/N0 (dB)", ylabel="Bit error probability",
           title=f"BPSK in real AWGN — {'symbol rate' if model == 'symbol' else 'RRC + preamble timing'}\n"
                 f"{bits:,} payload bits per point; seed {seed}")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    filename = "ber_curve.png" if model == "symbol" else "fullchain_ber.png"
    fig.savefig(images / filename, dpi=160)
    plt.close(fig)
    for row in records:
        estimate = f"BER={row['ber']:.6g}" if row["errors"] else f"BER < {row['upper_95']:.6g} (95% bound)"
        print(f"{model:6s} Eb/N0={row['ebn0_db']:2g} dB: {row['errors']}/{row['bits']} errors, {estimate}")
    return records


def illustrative_frame(seed: int):
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 2, 2000)
    return receive_frame(payload, 6.0, rng=rng, timing_offset=3)


def waveform_figures(output_dir: Path, seed: int):
    received = illustrative_frame(seed)
    images = output_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sps = 8
    time = (np.arange(2 * sps) - sps) / sps
    for i in range(100):
        center = received.payload_start + (i + 10) * sps
        ax.plot(time, received.matched_waveform[center - sps:center + sps], color="C0", alpha=0.12)
    ax.axvline(0, color="0.3", linestyle="--", linewidth=1)
    ax.set(xlabel="Time from acquired symbol center (symbols)", ylabel="Matched-filter amplitude",
           title="BPSK eye: unit-energy RRC pair, Eb/N0 = 6 dB")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(images / "eye_diagram.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(received.samples, np.zeros(len(received.samples)), s=8, alpha=0.15)
    ax.scatter([-1, 1], [0, 0], marker="x", s=80, color="C3", label="Ideal symbols")
    ax.set(xlabel="Real matched-filter sample", ylabel="Imaginary component (not simulated)",
           title="Real-baseband BPSK samples, Eb/N0 = 6 dB", ylim=(-0.25, 0.25))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(images / "constellation.png", dpi=160)
    plt.close(fig)


def run_single_ber(model: str):
    parser = argparse.ArgumentParser(description="Seeded BPSK BER with count-based uncertainty")
    parser.add_argument("--bits", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    ber_experiment(model, args.bits, args.seed, args.output_dir)
