"""Framed, coherent BPSK experiments with independent known training."""
from dataclasses import dataclass
import numpy as np
from .channel import awgn
from .filters import apply_filter, root_raised_cosine, upsample
from .modulation import bpsk_modulate
from .receiver import bit_error_count, bpsk_demodulate, estimate_timing
from .statistics import BerMeasurement

PREAMBLE_BITS = 1024


@dataclass(frozen=True)
class FrameReception:
    detected_bits: np.ndarray
    samples: np.ndarray
    matched_waveform: np.ndarray
    payload_start: int
    phase: int
    correlation: float


def receive_frame(payload: np.ndarray, ebn0_db: float | None, *, rng: np.random.Generator,
                  sps: int = 8, beta: float = 0.35, num_taps: int = 81,
                  timing_offset: int = 0) -> FrameReception:
    """Simulate Tx/channel/Rx. None Eb/N0 means a noiseless verification channel.

    Receiver acquisition receives only known interior preamble symbols. Actual
    timing_offset and payload truth are used by the channel/evaluator, never by
    the phase estimator. Quiet guards and full convolution retain filter tails.
    """
    pulse = root_raised_cosine(num_taps, sps, beta)
    if sps < 2:
        raise ValueError("The pulse-shaped real-baseband model requires sps >= 2")
    if not isinstance(timing_offset, int) or not 0 <= timing_offset < sps:
        raise ValueError("timing_offset must be an integer in 0..sps-1")
    if np.ndim(payload) != 1 or len(payload) < 1 or not np.all(np.isin(payload, [0, 1])):
        raise ValueError("A nonempty binary payload is required")
    guard = (num_taps - 1 + sps - 1) // sps
    if 2 * guard >= PREAMBLE_BITS:
        raise ValueError("Filter span leaves too few independent preamble symbols")
    known = bpsk_modulate(np.random.default_rng(271828).integers(0, 2, PREAMBLE_BITS))
    frame = np.concatenate((np.zeros(guard), known, bpsk_modulate(payload), np.zeros(guard)))
    tx = apply_filter(upsample(frame, sps), pulse)
    delayed = np.pad(tx, (timing_offset, 0))
    rx = delayed if ebn0_db is None else awgn(delayed, ebn0_db, rng=rng)
    matched = apply_filter(rx, pulse[::-1])
    # Two (L-1)/2 FIR delays add to L-1. Discard training edge symbols to
    # ensure no unknown payload waveform sample can influence acquisition.
    training_start = (guard + guard) * sps + num_taps - 1
    estimate = estimate_timing(matched, known[guard:-guard], sps, training_start)
    payload_start = (guard + PREAMBLE_BITS) * sps + num_taps - 1 + estimate.phase
    samples = matched[payload_start + np.arange(len(payload)) * sps]
    return FrameReception(bpsk_demodulate(samples), samples, matched,
                          payload_start, estimate.phase, estimate.correlation)


def simulate_symbol_rate(n_bits: int, ebn0_db: float, seed: int = 2026) -> BerMeasurement:
    if not isinstance(n_bits, int) or n_bits < 1:
        raise ValueError("n_bits must be a positive integer")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_bits)
    detected = bpsk_demodulate(awgn(bpsk_modulate(bits), ebn0_db, rng=rng))
    return BerMeasurement(ebn0_db, bit_error_count(bits, detected), n_bits, seed)


def simulate_full_chain(n_bits: int, ebn0_db: float, seed: int = 2026, *, sps: int = 8,
                        beta: float = 0.35, num_taps: int = 81,
                        frame_bits: int = 50000) -> BerMeasurement:
    if not isinstance(n_bits, int) or n_bits < 1 or not isinstance(frame_bits, int) or frame_bits < 1:
        raise ValueError("n_bits and frame_bits must be positive integers")
    rng = np.random.default_rng(seed)
    errors = frames = phase_mismatches = 0
    remaining = n_bits
    while remaining:
        count = min(frame_bits, remaining)
        payload = rng.integers(0, 2, count)
        offset = int(rng.integers(sps))
        received = receive_frame(payload, ebn0_db, rng=rng, sps=sps, beta=beta,
                                 num_taps=num_taps, timing_offset=offset)
        errors += bit_error_count(payload, received.detected_bits)
        frames += 1
        phase_mismatches += received.phase != offset
        remaining -= count
    # Acquisition failures propagate instead of silently dropping bad frames.
    return BerMeasurement(ebn0_db, errors, n_bits, seed, frames, int(phase_mismatches))
