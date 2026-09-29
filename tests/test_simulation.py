import math
import unittest
import numpy as np
from dsp.channel import awgn, noise_variance
from dsp.filters import apply_filter, root_raised_cosine, upsample
from dsp.modulation import bpsk_modulate
from dsp.prbs import generate_bits
from dsp.receiver import (TimingAcquisitionError, bit_error_count, bpsk_demodulate,
                          estimate_timing, sample_bpsk_from_waveform)
from dsp.simulation import receive_frame, simulate_full_chain, simulate_symbol_rate
from dsp.statistics import ber_interval, bpsk_theory


class ModulationTests(unittest.TestCase):
    def test_known_mapping_and_decisions(self):
        np.testing.assert_array_equal(bpsk_modulate(np.array([0, 1, 1, 0])), [-1, 1, 1, -1])
        np.testing.assert_array_equal(bpsk_demodulate(np.array([-4, 0, 0.01, 3])), [0, 0, 1, 1])

    def test_seeded_bits_are_reproducible_and_binary(self):
        a = generate_bits(1000, seed=42)
        np.testing.assert_array_equal(a, generate_bits(1000, seed=42))
        self.assertEqual(set(a), {0, 1})

    def test_unsigned_binary_bits_do_not_wrap_negative_symbols(self):
        np.testing.assert_array_equal(bpsk_modulate(np.array([0, 1], dtype=np.uint8)), [-1, 1])
        with self.assertRaises(ValueError):
            bpsk_modulate(np.array([0, 2]))

    def test_ber_counts_every_bit_and_rejects_truncation(self):
        self.assertEqual(bit_error_count(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])), 2)
        for a, b in [(np.array([]), np.array([])), (np.array([0, 1]), np.array([0]))]:
            with self.subTest(a=a):
                with self.assertRaises(ValueError):
                    bit_error_count(a, b)


class PulseAndNoiseTests(unittest.TestCase):
    def test_rrc_singularities_symmetry_and_unit_energy(self):
        for beta in (0, 0.25, 0.35, 1):
            with self.subTest(beta=beta):
                pulse = root_raised_cosine(81, 8, beta)
                self.assertTrue(np.all(np.isfinite(pulse)))
                np.testing.assert_allclose(pulse, pulse[::-1], atol=1e-14)
                self.assertAlmostEqual(float(pulse @ pulse), 1, places=14)

    def test_rrc_pair_is_nearly_nyquist_and_retains_known_delay(self):
        pulse = root_raised_cosine(81, 8, 0.35)
        impulse = apply_filter(apply_filter(upsample(np.array([1.]), 8), pulse), pulse[::-1])
        self.assertEqual(int(np.argmax(impulse)), 80)
        self.assertAlmostEqual(impulse[80], 1)
        other_centers = np.delete(impulse[:161:8], 10)
        self.assertLess(float(np.max(np.abs(other_centers))), 0.006)

    def test_noise_variance_uses_energy_not_observed_signal_power(self):
        self.assertAlmostEqual(noise_variance(0), 0.5)
        self.assertAlmostEqual(noise_variance(0, bit_energy=4), 2)
        quiet = awgn(np.zeros(300000), 0, rng=np.random.default_rng(20))
        active = awgn(np.full(300000, 10.0), 0, rng=np.random.default_rng(20)) - 10
        np.testing.assert_allclose(quiet, active, atol=1e-14)
        self.assertLess(abs(float(np.var(quiet)) - 0.5), 0.005)

    def test_unit_energy_matched_filter_preserves_noise_variance(self):
        noise = awgn(np.zeros(400000), 2, rng=np.random.default_rng(99))
        matched = apply_filter(noise, root_raised_cosine(81, 8, 0.35))[80:-80]
        self.assertLess(abs(float(np.var(matched)) - noise_variance(2)), 0.01)

    def test_invalid_filter_and_channel_parameters(self):
        for taps, sps, beta in [(80, 8, 0.35), (81, 0, 0.35), (81, 8, -1), (81, 8, math.nan)]:
            with self.subTest(taps=taps, sps=sps, beta=beta):
                with self.assertRaises(ValueError):
                    root_raised_cosine(taps, sps, beta)
        with self.assertRaises(ValueError):
            awgn(np.array([1 + 1j]), 0, rng=np.random.default_rng(1))


class TimingTests(unittest.TestCase):
    def test_every_integer_offset_recovers_unseen_noiseless_payload(self):
        payload = generate_bits(400, seed=194)
        for offset in range(8):
            with self.subTest(offset=offset):
                received = receive_frame(payload, None, rng=np.random.default_rng(0), timing_offset=offset)
                self.assertEqual(received.phase, offset)
                np.testing.assert_array_equal(received.detected_bits, payload)

    def test_timing_score_is_independent_of_unknown_payload(self):
        a = receive_frame(np.zeros(200, dtype=int), 0, rng=np.random.default_rng(33), timing_offset=5)
        b = receive_frame(np.ones(200, dtype=int), 0, rng=np.random.default_rng(33), timing_offset=5)
        self.assertEqual(a.phase, b.phase)
        self.assertEqual(a.correlation, b.correlation)
        self.assertFalse(np.array_equal(a.detected_bits, b.detected_bits))

    def test_noise_only_and_missing_preamble_are_rejected(self):
        training = bpsk_modulate(generate_bits(1000, seed=1))
        for samples in (np.zeros(8010), np.random.default_rng(5).normal(size=8010)):
            with self.subTest(noise=bool(np.any(samples))):
                with self.assertRaises(TimingAcquisitionError):
                    estimate_timing(samples, training, 8, 0)
        with self.assertRaises(ValueError):
            estimate_timing(np.zeros(10), training, 8, 0)

    def test_failed_acquisition_aborts_ber_instead_of_discarding_frame(self):
        with self.assertRaises(TimingAcquisitionError):
            simulate_full_chain(100, -80, seed=5)

    def test_truncated_payload_cannot_report_optimistic_partial_ber(self):
        with self.assertRaises(ValueError):
            sample_bpsk_from_waveform(np.ones(20), 8, 0, 4)


class BerStatisticsTests(unittest.TestCase):
    def test_theory_has_known_bpsk_reference_values(self):
        self.assertAlmostEqual(bpsk_theory(0), 0.07864960352514255, places=14)
        self.assertAlmostEqual(bpsk_theory(6), 0.0023882907809328075, places=14)

    def test_wilson_reference_interval_and_valid_range(self):
        lower, upper = ber_interval(10, 100)
        self.assertAlmostEqual(lower, 0.0552291370606751, places=12)
        self.assertAlmostEqual(upper, 0.174365661504913, places=12)
        self.assertEqual(ber_interval(100, 100)[1], 1)
        for errors, bits in [(-1, 10), (11, 10), (0, 0)]:
            with self.assertRaises(ValueError):
                ber_interval(errors, bits)

    def test_zero_error_bound_has_exact_one_sided_coverage_equation(self):
        lower, upper = ber_interval(0, 1000)
        self.assertEqual(lower, 0)
        self.assertGreater(upper, 0)
        self.assertAlmostEqual((1 - upper) ** 1000, 0.05, places=12)

    def test_symbol_rate_matches_theory_with_sampling_tolerance(self):
        for ebn0 in (-2, 0, 2, 4, 6):
            with self.subTest(ebn0=ebn0):
                result = simulate_symbol_rate(100000, ebn0, seed=100 + ebn0)
                expected = bpsk_theory(ebn0)
                tolerance = 6 * math.sqrt(expected * (1 - expected) / result.bits)
                self.assertLess(abs(result.ber - expected), tolerance)

    def test_full_chain_matches_theory_at_different_oversampling_rates(self):
        for sps in (4, 8, 16):
            with self.subTest(sps=sps):
                result = simulate_full_chain(50000, 4, seed=2026, sps=sps, num_taps=10 * sps + 1)
                expected = bpsk_theory(4)
                tolerance = 6 * math.sqrt(expected * (1 - expected) / result.bits)
                self.assertLess(abs(result.ber - expected), tolerance)
                self.assertEqual(result.bits, 50000)

    def test_frame_boundaries_and_reproducibility_preserve_all_payload_bits(self):
        first = simulate_full_chain(10003, 6, seed=21, frame_bits=4000)
        self.assertEqual(first.bits, 10003)
        self.assertEqual(first.frames, 3)
        self.assertEqual(first, simulate_full_chain(10003, 6, seed=21, frame_bits=4000))

    def test_zero_observed_errors_are_reported_as_a_bound(self):
        result = simulate_symbol_rate(1000, 30, seed=1)
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.ber, 0)
        self.assertGreater(result.record()["upper_95"], 0)
        self.assertEqual(result.record()["interval"], "one-sided zero-error bound")


if __name__ == "__main__":
    unittest.main()
