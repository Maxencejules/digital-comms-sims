"""Error-count reporting; no zero-error measurement is called zero true BER."""
from dataclasses import dataclass
from math import erfc, expm1, log, sqrt
from statistics import NormalDist


def bpsk_theory(ebn0_db: float) -> float:
    return 0.5 * erfc(sqrt(10 ** (ebn0_db / 10)))


def ber_interval(errors: int, bits: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson two-sided interval; zero errors use an exact one-sided upper bound.

    Both use a binomial (independent Bernoulli errors) model. For finite RRC
    filters/shared frame timing these are nominal approximate uncertainties.
    """
    if not isinstance(errors, int) or not isinstance(bits, int) or bits < 1 or not 0 <= errors <= bits:
        raise ValueError("Expected integer counts with 0 <= errors <= bits and bits > 0")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if errors == 0:
        return 0.0, -expm1(log(1 - confidence) / bits)
    p = errors / bits
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    denominator = 1 + z * z / bits
    center = (p + z * z / (2 * bits)) / denominator
    radius = z * sqrt(p * (1 - p) / bits + z * z / (4 * bits * bits)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class BerMeasurement:
    ebn0_db: float
    errors: int
    bits: int
    seed: int
    frames: int = 1
    timing_phase_mismatches: int = 0

    @property
    def ber(self) -> float:
        return self.errors / self.bits

    @property
    def interval(self) -> tuple[float, float]:
        return ber_interval(self.errors, self.bits)

    def record(self) -> dict:
        lower, upper = self.interval
        return {"ebn0_db": self.ebn0_db, "errors": self.errors, "bits": self.bits,
                "ber": self.ber, "lower_95": lower, "upper_95": upper,
                "interval": "one-sided zero-error bound" if self.errors == 0 else "Wilson two-sided",
                "theory": bpsk_theory(self.ebn0_db), "seed": self.seed, "frames": self.frames,
                "timing_phase_mismatches": self.timing_phase_mismatches}
