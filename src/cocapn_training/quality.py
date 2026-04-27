"""Quality scoring for individual tiles and entire Datasets."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DatasetReport:
    """Aggregated quality metrics for a Dataset."""

    min_quality: float
    max_quality: float
    mean_quality: float
    median_quality: float
    low_quality_count: int
    domain_coverage: float  # fraction of known domains present


def _normalise(value: float, min_val: float, max_val: float) -> float:
    """Clamp and scale *value* to [0, 1]."""
    if max_val == min_val:
        return 1.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


# ---------------------------------------------------------------------------
# Known domains for coverage calculation
# ---------------------------------------------------------------------------
_KNOWN_DOMAINS = {
    "navigation",
    "manipulation",
    "perception",
    "planning",
    "dialogue",
    "safety",
    "math",
    "coding",
}


def score_tile(tile: dict, target_domain: str | None = None) -> float:
    """Return a quality score in [0, 1] for a single tile.

    Factors:
      * question length (ideal ~20–100 chars)
      * answer length   (ideal ~30–300 chars)
      * confidence      (higher is better)
      * domain match    (1.0 if target_domain is None or matches)
    """
    q = tile.get("question", "")
    a = tile.get("answer", "")
    conf = float(tile.get("confidence", 0.0))
    domain = tile.get("domain", "")

    q_len = len(q)
    a_len = len(a)

    # Length scores — peaking in the "ideal" band
    q_score = 1.0 - abs(q_len - 60) / 100.0
    a_score = 1.0 - abs(a_len - 165) / 300.0

    # Confidence score (already 0–1)
    c_score = conf

    # Domain match
    if target_domain is None or domain == target_domain:
        d_score = 1.0
    else:
        d_score = 0.0

    # Weighted average
    total = (
        0.20 * _normalise(q_score, 0.0, 1.0)
        + 0.25 * _normalise(a_score, 0.0, 1.0)
        + 0.35 * c_score
        + 0.20 * d_score
    )
    return round(max(0.0, min(1.0, total)), 4)


def score_dataset(dataset) -> DatasetReport:
    """Compute aggregate quality metrics for a *Dataset* instance."""
    from cocapn_training.dataset import Dataset

    if not isinstance(dataset, Dataset):
        raise TypeError("expected Dataset instance")

    scores = [score_tile(t) for t in dataset]
    if not scores:
        return DatasetReport(
            min_quality=0.0,
            max_quality=0.0,
            mean_quality=0.0,
            median_quality=0.0,
            low_quality_count=0,
            domain_coverage=0.0,
        )

    low_threshold = 0.4
    low_quality_count = sum(1 for s in scores if s < low_threshold)

    present_domains = {t.get("domain", "") for t in dataset}
    domain_coverage = len(present_domains & _KNOWN_DOMAINS) / max(len(_KNOWN_DOMAINS), 1)

    return DatasetReport(
        min_quality=round(min(scores), 4),
        max_quality=round(max(scores), 4),
        mean_quality=round(statistics.mean(scores), 4),
        median_quality=round(statistics.median(scores), 4),
        low_quality_count=low_quality_count,
        domain_coverage=round(domain_coverage, 4),
    )
