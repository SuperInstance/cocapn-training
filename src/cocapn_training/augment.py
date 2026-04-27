"""Light-weight data augmentation for PLATO tiles."""

from __future__ import annotations

import copy
import random


# ---------------------------------------------------------------------------
# Paraphrase templates
# ---------------------------------------------------------------------------
_TEMPLATES = [
    "Can you explain {q}",
    "What is the answer to: {q}",
    "Please tell me about {q}",
    "I would like to know: {q}",
    "Could you clarify {q}",
    "How would you describe {q}",
    "In your own words, what is {q}",
    "What do you know regarding {q}",
]


def paraphrase_question(tile: dict, n: int = 3, seed: int | None = None) -> list[dict]:
    """Generate *n* rephrased copies of *tile* using simple templates.

    Returns a list of new tile dicts; the original is unmodified.
    """
    rng = random.Random(seed)
    original_q = tile.get("question", "")
    templates = rng.sample(_TEMPLATES, min(n, len(_TEMPLATES)))
    results: list[dict] = []
    for tmpl in templates[:n]:
        new_tile = copy.deepcopy(tile)
        new_tile["question"] = tmpl.format(q=original_q)
        results.append(new_tile)
    return results


def vary_confidence(tile: dict, sigma: float = 0.05, seed: int | None = None) -> dict:
    """Return a new tile with Gaussian noise added to its confidence score.

    Confidence is clamped to the range [0.0, 1.0].
    """
    rng = random.Random(seed)
    new_tile = copy.deepcopy(tile)
    conf = float(new_tile.get("confidence", 0.0))
    noise = rng.gauss(0.0, sigma)
    new_tile["confidence"] = round(max(0.0, min(1.0, conf + noise)), 4)
    return new_tile


def combine_tiles(tile_a: dict, tile_b: dict) -> dict:
    """Merge two tiles, concatenating their answers and averaging confidence.

    Domain is taken from *tile_a*.  Question is taken from *tile_a*.
    """
    return {
        "question": tile_a.get("question", ""),
        "answer": f"{tile_a.get('answer', '')} {tile_b.get('answer', '')}".strip(),
        "confidence": round(
            (float(tile_a.get("confidence", 0.0)) + float(tile_b.get("confidence", 0.0))) / 2, 4
        ),
        "domain": tile_a.get("domain", ""),
    }
