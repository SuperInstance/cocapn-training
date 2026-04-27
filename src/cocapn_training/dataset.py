"""Dataset management for PLATO training tiles."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence


class Dataset:
    """A collection of training tiles with load/save/filter/split operations."""

    def __init__(self, tiles: list[dict] | None = None):
        self._tiles: list[dict] = list(tiles) if tiles is not None else []

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @classmethod
    def load_jsonl(cls, path: str | Path) -> Dataset:
        """Load tiles from a JSONL file."""
        tiles: list[dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tiles.append(json.loads(line))
        return cls(tiles)

    def save_jsonl(self, path: str | Path) -> None:
        """Save tiles to a JSONL file."""
        with open(path, "w", encoding="utf-8") as fh:
            for tile in self._tiles:
                fh.write(json.dumps(tile, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Core ops
    # ------------------------------------------------------------------
    def shuffle(self, seed: int | None = None) -> Dataset:
        """Return a new Dataset with tiles shuffled."""
        rng = random.Random(seed)
        tiles = self._tiles.copy()
        rng.shuffle(tiles)
        return Dataset(tiles)

    def filter(self, predicate: Callable[[dict], bool]) -> Dataset:
        """Return a new Dataset containing only tiles matching *predicate*."""
        return Dataset([t for t in self._tiles if predicate(t)])

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Shuffle and split into train/val/test Datasets.

        Ratios are normalised so they need not sum to 1.0.
        """
        total = train + val + test
        if total == 0:
            raise ValueError("split ratios must sum to > 0")
        train_ratio = train / total
        val_ratio = val / total

        shuffled = self.shuffle(seed=seed)
        n = len(shuffled._tiles)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_tiles = shuffled._tiles[:n_train]
        val_tiles = shuffled._tiles[n_train : n_train + n_val]
        test_tiles = shuffled._tiles[n_train + n_val :]
        return Dataset(train_tiles), Dataset(val_tiles), Dataset(test_tiles)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._tiles)

    @property
    def domains(self) -> list[str]:
        """Unique domains present in the dataset."""
        return sorted({t.get("domain", "") for t in self._tiles})

    @property
    def confidence_distribution(self) -> dict[float, int]:
        """Mapping of confidence score -> count."""
        counter: Counter = Counter()
        for t in self._tiles:
            conf = t.get("confidence")
            if conf is not None:
                counter[round(float(conf), 4)] += 1
        return dict(counter)

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._tiles)

    def __iter__(self):
        return iter(self._tiles)

    def __getitem__(self, index: int) -> dict:
        return self._tiles[index]

    def to_list(self) -> list[dict]:
        """Return a shallow copy of the internal tile list."""
        return self._tiles.copy()
