# cocapn-training

Training data management for the Cocapn Fleet — load, augment, convert, and quality-check PLATO training tiles.

## What This Gives You

- **Dataset** — load/save/filter/split JSONL tile collections
- **Augmentation** — generate synthetic training variations
- **Converter** — transform tiles between formats
- **Quality checks** — validate tile completeness and distribution
- **Stratified splitting** — train/val/test splits with label balancing

## Quick Start

```bash
pip install cocapn-training

from cocapn_training import Dataset

ds = Dataset.load_jsonl("tiles.jsonl")
print(f"Loaded {len(ds)} tiles")

train, val, test = ds.split(ratios=(0.8, 0.1, 0.1))
train.save_jsonl("train.jsonl")
```

## How It Fits

The training infrastructure for the Cocapn Fleet. Part of the SuperInstance ecosystem.

Related repos:
- [cocapn-plato](https://github.com/SuperInstance/cocapn-plato) — PLATO framework
- [cocapn-curriculum](https://github.com/SuperInstance/cocapn-curriculum) — curriculum management
- [cocapn-pipeline](https://github.com/SuperInstance/cocapn-pipeline) — data pipeline

## License

Apache 2.0
