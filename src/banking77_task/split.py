"""Frozen data split: dev set, few-shot prompt examples, and the remaining training pool.

Created once by `banking77 make-split` and committed as data/split.json. Every later run loads it,
so the dev set and the few-shot examples never change. Few-shot examples are drawn from the
non-dev part of the training set and are excluded from `train_uids`, so a future training run
cannot see the examples that live in the prompt.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from banking77_task.data import DEV_FRACTION, HF_DATASET, HF_REVISION, Example, split_train_dev

SPLIT_PATH = Path("data/split.json")
SHOTS_PER_INTENT = 2


def make_split(
    train: Sequence[Example],
    num_labels: int,
    dev_fraction: float = DEV_FRACTION,
    shots_per_intent: int = SHOTS_PER_INTENT,
    seed: int = 0,
) -> dict:
    rest, dev = split_train_dev(list(train), dev_fraction, seed)
    rng = random.Random(seed)
    shots: list[Example] = []
    for label in range(num_labels):
        pool = [e for e in rest if e.label == label]
        shots.extend(rng.sample(pool, shots_per_intent))
    rng.shuffle(shots)  # train is label-sorted; mix intents so the prompt has no label blocks
    shot_uids = [e.uid for e in shots]
    used = set(shot_uids)
    return {
        "dataset": HF_DATASET,
        "revision": HF_REVISION,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "shots_per_intent": shots_per_intent,
        "dev_uids": sorted((e.uid for e in dev), key=_index),
        "fewshot_uids": shot_uids,  # prompt order
        "train_uids": [e.uid for e in rest if e.uid not in used],
    }


def _index(uid: str) -> int:
    return int(uid.split("-")[1])


def save_split(split: dict, path: Path = SPLIT_PATH) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; the split is frozen. Delete it deliberately to remake it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))


def load_split(path: Path = SPLIT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `banking77 make-split` once first.")
    split = json.loads(path.read_text())
    assert split["revision"] == HF_REVISION, "split was made for a different dataset revision"
    return split


def select(examples: Sequence[Example], uids: Sequence[str]) -> list[Example]:
    """Examples for `uids`, in the order of `uids`."""
    by_uid = {e.uid: e for e in examples}
    return [by_uid[u] for u in uids]
