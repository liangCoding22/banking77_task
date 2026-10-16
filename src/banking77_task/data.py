from __future__ import annotations

import json
import random
from dataclasses import dataclass

from datasets import load_dataset
from huggingface_hub import hf_hub_download

HF_DATASET = "PolyAI/banking77"
# The Hub's main branch only has a loading script, which `datasets` no longer runs;
# this PR revision holds the same data as parquet.
HF_REVISION = "refs/pr/6"
DEV_FRACTION = 0.2


def normalize_label(name: str) -> str:
    """Canonical intent name: lowercase. The source has one capitalized intent (Refund_not_showing_up)."""
    return name.lower()


def load_label_mapping() -> tuple[dict[int, str], dict[str, int]]:
    """(id2label, label2id) from the dataset repo's dataset_infos.json, names lowercased.

    Label names are the intents. Everything downstream (prompt, shots, parser, reports) uses these.
    """
    path = hf_hub_download(HF_DATASET, "dataset_infos.json", repo_type="dataset")
    with open(path) as f:
        names = json.load(f)["default"]["features"]["label"]["names"]
    id2label = {i: normalize_label(n) for i, n in enumerate(names)}
    assert len(set(id2label.values())) == len(id2label), "lowercasing made two intents collide"
    return id2label, {name: i for i, name in id2label.items()}


@dataclass(frozen=True)
class Example:
    uid: str
    text: str
    label: int


def load_label_names_and_splits() -> tuple[list[str], list[Example], list[Example]]:
    """(label names, train, test) as Example lists."""
    ds = load_dataset(HF_DATASET, revision=HF_REVISION)
    id2label, _ = load_label_mapping()
    names = [id2label[i] for i in range(len(id2label))]
    assert names == [normalize_label(n) for n in ds["train"].features["label"].names], "dataset_infos.json disagrees with parquet labels"
    def rows(split: str) -> list[Example]:
        return [Example(f"{split}-{i}", r["text"], r["label"]) for i, r in enumerate(ds[split])]
    return names, rows("train"), rows("test")


def split_train_dev(
    train: list[Example], dev_fraction: float = DEV_FRACTION, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Random (train, dev) split of the original training set; dev is `dev_fraction` of it."""
    idx = list(range(len(train)))
    random.Random(seed).shuffle(idx)
    n_dev = round(len(train) * dev_fraction)
    dev_idx = set(idx[:n_dev])
    dev = [train[i] for i in sorted(dev_idx)]
    rest = [train[i] for i in range(len(train)) if i not in dev_idx]
    return rest, dev
