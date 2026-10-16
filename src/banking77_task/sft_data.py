"""Chat-format SFT data for the cookbook's sft_loop, built from the frozen split.

Rows: train = `train_uids` only (dev and the few-shot examples are excluded, asserted below);
dev_eval = `dev_uids`, used by the trainer for eval loss only, never trained on.

Each row is {"messages": [system, user, assistant]}:
- system/user are prompt v3 (no shots, no `/no_think`), exactly what inference sends to an SFT model,
  so the trained prompt and the served prompt are byte-identical. The shots are dropped: SFT teaches the
  label space, and 154 examples would add ~3.7k tokens to every row.
- assistant content is the bare intent name, with no reasoning_content. With the cookbook's `qwen3`
  renderer this trains `<|im_start|>assistant\\n{intent}<|im_end|>`, which is the prompt the server's
  default chat template builds for a batch/chat request. The loss covers the intent and <|im_end|>.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from banking77_task import prompts
from banking77_task.data import load_label_mapping, load_label_names_and_splits
from banking77_task.split import load_split, select

DEFAULT_OUTPUT = Path("outputs/sft_data")
TRAIN_FILE, DEV_FILE, SUMMARY_FILE = "train_sft.jsonl", "dev_sft.jsonl", "summary.json"


def to_row(text: str, label: int, id2label: dict[int, str]) -> dict[str, Any]:
    messages = prompts.build_messages(text, id2label, no_think=False, shots=None)  # = prompt v3
    messages.append({"role": "assistant", "content": id2label[label]})
    return {"messages": messages}


def check_disjoint(split: dict) -> None:
    train, dev, shots = set(split["train_uids"]), set(split["dev_uids"]), set(split["fewshot_uids"])
    assert not train & dev, "train overlaps dev"
    assert not train & shots, "train overlaps few-shot examples"
    assert not dev & shots, "dev overlaps few-shot examples"


def build(output_dir: Path) -> dict[str, Any]:
    split = load_split()
    check_disjoint(split)
    _, train_all, _ = load_label_names_and_splits()
    id2label, _ = load_label_mapping()
    train, dev = select(train_all, split["train_uids"]), select(train_all, split["dev_uids"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in ((TRAIN_FILE, train), (DEV_FILE, dev)):
        with (output_dir / name).open("w") as f:
            for e in examples:
                f.write(json.dumps(to_row(e.text, e.label, id2label), ensure_ascii=False) + "\n")
    counts = Counter(id2label[e.label] for e in train)
    summary = {
        "prompt_version": "v3",
        "train_rows": len(train),
        "dev_rows": len(dev),
        "per_intent_train_min": min(counts.values()),
        "per_intent_train_max": max(counts.values()),
        "intents_in_train": len(counts),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build SFT data (train excl. dev and few-shot examples).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output_dir), indent=2))
    print(f"Wrote {args.output_dir}/{{{TRAIN_FILE},{DEV_FILE}}}")


if __name__ == "__main__":
    main()
