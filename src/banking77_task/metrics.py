from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence


def compute_metrics(
    gold: Sequence[int], pred: Sequence[Optional[int]], label_names: Sequence[str]
) -> dict:
    """Accuracy (unparsed counts as wrong), unparsed rate, and macro-F1 over all labels."""
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            if p is not None:
                fp[p] += 1
    f1s = []
    for i in range(len(label_names)):
        denom = 2 * tp[i] + fp[i] + fn[i]
        f1s.append(2 * tp[i] / denom if denom else 0.0)
    per_label_errors = Counter(label_names[g] for g, p in zip(gold, pred) if g != p)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(f1s),
        "unparsed": sum(p is None for p in pred),
        "worst_labels": per_label_errors.most_common(10),
    }


def per_intent_table(
    gold: Sequence[int], pred: Sequence[Optional[int]], label_names: Sequence[str]
) -> list[dict]:
    """One row per intent that appears in gold: support, correct, accuracy, most common wrong reply."""
    rows = []
    for i, name in enumerate(label_names):
        wrong = Counter(
            "<unparsed>" if p is None else label_names[p] for g, p in zip(gold, pred) if g == i and p != i
        )
        n = sum(g == i for g in gold)
        if not n:
            continue
        top = wrong.most_common(1)
        rows.append(
            {
                "intent": name,
                "n": n,
                "correct": n - sum(wrong.values()),
                "accuracy": (n - sum(wrong.values())) / n,
                "top_confusion": top[0][0] if top else "",
                "top_confusion_count": top[0][1] if top else 0,
            }
        )
    return sorted(rows, key=lambda r: (r["accuracy"], -r["n"]))
