"""banking77 make-split | prepare | submit | report

make-split: freeze the dev set and few-shot examples into data/split.json (run once).
prepare: build the dev set and the batch input JSONL locally (no network calls to Fireworks).
submit:  upload the prepared JSONL and create a Fireworks batch job (costs money).
report:  download a finished job's output, parse labels, write predictions + metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from banking77_task import prompts
from banking77_task.data import DEV_FRACTION, load_label_mapping, load_label_names_and_splits
from banking77_task.fireworks import DEFAULT_MODEL, BatchClient, build_batch_jsonl, parse_batch_output
from banking77_task.metrics import compute_metrics, per_intent_table
from banking77_task.parsers import parse_label
from banking77_task.split import SHOTS_PER_INTENT, load_split, make_split, save_split, select


def run_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / args.model.split("/")[-1] / prompts.VERSION


def cmd_make_split(args: argparse.Namespace) -> None:
    names, train, _ = load_label_names_and_splits()
    split = make_split(train, len(names), args.dev_fraction, args.shots_per_intent, args.seed)
    save_split(split)
    print(
        f"data/split.json: dev={len(split['dev_uids'])} fewshot={len(split['fewshot_uids'])} "
        f"train={len(split['train_uids'])}"
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    names, train, _ = load_label_names_and_splits()
    split = load_split()
    dev = select(train, split["dev_uids"])
    shots = select(train, split["fewshot_uids"])
    if args.limit:
        # Train is sorted by label, so a prefix would be one intent; sample at random instead.
        dev = sorted(random.Random(args.seed).sample(dev, args.limit), key=lambda e: int(e.uid.split("-")[1]))
    id2label, _ = load_label_mapping()
    rows = [
        {"custom_id": ex.uid, "messages": prompts.build_messages(ex.text, id2label, args.thinking, shots)} for ex in dev
    ]
    out = run_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.jsonl").write_text(build_batch_jsonl(rows, args.max_tokens, args.temperature))
    (out / "dev.jsonl").write_text(
        "\n".join(json.dumps({"uid": e.uid, "text": e.text, "label": e.label, "label_name": names[e.label]}) for e in dev)
    )
    print(f"{len(dev)} dev examples -> {out}/input.jsonl (nothing submitted)")
    print("sample prompt:\n" + rows[0]["messages"][0]["content"][-600:] + "\n---\n" + rows[0]["messages"][1]["content"])


def cmd_submit(args: argparse.Namespace) -> None:
    out = run_dir(args)
    jsonl = (out / "input.jsonl").read_text()
    job_id = BatchClient(args.account_id).submit(args.model, jsonl, jsonl.count("\n") + 1)
    (out / "job_id.txt").write_text(job_id)
    print(f"submitted {job_id}")


def cmd_report(args: argparse.Namespace) -> None:
    out = run_dir(args)
    job_id = args.job_id or (out / "job_id.txt").read_text().strip()
    client = BatchClient(args.account_id)
    job = client.wait(job_id) if args.wait else client.get_job(job_id)
    state = client.state(job)
    if state != "COMPLETED":
        raise SystemExit(f"job {job_id} is {state}")
    raw = client.download_output(job)
    (out / "output.jsonl").write_text(raw)
    names, _, _ = load_label_names_and_splits()
    results = parse_batch_output(raw)
    dev = [json.loads(l) for l in (out / "dev.jsonl").read_text().splitlines()]
    preds, finishes = [], []
    with (out / "predictions.jsonl").open("w") as f:
        for ex in dev:
            text, error, finish = results.get(ex["uid"], ("", "missing from batch output", None))
            pred = parse_label(text, names)
            preds.append(pred)
            finishes.append(finish)
            f.write(json.dumps({**ex, "reply": text, "error": error, "finish_reason": finish, "pred": pred}) + "\n")
    gold = [e["label"] for e in dev]
    metrics = compute_metrics(gold, preds, names)
    # Anything but "stop" (e.g. "length") means the reply may be cut off, so unparsed can be an artifact.
    metrics["finish_reasons"] = dict(Counter(finishes))
    table = per_intent_table(gold, preds, names)
    with (out / "per_intent.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("\nworst intents (accuracy, n, top confusion):")
    for r in table[:15]:
        print(f"  {r['accuracy']:.2f}  n={r['n']:<3} {r['intent']} -> {r['top_confusion']} ({r['top_confusion_count']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("make-split", cmd_make_split), ("prepare", cmd_prepare), ("submit", cmd_submit), ("report", cmd_report)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        if name == "make-split":
            s.add_argument("--dev-fraction", type=float, default=DEV_FRACTION)
            s.add_argument("--shots-per-intent", type=int, default=SHOTS_PER_INTENT)
            s.add_argument("--seed", type=int, default=0)
            continue
        s.add_argument("--model", default=DEFAULT_MODEL)
        s.add_argument("--output-dir", default="outputs")
        if name == "prepare":
            s.add_argument("--seed", type=int, default=0, help="seed for --limit sampling")
            s.add_argument("--limit", type=int, default=None, help="first N dev examples (smoke test)")
            s.add_argument("--max-tokens", type=int, default=32)
            s.add_argument("--temperature", type=float, default=0.0)
            s.add_argument("--thinking", action="store_true", help="leave Qwen3 thinking on (raise --max-tokens)")
        else:
            s.add_argument("--account-id", required=True)
        if name == "report":
            s.add_argument("--job-id")
            s.add_argument("--wait", action="store_true")
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
