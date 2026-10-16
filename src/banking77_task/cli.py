"""banking77 make-split | prepare | submit | report | chat-eval | classify

make-split: freeze the dev set and few-shot examples into data/split.json (run once).
prepare: build the dev set and the batch input JSONL locally (no network calls to Fireworks).
submit:  upload the prepared JSONL and create a Fireworks batch job (costs money).
chat-eval: score a deployment-served model with concurrent chat calls (no batch).
report:  download a finished job's output, parse labels, write predictions + metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

from banking77_task import prompts
from banking77_task.data import DEV_FRACTION, load_label_mapping, load_label_names_and_splits
from banking77_task.chat import complete_all, complete_one, read_input_rows
from banking77_task.fireworks import DEFAULT_MODEL, BatchClient, build_batch_jsonl, parse_batch_output
from banking77_task.metrics import compute_metrics, per_intent_table
from banking77_task.parsers import parse_label
from banking77_task.split import SHOTS_PER_INTENT, load_split, make_split, save_split, select


def run_dir(args: argparse.Namespace) -> Path:
    suffix = "-test" if args.eval_split == "test" else ""  # dev keeps the original paths
    return Path(args.output_dir) / args.model.split("/")[-1] / f"{args.prompt_version}{suffix}"


def cmd_make_split(args: argparse.Namespace) -> None:
    names, train, _ = load_label_names_and_splits()
    split = make_split(train, len(names), args.dev_fraction, args.shots_per_intent, args.seed)
    save_split(split)
    print(
        f"data/split.json: dev={len(split['dev_uids'])} fewshot={len(split['fewshot_uids'])} "
        f"train={len(split['train_uids'])}"
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    names, train, test = load_label_names_and_splits()
    split = load_split()
    dev = test if args.eval_split == "test" else select(train, split["dev_uids"])
    shots = select(train, split["fewshot_uids"]) if prompts.uses_shots(args.prompt_version) else None
    if args.limit:
        # Train is sorted by label, so a prefix would be one intent; sample at random instead.
        dev = sorted(random.Random(args.seed).sample(dev, args.limit), key=lambda e: int(e.uid.split("-")[1]))
    id2label, _ = load_label_mapping()
    no_think = prompts.uses_no_think(args.prompt_version) and not args.thinking
    rows = [
        {"custom_id": ex.uid, "messages": prompts.build_messages(ex.text, id2label, no_think, shots)} for ex in dev
    ]
    out = run_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.jsonl").write_text(build_batch_jsonl(rows, args.max_tokens, args.temperature))
    (out / "eval.jsonl").write_text(
        "\n".join(json.dumps({"uid": e.uid, "text": e.text, "label": e.label, "label_name": names[e.label]}) for e in dev)
    )
    print(f"{len(dev)} {args.eval_split} examples -> {out}/input.jsonl (nothing submitted)")
    print("sample prompt:\n" + rows[0]["messages"][0]["content"][-600:] + "\n---\n" + rows[0]["messages"][1]["content"])


def cmd_submit(args: argparse.Namespace) -> None:
    out = run_dir(args)
    jsonl = (out / "input.jsonl").read_text()
    job_id = BatchClient(args.account_id).submit(args.model, jsonl, jsonl.count("\n") + 1)
    (out / "job_id.txt").write_text(job_id)
    print(f"submitted {job_id}")


def write_report(out: Path, results: dict) -> None:
    """Score `results` (uid -> (reply, error, finish_reason)) against the run's eval file and write reports."""
    names, _, _ = load_label_names_and_splits()
    eval_file = out / "eval.jsonl" if (out / "eval.jsonl").exists() else out / "dev.jsonl"  # dev.jsonl: older runs
    dev = [json.loads(l) for l in eval_file.read_text().splitlines()]
    preds, finishes = [], []
    with (out / "predictions.jsonl").open("w") as f:
        for ex in dev:
            text, error, finish = results.get(ex["uid"], ("", "missing from output", None))
            pred = parse_label(text, names)
            preds.append(pred)
            finishes.append(finish)
            f.write(json.dumps({**ex, "reply": text, "error": error, "finish_reason": finish, "pred": pred}) + "\n")
    gold = [e["label"] for e in dev]
    metrics = compute_metrics(gold, preds, names)
    # Anything but "stop" (e.g. "length") means the reply may be cut off, so unparsed can be an artifact.
    metrics["finish_reasons"] = dict(Counter(finishes))
    metrics["errors"] = sum(1 for r in results.values() if r[1])
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
    write_report(out, parse_batch_output(raw))


def cmd_chat_eval(args: argparse.Namespace) -> None:
    """Score a model served from a deployment, e.g. 'accounts/<acct>/models/<id>#accounts/<acct>/deployments/<dep>'."""
    out = run_dir(args)
    rows = read_input_rows(out / "input.jsonl")
    previous: dict = {}
    if args.only_failed:  # rerun rows whose request failed last time, keep the rest
        for line in (out / "chat_output.jsonl").read_text().splitlines():
            r = json.loads(line)
            previous[r["custom_id"]] = (r["reply"], r["error"], r["finish_reason"])
        rows = [row for row in rows if previous.get(row["custom_id"], ("", "missing"))[1]]
        print(f"rerunning {len(rows)} failed rows")
    results = complete_all(os.environ.get("B77_SERVED_MODEL", args.model), rows, args.concurrency)  # served name may differ from the run-dir name
    results = {**previous, **results}
    (out / "chat_output.jsonl").write_text("\n".join(json.dumps({"custom_id": k, "reply": v[0], "error": v[1], "finish_reason": v[2]}) for k, v in results.items()))
    write_report(out, results)


def cmd_classify(args: argparse.Namespace) -> None:
    """Classify one or more messages with a served model (serverless base model, or '<model>#<deployment>')."""
    import requests

    id2label, _ = load_label_mapping()
    names = [id2label[i] for i in range(len(id2label))]
    shots = None
    if prompts.uses_shots(args.prompt_version):
        _, train, _ = load_label_names_and_splits()
        shots = select(train, load_split()["fewshot_uids"])
    no_think = prompts.uses_no_think(args.prompt_version)
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("Set FIREWORKS_API_KEY.")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"
    for text in args.texts:
        row = {"body": {"messages": prompts.build_messages(text, id2label, no_think, shots), "max_tokens": 32, "temperature": 0.0}}
        reply, error, _ = complete_one(session, args.model, row)
        if error:
            print(f"{text!r}\t<error: {error}>")
            continue
        pred = parse_label(reply, names)
        print(f"{text!r}\t{names[pred] if pred is not None else '<unparsed: ' + reply + '>'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("make-split", cmd_make_split), ("prepare", cmd_prepare), ("submit", cmd_submit), ("report", cmd_report), ("chat-eval", cmd_chat_eval), ("classify", cmd_classify)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        if name == "make-split":
            s.add_argument("--dev-fraction", type=float, default=DEV_FRACTION)
            s.add_argument("--shots-per-intent", type=int, default=SHOTS_PER_INTENT)
            s.add_argument("--seed", type=int, default=0)
            continue
        s.add_argument("--model", default=DEFAULT_MODEL)
        s.add_argument("--output-dir", default="outputs")
        s.add_argument("--eval-split", choices=("dev", "test"), default="dev",
                       help="test: the original 3,080-example test set (use once, on the final model)")
        s.add_argument("--prompt-version", choices=prompts.PROMPT_VERSIONS, default=prompts.DEFAULT_VERSION,
                       help="v1: zero-shot, v2: few-shot, v3: zero-shot without /no_think (for SFT models)")
        if name == "prepare":
            s.add_argument("--seed", type=int, default=0, help="seed for --limit sampling")
            s.add_argument("--limit", type=int, default=None, help="first N dev examples (smoke test)")
            s.add_argument("--max-tokens", type=int, default=32)
            s.add_argument("--temperature", type=float, default=0.0)
            s.add_argument("--thinking", action="store_true", help="leave Qwen3 thinking on (raise --max-tokens)")
        elif name == "classify":
            s.add_argument("texts", nargs="+", help="customer messages to classify")
        elif name == "chat-eval":
            s.add_argument("--concurrency", type=int, default=16)
            s.add_argument("--only-failed", action="store_true", help="rerun only rows that errored in chat_output.jsonl")
        else:
            s.add_argument("--account-id", required=True)
        if name == "report":
            s.add_argument("--job-id")
            s.add_argument("--wait", action="store_true")
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
