"""LoRA SFT with the Fireworks cookbook's sft_loop (`training/recipes/sft_loop.py`).

The loop lives in the cookbook, which is not part of this package (needs Python >= 3.11):

    git clone https://github.com/fw-ai/cookbook.git
    cd cookbook/training && pip install -e .
    export PYTHONPATH=/path/to/cookbook:/path/to/cookbook/training:$PYTHONPATH

Settings:
- Data from `banking77-sft-data`: train rows exclude the dev set and the few-shot examples.
- Renderer `qwen3`, loss on the assistant turn only (intent + <|im_end|>). See sft_data.py.
- The dev rows are passed as `evaluation_dataset` for eval loss each epoch; no train rows are carved out.
- A promotable checkpoint is saved at the end of every epoch, so one run covers all epoch counts
  up to `--epochs`; pick the epoch by dev accuracy afterwards.
- The trainer job id goes to `<log-path>/trainer_job.json`; `sft_job.json` maps checkpoints to epochs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

SFT_DATA = Path("outputs/sft_data")
BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
TOKENIZER_MODEL = "Qwen/Qwen3-4B"
RENDERER_NAME = "qwen3"
TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"
TRAIN_ON_WHAT = "all_assistant_messages"
SFT_JOB_FILE = "sft_job.json"
TRAINER_JOB_FILE = "trainer_job.json"
# Rows are ~700 tokens (system prompt with 77 intents + a short message); 1024 leaves headroom.
MAX_SEQ_LEN = 1024


def count_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def checkpoint_epochs(spe: int, epochs: int) -> dict[str, int]:
    """Checkpoint name -> epoch it ends, e.g. {"step-245": 1, "step-490": 2}."""
    return {f"step-{k * spe}": k for k in range(1, epochs + 1)}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    rows = count_rows(args.data_dir / "train_sft.jsonl")
    spe = math.ceil(rows / args.batch_size)
    return {
        "run_name": args.run_name,
        "train_rows": rows,
        "eval_rows": count_rows(args.data_dir / "dev_sft.jsonl"),
        "steps_per_epoch": spe,
        "total_steps": spe * args.epochs,
        "checkpoints": checkpoint_epochs(spe, args.epochs),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": MAX_SEQ_LEN,
        "renderer_name": RENDERER_NAME,
        "output_model_id": args.output_model_id,
    }


def build_config(args: argparse.Namespace, spe: int, log_path: Path) -> Any:
    from recipes.sft_loop import Config
    from training.utils import TrainerConfig, WandBConfig

    return Config(
        log_path=str(log_path),
        base_model=BASE_MODEL,
        dataset=str(args.data_dir / "train_sft.jsonl"),
        evaluation_dataset=str(args.data_dir / "dev_sft.jsonl"),
        tokenizer_model=TOKENIZER_MODEL,
        renderer_name=RENDERER_NAME,
        train_on_what=TRAIN_ON_WHAT,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=MAX_SEQ_LEN,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        sampler_save_interval=spe,
        save_final_checkpoint=True,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(training_shape_id=TRAINING_SHAPE),
        wandb=WandBConfig(project="banking77-sft"),
    )


def install_trainer_job_patch(log_path: Path) -> None:
    """Record the trainer job id in `log_path/trainer_job.json` (so a crashed run can be cleaned up)."""
    from training.utils import checkpoints as cp

    if getattr(cp.TrainingCheckpoints, "_b77_patched", False):
        return
    original_init = cp.TrainingCheckpoints.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if kwargs.get("trainer_id"):
            log_path.mkdir(parents=True, exist_ok=True)
            (log_path / TRAINER_JOB_FILE).write_text(json.dumps({"trainer_job_id": kwargs["trainer_id"]}, indent=2))

    cp.TrainingCheckpoints.__init__ = init
    cp.TrainingCheckpoints._b77_patched = True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT on BANKING77 (cookbook sft_loop).")
    p.add_argument("--run-name", required=True, help="log dir name under outputs/sft_runs/")
    p.add_argument("--data-dir", type=Path, default=SFT_DATA)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32, help="sequences per optimizer step")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None, help="default: 2 * rank")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-model-id", default=None, help="promote the final checkpoint to this model id")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit without training")
    args = p.parse_args(argv)
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_rank
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = plan(args)
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    from recipes.sft_loop import main as run_sft

    log_path = Path("outputs/sft_runs") / args.run_name
    install_trainer_job_patch(log_path)
    result = run_sft(build_config(args, summary["steps_per_epoch"], log_path))
    job = {**summary, "job_id": result["job_id"], "steps": result["steps"]}
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / SFT_JOB_FILE).write_text(json.dumps(job, indent=2))
    print(f"Wrote {log_path / SFT_JOB_FILE}")


if __name__ == "__main__":
    main()
