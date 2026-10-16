"""Hyperparameter sweep plan for SFT, selected on the dev set.

Stage 1 varies learning rate (rank fixed, 3 epochs); stage 2 varies LoRA rank at the best LR.
Epoch count is not a sweep axis: every run saves a checkpoint per epoch, and each is scored on dev.

    banking77-sweep --stage 1                 # print the runs (dry)
    banking77-sweep --stage 2 --best-lr 1e-4
    banking77-sweep --stage 1 --launch        # run them one after another (billed)
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from banking77_task import sft_train

STAGE1_LRS = (5e-5, 1e-4, 2e-4)
STAGE1_RANK = 16
STAGE2_RANKS = (8, 32)  # rank 32 is the Fireworks maximum
EPOCHS = 3
BATCH_SIZE = 32


def runs(stage: int, best_lr: Optional[float] = None) -> list[list[str]]:
    if stage == 1:
        grid = [(lr, STAGE1_RANK) for lr in STAGE1_LRS]
    else:
        if best_lr is None:
            raise SystemExit("--best-lr is required for stage 2")
        grid = [(best_lr, r) for r in STAGE2_RANKS]
    def tag(lr: float) -> str:
        return f"{lr:.0e}".replace("e-0", "e-")  # 5e-05 -> 5e-5

    return [
        [
            "--run-name", f"lr{tag(lr)}-r{rank}",
            "--learning-rate", str(lr),
            "--lora-rank", str(rank),
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--output-model-id", f"b77-lr{tag(lr).replace('e-', 'em')}-r{rank}",
        ]
        for lr, rank in grid
    ]


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--best-lr", type=float)
    p.add_argument("--launch", action="store_true", help="actually train (default: dry run)")
    args = p.parse_args(argv)
    for run_args in runs(args.stage, args.best_lr):
        print("banking77-sft", " ".join(run_args))
        sft_train.main(run_args + ([] if args.launch else ["--dry-run"]))


if __name__ == "__main__":
    main()
