# banking77_task

Prompt-based intent classification on [BANKING77](https://huggingface.co/datasets/PolyAI/banking77)
(77 banking intents) with Qwen3-4B (`accounts/fireworks/models/qwen3-4b`), compared against a LoRA-SFT
version of the same model. Base-model inference uses Fireworks batch jobs; the tuned model is scored
through a short-lived deployment.

## Results (original 3,080-example test set)

| Model | Prompt | Accuracy | Macro-F1 |
|---|---|---|---|
| Qwen3-4B, base | v2: few-shot (2 per intent), `/no_think` | 78.34% | 0.778 |
| Qwen3-4B + LoRA SFT, epoch 2 (`b77-lr1em4-r16-ep2`) | v3: zero-shot, no `/no_think` | **91.17%** | **0.912** |

- Accuracy is micro-F1 here (single-label). Unparsed replies count as wrong.
- Base model: 23 unparsed, 15 of them the invented name `get_virtual_card`. SFT model: 6 unparsed.
- Weakest intents after SFT are near-duplicate pairs: `topping_up_by_card` (50%, confused with
  `supported_cards_and_currencies`), `balance_not_updated_after_bank_transfer` (65%),
  `top_up_by_bank_transfer_charge` (65%), `direct_debit_payment_not_recognised` (68%), `top_up_failed` (68%).
- Each model was scored on test once, and nothing was tuned on test.

Not done: the zero-shot base model (prompt v1) was not scored on test, so the gap between the base and tuned
rows mixes the effect of tuning with the effect of the prompt. The epoch-3 checkpoint was promoted
(`b77-lr1em4-r16`) but never scored. No hyperparameter sweep was run (`banking77-sweep` only prints the plan).

First epoch-2 run scored 90.06% because 37 requests got HTTP 404 right after the deployment came up. The chat
client now retries 404s, and the 37 rows were rerun on a new deployment (`--only-failed`), giving 91.17%.

### SFT run

One run, `outputs/sft_runs/lr1e-4-r16` (not committed): learning rate 1e-4, LoRA rank 16 / alpha 32, 3 epochs,
batch 32, 738 steps, renderer `qwen3`, loss on the assistant turn only. These were best guesses, not tuned.

| Epoch | Dev eval loss | Perplexity |
|---|---|---|
| 0 | 3.376 | 29.2 |
| 1 | 0.0513 | 1.053 |
| 2 | 0.0425 | 1.043 |
| 3 | 0.0400 | 1.041 |

Epoch 2 was the checkpoint scored on test, chosen to test an intermediate checkpoint while epoch 3 was still training,
not because dev picked it.

## Data and split

The Hub's `main` branch only has a loading script, which current `datasets` refuses to run, so data is read
from the parquet revision `refs/pr/6`. Intent names come from the repo's `dataset_infos.json` and are lowercased
(`Refund_not_showing_up` -> `refund_not_showing_up`; `reverted_card_payment?` keeps its `?`, as in the source CSVs).

`data/split.json` is created once by `banking77 make-split` and committed. It never changes:

- `dev_uids`: 2,001 (seeded random 20% of the 10,003 training rows)
- `fewshot_uids`: 154 (2 random non-dev rows per intent, shuffled; the prompt order)
- `train_uids`: 7,848 (the rest; the SFT training set, so few-shot and dev rows are never trained on)

## Prompts

All three share the same system wording and the 77-intent list.

| Version | Examples in system prompt | `/no_think` | Used for |
|---|---|---|---|
| v1 | none | yes | base model, zero-shot |
| v2 | 154 lines `Customer message: "..." => Intent: <label>` | yes | base model, few-shot |
| v3 | none | no | SFT training rows and tuned-model inference |

The SFT model is trained and served with byte-identical v3 prompts. Training uses the `qwen3` cookbook renderer with a
bare-label assistant turn (`assistant\n<intent><|im_end|>`), which is the prompt the Fireworks server builds for chat
requests (checked by comparing prompt-token counts with the local template).

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
export FIREWORKS_API_KEY=...   # never commit this

banking77 make-split                                   # once; writes data/split.json
banking77 prepare --prompt-version v2 --eval-split test [--limit N]   # local only
banking77 submit --account-id <acct> --prompt-version v2 --eval-split test   # batch job (billed)
banking77 report --account-id <acct> --prompt-version v2 --eval-split test --wait
```

Outputs go to `outputs/<model>/<prompt-version>[-test]/`: `metrics.json`, `predictions.jsonl`, `per_intent.csv`
(with the most common wrong reply per intent), and `finish_reason` counts.

SFT (needs Python >= 3.11 and the [Fireworks cookbook](https://github.com/fw-ai/cookbook) on `PYTHONPATH`; see `sft_train.py`):

```bash
banking77-sft-data                                     # outputs/sft_data/{train,dev}_sft.jsonl, asserts no overlap
banking77-sft --run-name lr1e-4-r16 --learning-rate 1e-4 --lora-rank 16 --epochs 3 --dry-run
python scripts/plot_sft.py outputs/sft_runs/lr1e-4-r16 --steps-per-epoch 246    # loss chart as HTML
```

Tuned models are scored with a deployment, not batch:

```bash
banking77 prepare --model accounts/<acct>/models/<id> --prompt-version v3 --eval-split test
python scripts/eval_promoted.py --promoted-model accounts/<acct>/models/<id> --deployment-id <name> \
    --prompt-version v3 --eval-split test        # creates the deployment, scores, deletes it (GPU time is billed)
```

## Notes

- Replies are parsed as exactly one intent name, case-insensitive. A reply like `Intent: card_arrival` is unparsed.
- `--max-tokens` defaults to 32 with temperature 0; all replies in these runs ended with `finish_reason: stop`.
- Dev was used for eval loss and for iterating on the prompt, so dev scores are optimistic. Use test sparingly.
