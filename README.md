# banking77_task

Prompt-based intent classification on [BANKING77](https://huggingface.co/datasets/PolyAI/banking77)
(77 banking intents) with **Qwen3-4B**, and the same model after **LoRA SFT**, both served through Fireworks AI.

| Model (original 3,080-example test set) | Prompt | Accuracy | Macro-F1 |
|---|---|---|---|
| Qwen3-4B base | v2: few-shot, 2 examples per intent | 78.34% | 0.778 |
| Qwen3-4B + LoRA SFT (epoch 2) | v3: zero-shot | **91.17%** | **0.912** |

For reference, the BANKING77 paper reports 93.66% for a fine-tuned BERT trained on the full training set
([Casanueva et al., 2020, Table 3](https://arxiv.org/abs/2003.04807)). That BERT sees all 10,003 training rows; our
SFT run uses 7,848 (see [Data and splits](#data-and-splits)), so the two are not exactly comparable.

Contents: [Use the classifier](#use-the-classifier) · [Experiment set-up](#experiment-set-up) ·
[Results](#results) · [Reproduce](#reproduce) · [Limitations](#limitations-and-caveats) · [Repo layout](#repo-layout)

---

## Use the classifier

The classifier is a prompt plus a model. It sends a system prompt that lists the 77 intents and the customer message, and
the model replies with one intent name. A reply is accepted only if it is exactly one of the 77 names (case-insensitive).

**Install**

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
export FIREWORKS_API_KEY=...        # never commit this
```

**Classify messages.** Qwen3-4B is not on Fireworks' serverless pool, so the model has to be *deployed* first (an
on-demand deployment, billed by GPU time). The served name for a deployed LoRA model is `<model>#<deployment>`:

```bash
banking77 classify --prompt-version v3 \
  --model "accounts/<acct>/models/<tuned-model>#accounts/<acct>/deployments/<deployment>" \
  "My card still has not arrived after two weeks" "Why was I charged a fee for a top up?"
# 'My card still has not arrived after two weeks'   card_arrival
```

- Use `--prompt-version v3` for an SFT-tuned model and `v1` / `v2` for the base model (see [Prompts](#prompts)).
- Output is `message<TAB>intent`, or `<unparsed: ...>` if the reply is not exactly one intent name.
- Calls use temperature 0 and `max_tokens` 32.
- This command was unit-tested with a stubbed model. I did not run it live (a live call needs a deployment, which bills
  GPU time); the same request path (`chat.complete_one`) did serve the 3,080 test requests.

**From Python**

```python
from banking77_task.data import load_label_mapping
from banking77_task.prompts import build_messages   # v3 = no_think=False, no shots

id2label, _ = load_label_mapping()
messages = build_messages("Where is my card?", id2label, no_think=False, shots=None)
# POST {"model": "<model>#<deployment>", "messages": messages, "max_tokens": 32, "temperature": 0}
# to https://api.fireworks.ai/inference/v1/chat/completions, then parsers.parse_label(reply, names)
```

---

## Experiment set-up

### Dataset

[PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77): customer-service messages from the banking domain,
each labelled with one of **77 intents** (e.g. `card_arrival`, `top_up_failed`, `lost_or_stolen_card`).

| | Rows | Per intent | Notes |
|---|---|---|---|
| Train | 10,003 | 35–187 (imbalanced) | label-sorted in the file |
| Test | 3,080 | exactly 40 | balanced; used only for final scoring |

Messages are short: 11.9 words on average (median 10, max 79).

Data quirks that matter here:

- The Hub's `main` branch only has a loading script, which current `datasets` refuses to run. Data is read from the
  parquet revision `refs/pr/6` instead (`HF_REVISION` in `data.py`).
- The id-to-name mapping comes from the repo's `dataset_infos.json` (`load_label_mapping()` in `data.py`). Names are
  lowercased in one place (`normalize_label`), because the source has one capitalized intent (`Refund_not_showing_up`).
- `reverted_card_payment?` really has a trailing `?` in the source CSVs, `dataset_infos.json` and the HF script. It is
  kept as is.
- Several intents are near-duplicates (`verify_my_identity` vs `unable_to_verify_identity`, `top_up_failed` vs
  `top_up_reverted`, the three fee intents). This caps what any model can reach.

### Data and splits

`data/split.json` is created once by `banking77 make-split` (seed 0) and committed, so the splits never change:

| Set | Rows | How it is made | Used for |
|---|---|---|---|
| `dev_uids` | 2,001 | seeded random 20% of the 10,003 training rows | eval loss during SFT; prompt iteration |
| `fewshot_uids` | 154 | 2 random non-dev rows per intent, shuffled | the few-shot examples in prompt v2 |
| `train_uids` | 7,848 | everything else | SFT training |

The three sets are disjoint (asserted in code and tested), so the few-shot examples and dev rows are never trained on.
The original test set (3,080) is untouched until the final scoring.

### Prompts

All versions share the same system wording and the 77-intent list (in label-id order). They differ in two ways:

| Version | Examples in system prompt | `/no_think` in user message | Used for |
|---|---|---|---|
| v1 | none | yes | base model, zero-shot |
| v2 | 154 lines `Customer message: "..." => Intent: <label>` | yes | base model, few-shot |
| v3 | none | no | SFT training rows and tuned-model inference |

Shape of a request (abridged):

```
[system]  You are an expert assistant in the field of customer service. ... classify the customer's question ...
          Intents:
          activate_my_card
          age_limit
          ... (77 names)

          Examples:                                    <- v2 only
          Customer message: "How can I tell if my card payment is reverted?" => Intent: reverted_card_payment?
          ... (154 lines, 2 per intent, shuffled)

          Answer with exactly one intent from the list, copied verbatim, and nothing else.
[user]    Customer message: How do you come up with your exchange rate /no_think      <- "/no_think" in v1/v2 only
[assistant] exchange_rate
```

`/no_think` is Qwen3's soft switch that skips the thinking block, so the base model answers directly. A tuned model is
trained to answer directly, so v3 drops it and the training prompt is byte-identical to the inference prompt.

**How the prompt evolved** (all on dev, using batch jobs; nothing was tuned on test):

1. First prompt: intent list in the user message, zero-shot. A 10-example smoke test scored 0.1, but all 10 were
   `card_arrival`: the training data is sorted by label and `--limit` took a prefix. That was a sampling bug, not a model
   result. `--limit` now draws a random sample.
2. On 10 random dev examples, zero-shot scored 0.4. Four of the six errors were not valid intents: three invented names
   (`exchange_charge`, `receiving_money`, `get_physical_card`) and one `unknown` that the model produced on its own.
3. Adding 2 few-shot examples per intent (v2) scored 0.7 on the same 10 examples and removed the invented names. On a
   random 200-example dev slice it scored 77.5%. By then the system prompt had been reworded to tell the model to
   answer `unknown` when it could not classify, which is not an intent, so those replies could only count as errors.
4. Cleanups: fixed missing spaces and a curly apostrophe in the system prompt, removed the `unknown` instruction,
   lowercased all intents everywhere, and kept a zero-shot variant (v1) next to the few-shot one (v2) for
   apples-to-apples runs. The 200-example dev run was not repeated with the final wording; the final prompt was
   scored on test only.
5. For SFT, the examples are dropped (they would add about 3.7k tokens to every row) and `/no_think` is removed (v3).

Checks that mattered: `finish_reason` is recorded for every reply (all `stop`, so `max_tokens=32` never truncated), and
the server's prompt-token counts match the local Qwen3 chat template with default settings, so the server adds nothing
after `assistant\n` (important for matching the training format).

### SFT training data

Built by `banking77-sft-data` from `train_uids` only, as `{"messages": [system, user, assistant]}` rows:

- system and user: exactly the v3 prompt (full intent list, no examples, no `/no_think`);
- assistant: the bare intent name, no reasoning.

That gives 7,848 train rows (26–149 per intent after exclusions; about 530 tokens each, max 575 in a 200-row sample) and 2,001
dev rows, which the trainer uses only for eval loss. Rendered with the cookbook's `qwen3` renderer, a row trains
`assistant\n<intent><|im_end|>` with loss on those tokens only. The alternative renderer, `qwen3_disable_thinking`, puts
an empty think block in the prompt and expects the server to add it at inference. Our batch/chat requests cannot do
that, so I used `qwen3`.

### Training flow

[Fireworks cookbook](https://github.com/fw-ai/cookbook) `sft_loop`, wrapped by `banking77-sft` (`sft_train.py`). One run,
settings chosen as best guesses (cookbook defaults for epochs and learning rate), not tuned:

| Setting | Value |
|---|---|
| Base model | `accounts/fireworks/models/qwen3-4b`, LoRA |
| Learning rate | 1e-4 (constant) |
| LoRA rank / alpha | 16 / 32 |
| Epochs / batch size | 3 / 32 (246 steps per epoch, 738 total) |
| Max sequence length | 1024 |
| Training shape | `qwen3-4b-minimum-lora` |
| Cookbook commit | `5b0678b5561211381d30f382a7b79297df9ef4cd` |

A checkpoint is saved at the end of each epoch. Dev eval loss (token-level cross-entropy on the 2,001 dev rows):

| Epoch | Dev eval loss | Perplexity |
|---|---|---|
| 0 | 3.376 | 29.2 |
| 1 | 0.0513 | 1.053 |
| 2 | 0.0425 | 1.043 |
| 3 | 0.0400 | 1.041 |

The loss chart is `results/sft_run/metrics.html` (open it in a browser); the raw numbers are in
`results/sft_run/train_metrics.jsonl`.

### Final evaluation

- **Base model:** Fireworks batch inference (`banking77 submit` / `report`), prompt v2, on the 3,080 test rows.
- **Tuned model:** I did not try batch for it (a LoRA model most likely needs a deployment). The promoted checkpoint was
  deployed on a short-lived on-demand deployment, scored with 16 concurrent chat requests, and the deployment was
  deleted afterwards (`scripts/eval_promoted.py`).
- Both used temperature 0 and `max_tokens` 32 (verified in the saved request files).
- The test set was scored once per model. I scored the **epoch-2** checkpoint (step 492), chosen while epoch 3 was still
  training. The epoch-3 checkpoint was promoted but never scored, as a decision to limit spend.
- Metrics: accuracy (equal to micro-F1 for single-label classification; unparsed replies count as wrong), macro-F1
  over all 77 intents, unparsed count, `finish_reason` counts, and a per-intent table with each intent's most common
  wrong reply.

---

## Results

Test set, 3,080 examples (`results/`):

| Model | Prompt | Accuracy | Macro-F1 | Unparsed |
|---|---|---|---|---|
| Qwen3-4B base | v2 few-shot | 78.34% | 0.778 | 23 |
| Qwen3-4B + SFT, epoch 2 | v3 zero-shot | **91.17%** | **0.912** | 6 |

- **Base model errors.** 15 of its 23 unparsed replies are the invented name `get_virtual_card` (real name
  `getting_virtual_card`). Weakest intents: `get_physical_card` 17% (30 of 40 sent to `change_pin`), `failed_transfer`
  20% (`declined_transfer`), `topping_up_by_card` 30%, `order_physical_card` 32%, `beneficiary_not_allowed` 35%.
- **SFT model weakest intents** (all near-duplicate pairs): `topping_up_by_card` 50% (`supported_cards_and_currencies`),
  `balance_not_updated_after_bank_transfer` 65% (`transfer_not_received_by_recipient`),
  `top_up_by_bank_transfer_charge` 65% (`transfer_fee_charged`), `direct_debit_payment_not_recognised` 68%
  (`card_payment_not_recognised`), `top_up_failed` 68% (`top_up_reverted`).
- **A correction.** The first tuned-model run scored 90.06%, because 37 requests hit HTTP 404 right after the deployment
  came up (it was not routable yet) and counted as wrong. The client now retries 404s, and only those 37 rows were rerun
  on a fresh deployment (`--only-failed`), giving 91.17% with zero request errors. The first run's metrics are kept in
  `results/sft_epoch2_v3_test/metrics.first_run.json`.

---

## Reproduce

Needs a Fireworks account and `FIREWORKS_API_KEY`. Costs: the base-model test run is about 13M input tokens; SFT is
about 12.5M training tokens (3 epochs x 7,848 rows x ~530 tokens); the tuned-model eval needs a deployment billed by GPU
time. Prices are on Fireworks' pricing page; I did not confirm how cookbook trainer jobs are billed.

```bash
# 0. Setup and tests (the test suite needs network for the Hub's label metadata)
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest -q
export FIREWORKS_API_KEY=...

# 1. Split. data/split.json is committed, so skip this to reuse mine. Delete the file first to remake it.
banking77 make-split

# 2. Base model, few-shot, on test (batch). Use --eval-split dev for dev. --prompt-version v1 is zero-shot.
banking77 prepare --prompt-version v2 --eval-split test           # local only, nothing is sent
banking77 submit  --prompt-version v2 --eval-split test --account-id <acct>
banking77 report  --prompt-version v2 --eval-split test --account-id <acct> --wait
# -> outputs/qwen3-4b/v2-test/{metrics.json,predictions.jsonl,per_intent.csv}

# 3. SFT data (asserts train / dev / few-shot are disjoint)
banking77-sft-data                                                # -> outputs/sft_data/

# 4. SFT. Needs Python >= 3.11 and the Fireworks cookbook on PYTHONPATH:
#    git clone https://github.com/fw-ai/cookbook.git && git -C cookbook checkout 5b0678b
#    cd cookbook/training && pip install -e . && export PYTHONPATH=<cookbook>:<cookbook>/training:$PYTHONPATH
banking77-sft --run-name lr1e-4-r16 --learning-rate 1e-4 --lora-rank 16 --epochs 3 --batch-size 32 \
              --output-model-id b77-lr1em4-r16 --dry-run          # prints the plan; drop --dry-run to train (billed)
python scripts/plot_sft.py outputs/sft_runs/lr1e-4-r16 --steps-per-epoch 246     # loss chart

# 5. Promote the epoch-2 checkpoint to a model (any promotable step-N checkpoint works), then score it on test
#    (your own promotion step; list checkpoints with the cookbook / Fireworks API)
banking77 prepare --model accounts/<acct>/models/<promoted-id> --prompt-version v3 --eval-split test
python scripts/eval_promoted.py --promoted-model accounts/<acct>/models/<promoted-id> \
    --deployment-id <name> --prompt-version v3 --eval-split test   # creates, scores, deletes the deployment
```

Exact numbers can differ slightly on a rerun: SFT is not bit-reproducible (the shuffle seed is fixed but GPU kernels
are not), and temperature-0 decoding can still flip a rare tie. The splits and the few-shot examples are fixed by
`data/split.json`.

The promotion step has no script in this repo; I used the cookbook's checkpoint API from my ANLI project. The model ids in
the table above are `b77-lr1em4-r16` (epoch 3, unused) and `b77-lr1em4-r16-ep2` (the scored epoch 2).

---

## Limitations and caveats

- **Not a clean base-vs-tuned comparison.** The base row uses few-shot prompting and the tuned row uses zero-shot.
  I did not score the zero-shot base model (v1) on test, so the 12.8-point gap mixes tuning with the prompt change.
- **One SFT run, no sweep.** Learning rate, rank and epochs were best guesses. `banking77-sweep` only prints a plan
  (learning rate 5e-5/1e-4/2e-4, then rank 8/32) and was never launched.
- **Dev is optimistic.** Dev was used for eval loss and for iterating on the prompt, so dev numbers are not a clean
  estimate. Test was used once per model.
- **Fewer training rows than the BERT baseline.** 7,848 vs 10,003, so the gap to 93.66% is not fully like-for-like.
- **Eval loss is not accuracy.** It is per token and each answer is only a few tokens. The epoch-3 checkpoint was never
  scored, so I cannot say whether it beats epoch 2.
- **Label noise.** Near-duplicate intents mean there is a ceiling (published fine-tuned encoders reach about 93–94%).
- **Parsing is strict.** `Intent: card_arrival` or a slightly different name counts as wrong. Allowing an optional
  "Intent:" prefix would recover a few replies.
- **Single seed, 3,080 examples.** At about 91% accuracy the standard error is roughly +-0.5 points, so small differences
  are noise.

---

## Repo layout

```
data/split.json            frozen dev / few-shot / train uids
results/                   final metrics, per-intent tables, predictions, SFT loss log and chart
src/banking77_task/
  data.py                  dataset loading, label mapping, lowercase normalisation
  split.py                 frozen split (make / load / disjointness)
  prompts.py               prompt versions v1 / v2 / v3, few-shot rendering
  parsers.py  metrics.py   reply parsing; accuracy, macro-F1, per-intent table
  fireworks.py             batch API client      chat.py   concurrent chat client (retries 404/429/5xx)
  cli.py                   banking77 make-split | prepare | submit | report | chat-eval | classify
  sft_data.py  sft_train.py  sweep.py            SFT data builder, cookbook wrapper, sweep plan
scripts/eval_promoted.py   deploy -> score -> delete;   scripts/plot_sft.py   loss chart
tests/test_core.py
```
