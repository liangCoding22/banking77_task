# banking77_task

Prompt-based intent classification on [BANKING77](https://huggingface.co/datasets/PolyAI/banking77)
(77 intents) with Qwen3-4B (`accounts/fireworks/models/qwen3-4b`) via Fireworks batch inference.

The dev set is a seeded random 20% of the original training set (2,000 of 10,003 examples).
The Hub's `main` branch only has a loading script, so data is read from the parquet revision `refs/pr/6`.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
export FIREWORKS_API_KEY=...   # never commit this

banking77 prepare [--limit N]                 # local only: writes outputs/qwen3-4b/v1/{input,dev}.jsonl
banking77 submit --account-id <account>       # uploads input and creates the batch job (costs money)
banking77 report --account-id <account> --wait   # downloads output, writes predictions.jsonl + metrics.json
```

Thinking is off by default (`/no_think`, `--max-tokens 32`); pass `--thinking` and a larger `--max-tokens` to enable it.
