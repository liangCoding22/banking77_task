"""Fireworks Batch Inference API: upload JSONL, create a job, poll, download."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Optional

import requests

API_URL = "https://api.fireworks.ai/v1"
TERMINAL_STATES = {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"}
DEFAULT_MODEL = "accounts/fireworks/models/qwen3-4b"


def build_batch_jsonl(rows: list[dict[str, Any]], max_tokens: int, temperature: float = 0.0) -> str:
    """One row per request: {"custom_id", "body"}. The model is set on the job, not per row."""
    return "\n".join(
        json.dumps(
            {
                "custom_id": r["custom_id"],
                "body": {"messages": r["messages"], "max_tokens": max_tokens, "temperature": temperature},
            }
        )
        for r in rows
    )


def parse_batch_output(text: str) -> dict[str, tuple[str, Optional[str], Optional[str]]]:
    """custom_id -> (reply text, error, finish_reason)."""
    out: dict[str, tuple[str, Optional[str], Optional[str]]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("custom_id", ""))
        if row.get("error"):
            out[key] = ("", json.dumps(row["error"]), None)
            continue
        choices = (row.get("response") or row).get("choices") or []
        if not choices:
            out[key] = ("", "no choices in batch response", None)
            continue
        out[key] = (choices[0]["message"].get("content") or "", None, choices[0].get("finish_reason"))
    return out


class BatchClient:
    def __init__(self, account_id: str, api_key: Optional[str] = None, poll_seconds: float = 30.0) -> None:
        key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError("Set FIREWORKS_API_KEY to call Fireworks.")
        self.account_id = account_id
        self.poll_seconds = poll_seconds
        self._s = requests.Session()
        self._s.headers["Authorization"] = f"Bearer {key}"

    def _url(self, path: str) -> str:
        return f"{API_URL}/accounts/{self.account_id}/{path}"

    def _post(self, url: str, **kw: Any) -> dict[str, Any]:
        r = self._s.post(url, **kw)
        r.raise_for_status()
        return r.json() if r.content else {}

    def submit(self, model: str, jsonl: str, example_count: int, prefix: str = "b77") -> str:
        """Upload input and create the job; returns the job id."""
        suffix = uuid.uuid4().hex[:10]
        input_id, output_id, job_id = f"{prefix}-in-{suffix}", f"{prefix}-out-{suffix}", f"{prefix}-job-{suffix}"
        self._post(
            self._url("datasets"),
            json={"datasetId": input_id, "dataset": {"userUploaded": {}, "exampleCount": example_count}},
        )
        r = self._s.post(self._url(f"datasets/{input_id}:upload"), files={"file": ("input.jsonl", jsonl)})
        r.raise_for_status()
        acct = f"accounts/{self.account_id}"
        self._post(
            self._url("batchInferenceJobs"),
            params={"batchInferenceJobId": job_id},
            json={
                "model": model,
                "inputDatasetId": f"{acct}/datasets/{input_id}",
                "outputDatasetId": f"{acct}/datasets/{output_id}",
            },
        )
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any]:
        r = self._s.get(self._url(f"batchInferenceJobs/{job_id}"))
        r.raise_for_status()
        return r.json()

    @staticmethod
    def state(job: dict[str, Any]) -> str:
        return str(job.get("state", "")).replace("JOB_STATE_", "", 1)

    def wait(self, job_id: str) -> dict[str, Any]:
        while True:
            job = self.get_job(job_id)
            if self.state(job) in TERMINAL_STATES:
                return job
            time.sleep(self.poll_seconds)

    def download_output(self, job: dict[str, Any]) -> str:
        dataset_id = job["outputDatasetId"].split("/")[-1]
        r = self._s.get(self._url(f"datasets/{dataset_id}:getDownloadEndpoint"))
        r.raise_for_status()
        urls = r.json().get("filenameToSignedUrls", {})
        return "\n".join(requests.get(u).text for u in urls.values())
