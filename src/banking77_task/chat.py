"""Concurrent chat-completions calls, for models served from a deployment (no batch API)."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import requests

CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def complete_one(session: requests.Session, model: str, row: dict[str, Any], retries: int = 8) -> tuple[str, Optional[str], Optional[str]]:
    """(reply, error, finish_reason) for one batch-input row {"custom_id", "body"}; retries 404/429/5xx for ~2 min."""
    last = ""
    for attempt in range(retries):
        try:
            r = session.post(CHAT_URL, json={"model": model, **row["body"]}, timeout=120)
            if r.status_code == 200:
                choice = r.json()["choices"][0]
                return choice["message"].get("content") or "", None, choice.get("finish_reason")
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (404, 429, 500, 502, 503, 504):  # 404: a fresh deployment is not routable yet
                break
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2 ** attempt)
    return "", last, None


def complete_all(model: str, rows: list[dict[str, Any]], concurrency: int = 16) -> dict[str, tuple[str, Optional[str], Optional[str]]]:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise RuntimeError("Set FIREWORKS_API_KEY to call Fireworks.")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        replies = list(pool.map(lambda row: complete_one(session, model, row), rows))
    return {row["custom_id"]: reply for row, reply in zip(rows, replies)}


def read_input_rows(path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
