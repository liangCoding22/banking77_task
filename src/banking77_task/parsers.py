from __future__ import annotations

import re
from typing import Optional, Sequence

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop closed <think> blocks, and everything up to a lone closing tag."""
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def parse_label(text: Optional[str], labels: Sequence[str]) -> Optional[int]:
    """Label id when the final answer is exactly one label name (case-insensitive), else None."""
    if not text:
        return None
    answer = strip_thinking(text).strip().strip("`'\"").strip().lower()
    for i, name in enumerate(labels):
        if answer == name.lower():  # labels are already lowercase; this keeps the parser safe for any list
            return i
    return None
