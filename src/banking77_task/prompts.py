from __future__ import annotations

from typing import Mapping, Optional, Sequence

from banking77_task.data import Example, load_label_mapping

# v1: zero-shot (intent list only). v2: v1 plus 2 few-shot examples per intent (data/split.json).
# v3: v1 without the `/no_think` switch; the SFT training prompt, also used to serve SFT models.
# Same system wording throughout, so v1 vs v2 isolates the shots.
PROMPT_VERSIONS = ("v1", "v2", "v3")
DEFAULT_VERSION = "v2"


def uses_shots(version: str) -> bool:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version {version!r}; expected one of {PROMPT_VERSIONS}")
    return version == "v2"


def uses_no_think(version: str) -> bool:
    """The base model needs `/no_think` to skip thinking; a model SFT-trained on v3 answers directly."""
    uses_shots(version)  # validates the version
    return version != "v3"


def build_few_shots_part(shots: Sequence[Example], id2label: Mapping[int, str]) -> str:
    """One line per labelled example, joined by newlines, in the order given."""
    return "\n".join(
        f'Customer message: "{" ".join(e.text.split())}" => Intent: {id2label[e.label]}' for e in shots
    )


def build_system_prompt(
    id2label: Optional[Mapping[int, str]] = None, shots: Optional[Sequence[Example]] = None
) -> str:
    """System prompt listing every intent (label-id order, from the dataset's mapping) and the few-shot examples."""
    id2label = id2label if id2label is not None else load_label_mapping()[0]
    intents = "\n".join(id2label[i] for i in sorted(id2label))
    examples = f"Examples:\n{build_few_shots_part(shots, id2label)}\n\n" if shots else ""
    return (
        "You are an expert assistant in the field of customer service. "
        "Your task is to help agents in the customer service department of a company. "
        "Your task is to classify the customer's question in order to help the customer service agent to answer the question. "
        "In order to help the agent, you MUST respond with the name of one of the following intents you know. "
        "In case you reply with something else, you will be penalized.\n\n"
        f"Intents:\n{intents}\n\n"
        f"{examples}"
        "Answer with exactly one intent from the list, copied verbatim, and nothing else."
    )


def build_messages(
    text: str,
    id2label: Optional[Mapping[int, str]] = None,
    no_think: bool = True,
    shots: Optional[Sequence[Example]] = None,
) -> list[dict[str, str]]:
    """Chat messages for one example. With `no_think`, Qwen3's `/no_think` soft switch skips the thinking block."""
    user = f"Customer message: {text}"
    if no_think:
        user += " /no_think"
    return [
        {"role": "system", "content": build_system_prompt(id2label, shots)},
        {"role": "user", "content": user},
    ]
