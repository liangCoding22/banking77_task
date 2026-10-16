from banking77_task.data import Example, split_train_dev
from banking77_task.fireworks import build_batch_jsonl, parse_batch_output
from banking77_task.metrics import compute_metrics
from banking77_task.parsers import parse_label
from banking77_task.prompts import build_few_shots_part, build_messages
from banking77_task.split import make_split

LABELS = ["card_arrival", "lost_or_stolen_card"]


def test_split_is_20_percent_disjoint_and_deterministic():
    train = [Example(str(i), "t", i % 2) for i in range(100)]
    rest, dev = split_train_dev(train)
    assert len(dev) == 20 and len(rest) == 80
    assert not {e.uid for e in dev} & {e.uid for e in rest}
    assert dev == split_train_dev(train)[1]


def test_parse_label():
    assert parse_label("card_arrival", LABELS) == 0
    assert parse_label("<think>\n\n</think>\n\nLost_or_Stolen_Card\n", LABELS) == 1
    assert parse_label("card_arrival because...", LABELS) is None
    assert parse_label("", LABELS) is None


def test_prompt_lists_all_labels_and_no_think_switch():
    system, user = (m["content"] for m in build_messages("hi", dict(enumerate(LABELS))))
    assert "card_arrival\nlost_or_stolen_card" in system and user.endswith("/no_think")
    assert "/no_think" not in build_messages("hi", dict(enumerate(LABELS)), no_think=False)[1]["content"]


def test_batch_roundtrip():
    jsonl = build_batch_jsonl([{"custom_id": "a", "messages": [{"role": "user", "content": "x"}]}], 8)
    assert '"custom_id": "a"' in jsonl
    out = '{"custom_id":"a","response":{"choices":[{"message":{"content":"card_arrival"},"finish_reason":"stop"}]}}\n{"custom_id":"b","error":{"m":1}}'
    parsed = parse_batch_output(out)
    assert parsed["a"] == ("card_arrival", None, "stop") and parsed["b"][1]


def test_metrics():
    m = compute_metrics([0, 1, 1], [0, 1, None], LABELS)
    assert m["accuracy"] == 2 / 3 and m["unparsed"] == 1


def test_label_mapping_matches_dataset():
    from banking77_task.data import load_label_mapping

    id2label, label2id = load_label_mapping()
    assert len(id2label) == 77 and id2label[0] == "activate_my_card"
    assert all(label2id[n] == i for i, n in id2label.items())


def test_dev_is_not_label_sorted_prefix_safe():
    # BANKING77 train is sorted by label; the dev split must still mix labels when sampled.
    train = [Example(f"train-{i}", "t", i // 50) for i in range(1000)]
    _, dev = split_train_dev(train)
    assert len({e.label for e in dev[:10]}) > 1


def test_split_freezes_dev_and_excludes_fewshots_from_train():
    train = [Example(f"train-{i}", f"msg {i}", i // 50) for i in range(1000)]  # 20 labels, label-sorted
    sp = make_split(train, 20)
    dev, shots, rest = set(sp["dev_uids"]), sp["fewshot_uids"], set(sp["train_uids"])
    assert len(dev) == 200 and len(shots) == 40 and len(set(shots)) == 40
    assert not dev & set(shots) and not dev & rest and not set(shots) & rest
    assert dev | set(shots) | rest == {e.uid for e in train}
    by_uid = {e.uid: e for e in train}
    assert all(sum(by_uid[u].label == k for u in shots) == 2 for k in range(20))
    assert sp == make_split(train, 20)  # deterministic


def test_few_shots_part_is_newline_joined_and_in_system_prompt():
    shots = [Example("train-1", "lost\nmy  card", 1), Example("train-2", "where is it", 0)]
    part = build_few_shots_part(shots, dict(enumerate(LABELS)))
    assert part.split("\n") == [
        'Customer message: "lost my card" => Intent: lost_or_stolen_card',
        'Customer message: "where is it" => Intent: card_arrival',
    ]
    assert part in build_messages("hi", dict(enumerate(LABELS)), shots=shots)[0]["content"]


def test_per_intent_table():
    from banking77_task.metrics import per_intent_table

    rows = per_intent_table([0, 0, 1], [1, 0, None], LABELS)
    by = {r["intent"]: r for r in rows}
    assert by["card_arrival"]["correct"] == 1 and by["card_arrival"]["top_confusion"] == "lost_or_stolen_card"
    assert by["lost_or_stolen_card"]["top_confusion"] == "<unparsed>"


def test_labels_are_lowercase_and_parser_ignores_reply_case():
    from banking77_task.data import load_label_mapping

    id2label, label2id = load_label_mapping()
    assert all(n == n.lower() for n in id2label.values())
    assert "refund_not_showing_up" in label2id and "Refund_not_showing_up" not in label2id
    names = [id2label[i] for i in range(len(id2label))]
    assert parse_label("Refund_Not_Showing_Up", names) == label2id["refund_not_showing_up"]


def test_sft_rows_and_sweep():
    from banking77_task.sft_data import check_disjoint, to_row
    from banking77_task.sweep import runs

    row = to_row("hi", 1, dict(enumerate(LABELS)))["messages"]
    assert [m["role"] for m in row] == ["system", "user", "assistant"]
    assert "/no_think" not in row[1]["content"]
    assert row[2] == {"role": "assistant", "content": "lost_or_stolen_card"} and "Examples:" not in row[0]["content"]
    check_disjoint({"train_uids": ["a"], "dev_uids": ["b"], "fewshot_uids": ["c"]})
    import pytest

    with pytest.raises(AssertionError):
        check_disjoint({"train_uids": ["a"], "dev_uids": ["b"], "fewshot_uids": ["a"]})
    assert runs(1)[0][1] == "lr5e-5-r16" and runs(2, 1e-4)[1][1] == "lr1e-4-r32"


def test_prompt_versions():
    from banking77_task.prompts import uses_shots

    from banking77_task.prompts import uses_no_think

    assert not uses_shots("v1") and uses_shots("v2") and not uses_shots("v3")
    assert uses_no_think("v1") and uses_no_think("v2") and not uses_no_think("v3")


def test_chat_complete_one_parses_reply_and_retries(monkeypatch):
    from banking77_task import chat

    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._b, self.text = code, body, "x"

        def json(self):
            return self._b

    calls = [Resp(429), Resp(200, {"choices": [{"message": {"content": "card_arrival"}, "finish_reason": "stop"}]})]

    class Sess:
        def post(self, *a, **k):
            return calls.pop(0)

    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    assert chat.complete_one(Sess(), "m", {"custom_id": "a", "body": {"messages": []}}) == ("card_arrival", None, "stop")


def test_chat_retries_404(monkeypatch):
    from banking77_task import chat

    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._b, self.text = code, body, "x"

        def json(self):
            return self._b

    calls = [Resp(404), Resp(404), Resp(200, {"choices": [{"message": {"content": "a"}, "finish_reason": "stop"}]})]

    class Sess:
        def post(self, *a, **k):
            return calls.pop(0)

    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    assert chat.complete_one(Sess(), "m", {"custom_id": "x", "body": {}})[1] is None
