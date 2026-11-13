"""The teacher-written data pipeline (facts, training conversations, eval questions), run end to end with a
stand-in teacher that returns canned replies. The real teacher needs a cloud GPU; this checks everything around it."""
import json

from batlm import evals, facts, rag, sft_data, sft_gen
from batlm.teacher import parse_json

INDEX = rag.load()


class Stub:
    """Answers each kind of prompt the pipeline sends with a fixed, well-formed reply."""

    def ask(self, prompts, system=None, **_):
        out = []
        for p in prompts:
            if "BOTH subjects" in p:
                out.append(json.dumps(["Who appeared first, Batman or the Joker?", "How are they connected?", 7]))
            elif p.startswith("You are writing test questions"):
                out.append("Sure!\n```json\n" + json.dumps({
                    "questions": [{"style": "yesno", "q": "Is the Joker Batman's archenemy?"},
                                  {"style": "unknowable", "q": "What is the Joker's shoe size?"},
                                  {"style": "nonsense", "q": "ignored"}],
                    "conversations": [["Who is the Joker?", "who created him?"], ["too short"]]}) + "\n```")
            elif "NOTHING to do with Batman" in p:
                out.append(json.dumps(["What is the capital of France?", "what time is it", "how do i bake bread"]))
            elif "SUPPORTED or UNSUPPORTED" in p:
                out.append("UNSUPPORTED" if "shoe size" in p else "SUPPORTED.")
            elif system and "what time is it" in p:
                out.append("That is outside my case files. I only answer questions about the Batman universe.")
            elif system:
                out.append("Answer: Yes. The Joker is Batman's archenemy.")
            else:
                out.append("")
        return out


def test_parse_json_tolerates_chatter():
    assert parse_json('Here you go:\n```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json('{"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert parse_json("no json here") is None


def test_a_fact_must_be_grounded_and_standalone():
    source = "The Joker first appeared in Batman #1 in 1940. He was created by Bill Finger, Bob Kane and Jerry Robinson."
    ok = {"fact": "The Joker first appeared in Batman #1 in 1940.", "keys": ["Batman #1", "1940"],
          "asks": ["When did the Joker first appear?", "What was the Joker's first comic?", "Which issue introduced the Joker?"]}
    got = facts.accept(ok, source)
    assert got and got["held"] == "Which issue introduced the Joker?" and len(got["asks"]) == 2 and got["keys"] == "Batman #1, 1940"
    assert facts.accept({**ok, "keys": ["1939"]}, source) is None                                  # key not in the fact
    assert facts.accept({**ok, "fact": "The Joker first appeared in Batman #2 in 1941.", "keys": ["1941"]}, source) is None  # not in the source
    assert facts.accept({**ok, "fact": "He first appeared in Batman #1 in 1940."}, source) is None  # does not name its subject
    assert facts.accept("not a dict", source) is None


def test_windows_follow_the_page():
    text = "\n".join(["Lead sentence about the subject that is long enough to keep here."] * 40 + ["== Plot ==", "word " * 300])
    ws = facts.windows(text, size=100, most=3)
    assert 1 <= len(ws) <= 3 and all(len(w.split()) <= 320 for w in ws)


def test_teacher_data_end_to_end():
    stub = Stub()
    convs = sft_gen.write_questions(stub, INDEX, limit=40)
    assert any(len(c["turns"]) == 2 for c in convs) and any(c["styles"] == ["compare"] for c in convs) and any(c["styles"] == ["off"] for c in convs)
    assert all(isinstance(q, str) for c in convs for q in c["turns"])
    examples = sft_gen.walk(INDEX, stub, convs)
    assert examples
    for e in examples:
        assert e["prompt"].startswith("<|im_start|>system") and e["prompt"].endswith("<|im_start|>assistant\n") and e["answer"]
    follow = [e for e in examples if e["kind"] == "followup"]
    assert follow and "QUESTION: Who is the Joker?<|im_end|>" in follow[0]["prompt"]   # the earlier turn is in the prompt
    assert follow[0]["answer"] == "Yes. The Joker is Batman's archenemy."              # "Answer:" prefix removed
    assert not any("shoe size" in e["q"] for e in examples)                            # the check rejected its reply
    refs = sft_gen.references(stub, convs)
    kinds = {t["kind"] for c in refs for t in c["turns"]}
    assert {"yesno", "missing", "off_topic", "followup"} <= kinds and kinds <= set(evals.KINDS)
    assert all(t["ref"] for c in refs for t in c["turns"])
    missing = [t for c in refs for t in c["turns"] if t["kind"] == "missing"]
    assert missing and all(t["ref"] == sft_data.MISSING for t in missing)


def test_reference_answers_wait_for_the_teacher():
    turn = {"kind": "yesno", "keys": "", "bad": "", "ref": "Yes."}
    assert evals.judge(turn, "Yes, he is.", False) is None
    assert evals.judge({**turn, "verdict": True}, "Yes, he is.", False) is True
    assert evals.judge(turn, "I don't have that in my case files.", False) is False
    rows = [{"conv": "gen-1", "kind": "yesno", "pred": "Yes.", "capped": False, "correct": None},
            {"conv": "hand-1", "kind": "lookup", "pred": "Bruce Wayne", "capped": False, "correct": True}]
    s = evals.summarize(rows)
    assert s["overall"]["n"] == 1 and s["overall"]["waiting_for_the_teacher"] == 1
