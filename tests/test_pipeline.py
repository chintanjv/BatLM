"""Guards against the two silent failures this project has had: prompt skew between training and the
browser, and search that cannot find the obvious passage."""
import json
import pathlib
import subprocess

from transformers import AutoTokenizer

from batlm import cards, rag, sft_data
from batlm.prompt import BASE_MODEL, chat, render

ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = rag.load()


def test_render_matches_chat_template():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    hits = [c for _, c in INDEX.search("Who is the Joker?")]
    assert render("Who is the Joker?", hits) == tok.apply_chat_template(
        chat("Who is the Joker?", hits), tokenize=False, add_generation_prompt=True)


def test_simple_questions_put_the_answer_first():
    for q, fact in [("who is the batman?", "Batman is Bruce Wayne"), ("What is the Joker's real identity?", "no definitive real name"),
                    ("Who is Alfred?", "butler"), ("What are the top 5 Batman villains?", "1. The Joker"),
                    ("Who broke Batman's back?", "Bane broke"), ("Who played Batman in The Dark Knight?", "Christian Bale"),
                    ("Who is the Batman Who Laughs?", "evil Bruce Wayne")]:
        assert fact in INDEX.search(q)[0][1]["text"], q


def test_every_card_fact_is_self_consistent():
    for card in cards.CARDS:
        for field, answer, keys in cards.facts(card):
            assert cards.key_hit(answer, keys), (card["title"], field)


def test_training_and_eval_questions_do_not_overlap():
    train = {it["q"].lower().rstrip("?.") for it in sft_data.train_items(INDEX) if it["kind"] in ("card", "wiki")}
    held = [it["q"] for it in sft_data.eval_items(INDEX) if it["kind"] in ("unseen", "reworded", "wiki")]
    assert not [q for q in held if q.lower().rstrip("?.") in train]


def test_browser_search_matches_python(tmp_path):
    """web/src/lib/search.ts must return the same passages, in the same order, as batlm/rag.py."""
    qs = [it["q"] for it in sft_data.eval_items(INDEX)][::3]
    (tmp_path / "index.json").write_text(json.dumps({"passages": INDEX.passages}))
    (tmp_path / "expected.json").write_text(json.dumps(
        [{"q": q, "ids": [c["id"] for _, c in INDEX.search(q)], "named": len(INDEX.mentions(q))} for q in qs]))
    r = subprocess.run(["node", "web/scripts/search-parity.mjs", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
