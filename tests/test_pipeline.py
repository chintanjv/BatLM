"""Guards against the silent failures this project has had: prompt skew between training and the
browser, search that cannot find the obvious passage, and follow-up questions losing their subject."""
import json
import pathlib
import subprocess

from transformers import AutoTokenizer

from batlm import cards, evals, rag, sft_data
from batlm.agent import plan
from batlm.prompt import BASE_MODEL, chat, render

ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = rag.load()


def titles(question, history=()):
    return [h["title"] for h in plan(INDEX, question, history).hits]


def test_render_matches_chat_template():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    hits = [c for _, c in INDEX.search("Who is the Joker?")]
    history = [{"q": "Who is Batman?", "a": "Batman is Bruce Wayne.", "topic": []}]
    for h in ([], history):
        assert render("Who is the Joker?", hits, h) == tok.apply_chat_template(
            chat("Who is the Joker?", hits, h), tokenize=False, add_generation_prompt=True)


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


def test_typos_still_find_the_subject():
    for typo, right in [("who is the jokar", "who is the joker"), ("who is alferd", "who is alfred"), ("pengiun real name", "penguin real name")]:
        assert INDEX.mentions(typo) and INDEX.mentions(typo) == INDEX.mentions(right), typo
    assert INDEX.q_terms("batman villians") == INDEX.q_terms("batman villains")
    assert INDEX.q_terms("the capital of France") == rag.terms("the capital of France", expand=True)  # real words are left alone


def test_follow_ups_keep_their_subject():
    joker = [{"q": "what is the joker's real name?", "a": "The Joker has no definitive real name.", "topic": INDEX.mentions("the joker")}]
    assert "Joker" in titles("was he an arkham patient?", joker)
    assert plan(INDEX, "who created him?", joker).prompt is not None  # no subject named, but not off-topic
    assert plan(INDEX, "who created him?", joker).topic == INDEX.mentions("the joker")
    catwoman = [{"q": "Is Catwoman a villain?", "a": "Catwoman is sometimes Batman's enemy.", "topic": INDEX.mentions("catwoman")}]
    assert "Catwoman" in titles("You listed her as a Batman villain.", catwoman)
    assert INDEX.subjects("What is the capital of France?", joker[-1]) == ([], [])  # an unrelated question does not inherit one


def test_handwritten_eval_is_well_formed():
    turns = [t for c in evals.handwritten() for t in c["turns"]]
    assert len(turns) > 150
    for t in turns:
        assert t["kind"] in evals.KINDS, t
        assert bool(t["keys"]) != (t["kind"] in evals.DECLINE), t  # answerable turns need keys; the others must not have any
    ids = [c["id"] for c in evals.conversations(INDEX)]
    assert len(ids) == len(set(ids))


def test_browser_search_matches_python(tmp_path):
    """web/src/lib/search.ts must return the same subjects and the same passages, in the same order, as batlm/rag.py."""
    convs = evals.conversations(INDEX)
    cases = []
    for c in [c for c in convs if c["id"].startswith("auto-")][::3] + [c for c in convs if c["id"].startswith("hand-")]:
        last = None
        for t in c["turns"]:
            named, carried = INDEX.subjects(t["q"], last)
            cases.append({"q": t["q"], "last": last, "named": named, "carried": carried,
                          "ids": [h["id"] for _, h in INDEX.search(t["q"], 3, carried)]})
            last = {"a": t["a"] or t["keys"], "topic": carried[:1] + named if named else carried[:2]}
    (tmp_path / "index.json").write_text(json.dumps({"passages": INDEX.passages}))
    (tmp_path / "expected.json").write_text(json.dumps(cases))
    r = subprocess.run(["node", "web/scripts/search-parity.mjs", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
