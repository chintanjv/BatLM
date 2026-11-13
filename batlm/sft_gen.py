"""Stage 4b — Training conversations written by the teacher.

    python -m batlm.sft_gen [--limit N]     # needs the teacher (a cloud GPU)

The templates in batlm.sft_data only produce lookups ("Who is X?") answered by copying a sentence. Real
users also ask yes/no questions, push back, compare, ask two things at once, mistype, and follow up
with "was he ...?". This stage makes training data for all of that, in three steps:

  1. The teacher is shown the facts about one subject (or two) and writes questions in each style,
     plus short conversations whose later turns lean on the first.
  2. Every question goes through the real search (batlm.agent.plan), with its conversation so far.
  3. The teacher writes the reply BatLM should give from exactly those passages, under BatLM's rules,
     and a second pass drops any reply the passages do not support.

So the student is trained on what the product will really show it, as batlm.train_lora already insists.
Subjects in the held-out bucket are never trained on: their questions become evals/generated.json, with
reference answers written from the subject's own facts rather than from search.

Writes checkpoints/sft_teacher.jsonl and evals/generated.json.
"""
import argparse
import json
import pathlib
import random
import re
from collections import Counter, defaultdict

from . import rag, sft_data
from .agent import plan, settle
from .prompt import HISTORY
from .teacher import parse_json

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = ROOT / "checkpoints" / "sft_teacher.jsonl"
EVAL = ROOT / "evals" / "generated.json"

ASK = """You are writing test questions for a chatbot that answers questions about the Batman universe.
Here is what its files say about "{title}":
{facts}

Write questions a real person might type. Use only things these facts answer, except for the "unknowable" style.
Give one question for each style that fits (skip a style the facts do not support):
- "yesno": a yes/no question whose answer is yes
- "false": a yes/no question or a statement built on a wrong assumption that these facts correct (wrong name, wrong person, wrong date)
- "challenge": the user doubts one of the facts or pushes back ("Are you sure ...?", "I thought ...", "No way, ...")
- "why": a why or how question
- "casual": terse lowercase chat style, no punctuation, with one small spelling mistake
- "multipart": one question that asks for two different things
- "list": asks for several items, with a number if natural ("name three ...")
- "plain": an ordinary question, worded differently from "Who is ...?" or "What is ...?"
- "unknowable": a sensible-sounding question about "{title}" that no encyclopedia could answer (a private or never-stated detail)
Then write two short conversations of 2 or 3 user messages each. The first message names the subject. The later ones lean on it without naming it, the way people really talk ("what about his son?", "who played her?", "and when was that?", "are you sure?").

Reply with JSON and nothing else:
{{"questions": [{{"style": "yesno", "q": "..."}}], "conversations": [["...", "...", "..."], ["...", "..."]]}}"""

PAIR = """You are writing test questions for a chatbot that answers questions about the Batman universe.
Here is what its files say about "{a}":
{facts_a}

And about "{b}":
{facts_b}

Write three questions a real person might type that involve BOTH subjects and that these facts answer: one comparing them, one asking how they are connected or different, and one asking which of the two something is true of.

Reply with a JSON list of three strings and nothing else."""

OFF = """Write 40 different messages a user might send to a chatbot that have NOTHING to do with Batman, Gotham City or any Batman character, comic, film, show or game.
Theme of this batch: {theme}.
Vary the length and style: full questions, terse lowercase fragments, requests, statements.

Reply with a JSON list of 40 strings and nothing else."""

THEMES = """Marvel superheroes and films | Superman, Wonder Woman, the Flash and other DC heroes with no Batman involved | Star Wars, Harry Potter and other fiction |
cooking and recipes | writing code and fixing bugs | arithmetic and maths homework | geography and capitals | world history | physics, chemistry and biology |
sports results and rules | music and musicians | shopping and product recommendations | health, fitness and diet | travel plans | money, investing and taxes |
greetings, thanks and small talk | requests for jokes, poems, essays and emails | weather, time and today's news | the other films of actors who also played Batman characters, without mentioning Batman |
tech support for phones and computers | relationship and career advice | translation and grammar | philosophy and big questions | cars and repairs | real bats and other animals |
video games that are not Batman games | television shows that are not about Batman | random gibberish, single words and emoji | politics and world leaders | questions about the chatbot itself""".replace("\n", " ").split(" | ")

RULES = """You write the ideal reply for BatLM, a small assistant that answers questions about the Batman universe using only the CASE FILES it is shown.

Rules for the reply:
1. Use only what the CASE FILES and the earlier turns say. Never add anything you know from elsewhere.
2. Answer the exact question in one or two sentences, or a short numbered list when several items are asked for. No preamble, and do not mention the files.
3. For a yes/no question, start with "Yes" or "No", then give the fact that settles it. If the user assumes something the files contradict, say so and give the correct fact.
4. Check who each file is about. A file about another character counts only if it actually states the answer.
5. In a conversation, "he", "she", "it", "that" and similar words refer to the earlier turns.
6. If the user asks for more items than the files list, give the ones the files list and say that is all the files have.
7. If the question asks two things and the files answer only one, answer that one and say the files do not cover the other.
8. If the question is about the Batman universe but the files do not contain the answer, reply exactly: I don't have that in my case files.
9. If the question is not about the Batman universe at all, reply exactly: That is outside my case files. I only answer questions about the Batman universe.

Reply with the answer text only."""

CHECK = """{context}

REPLY: {answer}

Is every statement in the REPLY supported by the case files or the earlier turns above, and does it answer the QUESTION? Reply with one word: SUPPORTED or UNSUPPORTED."""

# teacher style -> eval kind (batlm.evals.KINDS)
KIND = {"yesno": "yesno", "false": "yesno", "challenge": "challenge", "why": "why", "casual": "typo", "multipart": "multipart",
        "list": "list", "plain": "lookup", "unknowable": "missing", "compare": "compare", "off": "off_topic"}


def declined(a: str) -> bool:
    a = a.lower()
    return "outside my case files" in a or "don't have that" in a


def subjects(index) -> dict[str, list[str]]:
    """Subject -> the standalone facts the files hold about it: card facts, teacher-written facts, and the page's opening sentence."""
    out, leads = defaultdict(list), set(index.lead.values())
    for i, p in enumerate(index.passages):
        if p.get("card"):
            out[p.get("page") or p["title"]].append(p["text"])
        elif p.get("fact"):
            out[p["title"]].append(p["text"])
        elif i in leads and (s := sft_data.lead_sentence(p["text"])):
            out[p["title"]].insert(0, s)
    return {t: f for t, f in out.items() if len(f) >= 2}


def bullets(facts: list[str]) -> str:
    return "\n".join(f"- {f}" for f in facts)


def good(q) -> bool:
    return isinstance(q, str) and 1 <= len(q.split()) <= 40 and len(q) <= 300


def context(question: str, hits: list[dict], history: list[dict]) -> str:
    """What the student will see for this turn, as plain text for the teacher."""
    earlier = "".join(f"User: {t['q']}\nBatLM: {t['a']}\n" for t in history[-HISTORY:])
    return (f"EARLIER TURNS:\n{earlier}\n" if earlier else "") + f"{rag.format_context(hits)}\n\nQUESTION: {question}"


def tidy(a: str) -> str:
    a = re.sub(r"^(?:BatLM|Answer|Reply)\s*:\s*", "", a.strip().strip('"').strip())
    if "outside my case files" in a.lower():
        return sft_data.REFUSAL
    if "don't have that" in a.lower() or "do not have that" in a.lower():
        return sft_data.MISSING
    return a


def answer(teacher, contexts: list[str]) -> list[str]:
    """The teacher's reply for each context, or "" where it broke the length limit or the check found it unsupported."""
    replies = [tidy(a) for a in teacher.ask(contexts, system=RULES, max_tokens=220, temperature=0.0)]
    replies = [a if 1 <= len(a.split()) <= 90 else "" for a in replies]
    todo = [i for i, a in enumerate(replies) if a and not declined(a)]
    verdicts = teacher.ask([CHECK.format(context=contexts[i], answer=replies[i]) for i in todo], max_tokens=6, temperature=0.0)
    for i, v in zip(todo, verdicts):
        if not v.strip().upper().startswith("SUPPORTED"):
            replies[i] = ""
    return replies


def write_questions(teacher, index, limit=None, rng=None):
    """Conversations to label: [{"id", "styles": [style per turn], "turns": [question...], "facts": [...], "held": bool}]."""
    rng = rng or random.Random(1939)
    subs = subjects(index)
    titles = sorted(subs)[:limit]
    held = {t for t in titles if sft_data._bucket("gen/" + t) < 8}
    jobs = []  # (titles, facts, prompt)
    for t in titles:
        for _ in range(2 if len(subs[t]) > 8 else 1):
            facts = rng.sample(subs[t], min(8, len(subs[t])))
            jobs.append(((t,), facts, ASK.format(title=t, facts=bullets(facts))))
    rich = [t for t in titles if len(subs[t]) >= 4]
    for _ in range(min(1200, len(rich) * 2) if len(rich) > 1 else 0):
        a, b = rng.sample(rich, 2)
        fa, fb = rng.sample(subs[a], min(5, len(subs[a]))), rng.sample(subs[b], min(5, len(subs[b])))
        jobs.append(((a, b), fa + fb, PAIR.format(a=a, b=b, facts_a=bullets(fa), facts_b=bullets(fb))))
    replies = teacher.ask([j[2] for j in jobs], max_tokens=900, temperature=0.9)
    convs = []

    def add(titles_, facts, styles, turns):
        if turns and all(good(q) for q in turns):
            convs.append({"id": f"gen-{len(convs)}", "styles": styles, "turns": [q.strip() for q in turns], "facts": facts,
                          "held": any(t in held for t in titles_)})

    for (titles_, facts, _), reply in zip(jobs, replies):
        data = parse_json(reply)
        if len(titles_) == 2:
            for q in data if isinstance(data, list) else []:
                add(titles_, facts, ["compare"], [q])
            continue
        if not isinstance(data, dict):
            continue
        for item in data.get("questions") or []:
            if isinstance(item, dict) and item.get("style") in KIND:
                add(titles_, facts, [item["style"]], [item.get("q")])
        for turns in data.get("conversations") or []:
            if isinstance(turns, list) and 2 <= len(turns) <= 3:
                add(titles_, facts, ["plain"] + ["followup"] * (len(turns) - 1), turns)
    # Off-topic messages: alone, and dropped into the middle of a Batman conversation.
    openers = [c["turns"][0] for c in convs if len(c["turns"]) > 1] or ["Who is Batman?"]
    for n, reply in enumerate(teacher.ask([OFF.format(theme=t) for t in THEMES[:limit]], max_tokens=1800, temperature=1.0)):
        data = parse_json(reply)
        for m, q in enumerate(q for q in (data if isinstance(data, list) else []) if good(q)):
            lone = m % 2 == 0
            convs.append({"id": f"gen-{len(convs)}", "styles": ["off"] if lone else ["plain", "off"],
                          "turns": [q.strip()] if lone else [rng.choice(openers), q.strip()], "facts": [], "held": n % 10 == 9})
    return convs


def walk(index, teacher, convs) -> list[dict]:
    """Training examples: every turn of every conversation, searched and answered as the site would, labelled by the teacher."""
    examples, state = [], [{"history": [], "dead": False} for _ in convs]
    for n in range(max((len(c["turns"]) for c in convs), default=0)):
        batch = []
        for c, s in zip(convs, state):
            if n >= len(c["turns"]) or s["dead"]:
                continue
            p = plan(index, c["turns"][n], s["history"])
            if p.prompt is not None:  # otherwise the gate answers: nothing for the model to learn, and the turn leaves no history
                batch.append((c, s, p))
        replies = answer(teacher, [context(c["turns"][n], p.hits, s["history"]) for c, s, p in batch])
        for (c, s, p), a in zip(batch, replies):
            a = settle(a, p.named)
            if not a:
                s["dead"] = True  # the later turns lean on this one
                continue
            examples.append({"prompt": p.prompt, "answer": a, "kind": "followup" if n and c["styles"][n] != "off" else c["styles"][n],
                             "q": c["turns"][n], "conv": c["id"], "turn": n})
            s["history"].append({"q": c["turns"][n], "a": a, "topic": p.topic})
    return examples


def references(teacher, convs) -> list[dict]:
    """Eval conversations: each turn with a reference answer written from the subject's own facts, not from search."""
    out, state = [], [{"history": [], "turns": []} for _ in convs]
    facts = [[{"title": "Files", "section": "Facts", "text": f} for f in c["facts"]] for c in convs]
    for n in range(max((len(c["turns"]) for c in convs), default=0)):
        batch = [(i, c) for i, c in enumerate(convs) if n < len(c["turns"]) and len(state[i]["turns"]) == n]
        fixed = {"off": sft_data.REFUSAL, "unknowable": sft_data.MISSING}
        ask = [(i, c) for i, c in batch if c["styles"][n] not in fixed]
        replies = dict(zip((i for i, _ in ask), answer(teacher, [context(c["turns"][n], facts[i], state[i]["history"]) for i, c in ask])))
        for i, c in batch:
            style = c["styles"][n]
            ref = fixed.get(style) or replies.get(i, "")
            if not ref or (style not in fixed and declined(ref)):
                continue  # no usable reference: the conversation ends here
            kind = "followup" if style == "followup" else KIND[style]
            state[i]["turns"].append({"q": c["turns"][n], "kind": kind, "ref": ref})
            state[i]["history"].append({"q": c["turns"][n], "a": ref})
    for c, s in zip(convs, state):
        if s["turns"]:
            out.append({"id": c["id"], "turns": s["turns"]})
    return out


def main(teacher=None, limit=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N subjects (for a quick trial)")
    args = ap.parse_args([] if teacher else None)
    limit = limit or args.limit
    own = teacher is None
    if own:
        from .teacher import Teacher
        teacher = Teacher()
    try:
        index, rng = rag.load(), random.Random(1939)
        convs = write_questions(teacher, index, limit, rng)
        print(f"{len(convs)} conversations written, {sum(c['held'] for c in convs)} held out for evals", flush=True)
        examples = walk(index, teacher, [c for c in convs if not c["held"]])
        refusals = [e for e in examples if declined(e["answer"])]
        rest = [e for e in examples if not declined(e["answer"])]
        rng.shuffle(refusals)
        examples = rest + refusals[:max(int(len(rest) * 0.15 / 0.85), 20)]  # declining stays a minority habit
        TRAIN.parent.mkdir(exist_ok=True)
        with open(TRAIN, "w") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{len(examples)} training examples {dict(Counter(e['kind'] for e in examples))} -> {TRAIN}", flush=True)
        held = [c for c in convs if c["held"]]
        rng.shuffle(held)
        evals = references(teacher, held[:700])
        EVAL.write_text(json.dumps(evals, ensure_ascii=False, indent=1))
        print(f"{len(evals)} eval conversations {dict(Counter(t['kind'] for c in evals for t in c['turns']))} -> {EVAL}", flush=True)
    finally:
        if own:
            teacher.close()


if __name__ == "__main__":
    main()
