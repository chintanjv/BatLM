"""Stage 2b — Facts: the teacher reads every Wikipedia page and writes short, standalone facts from it.

    python -m batlm.facts [--limit N]        # needs the teacher (a cloud GPU) and data/raw/ from batlm.scrape

A hand-written card covers what people ask most about ~140 subjects. This does the same job for every
page: each fact is one sentence that names its subject, with the questions it answers. Search finds a
fact through those questions, and a small model answers best from a single sentence that says the thing.

A fact is kept only if its key words (a name, a number, a place) appear in the text the teacher was shown,
the same check batlm.cards applies to hand-written facts. Writes data/facts.json.
"""
import argparse
import json
import pathlib
import re

from . import scrape
from .teacher import parse_json

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "facts.json"
FIRST_ID = 1_000_000  # fact ids start here, clear of card ids (negative) and passage ids

PROMPT = """You are building a question-answering knowledge base about the Batman universe.
Below is part of the Wikipedia article "{title}".

Write the facts from this text that a fan might ask about: who someone is, real names, relationships, what happened and why, who created, wrote, directed, played or voiced something, dates, places and firsts.

Rules:
- Each fact is ONE standalone sentence of at most 35 words that names its subject in full. Write "The Joker" or "Batman Begins", never "he", "she", "it", "the film" or "the character".
- Every fact must be stated in the text below. Do not use anything you know from elsewhere.
- For each fact give three different questions a person would naturally type that the fact answers.
- For each fact give "keys": one or two short strings copied exactly from the fact (a name, a number, a place) that any correct answer must contain.
- At most {n} facts. Skip opinions, trivia and publication details nobody would ask about.

Reply with a JSON list and nothing else:
[{{"fact": "...", "asks": ["...?", "...?", "...?"], "keys": ["..."]}}]

TEXT:
{text}"""

VAGUE = re.compile(r"^(?:he|she|it|they|his|her|its|their|this|these|the (?:film|movie|series|show|game|character|story|storyline|comic|book|episode|issue))\b", re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).casefold().strip()


def windows(text: str, size: int = 800, most: int = 3) -> list[str]:
    """The page as up to `most` stretches of whole paragraphs, about `size` words each, in the order batlm.scrape reads it."""
    out, buf, n = [], [], 0
    for _, p in scrape.sections(text, cap=size * most):
        w = len(p.split())
        if buf and n + w > size:
            out.append("\n".join(buf))
            buf, n = [], 0
        buf.append(p)
        n += w
    return (out + ["\n".join(buf)] if buf else out)[:most]


def accept(item, source: str) -> dict | None:
    """A teacher-written fact, cleaned, or None if it breaks a rule or its keys are not in the source text."""
    if not isinstance(item, dict):
        return None
    fact, asks, keys = item.get("fact"), item.get("asks"), item.get("keys")
    if not isinstance(fact, str) or not isinstance(asks, list) or not isinstance(keys, list):
        return None
    fact = re.sub(r"\s+", " ", fact).strip()
    asks = list(dict.fromkeys(re.sub(r"\s+", " ", a).strip() for a in asks if isinstance(a, str)))
    asks = [a for a in asks if a.endswith("?") and 3 <= len(a.split()) <= 25]
    keys = [k.strip() for k in keys if isinstance(k, str) and 2 <= len(k.strip()) <= 60 and "," not in k and "|" not in k]
    if not 6 <= len(fact.split()) <= 45 or not fact.endswith(".") or VAGUE.match(fact) or len(asks) < 2 or not keys:
        return None
    if not all(norm(k) in norm(fact) and norm(k) in norm(source) for k in keys):
        return None
    return {"text": fact, "asks": asks[:-1], "held": asks[-1], "keys": ", ".join(keys[:2])}


def pages(limit: int | None = None) -> dict[str, dict]:
    """Display title -> {"text", "url", "words"} for the pages in the corpus (the first `limit`, if given), in corpus
    order. The full text comes from the download cache, or from Wikipedia for any page not in it."""
    chunks = json.loads((ROOT / "data" / "chunks.json").read_text())
    size, url = {}, {}
    for c in chunks:
        size[c["title"]] = size.get(c["title"], 0) + len(c["text"].split())
        url.setdefault(c["title"], c["url"])
    cached = {}
    for f in sorted(scrape.RAW.glob("*.json")):
        page = json.loads(f.read_text()).get("query", {}).get("pages", [{}])[0]
        if page.get("extract"):
            cached.setdefault(scrape.display(page["title"]), page["extract"])
    out = {}
    for title in list(url)[:limit]:
        text = cached.get(title)
        if text is None:
            page = scrape.fetch(url[title].split("/wiki/", 1)[1].replace("_", " "))
            text = page and page["text"]
        if text:
            out[title] = {"text": text, "url": url[title], "words": size[title]}
    return out


def build(teacher, limit: int | None = None) -> list[dict]:
    jobs = []
    for title, page in pages(limit).items():
        for w in windows(page["text"], most=min(4, max(1, round(page["words"] / 600)))):
            jobs.append((title, page["url"], w))
    print(f"{len(jobs)} stretches of text from {len({j[0] for j in jobs})} pages", flush=True)
    replies = teacher.ask([PROMPT.format(title=t, n=10, text=w) for t, _, w in jobs], max_tokens=1500, temperature=0.2)
    out, seen, raw = [], set(), 0
    for (title, url, w), reply in zip(jobs, replies):
        items = parse_json(reply)
        for item in items if isinstance(items, list) else []:
            raw += 1
            f = accept(item, w)
            if f is None or norm(f["text"]) in seen:
                continue
            seen.add(norm(f["text"]))
            out.append({"id": FIRST_ID + len(out), "title": title, "section": "Facts", "url": url, "fact": True, **f})
    print(f"{len(out)} facts kept of {raw} written", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N pages (for a quick trial)")
    args = ap.parse_args()
    from .teacher import Teacher
    teacher = Teacher()
    try:
        OUT.write_text(json.dumps(build(teacher, args.limit), ensure_ascii=False, separators=(",", ":")))
    finally:
        teacher.close()


if __name__ == "__main__":
    main()
