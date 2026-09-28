"""Stage 6 — Retrieval: Okapi BM25 over the scraped chunks, from scratch.

Mirrored line-for-line in web/src/lib/bm25.ts so the browser retrieves exactly what evals measured.
"""
import json
import math
import pathlib
import re
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent

STOP = set("""a an the and or but of to in on at by for with from as is are was were be been being it its this that
these those he she they his her their him them who whom which what when where why how does do did has have had
not no can could would should will shall may might i you me my your we our us about into than then there also
tell name describe list give s""".split())


# Query expansion: bridges vocabulary gaps between how people ask and how Wikipedia writes.
SYNONYMS = {
    "mother": "parents", "father": "parents", "mom": "parents", "dad": "parents",
    "children": "son", "child": "son", "kid": "son", "girlfriend": "love interest", "wife": "love interest",
    "villain": "foe rogues", "enemy": "foe rogues", "enemie": "foe rogues", "nemesi": "archenemy foe", "worst": "archenemy",
    "superhuman": "superpowers powers", "power": "superpowers", "creator": "created", "made": "created",
    "came": "created", "debut": "debuted first appeared", "appearance": "debuted appeared",
    "live": "resides residence", "reside": "residence", "home": "residence manor", "drink": "ginger ale teetotaler",
    "raised": "raised brought", "alias": "identity", "real": "identity", "sale": "sold copies", "sold": "copies",
    "car": "vehicle batmobile", "headquarter": "batcave", "gadget": "utility belt", "undercover": "disguise",
    "asylum": "arkham", "treated": "psychiatric", "cowl": "identity persona batman", "worn": "assume",
    "inspired": "inspiration", "pulp": "pulp sherlock", "reason": "because", "movie": "films", "actor": "portrayed",
}


def stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def terms(text: str, expand: bool = False) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOP:
            continue
        s = stem(t)
        out.append(s)
        if expand and s in SYNONYMS:
            out += [stem(x) for x in SYNONYMS[s].split()]
    return out


class BM25:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # Section titles are indexed too: they are strong topical signals ("Enemies", "Technology").
        self.docs = [terms(c["section"] + " " + c["text"]) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # Domain anchors: terms in >30% of chunks ("batman"). BM25 gives them ~0 weight precisely because
        # they are everywhere — so a relevance gate must treat them separately, or "Who is Batman?" gets refused.
        self.anchors = {t for t, f in df.items() if f / n > 0.3}

    def mentions_anchor(self, query: str) -> bool:
        return any(t in self.anchors for t in terms(query))

    def score(self, q: list[str], i: int) -> float:
        tf, dl, s = self.tf[i], len(self.docs[i]), 0.0
        for t in q:
            if t in tf:
                s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        q = terms(query, expand=True)
        scored = sorted(((self.score(q, i), c) for i, c in enumerate(self.chunks)), key=lambda x: -x[0])
        return [(s, c) for s, c in scored[:k] if s > 0]


def load() -> BM25:
    return BM25(json.loads((ROOT / "data" / "chunks.json").read_text()))


def gold_chunk(index, answer_keys):
    """Best chunk containing the most answer keys (used for oracle insertion)."""
    def hits(c):
        return sum(k.lower() in c["text"].lower() for k in answer_keys)
    best = max(index.chunks, key=hits)
    return best if hits(best) else None


def format_context(hits: list[dict]) -> str:
    if not hits:
        return "CASE FILES: (no matching files)"
    return "CASE FILES:\n" + "\n".join(f"[{i + 1}] ({c['section']}) {c['text']}" for i, c in enumerate(hits))


if __name__ == "__main__":
    import sys
    for s, c in load().search(" ".join(sys.argv[1:]) or "Who is the Joker?", k=3):
        print(f"{s:6.2f}  [{c['source']}/{c['section']}] {c['text'][:140]}...")
