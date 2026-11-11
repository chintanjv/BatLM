"""Stage 3 — Search: BM25 over fact cards and Wikipedia passages, plus name matching.

Plain BM25 fails on the commonest questions: "who is the batman?" reduces to the single word
"batman", which is in most passages. So the index also knows every card's name and aliases; a
passage about something the question names is ranked up, and that subject's card ranks first.

Mirrored line-for-line in web/src/lib/search.ts so the browser retrieves exactly what evals measured.
"""
import json
import math
import pathlib
import re
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent

STOP = set("""a an the and or but of to in on at by for with from as is are was were be been being it its this that
these those he she they his her their him them who whom which what when where why how does do did has have had
not no can could would should will shall may might i you me my your we our us about into than then there also
tell name describe list give s""".split())

# Query expansion: bridges vocabulary gaps between how people ask and how the passages are written.
SYNONYMS = {
    "mother": "parents", "father": "parents", "mom": "parents", "dad": "parents",
    "villain": "enemy foe rogues", "enemie": "enemy villain", "enemy": "villain foe", "nemesi": "archenemy", "worst": "archenemy",
    "superhuman": "superpowers powers", "power": "superpowers", "creator": "created", "made": "created", "invented": "created",
    "debut": "first appeared", "introduced": "first appeared", "appearance": "appeared",
    "live": "home", "home": "manor", "car": "vehicle batmobile", "headquarter": "batcave", "base": "headquarters batcave",
    "gadget": "equipment", "movie": "film", "movies": "films", "actor": "played", "show": "series television",
    "wrote": "written writer", "author": "written writer", "directed": "director", "voice": "voiced",
    "real": "name", "identity": "name", "girlfriend": "love interest", "kid": "son", "child": "son", "children": "son",
    "die": "killed", "died": "killed", "dead": "killed", "murdered": "killed", "sidekick": "robin partner",
    "game": "video", "games": "video", "comic": "comics", "book": "comics",
    "brought": "raised", "imprisoned": "locked held", "jail": "prison locked", "contact": "call", "summon": "call",
}

CARD_WEIGHT = 1.5  # a card's BM25 score counts double: cards are short, curated and written to be found
CARD_BOOST = 12.0  # the card of a subject the question names (it is also pinned to the top, see search)
PAGE_BOOST = 1.5   # Wikipedia passages from that subject's page
LEAD_BOOST = 1.0   # ...and a little more for the page's opening passage
TITLE_BOOST = 10.0 # the opening passage of a page named in the question, when that page has no card


def stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def words(text: str) -> list[str]:
    """Every word, stemmed, stopwords kept: used to spot names ("the batman who laughs")."""
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]


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


class Index:
    def __init__(self, passages: list[dict], k1: float = 1.5, b: float = 0.75):
        self.passages, self.k1, self.b = passages, k1, b
        docs = [terms(f"{p['title']} {p['section']} {p['text']} {p.get('keywords', '')}") for p in passages]
        self.len = [len(d) for d in docs]
        self.weight = [CARD_WEIGHT if p.get("card") else 1.0 for p in passages]
        self.avgdl = sum(self.len) / len(docs)
        self.postings = defaultdict(list)  # term -> [(passage index, term frequency)]
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.postings[t].append((i, f))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self.pages, self.lead = defaultdict(list), {}
        for i, p in enumerate(passages):
            if not p.get("card"):
                self.pages[p["title"]].append(i)
                self.lead.setdefault(p["title"], i)
        # names -> what they refer to, as (card index or None, Wikipedia page title or None).
        # Longest names first, so "the batman who laughs" beats "batman".
        names = defaultdict(list)
        for i, p in enumerate(passages):
            for a in p.get("aliases", []):
                names[tuple(words(a))].append((i, p.get("page")))
        carded = {p.get("page") for p in passages if p.get("card")}
        for title in self.pages:  # a page without a card is still findable by its own title
            if title not in carded:
                names[tuple(words(title))].append((None, title))
                if m := re.match(r"^(.*?)\s*\((.+)\)$", title):  # "Batman (1966 film)" is also "the 1966 film Batman"
                    names[tuple(words(f"{m.group(2)} {m.group(1)}"))].append((None, title))
        self.names = sorted(names.items(), key=lambda kv: -len(kv[0]))

    def mentions(self, query: str) -> list[tuple]:
        """Subjects named in the query. A longer name hides shorter names inside it."""
        q, used, out = words(query), set(), []
        for name, targets in self.names:
            n = len(name)
            for s in range(len(q) - n + 1):
                if tuple(q[s:s + n]) == name and not used & set(range(s, s + n)):
                    used |= set(range(s, s + n))
                    for t in targets:
                        if t not in out:
                            out.append(t)
        return out

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        score = defaultdict(float)
        for t in terms(query, expand=True):
            for i, tf in self.postings.get(t, ()):
                score[i] += self.weight[i] * self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl))
        pinned = []  # what the question names goes first, whatever else scores well: its card, or its page's opening
        for card, page in self.mentions(query):
            if card is not None:
                score[card] += CARD_BOOST
                pinned.append(card)
            for i in self.pages.get(page, ()):
                score[i] += PAGE_BOOST
            if page in self.lead:
                score[self.lead[page]] += LEAD_BOOST if card is not None else TITLE_BOOST
                if card is None:
                    pinned.append(self.lead[page])
        pinned = pinned[:k]
        rest = sorted((x for x in score.items() if x[0] not in pinned), key=lambda x: (-x[1], x[0]))[:k - len(pinned)]
        return [(score[i], self.passages[i]) for i in pinned] + [(s, self.passages[i]) for i, s in rest if s > 0]


def load() -> Index:
    from . import cards
    return Index(cards.passages() + json.loads((ROOT / "data" / "chunks.json").read_text()))


def format_context(hits: list[dict]) -> str:
    if not hits:
        return "CASE FILES: (no matching files)"
    return "CASE FILES:\n" + "\n".join(f"[{i + 1}] ({c['title']}: {c['section']}) {c['text']}" for i, c in enumerate(hits))


if __name__ == "__main__":
    import sys
    index = load()
    q = " ".join(sys.argv[1:]) or "who is the batman?"
    print("names:", [index.passages[c]["title"] if c is not None else page for c, page in index.mentions(q)])
    for s, c in index.search(q, k=3):
        print(f"{s:6.2f}  [{c['title']}: {c['section']}] {c['text'][:110]}...")
