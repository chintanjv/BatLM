"""Stage 3 — Search: BM25 over fact cards and Wikipedia passages, plus name and question matching.

Plain BM25 fails on the commonest questions: "who is the batman?" reduces to the single word
"batman", which is in most passages. So the index also knows:
  * every card's name and aliases: the best fact about a subject the question names is always shown first;
  * the ways people ask for each fact ("who is", "real name", "first appear"): these pick the right fact;
  * one-letter typos ("jokar", "villian"): an unknown word is read as the known word it is closest to;
  * who a follow-up is about ("was he an Arkham patient?"): the subjects of the previous turn carry over.
A 360M model answers reliably when the sentence it needs is the first thing it reads.

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

CARD_WEIGHT = 1.5  # a fact's BM25 score counts extra: facts are short, curated and written to be found
ASK_WEIGHT = 20.0  # a fact whose usual phrasing matches the question word for word scores this much
FACT_BOOST = 3.0   # facts about a subject the question names
PAGE_BOOST = 1.5   # Wikipedia passages from that subject's page
LEAD_BOOST = 1.0   # ...and a little more for the page's opening passage
TITLE_BOOST = 10.0  # the opening passage of a page named in the question, when that page has no card
OFF_SUBJECT = 0.6  # a passage that never mentions any subject the question is about keeps this share of its score

# A question that leans on the previous turn: it uses a pronoun, points back at what was said, or starts mid-thought.
FOLLOW = re.compile(
    r"\b(?:he|him|his|she|her|hers|they|them|their|theirs|it|its|himself|herself|that|this|those|these|"
    r"you said|you listed|you mentioned|you told|are you sure|what about|how about)\b"
    r"|^(?:and|but|so|also|then|why|really|sure)\b")


def follows(question: str) -> bool:
    return bool(FOLLOW.search(question.lower().strip()))


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


def near(a: str, b: str) -> bool:
    """True when b is one typo away from a: one letter changed, added or dropped, or two neighbours swapped."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or a == b:
        return False
    i = 0
    while i < min(la, lb) and a[i] == b[i]:
        i += 1
    if la == lb:
        return a[i + 1:] == b[i + 1:] or (i + 1 < la and a[i] == b[i + 1] and a[i + 1] == b[i] and a[i + 2:] == b[i + 2:])
    return a[i + 1:] == b[i:] if la > lb else a[i:] == b[i + 1:]


def bm25(tf: int, idf: float, length: int, avg: float, k1: float = 1.5, b: float = 0.75) -> float:
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg))


class Index:
    def __init__(self, passages: list[dict]):
        self.passages = passages
        docs = [terms(f"{p['title']} {p['section']} {p['text']} {p.get('keywords', '')}") for p in passages]
        self.len = [len(d) for d in docs]
        self.avgdl = sum(self.len) / len(docs)
        self.postings = defaultdict(list)  # term -> [(passage index, term frequency)]
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.postings[t].append((i, f))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        # The ways of asking for each fact, as (fact index, words). Stopwords are kept: "who is" must match.
        self.asks = [(i, list(dict.fromkeys(words(a)))) for i, p in enumerate(passages) for a in p.get("asks", [])]
        df = Counter(w for _, ws in self.asks for w in ws)
        n = len(self.asks)
        self.ask_idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.ask_rare = math.log(1 + (n + 0.5) / 0.5)  # weight of a word no phrasing contains
        self.pages, self.lead, self.facts = defaultdict(list), {}, defaultdict(list)
        for i, p in enumerate(passages):
            if p.get("card"):
                self.facts[p["title"]].append(i)
            else:
                self.pages[p["title"]].append(i)
                self.lead.setdefault(p["title"], i)
        # names -> what they refer to, as (card title or None, Wikipedia page title or None).
        # Longest names first, so "the batman who laughs" beats "batman".
        names = defaultdict(list)
        for p in passages:
            for a in p.get("aliases", []):
                target = (p["title"], p.get("page"))
                if target not in names[tuple(words(a))]:
                    names[tuple(words(a))].append(target)
        carded = {p.get("page") for p in passages if p.get("card")}
        for title in self.pages:  # a page without a card is still findable by its own title
            if title not in carded:
                names[tuple(words(title))].append((None, title))
                if m := re.match(r"^(.*?)\s*\((.+)\)$", title):  # "Batman (1966 film)" is also "the 1966 film Batman"
                    names[tuple(words(f"{m.group(2)} {m.group(1)}"))].append((None, title))
        self.names = sorted(names.items(), key=lambda kv: -len(kv[0]))
        # Every word the index knows, with how many passages use it: what a mistyped word is corrected to.
        self.known = {t: len(p) for t, p in self.postings.items()}
        for name, _ in self.names:
            for w in name:
                self.known.setdefault(w, 0)
        for w in self.ask_idf:
            self.known.setdefault(w, 0)
        for w in STOP:
            self.known[w] = 10 ** 9
        self.by_shape = defaultdict(list)  # (first letter, length) -> known words
        for w in self.known:
            if w:
                self.by_shape[(w[0], len(w))].append(w)
        self.fixed = {}
        # Each passage as a run of words, and each subject's names in the same form, to test "does this passage mention them?"
        self.wtext = [" " + " ".join(words(f"{p['title']} {p['text']}")) + " " for p in passages]
        self.called = defaultdict(list)
        for name, targets in self.names:
            if name:
                for t in targets:
                    self.called[t].append(" " + " ".join(name) + " ")

    def fix(self, t: str) -> str:
        """A question word as the index knows it: stemmed, and corrected if it is an unknown word one typo away from a known one."""
        s = stem(t)
        if s in self.known or t in self.known or len(s) < 5 or not s.isalpha():
            return s
        if s not in self.fixed:
            best = None
            for n in (len(s) - 1, len(s), len(s) + 1):
                for c in self.by_shape.get((s[0], n), ()):
                    if near(s, c) and (best is None or (-self.known[c], c) < (-self.known[best], best)):
                        best = c
            self.fixed[s] = best or s
        return self.fixed[s]

    def q_words(self, query: str) -> list[str]:
        return [self.fix(t) for t in re.findall(r"[a-z0-9]+", query.lower())]

    def q_terms(self, query: str) -> list[str]:
        out = []
        for t in re.findall(r"[a-z0-9]+", query.lower()):
            if t in STOP:
                continue
            s = self.fix(t)
            if s != stem(t) and s in STOP:  # a mistyped stopword ("thier")
                continue
            out.append(s)
            if s in SYNONYMS:
                out += [stem(x) for x in SYNONYMS[s].split()]
        return out

    def mentions(self, query: str) -> list[tuple]:
        """Subjects named in the query. A longer name hides shorter names inside it."""
        q, used, out = self.q_words(query), set(), []
        for name, targets in self.names:
            n = len(name)
            for s in range(len(q) - n + 1):
                if tuple(q[s:s + n]) == name and not used & set(range(s, s + n)):
                    used |= set(range(s, s + n))
                    for t in targets:
                        if t not in out:
                            out.append(t)
        return out

    def subjects(self, question: str, last: dict | None = None) -> tuple[list, list]:
        """(named, carried): the subjects the question names, and those it inherits from the previous turn.
        `last` is that turn: {"a": its answer, "topic": its subjects}. Only a follow-up inherits: first who the
        previous question was about, then anyone its answer introduced."""
        named = self.mentions(question)
        if not last or not follows(question):
            return named, []
        topic = [tuple(t) for t in last["topic"]]
        prev = topic[:2] + [t for t in self.mentions(last["a"]) if t not in topic][:2]
        return named, [t for t in prev if t not in named]

    def search(self, query: str, k: int = 3, carried=()) -> list[tuple[float, dict]]:
        score = defaultdict(float)
        for t in self.q_terms(query):
            for i, tf in self.postings.get(t, ()):
                score[i] += (CARD_WEIGHT if self.passages[i].get("card") else 1.0) * bm25(tf, self.idf[t], self.len[i], self.avgdl)
        # How closely does the question match a known way of asking for a fact? Shared words over all words,
        # each weighted by rarity, so "how many" alone counts for little and a full paraphrase for a lot.
        q = list(dict.fromkeys(self.q_words(query)))
        q_weight = sum(self.ask_idf.get(w, self.ask_rare) for w in q)
        match = {}
        for i, ask in self.asks:
            shared = sum(self.ask_idf[w] for w in q if w in ask)  # `ask` is short: a list scan is fine
            if shared:
                sim = shared / (q_weight + sum(self.ask_idf[w] for w in ask if w not in q))
                if sim > match.get(i, 0.0):
                    match[i] = sim
        for i, sim in match.items():
            score[i] += ASK_WEIGHT * sim
        named = self.mentions(query)
        subjects = named + [t for t in (tuple(c) for c in carried) if t not in named]
        if subjects:  # a passage that is neither about a subject nor mentions one is probably about someone else
            calls = [n for t in subjects for n in self.called.get(t, ())]
            for i in score:
                p = self.passages[i]
                if any(p["title"] == (card if p.get("card") else page) for card, page in subjects):
                    continue
                if not any(n in self.wtext[i] for n in calls):
                    score[i] *= OFF_SUBJECT
        # What the question names goes first: the best fact about it, or its page's opening passage.
        # A subject carried over from the previous turn is pinned too, and is always given a place.
        pinned, late = [], []
        for t in subjects:
            card, page = t
            pins = pinned if t in named else late
            if card is not None:
                for i in self.facts[card]:
                    score[i] += FACT_BOOST
                best = max(self.facts[card], key=lambda i: (score[i], -i))
                if best not in pins:
                    pins.append(best)
            for i in self.pages.get(page, ()):
                score[i] += PAGE_BOOST
            if page in self.lead:
                score[self.lead[page]] += LEAD_BOOST if card is not None else TITLE_BOOST
                if card is None and self.lead[page] not in pins:
                    pins.append(self.lead[page])
        pinned = pinned[:k - 1 if late else k]
        room = min(k - len(pinned), 1 if pinned else k - 1)
        pinned += [i for i in late if i not in pinned][:max(room, 0)]
        rest = sorted((x for x in score.items() if x[0] not in pinned), key=lambda x: (-x[1], x[0]))[:k - len(pinned)]
        top = [(i, score[i]) for i in pinned] + [x for x in rest if x[1] > 0]
        return [(s, self.passages[i]) for i, s in sorted(top, key=lambda x: (-x[1], x[0]))]  # best match is read first


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
    print("names:", [index.passages[index.facts[c][0]]["title"] if c is not None else page for c, page in index.mentions(q)])
    for s, c in index.search(q, k=3):
        print(f"{s:6.2f}  [{c['title']}: {c['section']}] {c['text'][:110]}...")
