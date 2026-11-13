"""Stage 4 — Questions and answers for training and for evals, generated from the fact cards and the corpus.

Four kinds of training question:
  * card facts       many phrasings of each fact on a card -> that card's sentence
  * Wikipedia leads  "What is <page>?" for pages without a card -> the page's opening sentence
  * missing          Batman questions the files cannot answer -> MISSING
  * off-topic        everything else -> REFUSAL

Held out for evals, so they measure reading rather than memory:
  * `unseen`    whole facts that never appear in training (about one in eight)
  * `reworded`  one phrasing of every trained fact
  * `wiki`      pages whose lead question is never trained
  * held-out missing and off-topic questions
"""
import hashlib
import re

from . import cards
from .cards import WHAT, WHO

PERSONA = ("You are BatLM, an expert on the Batman universe. Answer the question directly in one or two sentences, "
           "or a short numbered list, using only the CASE FILES. If the files do not contain the answer, say so.")

REFUSAL = "That is outside my case files. I only answer questions about the Batman universe."
MISSING = "I don't have that in my case files."

# Batman questions no file answers. The model must say so rather than invent something.
UNANSWERABLE = ["What is {n}'s favorite food?", "What is {n}'s blood type?", "What is {n}'s phone number?", "What is {n}'s shoe size?",
                "What is {n}'s favorite color?", "What is {n}'s email address?", "What is {n}'s zodiac sign?",
                "What did {n} eat for breakfast today?", "What is {n}'s favorite song?", "How many pets does {n} have?",
                "What is {n}'s favorite band?"]  # last: eval only

OFF_TOPIC = """What is the capital of France? | Who is Iron Man? | Write me a Python function to sort a list. | What's the weather today? |
Who won the 2018 World Cup? | How do I bake bread? | What is quantum computing? | Who is Spider-Man's aunt? | Explain the stock market. |
What is 17 times 23? | Who painted the Mona Lisa? | Recommend a good laptop. | Who is Superman's father? | Where is Wakanda? |
What is Iron Man's suit made of? | Who leads the X-Men? | What is Spider-Man's real name? | Who is Thor's brother? |
What planet is Superman from? | What is the Flash's top speed? | Who is Green Lantern? | Where do the Avengers live? |
Write a poem about the ocean. | Translate hello into Spanish. | What is the square root of 144? | Who invented the telephone? |
What is the speed of light? | How do I cook pasta? | What is the tallest building in the world? | Summarize World War II. |
What is Bitcoin? | Give me a workout plan. | Who is Taylor Swift? | What year did the Titanic sink? | How do vaccines work? |
What is the best programming language? | Who discovered penicillin? | Recommend a movie that isn't about superheroes. |
hello | hi there | how are you? | what can you do? | thanks | What time is it? | Tell me a joke. | What is your name? |
Who is the president of the United States? | How far is the moon? | What is photosynthesis? | Who wrote Hamlet? |
What is the population of India? | How do I learn to code? | What is the meaning of life? | Who is Wolverine? | Who is Captain Marvel? |
What is Thanos's goal? | Who is Black Panther? | What is Hogwarts? | Who is Darth Vader? | Who is Luke Skywalker's father? |
Who is Harry Potter? | Who is Sherlock Holmes's assistant? | What is the plot of Titanic? | Who directed Jaws? | What is Pikachu? |
How do I fix a flat tire? | What is the boiling point of water? | How many continents are there? | What is the largest ocean? |
Who was Albert Einstein? | What is machine learning? | How do airplanes fly? | What is the GDP of Japan? | Write an essay about climate change. |
What should I eat for dinner? | How do I lose weight? | Is it going to rain tomorrow? | What is 2 plus 2? | Solve x squared equals 9. |
Who is Elon Musk? | What is the stock price of Apple? | How do I invest money? | Explain blockchain. | What is the theory of relativity? |
Who won the Super Bowl? | Who is Lionel Messi? | How long is a marathon? | What is the offside rule? | Who is LeBron James? |
What is the best phone to buy? | How do I reset my password? | Write a cover letter for me. | Draft an email to my boss. |
What is Wonder Woman's lasso called? | Who is Aquaman? | Who is Lex Luthor? | What is Kryptonite? | Who is Lois Lane? |
Who is Deadpool? | Who is Doctor Strange? | What is the Infinity Gauntlet? | Who is Hulk? | Who is Loki? | Who is Venom's host? |
What is Star Trek about? | Who is Frodo? | What is Game of Thrones? | Who is James Bond? | Who is Mario? | What is Minecraft? |
asdfghjkl | ??? | 12345 | Give me a recipe for pancakes. | How do I meditate? | What is a black hole? | What causes earthquakes? |
Who built the pyramids? | When was the French Revolution? | What language is spoken in Brazil? | How do I tie a tie? |
What's a good name for a dog? | How many days are in a leap year? | Convert 10 miles to kilometers. | Define the word ephemeral.""".replace("\n", " ").split(" | ")
OFF_TOPIC_EVAL = """Who founded Microsoft? | What is the Avengers' headquarters? | How tall is Mount Everest? | Can you help me with my taxes? |
Who is Wonder Woman's mother? | What is Captain America's shield made of? | Who is the president of France? | How many moons does Mars have? |
How do I change a car tire? | Who is Superman's girlfriend? | What is the capital of Australia? | Write a haiku about autumn. |
Who is Tony Stark's father? | What is the fastest animal on Earth? | Who discovered America? | good morning | What is 9 times 8? |
Who is Peter Parker's best friend? | How do I make coffee? | Who directed The Godfather?""".replace("\n", " ").split(" | ")

# The questions the owner asked for by name. Always in the eval set, whatever else is sampled.
OWNER = [
    ("who is the batman?", "Bruce Wayne", ""),
    ("What is the Joker's real identity?", "no definitive|unknown|never|uncertain|not known", "Ra's al Ghul"),
    ("Who is Alfred?", "butler", ""),
    ("What are the top 5 Batman villains?", "Joker, Two-Face, Penguin", ""),
    ("who is the joker", "archenemy|arch-enemy|enemy", "Ra's al Ghul"),
    ("Who killed Batman's parents?", "Joe Chill", ""),
    ("Who is Harley Quinn?", "Harleen Quinzel|psychiatrist", ""),
    ("Who broke Batman's back?", "Bane", ""),
    ("Who played Batman in The Dark Knight?", "Bale", "Ledger"),
    ("Who played the Joker in the 1989 movie?", "Nicholson", ""),
    ("Who directed The Batman?", "Reeves", ""),
    ("best batman comics to read", "Year One, Dark Knight Returns", ""),
    ("Where does Batman live?", "Wayne Manor", ""),
]


def _bucket(s: str, n: int = 100) -> int:
    """Stable pseudo-random bucket, so the train/eval split never changes between runs."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % n


def unseen(title: str, field: str) -> bool:
    return _bucket(f"{title}/{field}") < 12


def unseen_fact(card, field) -> bool:
    """Lists are a format the model only learns from a handful of examples, so they are always trained."""
    return card["kind"] != "topic" and unseen(card["title"], field)


def lead_sentence(text: str) -> str | None:
    """A page's opening sentence, if it is a usable one- or two-line answer."""
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)[0].strip()
    return s if 8 <= len(s.split()) <= 45 and s.endswith(".") else None


PERSON = re.compile(r"\b(?:is|was) (?:an?|the) [^.]*?\b(actor|actress|writer|artist|director|filmmaker|composer|editor|cartoonist|"
                    r"screenwriter|producer|animator|character|supervillain|superhero|antihero|villain|vigilante)\b")


def natural(title: str) -> str:
    """'Batman (1966 film)' -> 'the 1966 film Batman': how someone would actually say it."""
    m = re.match(r"^(.*?)\s*\((.+)\)$", title)
    return f"the {m.group(2)} {m.group(1)}" if m else title


def wiki_questions(index) -> list[tuple[str, str, str, list[str]]]:
    """(page title, answer, eval question, train questions) for pages that have no card."""
    carded = {p.get("page") for p in index.passages if p.get("card")}
    out = []
    for title, i in index.lead.items():
        a = lead_sentence(index.passages[i]["text"])
        if title in carded or title in index.noname or a is None:
            continue
        n = natural(title)
        t = WHO if PERSON.search(a) else WHAT
        out.append((title, a, cards.fill(t[-1], n), [cards.fill(t[0], n), cards.fill(t[2], n), cards.fill(t[1], n)]))
    return out


def train_items(index) -> list[dict]:
    items = []
    for card in cards.CARDS:
        for field, a, _, _, qs in cards.questions(card):
            if not unseen_fact(card, field):
                items += [{"q": q, "a": a, "kind": "card"} for q in qs]
    for title, a, _, qs in wiki_questions(index):
        if _bucket("wiki/" + title) >= 10:
            items += [{"q": q, "a": a, "kind": "wiki"} for q in qs]
    for p in index.passages:  # teacher-written facts: one of the ways of asking, for a share of them
        if p.get("fact") and p["asks"] and _bucket("fact/" + p["text"]) < 40:
            items.append({"q": p["asks"][0], "a": p["text"], "kind": "fact"})
    named = [c["n"] for c in cards.CARDS if c["n"] and c["kind"] == "who"]
    for i, t in enumerate(UNANSWERABLE[:-1]):
        items += [{"q": cards.fill(t, n), "a": MISSING, "kind": "missing"} for n in named[i::8]]
    items += [{"q": q, "a": REFUSAL, "kind": "off_topic"} for q in OFF_TOPIC]
    for i, it in enumerate(items):  # people often type without capitals or a question mark
        if i % 4 == 3:
            it["q"] = it["q"].lower().rstrip("?.")
    return items


def eval_items(index) -> list[dict]:
    items = [{"q": q, "a": "", "keys": k, "bad": bad, "kind": "owner"} for q, k, bad in OWNER]
    for card in cards.CARDS:
        for field, a, k, q, qs in cards.questions(card):
            if unseen_fact(card, field):
                items.append({"q": q, "a": a, "keys": k, "kind": "unseen"})
            elif _bucket(f"reworded/{card['title']}/{field}") < 30:
                items.append({"q": q, "a": a, "keys": k, "kind": "reworded"})
    for title, a, q, _ in wiki_questions(index):
        if _bucket("wiki/" + title) < 10:
            # the opening sentence names its subject; a right answer repeats the two most telling words
            words = [w for w in re.findall(r"[A-Za-z][a-z]{3,}", a.split(" is ", 1)[-1].split(" was ", 1)[-1])][:2]
            items.append({"q": q, "a": a, "keys": ", ".join(words), "kind": "wiki"})
    for p in index.passages:  # teacher-written facts, asked in the wording kept out of search and training
        if p.get("fact") and _bucket("fact-eval/" + p["text"]) < 2:
            items.append({"q": p["held"], "a": p["text"], "keys": p["keys"], "kind": "fact"})
    named = [c["n"] for c in cards.CARDS if c["n"] and c["kind"] == "who"]
    items += [{"q": cards.fill(UNANSWERABLE[-1], n), "a": MISSING, "keys": "", "kind": "missing"} for n in named[::6]]
    items += [{"q": q, "a": REFUSAL, "keys": "", "kind": "off_topic"} for q in OFF_TOPIC_EVAL]
    return items
