"""Stage 1 — Corpus: fetch the Batman pages from Wikipedia and cut them into passages.

    python -m batlm.scrape

The pages are the hand-picked list below plus every article in Wikipedia's Batman categories.
Each page is fetched once as plain text (cached under data/raw/, not tracked in git) and written to
data/chunks.json as ~80-word passages that keep their page title and section.
Text is CC BY-SA 4.0, from en.wikipedia.org.
"""
import json
import pathlib
import re
import time
from collections import Counter

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "corpus-builder/1.0 (educational open-source project)"}

# (word cap per page, titles). The cap keeps the lead and the first sections, where the defining facts are.
PAGES = [
    (4000, """Batman | Joker (character) | Gotham City | Robin (character) | Alfred Pennyworth | Catwoman |
        James Gordon (character) | List of Batman family enemies | List of Batman supporting characters"""),
    # heroes and allies
    (1600, """Dick Grayson | Jason Todd | Tim Drake | Damian Wayne | Stephanie Brown (character) | Barbara Gordon |
        Cassandra Cain | Batgirl | Nightwing | Batwoman | Red Hood | Lucius Fox | Harvey Bullock (character) |
        Renee Montoya | Leslie Thompkins | Vicki Vale | Thomas Wayne | Martha Wayne | Azrael (DC Comics) |
        Huntress (Helena Bertinelli) | Batwing (DC Comics) | Duke Thomas (character) | Harper Row | Ace the Bat-Hound |
        Bat-Mite | Talia al Ghul | Julie Madison | Joe Chill | Terry McGinnis | Carrie Kelley | Batman Incorporated |
        Outsiders (comics) | Birds of Prey (team) | Gotham City Police Department | Sarah Essen | Crispus Allen |
        Silver St. Cloud | Rachel Dawes | Andrea Beaumont | Oracle (DC Comics) | Luke Fox (character) | Jace Fox |
        Kathy Kane | Bette Kane | Holly Robinson (character)"""),
    # villains
    (1600, """Harley Quinn | Penguin (character) | Riddler | Two-Face | Bane (DC Comics) | Scarecrow (DC Comics) |
        Ra's al Ghul | Mr. Freeze | Poison Ivy (character) | Killer Croc | Clayface | Mad Hatter (DC Comics) |
        Hush (character) | Black Mask (character) | Deathstroke | Court of Owls | Man-Bat | Hugo Strange |
        Victor Zsasz | Carmine Falcone | Sal Maroni | Firefly (DC Comics) | Calendar Man | Professor Pyg | Killer Moth |
        Ventriloquist (character) | Deadshot | Solomon Grundy (character) | Anarky | Lady Shiva | League of Assassins |
        Owlman (character) | Punchline (DC Comics) | Catman (comics) | KGBeast | Maxie Zeus | Tweedledum and Tweedledee (comics) |
        The Batman Who Laughs | Talon (DC Comics) | Cluemaster | Rupert Thorne | Tony Zucco | Nyssa Raatko |
        Simon Hurt | Crazy Quilt | Copperhead (DC Comics) | Arkham Knight | Jeremiah Arkham | Kite Man | Condiment King |
        Joker (The Dark Knight) | Great White Shark (character) |
        Roland Daggett | Doctor Death (character) | Monk (comics) | Egghead (DC Comics) |
        King Tut (comics) | Bookworm (comics) | Mr. Bloom | Flamingo (comics) | Roxy Rocket | Baby-Doll |
        Joker's Daughter | Gotham City Sirens"""),
    # places and gear
    (1400, """Arkham Asylum | Batcave | Wayne Manor | Wayne Enterprises | Batmobile | Batsuit | Batman's utility belt |
        Bat-Signal | Batarang | Batplane | Batcycle | Batboat | Blackgate Penitentiary | Batcomputer | Iceberg Lounge | Batcopter | Ace Chemicals | Batman's rogues gallery"""),
    # comics and storylines
    (1200, """Batman: Year One | Batman: The Killing Joke | The Dark Knight Returns | Batman: Knightfall |
        Batman: The Long Halloween | Batman: Hush | Batman: A Death in the Family | Batman: No Man's Land |
        Batman: Under the Hood | Batman: Death of the Family | Batman R.I.P. |
        Batman: Dark Victory | Arkham Asylum: A Serious House on Serious Earth | Batman: The Man Who Laughs |
        Batman: Zero Year | Batman: Endgame | Batman and Son | Batman: War Games | Batman: Cataclysm |
        Batman: Contagion | Batman: Year Two | Batman: A Lonely Place of Dying | Batman: The Cult |
        Batman: Son of the Demon | Gotham by Gaslight | Batman: White Knight | Batman: Earth One |
        Batman: Battle for the Cowl | Batman: The Return of Bruce Wayne | The Joker War | Batman: Three Jokers |
        Dark Nights: Metal | Detective Comics | Batman (comic book) | Batman: Legends of the Dark Knight |
        Detective Comics 27 | The Dark Knight Strikes Again | Batman: Ego | Batman Black and White | Absolute Batman |
        Batman: Shadow of the Bat | Batman Eternal | Batman: The Black Mirror | Batman: Prey | Batman: Gothic | Batman: The 12 Cent Adventure |
        Batman: Night of the Owls | Batman: Last Knight on Earth | Batman: Damned | Batman: The Imposter |
        Batman/Superman | World's Finest Comics | The Brave and the Bold | Batman and Robin (comic book) |
        Batman: KnightsEnd | Batman: Haunted Knight | Catwoman: When in Rome | Gotham Central | Joker (graphic novel) |
        Batman: The Doom That Came to Gotham | Batman: Holy Terror | Kingdom Come (comics)"""),
    # film, television and games
    (1300, """Batman (TV series) | Batman (1966 film) | Batman (1989 film) | Batman Returns | Batman Forever |
        Batman & Robin (film) | Batman Begins | The Dark Knight | The Dark Knight Rises |
        Batman v Superman: Dawn of Justice | The Batman (film) | Joker (2019 film) | Joker: Folie à Deux |
        The Lego Batman Movie | Batman: The Animated Series | The New Batman Adventures | Batman Beyond |
        The Batman (TV series) | Batman: The Brave and the Bold | Beware the Batman | Batman: Caped Crusader |
        Gotham (TV series) | Batwoman (TV series) | Pennyworth (TV series) | Batman: Mask of the Phantasm |
        Batman: Under the Red Hood | Batman: Arkham Asylum | Batman: Arkham City | Batman: Arkham Origins |
        Batman: Arkham Knight | Batman: Arkham | The Penguin (TV series) | Batman in film | Batman (serial) |
        Batman and Robin (serial) | Harley Quinn (TV series) | Suicide Squad (2016 film) | Birds of Prey (2020 film) |
        Batman: The Telltale Series | Lego Batman: The Videogame | The Dark Knight Trilogy | Batman franchise media |
        Joker in other media | DC Animated Universe | Batman: Return of the Caped Crusaders | Batman: Gotham Knight |
        Batman Ninja | Batman: The Killing Joke (film) | Batman: Hush (film) | Gotham Knights (video game) |
        Batman: Year One (film) | Batman: The Dark Knight Returns (film) | Batman: The Long Halloween (film) |
        Batman Beyond: Return of the Joker | Batman & Mr. Freeze: SubZero | Batman: Mystery of the Batwoman |
        Son of Batman | Batman vs. Robin | Batman: Bad Blood | Batman: Assault on Arkham | The Batman Part II |
        Batman: Arkham Shadow | Batman: The Enemy Within | Justice League (film) | Batman in other media |
        Catwoman (film) | Batman: Gotham by Gaslight (film) | Merry Little Batman | Batman: Soul of the Dragon |
        Batman: The Doom That Came to Gotham (film) | The Batman vs. Dracula | Batman Unlimited |
        Batman: Death in the Family | Lego Batman 2: DC Super Heroes | Lego Batman 3: Beyond Gotham |
        Batman: Arkham Origins Blackgate | Batman: Vengeance | Batman: Rise of Sin Tzu | Batman Begins (video game) |
        Batman: Dark Tomorrow | Harley Quinn in other media | Catwoman in other media | Robin in other media |
        Penguin in other media | Riddler in other media | Two-Face in other media | Bane in other media |
        Scarecrow in other media | Mr. Freeze in other media | Poison Ivy in other media"""),
    # the people behind Batman (lead only: the rest of these pages is not about Batman)
    (450, """Bob Kane | Bill Finger | Jerry Robinson | Frank Miller | Dennis O'Neil | Neal Adams | Grant Morrison |
        Scott Snyder | Jeph Loeb | Tim Sale | Paul Dini | Bruce Timm | Kevin Conroy | Mark Hamill | Jim Lee |
        Greg Capullo | Tom King (writer) | Chuck Dixon | Dick Sprang | Gardner Fox | Sheldon Moldoff |
        Christopher Nolan | Tim Burton | Matt Reeves | Joel Schumacher | Adam West | Burt Ward | Michael Keaton |
        Val Kilmer | George Clooney | Christian Bale | Ben Affleck | Robert Pattinson | Heath Ledger |
        Jack Nicholson | Joaquin Phoenix | Cesar Romero | Michael Caine | Danny DeVito | Michelle Pfeiffer |
        Anne Hathaway | Zoë Kravitz | Tom Hardy | Colin Farrell | Paul Dano | Gary Oldman | Jeffrey Wright |
        Arleen Sorkin | Alan Moore | Brian Bolland | Marv Wolfman | Doug Moench | Norm Breyfogle | Alan Grant (writer) |
        Jim Aparo | Carmine Infantino | Julius Schwartz | Steve Englehart | Marshall Rogers | Lee Bermejo |
        James Tynion IV | Chip Zdarsky | Sean Murphy (artist) | Hans Zimmer | Danny Elfman"""),
]

# Wikipedia categories walked for more pages, after the list above: (word cap per page, levels of subcategories, category).
CATEGORIES = [
    (1000, 1, "Batman characters"), (1000, 1, "Batman storylines"), (1300, 2, "Batman films"),
    (1000, 2, "Batman television series"), (800, 1, "Batman video games"), (600, 1, "Batman titles"),
    (800, 1, "Batman in other media"), (800, 0, "Batman objects"), (800, 1, "Gotham City"),
]
AUTO_LIMIT = 1200  # at most this many pages from the categories
SKIP_TITLE = re.compile(r"^(?:Lists? of|Outline of|Index of)\b|\((?:soundtrack|album|song|score)\)$|discography", re.I)

# Top-level sections that are lists, credits or commentary rather than facts about the subject.
SKIP = {"see also", "references", "notes", "further reading", "external links", "sources", "citations",
        "bibliography", "collected editions", "footnotes", "accolades", "awards", "awards and nominations",
        "discography", "filmography", "soundtrack", "music", "home media", "merchandise", "marketing",
        "box office", "reception", "critical reception", "critical response", "sales", "works", "publications",
        "explanatory notes", "general and cited references", "works cited", "analysis", "themes", "release"}
# Sections read straight after the lead, because people ask about them and the word cap would otherwise cut them off.
FIRST = {"cast", "voice cast", "cast and characters", "characters", "main characters", "premise"}


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def get(params: dict) -> str:
    """One Wikipedia API call, as text. Waits and retries when rate-limited or disconnected."""
    r = None
    for attempt in range(8):
        try:
            r = requests.get(API, headers=UA, timeout=30, params={**params, "format": "json", "formatversion": 2, "maxlag": 5})
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(10 * (attempt + 1))
            continue
        try:
            if r.status_code != 429 and "error" not in r.json():
                break
        except ValueError:  # an error page instead of JSON
            pass
        time.sleep(int(r.headers.get("Retry-After", 0)) or 5 * (attempt + 1))
    if r is None:
        raise RuntimeError(f"could not reach Wikipedia for {params}")
    r.raise_for_status()
    time.sleep(1.0)
    return r.text


def fetch(title: str) -> dict | None:
    """Return {'title', 'text'} for a page, following redirects. Cached; None if the page does not exist."""
    cache = RAW / f"{slug(title)}.json"
    if not cache.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        cache.write_text(get({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title}))
    page = json.loads(cache.read_text())["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    return {"title": page["title"], "text": page["extract"]}


def members(category: str, depth: int) -> list[str]:
    """Titles of the articles in a category and `depth` levels of its subcategories. Cached."""
    cache = RAW / "categories" / f"{slug(category)}_{depth}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out, todo, seen = [], [(category, depth)], set()
    while todo:
        cat, d = todo.pop(0)
        if cat in seen:
            continue
        seen.add(cat)
        more = {}
        while True:
            r = json.loads(get({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
                                "cmlimit": 500, "cmtype": "page|subcat", **more}))
            for m in r.get("query", {}).get("categorymembers", []):
                if m["ns"] == 0 and m["title"] not in out:
                    out.append(m["title"])
                elif m["ns"] == 14 and d > 0:
                    todo.append((m["title"].split(":", 1)[1], d - 1))
            if "continue" not in r:
                break
            more = r["continue"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def sections(text: str, cap: int) -> list[tuple[str, str]]:
    """(section, paragraph) pairs, skipping list-like sections, up to `cap` words: the lead, then cast lists
    (at most 40% of the cap), then the rest in page order."""
    lead, first, rest, top, section = [], [], [], "Overview", "Overview"
    for line in text.split("\n"):
        line = line.strip()
        if m := re.match(r"^(=+)\s*(.+?)\s*=+$", line):
            section = m.group(2)
            if len(m.group(1)) == 2:
                top = section
            continue
        n = len(line.split())
        early = top.lower() in FIRST
        if n < (4 if early else 8) or top.lower() in SKIP:  # a cast entry can be short: "Michael Caine as Alfred Pennyworth"
            continue
        (lead if top == "Overview" else first if early else rest).append((section, re.sub(r"\s+", " ", line), n))
    out, words, early_words = [], 0, 0
    for group in (lead, first, rest):
        for section, line, n in group:
            if group is first:
                if early_words + n > 0.4 * cap:
                    break
                early_words += n
            if words + n > cap and words:
                return out
            out.append((section, line))
            words += n
    return out


def chunk(paras: list[tuple[str, str]], target_words: int = 80) -> list[tuple[str, str]]:
    """Greedy sentence packing: never split a sentence, never cross a section boundary."""
    chunks, buf, sec = [], [], None
    for section, p in paras:
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", p):
            if buf and (section != sec or sum(len(b.split()) for b in buf) + len(sent.split()) > target_words):
                chunks.append((sec, " ".join(buf)))
                buf = []
            sec = section
            buf.append(sent)
    if buf:
        chunks.append((sec, " ".join(buf)))
    return chunks


def display(title: str) -> str:
    """'Joker (character)' -> 'Joker': the disambiguation suffix is noise in a prompt."""
    return re.sub(r"\s+\((character|DC Comics|comics|comic book|identity|team)\)$", "", title)


def main():
    out, seen, shown, missing, auto = [], set(), set(), [], set()
    wanted = [(cap, t.strip(), False) for cap, block in PAGES for t in block.replace("\n", " ").split("|")]
    listed = {t for _, t, _ in wanted}
    found = []
    for cap, depth, category in CATEGORIES:
        titles = [t for t in members(category, depth) if not SKIP_TITLE.search(t)]
        print(f"category {category!r}: {len(titles)} pages", flush=True)
        found += [(cap, t, True) for t in titles if t not in listed and t not in {x[1] for x in found}]
    wanted += found[:AUTO_LIMIT]
    for n, (cap, title, from_category) in enumerate(wanted):
        if n % 100 == 0:
            print(f"page {n} of {len(wanted)}", flush=True)
        page = fetch(title)
        if page is None:
            missing.append(title)
            continue
        name = display(page["title"])
        if page["title"] in seen or name in shown:  # two entries redirected to the same page, or two pages that display alike
            continue
        seen.add(page["title"])
        shown.add(name)
        if from_category:
            auto.add(name)
        url = "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_")
        for section, text in chunk(sections(page["text"], cap)):
            out.append({"id": len(out), "title": name, "section": section, "text": text, "url": url})
    # A category page titled with one everyday word ("Legacy") must not turn that word into a Batman subject:
    # if the word also turns up in lowercase in running text, search will not treat it as a name.
    lower = Counter(w for c in out for w in set(re.findall(r"\b[a-z]{3,}\b", c["text"])))
    plain = {t for t in auto if re.fullmatch(r"(?:The )?[A-Za-z]+", t) and lower[t.split()[-1].lower()] >= 3}
    for c in out:
        if c["title"] in plain:
            c["name"] = False
    (ROOT / "data" / "chunks.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    words = sum(len(c["text"].split()) for c in out)
    print(f"pages={len(seen)} (from categories: {len(auto)}) passages={len(out)} words={words}")
    if plain:
        print("not used as names:", " | ".join(sorted(plain)))
    if missing:
        print("missing:", " | ".join(missing))


if __name__ == "__main__":
    main()
