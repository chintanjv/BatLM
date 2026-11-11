"""Stage 1 — Corpus: fetch the Batman pages from Wikipedia and cut them into passages.

    python -m batlm.scrape

Each page is fetched once as plain text (cached under data/raw/, not tracked in git) and written to
data/chunks.json as ~80-word passages that keep their page title and section.
Text is CC BY-SA 4.0, from en.wikipedia.org.
"""
import json
import pathlib
import re
import time

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
    (1000, """Batman (TV series) | Batman (1966 film) | Batman (1989 film) | Batman Returns | Batman Forever |
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

# Top-level sections that are lists, credits or commentary rather than facts about the subject.
SKIP = {"see also", "references", "notes", "further reading", "external links", "sources", "citations",
        "bibliography", "collected editions", "footnotes", "accolades", "awards", "awards and nominations",
        "discography", "filmography", "soundtrack", "music", "home media", "merchandise", "marketing",
        "box office", "reception", "critical reception", "critical response", "sales", "works", "publications",
        "explanatory notes", "general and cited references", "works cited", "analysis", "themes", "release"}


def slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def fetch(title: str) -> dict | None:
    """Return {'title', 'text'} for a page, following redirects. Cached; None if the page does not exist."""
    cache = RAW / f"{slug(title)}.json"
    if not cache.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        for attempt in range(8):
            try:
                r = requests.get(API, headers=UA, timeout=30, params={
                    "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                    "titles": title, "format": "json", "formatversion": 2, "maxlag": 5})
            except requests.ConnectionError:  # dropped connection: wait and try again
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code != 429 and "error" not in r.json():
                break
            time.sleep(int(r.headers.get("Retry-After", 0)) or 5 * (attempt + 1))  # rate-limited: back off
        r.raise_for_status()
        cache.write_text(r.text)
        time.sleep(1.0)
    page = json.loads(cache.read_text())["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    return {"title": page["title"], "text": page["extract"]}


def sections(text: str, cap: int) -> list[tuple[str, str]]:
    """(section, paragraph) pairs in page order, skipping list-like sections, up to `cap` words."""
    out, top, section, words = [], "Overview", "Overview", 0
    for line in text.split("\n"):
        line = line.strip()
        if m := re.match(r"^(=+)\s*(.+?)\s*=+$", line):
            section = m.group(2)
            if len(m.group(1)) == 2:
                top = section
            continue
        n = len(line.split())
        if n < 8 or top.lower() in SKIP:
            continue
        if words + n > cap and words:
            break
        out.append((section, re.sub(r"\s+", " ", line)))
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
    out, seen, missing = [], set(), []
    for cap, block in PAGES:
        for title in (t.strip() for t in block.replace("\n", " ").split("|")):
            page = fetch(title)
            if page is None:
                missing.append(title)
                continue
            if page["title"] in seen:  # two list entries redirected to the same page
                continue
            seen.add(page["title"])
            url = "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_")
            for section, text in chunk(sections(page["text"], cap)):
                out.append({"id": len(out), "title": display(page["title"]), "section": section, "text": text, "url": url})
    (ROOT / "data" / "chunks.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    words = sum(len(c["text"].split()) for c in out)
    print(f"pages={len(seen)} passages={len(out)} words={words}")
    if missing:
        print("missing:", " | ".join(missing))


if __name__ == "__main__":
    main()
