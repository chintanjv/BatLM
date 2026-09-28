"""Stage 1 — Data acquisition & cleaning.

Fetches the two source pages (cached under data/raw/), strips navigation,
references and tables, and writes:
  data/corpus/<source>.txt   plain prose, one paragraph per line
  data/chunks.json           ~120-word passages with section titles (for RAG)
"""
import json
import pathlib
import re

import requests
from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CORPUS = ROOT / "data" / "corpus"
UA = {"User-Agent": "BatLM/0.1 (educational open-source fan project)"}

SOURCES = {
    "wikipedia": {
        "url": "https://en.wikipedia.org/wiki/Batman",
        "api": "https://en.wikipedia.org/w/api.php?action=parse&page=Batman&prop=text&format=json&formatversion=2",
    },
    "fandom": {
        "url": "https://batman.fandom.com/wiki/Batman_Wiki",
        "api": "https://batman.fandom.com/api.php?action=parse&page=Batman_Wiki&prop=text&format=json&formatversion=2",
    },
}

# Sections that are lists of links / citations rather than prose.
SKIP_SECTIONS = {"see also", "references", "notes", "further reading", "external links", "sources", "citations", "bibliography"}


def fetch(name: str) -> str:
    cache = RAW / f"{name}.json"
    if not cache.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get(SOURCES[name]["api"], headers=UA, timeout=30)
        r.raise_for_status()
        cache.write_text(r.text)
    return json.loads(cache.read_text())["parse"]["text"]


def clean(text: str) -> str:
    text = re.sub(r"\[(\d+|[a-z]|citation needed|note \d+|nb \d+)\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def extract(html: str) -> list[tuple[str, str]]:
    """Return (section, paragraph) pairs in document order."""
    soup = BeautifulSoup(html, "lxml")
    for sel in ["table", "style", "script", "sup.reference", ".reflist", ".navbox", ".hatnote",
                ".mw-editsection", "figure", ".thumb", ".gallery", ".infobox", ".shortdescription",
                ".mw-empty-elt", "noscript"]:
        for el in soup.select(sel):
            el.decompose()

    out, section = [], "Overview"
    for el in soup.find_all(["h2", "h3", "h4", "p", "li", "div"]):
        if el.name in ("h2", "h3", "h4"):
            section = clean(el.get_text(" "))
            continue
        if section.lower() in SKIP_SECTIONS:
            continue
        if el.name == "div":
            # Fandom main page uses <div class="header"/"body"> blocks instead of <p>.
            classes = el.get("class") or []
            if "header" in classes:
                section = clean(el.get_text(" "))
                continue
            if el.find(["p", "div", "li", "ul"]) is not None or "body" not in classes:
                continue
        if el.name == "li" and el.find_parent(["li"]) is not None:
            continue
        t = clean(el.get_text(" "))
        min_words = 5 if el.name == "li" else 8
        if len(t.split()) >= min_words:
            out.append((section, t))
    return out


def chunk(paras: list[tuple[str, str]], source: str, target_words: int = 80) -> list[dict]:
    """Greedy sentence packing: never split a sentence, never cross a section boundary."""
    chunks, buf, sec = [], [], None
    for section, p in paras:
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", p):
            if buf and (section != sec or sum(len(b.split()) for b in buf) + len(sent.split()) > target_words):
                chunks.append({"source": source, "section": sec, "text": " ".join(buf)})
                buf = []
            sec = section
            buf.append(sent)
    if buf:
        chunks.append({"source": source, "section": sec, "text": " ".join(buf)})
    return chunks


def main():
    CORPUS.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for name in SOURCES:
        paras = extract(fetch(name))
        (CORPUS / f"{name}.txt").write_text("\n".join(p for _, p in paras) + "\n")
        c = chunk(paras, name)
        all_chunks += c
        words = sum(len(p.split()) for _, p in paras)
        print(f"{name:10s} paragraphs={len(paras):4d} words={words:6d} chunks={len(c)}")
    for i, c in enumerate(all_chunks):
        c["id"] = i
        c["url"] = SOURCES[c["source"]]["url"]
    (ROOT / "data" / "chunks.json").write_text(json.dumps(all_chunks, indent=1))


if __name__ == "__main__":
    main()
