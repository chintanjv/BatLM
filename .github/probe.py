"""One-off diagnostics for the try branch."""
import json
from batlm import cards, rag

chunks = json.loads((rag.ROOT / "data" / "chunks.json").read_text())
cast = rag.cast_facts(chunks, cards.passages())
print(len(cast), "cast passages; from cards:")
for p in [p for p in cast if p["url"] == ""][:30]:
    print("   ", p["title"], "|", p["text"])
print("The Batman (film) cast lines:", [p["text"] for p in cast if p["title"] == "The Batman (film)"])
index = rag.load()
for q in ["who is the batman?", "Who plays the Penguin in The Batman?", "Who played Alfred in The Dark Knight?", "Who is the Dark Knight?"]:
    print("\\n=====", q, index.groups(q))
    for s, p in index.search(q, k=3):
        print(f"{s:7.2f}  [{p['title']}: {p['section']}] {p['text'][:120]}")
