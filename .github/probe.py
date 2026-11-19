"""One-off diagnostics for the try branch."""
import json
from batlm import rag

chunks = json.loads((rag.ROOT / "data" / "chunks.json").read_text())
cast = rag.cast_facts(chunks)
print(len(cast), "cast passages, e.g.:")
for p in cast[::max(len(cast) // 25, 1)][:25]:
    print("   ", p["title"], "|", p["text"])
index = rag.load()
for q in ["Who played Alfred in The Dark Knight?", "Who played Catwoman in The Dark Knight Rises?", "Who plays the Penguin in The Batman?",
          "Who played the Riddler in The Batman?", "Who voiced Batman in Batman: The Animated Series?", "Who played Bane in The Dark Knight Rises?"]:
    print("\\n=====", q)
    for s, p in index.search(q, k=3):
        print(f"{s:7.2f}  [{p['title']}: {p['section']}] {p['text'][:120]}")
