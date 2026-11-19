"""One-off diagnostics for the try branch."""
import json
from batlm import cards, rag

cast = rag.cast_facts(json.loads((rag.ROOT / "data" / "chunks.json").read_text()), cards.passages())
print("cast sentences ending badly:", [p["text"] for p in cast if ". in " in p["text"]])
