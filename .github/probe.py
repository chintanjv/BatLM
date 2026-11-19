"""One-off diagnostics for the try branch: the full scoring behind questions that still fail."""
from batlm import rag
from batlm.agent import plan

index = rag.load()
for q in ["Who played Alfred in The Dark Knight?", "Who played Catwoman in The Dark Knight Rises?", "Who plays the Penguin in The Batman?"]:
    print("\n=====", q)
    print("named:", index.mentions(q))
    for s, p in index.search(q, k=10):
        print(f"{s:7.2f}  [{p['title']}: {p['section']}] {p['text'][:120]}")
    alf = [(i, p) for i, p in enumerate(index.passages) if "Caine" in p["text"] and "Dark Knight" in p["title"]]
    print("passages naming Caine in a Dark Knight page:", [(p["title"], p["section"], p["text"][:80]) for _, p in alf[:4]])
