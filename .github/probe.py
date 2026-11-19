"""One-off diagnostics for the try branch."""
import collections, json
from batlm import evals, rag
index = rag.load()
convs = evals.conversations(index)
print("conversations by source:", collections.Counter(c["id"].split("-")[0] for c in convs))
print("teacher-judged turns:", sum(1 for c in convs for t in c["turns"] if not t["keys"] and t["ref"]))
