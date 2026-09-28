"""Stage 2 — A byte-level BPE tokenizer written from scratch (GPT-2 style, minbpe-inspired).

Text is pre-split with a regex (so merges never cross word boundaries), turned into
UTF-8 bytes (ids 0-255), then the most frequent adjacent pair is merged repeatedly
until we reach `vocab_size`. The same algorithm is re-implemented in the website
(web/src/lib/bpe.ts) so the browser tokenizes exactly like training did.
"""
import json
from collections import Counter

import regex

SPLIT = r"""'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"""
SPECIALS = {"<|endoftext|>": None}  # id assigned after merges


def _pairs(ids):
    return Counter(zip(ids, ids[1:]))


def _merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = merges or []  # list of (a, b) in rank order; new id = 256 + rank
        self._build()

    def _build(self):
        self.ranks = {tuple(p): i for i, p in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eot = 256 + len(self.merges)

    @property
    def vocab_size(self):
        return self.eot + 1

    @classmethod
    def train(cls, text, vocab_size=512, verbose=False):
        chunks = [list(c.encode("utf-8")) for c in regex.findall(SPLIT, text)]
        merges = []
        for i in range(vocab_size - 257):
            stats = Counter()
            for c in chunks:
                stats.update(_pairs(c))
            if not stats:
                break
            pair, n = stats.most_common(1)[0]
            if n < 2:
                break
            merges.append(pair)
            chunks = [_merge(c, pair, 256 + i) for c in chunks]
            if verbose and i % 50 == 0:
                tok = cls(merges)
                print(f"merge {i:4d}: {tok.vocab[256 + i]!r} ({n}x)")
        return cls(merges)

    def encode(self, text):
        out = []
        for part in regex.findall(SPLIT, text):
            ids = list(part.encode("utf-8"))
            while len(ids) >= 2:
                pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 1e9))
                if pair not in self.ranks:
                    break
                ids = _merge(ids, pair, 256 + self.ranks[pair])
            out += ids
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids if i != self.eot).decode("utf-8", errors="replace")

    def save(self, path):
        json.dump({"type": "byte-bpe", "split": SPLIT, "merges": self.merges, "eot": self.eot}, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls([tuple(m) for m in json.load(open(path))["merges"]])
