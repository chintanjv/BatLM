// Mirror of batlm/tokenizer.py — byte-level BPE, so BatLM-nano sees identical token ids in the browser.

export class BPE {
  private ranks = new Map<string, number>();
  private vocab: Uint8Array[] = [];
  private split: RegExp;
  readonly eot: number;

  constructor(json: { merges: [number, number][]; eot: number }) {
    // Python's `regex` \p{L}/\p{N} classes map directly onto JS unicode property escapes.
    this.split = /'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/gu;
    for (let i = 0; i < 256; i++) this.vocab.push(new Uint8Array([i]));
    json.merges.forEach(([a, b], i) => {
      this.ranks.set(`${a},${b}`, i);
      const m = new Uint8Array(this.vocab[a].length + this.vocab[b].length);
      m.set(this.vocab[a]);
      m.set(this.vocab[b], this.vocab[a].length);
      this.vocab.push(m);
    });
    this.eot = json.eot;
  }

  encode(text: string): number[] {
    const enc = new TextEncoder();
    const out: number[] = [];
    for (const part of text.match(this.split) ?? []) {
      let ids = Array.from(enc.encode(part));
      while (ids.length >= 2) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < ids.length - 1; i++) {
          const r = this.ranks.get(`${ids[i]},${ids[i + 1]}`);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        const pair = [ids[best], ids[best + 1]];
        const merged: number[] = [];
        for (let i = 0; i < ids.length; i++) {
          if (i < ids.length - 1 && ids[i] === pair[0] && ids[i + 1] === pair[1]) { merged.push(256 + bestRank); i++; }
          else merged.push(ids[i]);
        }
        ids = merged;
      }
      out.push(...ids);
    }
    return out;
  }

  /** Bytes for one token — streamed decoding must buffer bytes, since a UTF-8 char can span tokens. */
  bytes(id: number): Uint8Array {
    return id === this.eot ? new Uint8Array() : this.vocab[id];
  }
}
