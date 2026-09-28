// Mirror of batlm/rag.py — keep in sync so the browser retrieves exactly what evals measured.

export type Chunk = { id: number; source: string; section: string; text: string; url: string };
export type Hit = { score: number; chunk: Chunk };

const STOP = new Set(
  `a an the and or but of to in on at by for with from as is are was were be been being it its this that
these those he she they his her their him them who whom which what when where why how does do did has have had
not no can could would should will shall may might i you me my your we our us about into than then there also
tell name describe list give s`.split(/\s+/),
);

const SYNONYMS: Record<string, string> = {
  mother: "parents", father: "parents", mom: "parents", dad: "parents",
  children: "son", child: "son", kid: "son", girlfriend: "love interest", wife: "love interest",
  villain: "foe rogues", enemy: "foe rogues", enemie: "foe rogues", nemesi: "archenemy foe", worst: "archenemy",
  superhuman: "superpowers powers", power: "superpowers", creator: "created", made: "created",
  came: "created", debut: "debuted first appeared", appearance: "debuted appeared",
  live: "resides residence", reside: "residence", home: "residence manor", drink: "ginger ale teetotaler",
  raised: "raised brought", alias: "identity", real: "identity", sale: "sold copies", sold: "copies",
  car: "vehicle batmobile", headquarter: "batcave", gadget: "utility belt", undercover: "disguise",
  asylum: "arkham", treated: "psychiatric", cowl: "identity persona batman", worn: "assume",
  inspired: "inspiration", pulp: "pulp sherlock", reason: "because", movie: "films", actor: "portrayed",
};

function stem(t: string): string {
  for (const suf of ["ing", "ed", "es", "s"]) {
    if (t.length > 4 && t.endsWith(suf)) return t.slice(0, -suf.length);
  }
  return t;
}

export function terms(text: string, expand = false): string[] {
  const out: string[] = [];
  for (const t of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (STOP.has(t)) continue;
    const s = stem(t);
    out.push(s);
    if (expand && SYNONYMS[s]) out.push(...SYNONYMS[s].split(" ").map(stem));
  }
  return out;
}

export class BM25 {
  private docs: string[][];
  private tf: Map<string, number>[];
  private idf = new Map<string, number>();
  private avgdl: number;
  /** Terms in >30% of chunks ("batman"): ~0 BM25 weight, so the relevance gate treats them separately. */
  anchors = new Set<string>();

  constructor(public chunks: Chunk[], private k1 = 1.5, private b = 0.75) {
    this.docs = chunks.map((c) => terms(`${c.section} ${c.text}`));
    this.tf = this.docs.map((d) => {
      const m = new Map<string, number>();
      d.forEach((t) => m.set(t, (m.get(t) ?? 0) + 1));
      return m;
    });
    this.avgdl = this.docs.reduce((s, d) => s + d.length, 0) / this.docs.length;
    const df = new Map<string, number>();
    this.docs.forEach((d) => new Set(d).forEach((t) => df.set(t, (df.get(t) ?? 0) + 1)));
    const n = this.docs.length;
    df.forEach((f, t) => {
      this.idf.set(t, Math.log(1 + (n - f + 0.5) / (f + 0.5)));
      if (f / n > 0.3) this.anchors.add(t);
    });
  }

  mentionsAnchor(query: string): boolean {
    return terms(query).some((t) => this.anchors.has(t));
  }

  private score(q: string[], i: number): number {
    const tf = this.tf[i], dl = this.docs[i].length;
    let s = 0;
    for (const t of q) {
      const f = tf.get(t);
      if (f) s += (this.idf.get(t)! * f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * dl) / this.avgdl));
    }
    return s;
  }

  search(query: string, k = 3): Hit[] {
    const q = terms(query, true);
    return this.chunks
      .map((chunk, i) => ({ score: this.score(q, i), chunk }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .filter((h) => h.score > 0);
  }
}
