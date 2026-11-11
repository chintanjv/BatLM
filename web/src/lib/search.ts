// Mirror of batlm/rag.py — keep in sync so the browser retrieves exactly what evals measured.

export type Passage = {
  id: number; title: string; section: string; text: string; url: string;
  /** Fact passages only: a fact card's name for the subject, and the usual ways of asking for this fact. */
  card?: boolean; page?: string | null; aliases?: string[]; asks?: string[]; keywords?: string;
};
export type Hit = { score: number; passage: Passage };
/** A subject named in a question: its fact card's title and/or its Wikipedia page title. */
type Target = [card: string | null, page: string | null];

const STOP = new Set(
  `a an the and or but of to in on at by for with from as is are was were be been being it its this that
these those he she they his her their him them who whom which what when where why how does do did has have had
not no can could would should will shall may might i you me my your we our us about into than then there also
tell name describe list give s`.split(/\s+/),
);

const SYNONYMS: Record<string, string> = {
  mother: "parents", father: "parents", mom: "parents", dad: "parents",
  villain: "enemy foe rogues", enemie: "enemy villain", enemy: "villain foe", nemesi: "archenemy", worst: "archenemy",
  superhuman: "superpowers powers", power: "superpowers", creator: "created", made: "created", invented: "created",
  debut: "first appeared", introduced: "first appeared", appearance: "appeared",
  live: "home", home: "manor", car: "vehicle batmobile", headquarter: "batcave", base: "headquarters batcave",
  gadget: "equipment", movie: "film", movies: "films", actor: "played", show: "series television",
  wrote: "written writer", author: "written writer", directed: "director", voice: "voiced",
  real: "name", identity: "name", girlfriend: "love interest", kid: "son", child: "son", children: "son",
  die: "killed", died: "killed", dead: "killed", murdered: "killed", sidekick: "robin partner",
  game: "video", games: "video", comic: "comics", book: "comics",
  brought: "raised", imprisoned: "locked held", jail: "prison locked", contact: "call", summon: "call",
};

const CARD_WEIGHT = 1.5;  // a fact's BM25 score counts extra: facts are short, curated and written to be found
const ASK_WEIGHT = 20.0;  // a fact whose usual phrasing matches the question word for word scores this much
const FACT_BOOST = 3.0;   // facts about a subject the question names
const PAGE_BOOST = 1.5;   // Wikipedia passages from that subject's page
const LEAD_BOOST = 1.0;   // ...and a little more for the page's opening passage
const TITLE_BOOST = 10.0; // the opening passage of a page named in the question, when that page has no card
const K1 = 1.5, B = 0.75;

function stem(t: string): string {
  for (const suf of ["ing", "ed", "es", "s"]) {
    if (t.length > 4 && t.endsWith(suf)) return t.slice(0, -suf.length);
  }
  return t;
}

/** Every word, stemmed, stopwords kept: used to spot names ("the batman who laughs"). */
const words = (text: string): string[] => (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).map(stem);

function terms(text: string, expand = false): string[] {
  const out: string[] = [];
  for (const t of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (STOP.has(t)) continue;
    const s = stem(t);
    out.push(s);
    if (expand && Object.hasOwn(SYNONYMS, s)) out.push(...SYNONYMS[s].split(" ").map(stem));
  }
  return out;
}

const unique = (xs: string[]): string[] => [...new Set(xs)];

export class Index {
  passages: Passage[];
  private len: number[];
  private avgdl: number;
  private postings = new Map<string, [number, number][]>(); // term -> [passage index, term frequency]
  private idf = new Map<string, number>();
  private asks: [number, string[]][] = []; // the ways of asking for each fact, as [fact index, words]
  private askIdf = new Map<string, number>();
  private askRare: number;
  private pages = new Map<string, number[]>();
  private lead = new Map<string, number>();
  private facts = new Map<string, number[]>();
  private names: [string[], Target[]][];

  constructor(passages: Passage[]) {
    this.passages = passages;
    const docs = passages.map((p) => terms(`${p.title} ${p.section} ${p.text} ${p.keywords ?? ""}`));
    this.len = docs.map((d) => d.length);
    this.avgdl = this.len.reduce((s, n) => s + n, 0) / docs.length;
    const push = <K, V>(m: Map<K, V[]>, k: K, v: V) => { const l = m.get(k); if (l) l.push(v); else m.set(k, [v]); };
    docs.forEach((d, i) => {
      const tf = new Map<string, number>();
      for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1);
      tf.forEach((f, t) => push(this.postings, t, [i, f] as [number, number]));
    });
    this.postings.forEach((p, t) => this.idf.set(t, Math.log(1 + (docs.length - p.length + 0.5) / (p.length + 0.5))));
    // Stopwords are kept in the phrasings: "who is" must match.
    passages.forEach((p, i) => p.asks?.forEach((a) => this.asks.push([i, unique(words(a))])));
    const df = new Map<string, number>();
    for (const [, ws] of this.asks) for (const w of ws) df.set(w, (df.get(w) ?? 0) + 1);
    const n = this.asks.length;
    df.forEach((f, w) => this.askIdf.set(w, Math.log(1 + (n - f + 0.5) / (f + 0.5))));
    this.askRare = Math.log(1 + (n + 0.5) / 0.5); // weight of a word no phrasing contains
    passages.forEach((p, i) => {
      if (p.card) return push(this.facts, p.title, i);
      if (!this.pages.has(p.title)) this.lead.set(p.title, i);
      push(this.pages, p.title, i);
    });
    // Names -> what they refer to. Longest names first, so "the batman who laughs" beats "batman".
    const names = new Map<string, Target[]>();
    const add = (name: string, t: Target) => {
      const key = words(name).join(" ");
      const list = names.get(key);
      if (!list) names.set(key, [t]);
      else if (!list.some((x) => x[0] === t[0] && x[1] === t[1])) list.push(t);
    };
    for (const p of passages) p.aliases?.forEach((a) => add(a, [p.title, p.page ?? null]));
    const carded = new Set(passages.filter((p) => p.card).map((p) => p.page));
    for (const title of this.pages.keys()) {
      if (carded.has(title)) continue;
      add(title, [null, title]); // a page without a card is still findable by its own title
      const m = title.match(/^(.*?)\s*\((.+)\)$/); // "Batman (1966 film)" is also "the 1966 film Batman"
      if (m) add(`${m[2]} ${m[1]}`, [null, title]);
    }
    this.names = [...names.entries()]
      .map(([k, t]): [string[], Target[]] => [k ? k.split(" ") : [], t])
      .sort((a, b) => b[0].length - a[0].length);
  }

  /** Subjects named in the query. A longer name hides shorter names inside it. */
  mentions(query: string): Target[] {
    const q = words(query), used = new Set<number>(), out: Target[] = [];
    for (const [name, targets] of this.names) {
      const n = name.length;
      for (let s = 0; s + n <= q.length; s++) {
        let ok = true;
        for (let j = 0; j < n && ok; j++) ok = q[s + j] === name[j] && !used.has(s + j);
        if (!ok) continue;
        for (let j = 0; j < n; j++) used.add(s + j);
        for (const t of targets) if (!out.some((x) => x[0] === t[0] && x[1] === t[1])) out.push(t);
      }
    }
    return out;
  }

  search(query: string, k = 3): Hit[] {
    const score = new Map<number, number>();
    const bump = (i: number, s: number) => score.set(i, (score.get(i) ?? 0) + s);
    for (const t of terms(query, true)) {
      const idf = this.idf.get(t);
      if (idf === undefined) continue;
      for (const [i, tf] of this.postings.get(t)!)
        bump(i, (this.passages[i].card ? CARD_WEIGHT : 1.0) * ((idf * tf * (K1 + 1)) / (tf + K1 * (1 - B + (B * this.len[i]) / this.avgdl))));
    }
    // How closely does the question match a known way of asking for a fact? Shared words over all words,
    // each weighted by rarity, so "how many" alone counts for little and a full paraphrase for a lot.
    const q = unique(words(query));
    let qWeight = 0;
    for (const w of q) qWeight += this.askIdf.get(w) ?? this.askRare;
    const match = new Map<number, number>();
    for (const [i, ask] of this.asks) {
      let shared = 0;
      for (const w of q) if (ask.includes(w)) shared += this.askIdf.get(w)!;
      if (!shared) continue;
      let extra = 0;
      for (const w of ask) if (!q.includes(w)) extra += this.askIdf.get(w)!;
      const sim = shared / (qWeight + extra);
      if (sim > (match.get(i) ?? 0)) match.set(i, sim);
    }
    match.forEach((sim, i) => bump(i, ASK_WEIGHT * sim));
    let pinned: number[] = []; // what the question names goes first: the best fact about it, or its page's opening passage
    for (const [card, page] of this.mentions(query)) {
      if (card !== null) {
        const facts = this.facts.get(card)!;
        for (const i of facts) bump(i, FACT_BOOST);
        let best = facts[0];
        for (const i of facts) if (score.get(i)! > score.get(best)!) best = i;
        if (!pinned.includes(best)) pinned.push(best);
      }
      if (page === null) continue;
      for (const i of this.pages.get(page) ?? []) bump(i, PAGE_BOOST);
      const lead = this.lead.get(page);
      if (lead === undefined) continue;
      bump(lead, card !== null ? LEAD_BOOST : TITLE_BOOST);
      if (card === null && !pinned.includes(lead)) pinned.push(lead);
    }
    pinned = pinned.slice(0, k);
    const rest = [...score.entries()]
      .filter(([i]) => !pinned.includes(i))
      .sort((a, b) => b[1] - a[1] || a[0] - b[0])
      .slice(0, k - pinned.length)
      .filter(([, s]) => s > 0);
    return [...pinned.map((i): [number, number] => [i, score.get(i)!]), ...rest]
      .sort((a, b) => b[1] - a[1] || a[0] - b[0]) // best match is read first
      .map(([i, s]) => ({ score: s, passage: this.passages[i] }));
  }
}
