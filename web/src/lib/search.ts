// Mirror of batlm/rag.py — keep in sync so the browser retrieves exactly what evals measured.

export type Passage = {
  id: number; title: string; section: string; text: string; url: string;
  card?: boolean; page?: string | null; aliases?: string[]; keywords?: string;
};
export type Hit = { score: number; passage: Passage };
/** A subject named in a question: its card (index into passages) and/or its Wikipedia page title. */
type Target = [card: number | null, page: string | null];

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

const CARD_WEIGHT = 1.5; // a card's BM25 score counts double: cards are short, curated and written to be found
const CARD_BOOST = 12.0; // the card of a subject the question names (it is also pinned to the top, see search)
const PAGE_BOOST = 1.5;  // Wikipedia passages from that subject's page
const LEAD_BOOST = 1.0;  // ...and a little more for the page's opening passage
const TITLE_BOOST = 10.0; // the opening passage of a page named in the question, when that page has no card

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

export class Index {
  private len: number[];
  private avgdl: number;
  private postings = new Map<string, [number, number][]>(); // term -> [passage index, term frequency]
  private idf = new Map<string, number>();
  private pages = new Map<string, number[]>();
  private lead = new Map<string, number>();
  private names: [string[], Target[]][];
  passages: Passage[];
  private k1 = 1.5;
  private b = 0.75;

  constructor(passages: Passage[]) {
    this.passages = passages;
    const docs = passages.map((p) => terms(`${p.title} ${p.section} ${p.text} ${p.keywords ?? ""}`));
    this.len = docs.map((d) => d.length);
    this.avgdl = this.len.reduce((s, n) => s + n, 0) / docs.length;
    docs.forEach((d, i) => {
      const tf = new Map<string, number>();
      for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1);
      tf.forEach((f, t) => {
        const list = this.postings.get(t);
        if (list) list.push([i, f]); else this.postings.set(t, [[i, f]]);
      });
    });
    const n = docs.length;
    this.postings.forEach((p, t) => this.idf.set(t, Math.log(1 + (n - p.length + 0.5) / (p.length + 0.5))));
    passages.forEach((p, i) => {
      if (p.card) return;
      const list = this.pages.get(p.title);
      if (list) list.push(i); else { this.pages.set(p.title, [i]); this.lead.set(p.title, i); }
    });
    // Names -> what they refer to. Longest names first, so "the batman who laughs" beats "batman".
    const names = new Map<string, Target[]>();
    const add = (name: string, t: Target) => {
      const key = words(name).join(" ");
      const list = names.get(key);
      if (list) list.push(t); else names.set(key, [t]);
    };
    passages.forEach((p, i) => p.aliases?.forEach((a) => add(a, [i, p.page ?? null])));
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
    const q = words(query), used = new Set<number>(), out: Target[] = [], seen = new Set<string>();
    for (const [name, targets] of this.names) {
      const n = name.length;
      for (let s = 0; s + n <= q.length; s++) {
        let ok = true;
        for (let j = 0; j < n && ok; j++) ok = q[s + j] === name[j] && !used.has(s + j);
        if (!ok) continue;
        for (let j = 0; j < n; j++) used.add(s + j);
        for (const t of targets) {
          const key = `${t[0]}|${t[1]}`;
          if (!seen.has(key)) { seen.add(key); out.push(t); }
        }
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
        bump(i, ((this.passages[i].card ? CARD_WEIGHT : 1) * idf * tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + (this.b * this.len[i]) / this.avgdl)));
    }
    let pinned: number[] = []; // what the question names goes first, whatever else scores well: its card, or its page's opening
    for (const [card, page] of this.mentions(query)) {
      if (card !== null) { bump(card, CARD_BOOST); pinned.push(card); }
      if (page === null) continue;
      for (const i of this.pages.get(page) ?? []) bump(i, PAGE_BOOST);
      const lead = this.lead.get(page);
      if (lead === undefined) continue;
      bump(lead, card !== null ? LEAD_BOOST : TITLE_BOOST);
      if (card === null) pinned.push(lead);
    }
    pinned = pinned.slice(0, k);
    const rest = [...score.entries()]
      .filter(([i]) => !pinned.includes(i))
      .sort((a, b) => b[1] - a[1] || a[0] - b[0])
      .slice(0, k - pinned.length)
      .filter(([, s]) => s > 0);
    return [...pinned.map((i): [number, number] => [i, score.get(i)!]), ...rest].map(([i, s]) => ({ score: s, passage: this.passages[i] }));
  }
}
