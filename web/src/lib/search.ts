// Mirror of batlm/rag.py — keep in sync so the browser retrieves exactly what evals measured.

export type Passage = {
  id: number; title: string; section: string; text: string; url: string;
  /** Fact passages only: a fact card's name for the subject, and the usual ways of asking for this fact. */
  card?: boolean; page?: string | null; aliases?: string[]; asks?: string[]; keywords?: string;
  /** false on pages titled with an everyday word ("Legacy"): the title is not used as a subject name. */
  name?: boolean;
};
export type Hit = { score: number; passage: Passage };
/** A subject named in a question: its fact card's title and/or its Wikipedia page title. */
export type Target = [card: string | null, page: string | null];
/** The previous turn, as far as search needs it: its answer and who it was about. */
export type Last = { a: string; topic: Target[] };

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
const OFF_SUBJECT = 0.6;  // a passage that never mentions any subject the question is about keeps this share of its score
const K1 = 1.5, B = 0.75;

// A question that leans on the previous turn: it uses a pronoun, points back at what was said, or starts mid-thought.
const FOLLOW = new RegExp(
  "\\b(?:he|him|his|she|her|hers|they|them|their|theirs|it|its|himself|herself|that|this|those|these|" +
  "you said|you listed|you mentioned|you told|are you sure|what about|how about)\\b" +
  "|^(?:and|but|so|also|then|why|really|sure)\\b");

const SMALLTALK = /^(?:hi|hello|hey|yo|thanks|thank you|thx|ok|okay|cool|nice|great|bye|goodbye|good morning|good night|good evening|lol|wow|how are you)\b/;

export const follows = (question: string): boolean => FOLLOW.test(question.toLowerCase().trim());

/** A few words said mid-conversation ("in which story?") continue it, unless they are small talk. */
function brief(question: string): boolean {
  const q = question.toLowerCase().trim();
  return (q.match(/[a-z0-9]+/g) ?? []).length <= 5 && !SMALLTALK.test(q);
}

function stem(t: string): string {
  for (const suf of ["ing", "ed", "es", "s"]) {
    if (t.length > 4 && t.endsWith(suf)) return t.slice(0, -suf.length);
  }
  return t;
}

const tokens = (text: string): string[] => text.toLowerCase().match(/[a-z0-9]+/g) ?? [];

/** Every word, stemmed, stopwords kept: used to spot names ("the batman who laughs"). */
const words = (text: string): string[] => tokens(text).map(stem);

function terms(text: string): string[] {
  const out: string[] = [];
  for (const t of tokens(text)) if (!STOP.has(t)) out.push(stem(t));
  return out;
}

/** True when b is one typo away from a: one letter changed, added or dropped, or two neighbours swapped. */
function near(a: string, b: string): boolean {
  const la = a.length, lb = b.length;
  if (Math.abs(la - lb) > 1 || a === b) return false;
  let i = 0;
  while (i < Math.min(la, lb) && a[i] === b[i]) i++;
  if (la === lb)
    return a.slice(i + 1) === b.slice(i + 1) || (i + 1 < la && a[i] === b[i + 1] && a[i + 1] === b[i] && a.slice(i + 2) === b.slice(i + 2));
  return la > lb ? a.slice(i + 1) === b.slice(i) : a.slice(i) === b.slice(i + 1);
}

const unique = (xs: string[]): string[] => [...new Set(xs)];
const same = (a: Target, b: Target): boolean => a[0] === b[0] && a[1] === b[1];
const has = (list: Target[], t: Target): boolean => list.some((x) => same(x, t));

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
  private stems = new Set<string>();             // every stemmed word the index knows
  private raw = new Map<string, number>();       // every word as written -> how many passages use it
  private byShape = new Map<string, string[]>(); // first letter + length -> known words
  private fixed = new Map<string, string>();
  private wtext: string[];                       // each passage as a run of words
  private called = new Map<string, string[]>();  // subject -> its names, in the same form

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
      else if (!has(list, t)) list.push(t);
    };
    for (const p of passages) p.aliases?.forEach((a) => add(a, [p.title, p.page ?? null]));
    const carded = new Set(passages.filter((p) => p.card).map((p) => p.page));
    const noname = new Set(passages.filter((p) => p.name === false).map((p) => p.title));
    for (const title of this.pages.keys()) {
      if (carded.has(title) || noname.has(title)) continue;
      add(title, [null, title]); // a page without a card is still findable by its own title
      const m = title.match(/^(.*?)\s*\((.+)\)$/); // "Batman (1966 film)" is also "the 1966 film Batman"
      if (m) add(`${m[2]} ${m[1]}`, [null, title]);
    }
    this.names = [...names.entries()]
      .map(([k, t]): [string[], Target[]] => [k ? k.split(" ") : [], t])
      .sort((a, b) => b[0].length - a[0].length);
    // Every word the index knows, stemmed and as written (with how many passages use it): a mistyped
    // word is corrected to the commonest known word one typo away.
    this.postings.forEach((_, t) => this.stems.add(t));
    for (const [name] of this.names) for (const w of name) this.stems.add(w);
    this.askIdf.forEach((_, w) => this.stems.add(w));
    for (const p of passages)
      for (const w of new Set(tokens(`${p.title} ${p.section} ${p.text} ${p.keywords ?? ""}`))) this.raw.set(w, (this.raw.get(w) ?? 0) + 1);
    for (const p of passages)
      for (const a of [...(p.aliases ?? []), ...(p.asks ?? [])])
        for (const w of tokens(a)) if (!this.raw.has(w)) this.raw.set(w, 0);
    STOP.forEach((w) => this.raw.set(w, 1e9));
    this.raw.forEach((_, w) => push(this.byShape, `${w[0]}${w.length}`, w));
    this.wtext = passages.map((p) => ` ${words(`${p.title} ${p.text}`).join(" ")} `);
    for (const [name, targets] of this.names) {
      if (!name.length) continue;
      for (const t of targets) push(this.called, JSON.stringify(t), ` ${name.join(" ")} `);
    }
  }

  /** A question word as the index spells it: an unknown word one typo away from a known one becomes that word. */
  private correct(t: string): string {
    if (this.raw.has(t) || this.stems.has(stem(t)) || t.length < 5 || !/^[a-z]+$/.test(t)) return t;
    let best = this.fixed.get(t);
    if (best === undefined) {
      let b: string | null = null;
      for (const n of [t.length - 1, t.length, t.length + 1]) {
        for (const c of this.byShape.get(`${t[0]}${n}`) ?? []) {
          if (!near(t, c)) continue;
          if (b === null || this.raw.get(c)! > this.raw.get(b)! || (this.raw.get(c) === this.raw.get(b) && c < b)) b = c;
        }
      }
      best = b ?? t;
      this.fixed.set(t, best);
    }
    return best;
  }

  private qWords(query: string): string[] {
    return tokens(query).map((t) => stem(this.correct(t)));
  }

  private qTerms(query: string): string[] {
    const out: string[] = [];
    for (const raw of tokens(query)) {
      const t = this.correct(raw);
      if (STOP.has(t)) continue;
      const s = stem(t);
      out.push(s);
      if (Object.hasOwn(SYNONYMS, s)) out.push(...SYNONYMS[s].split(" ").map(stem));
    }
    return out;
  }

  /** Subjects named in the query. A longer name hides shorter names inside it. */
  mentions(query: string): Target[] {
    const q = this.qWords(query), used = new Set<number>(), out: Target[] = [];
    for (const [name, targets] of this.names) {
      const n = name.length;
      for (let s = 0; s + n <= q.length; s++) {
        let ok = true;
        for (let j = 0; j < n && ok; j++) ok = q[s + j] === name[j] && !used.has(s + j);
        if (!ok) continue;
        for (let j = 0; j < n; j++) used.add(s + j);
        for (const t of targets) if (!has(out, t)) out.push(t);
      }
    }
    return out;
  }

  /** The subjects the question names, and those it inherits from the previous turn. Only a follow-up inherits:
   *  (a question that points back, or a few words naming nobody): first who the previous question was about,
   *  then anyone its answer introduced. */
  subjects(question: string, last?: Last | null): { named: Target[]; carried: Target[] } {
    const named = this.mentions(question);
    if (!last || !(follows(question) || (!named.length && brief(question)))) return { named, carried: [] };
    const prev = [...last.topic.slice(0, 2), ...this.mentions(last.a).filter((t) => !has(last.topic, t)).slice(0, 2)];
    return { named, carried: prev.filter((t) => !has(named, t)) };
  }

  search(query: string, k = 3, carried: Target[] = []): Hit[] {
    const score = new Map<number, number>();
    const bump = (i: number, s: number) => score.set(i, (score.get(i) ?? 0) + s);
    for (const t of this.qTerms(query)) {
      const idf = this.idf.get(t);
      if (idf === undefined) continue;
      for (const [i, tf] of this.postings.get(t)!)
        bump(i, (this.passages[i].card ? CARD_WEIGHT : 1.0) * ((idf * tf * (K1 + 1)) / (tf + K1 * (1 - B + (B * this.len[i]) / this.avgdl))));
    }
    // How closely does the question match a known way of asking for a fact? Shared words over all words,
    // each weighted by rarity, so "how many" alone counts for little and a full paraphrase for a lot.
    const q = unique(this.qWords(query));
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
    const named = this.mentions(query);
    const subjects = [...named, ...carried.filter((t) => !has(named, t))];
    if (subjects.length) { // a passage that is neither about a subject nor mentions one is probably about someone else
      const calls = subjects.flatMap((t) => this.called.get(JSON.stringify(t)) ?? []);
      for (const [i, s] of [...score.entries()]) {
        const p = this.passages[i];
        if (subjects.some(([card, page]) => p.title === (p.card ? card : page))) continue;
        if (!calls.some((n) => this.wtext[i].includes(n))) score.set(i, s * OFF_SUBJECT);
      }
    }
    // What the question names goes first: the best fact about it, or its page's opening passage.
    // A subject carried over from the previous turn is pinned too, and is always given a place.
    let pinned: number[] = [];
    const late: number[] = [];
    for (const t of subjects) {
      const [card, page] = t;
      const pins = has(named, t) ? pinned : late;
      if (card !== null) {
        const facts = this.facts.get(card)!;
        for (const i of facts) bump(i, FACT_BOOST);
        let best = facts[0];
        for (const i of facts) if (score.get(i)! > score.get(best)!) best = i;
        if (!pins.includes(best)) pins.push(best);
      }
      if (page === null) continue;
      for (const i of this.pages.get(page) ?? []) bump(i, PAGE_BOOST);
      const lead = this.lead.get(page);
      if (lead === undefined) continue;
      bump(lead, card !== null ? LEAD_BOOST : TITLE_BOOST);
      if (card === null && !pins.includes(lead)) pins.push(lead);
    }
    pinned = pinned.slice(0, late.length ? k - 1 : k);
    const room = Math.min(k - pinned.length, pinned.length ? 1 : k - 1);
    pinned.push(...late.filter((i) => !pinned.includes(i)).slice(0, Math.max(room, 0)));
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
