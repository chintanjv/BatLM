import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Atmosphere } from "./components/Atmosphere";
import { Boot } from "./components/Boot";
import { Diagnostics, type Report } from "./components/Diagnostics";
import { Dossier } from "./components/Dossier";
import { BatMark, Emblem } from "./components/Emblem";
import { Trace } from "./components/Trace";
import { Oracle, type Fact, type Step } from "./lib/agent";
import { BM25, type Chunk, type Hit } from "./lib/bm25";
import { sfx } from "./lib/sfx";
import type { ModelKey, WorkerIn, WorkerOut } from "./worker";

type ModelState = {
  state: "idle" | "loading" | "ready" | "error";
  files: Record<string, { loaded: number; total: number }>;
  device?: string; dtype?: string; params?: string; error?: string;
};
type Stats = { tokens: number; ms: number; ttftMs: number };
type Msg = {
  id: string; role: "user" | "bot"; model: ModelKey; text: string;
  kind?: "answer" | "raw" | "refuse" | "quiz" | "grade"; streaming?: boolean;
  trace?: Step[]; evidence?: Hit[]; stats?: Stats; status?: string;
};
type Tab = "trace" | "diag" | "dossier";

const MODELS: Record<ModelKey, { name: string; blurb: string; size: string }> = {
  batlm: { name: "BATLM-360M", blurb: "Fine-tuned (LoRA) Q&A analyst. Grounded answers from retrieved case files.", size: "~270 MB · 4-bit" },
  nano: { name: "BATLM-NANO", blurb: "Trained from scratch on 40 KB of Batman text. Raw text completion, charmingly unhinged.", size: "~8 MB · fp32" },
};
const SUGGEST: Record<ModelKey, string[]> = {
  batlm: ["Who created Batman?", "Who broke Batman's back?", "What is inside the Batcave?", "Compare the Joker and Two-Face",
    "What happened in 1940?", "Quiz me", "What's the capital of France?"],
  nano: ["Batman is", "The Joker", "Bruce Wayne", "In the 1970s,", "Gotham City"],
};

const uid = () => Math.random().toString(36).slice(2, 10);
const progressOf = (m: ModelState) => {
  const f = Object.values(m.files);
  const t = f.reduce((s, x) => s + x.total, 0);
  return t ? f.reduce((s, x) => s + x.loaded, 0) / t : 0;
};

function useClock() {
  const [t, setT] = useState(() => new Date());
  useEffect(() => { const i = setInterval(() => setT(new Date()), 1000); return () => clearInterval(i); }, []);
  return t.toLocaleTimeString("en-GB");
}

export default function App() {
  const [booted, setBooted] = useState(() => {
    if (new URLSearchParams(location.search).get("boot") === "0") return true;
    try { return sessionStorage.getItem("batlm.booted") === "1"; } catch { return false; }
  });
  const [active, setActive] = useState<ModelKey>("batlm");
  const [models, setModels] = useState<Record<ModelKey, ModelState>>({ batlm: { state: "idle", files: {} }, nano: { state: "idle", files: {} } });
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("trace");
  const [focusMsg, setFocusMsg] = useState<string | null>(null);
  const [flash, setFlash] = useState<number | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [oracle, setOracle] = useState<Oracle | null>(null);
  const [sound, setSound] = useState(sfx.enabled);
  const [mobileView, setMobileView] = useState<"console" | Tab>("console");
  const [toast, setToast] = useState<string | null>(null);
  const [session, setSession] = useState({ tokens: 0, queries: 0 });
  const worker = useRef<Worker | null>(null);
  const waiters = useRef<Record<ModelKey, (() => void)[]>>({ batlm: [], nano: [] });
  const feed = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const clock = useClock();

  // ---- data ----
  useEffect(() => {
    Promise.all([fetch("/data/chunks.json").then((r) => r.json()), fetch("/data/facts.json").then((r) => r.json())])
      .then(([chunks, f]: [Chunk[], { persona: string; refusal: string; facts: Fact[] }]) =>
        setOracle(new Oracle(new BM25(chunks), f.facts, f.persona, f.refusal)))
      .catch(() => setToast("Could not load case files."));
    fetch("/data/report.json").then((r) => (r.ok ? r.json() : null)).then(setReport).catch(() => {});
  }, []);

  // ---- worker ----
  useEffect(() => {
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    w.onmessage = (e: MessageEvent<WorkerOut>) => {
      const m = e.data;
      if (m.type === "progress") {
        setModels((s) => ({ ...s, [m.model]: { ...s[m.model], state: "loading", files: { ...s[m.model].files, [m.file]: { loaded: m.loaded, total: m.total } } } }));
      } else if (m.type === "ready") {
        setModels((s) => ({ ...s, [m.model]: { ...s[m.model], state: "ready", device: m.device, dtype: m.dtype, params: m.params } }));
        waiters.current[m.model].splice(0).forEach((f) => f());
      } else if (m.type === "token") {
        setMsgs((ms) => ms.map((x) => (x.id === m.id ? { ...x, text: x.text + m.text, status: undefined } : x)));
        sfx.tick();
      } else if (m.type === "done") {
        setMsgs((ms) => ms.map((x) => (x.id === m.id ? { ...x, text: x.text.trim(), streaming: false, stats: { tokens: m.tokens, ms: m.ms, ttftMs: m.ttftMs } } : x)));
        setSession((s) => ({ ...s, tokens: s.tokens + m.tokens }));
        setBusy(null);
        sfx.done();
      } else if (m.type === "error") {
        if (m.model) setModels((s) => ({ ...s, [m.model!]: { ...s[m.model!], state: "error", error: m.message } }));
        setMsgs((ms) => ms.map((x) => (x.streaming ? { ...x, streaming: false, kind: "refuse", text: `SYSTEM FAULT: ${m.message}` } : x)));
        setBusy(null);
        setToast(m.message);
      }
    };
    return () => w.terminate();
  }, []);

  const post = (m: WorkerIn) => worker.current?.postMessage(m);
  const ensure = useCallback((model: ModelKey) => new Promise<void>((resolve) => {
    setModels((s) => {
      if (s[model].state === "ready") { resolve(); return s; }
      waiters.current[model].push(resolve);
      if (s[model].state !== "loading") post({ type: "load", model });
      return { ...s, [model]: { ...s[model], state: "loading", error: undefined } };
    });
  }), []);

  // nano is tiny: arm it as soon as the cave is open
  useEffect(() => { if (booted) ensure("nano"); }, [booted, ensure]);
  useEffect(() => { if (toast) { const t = setTimeout(() => setToast(null), 5000); return () => clearTimeout(t); } }, [toast]);
  useEffect(() => { feed.current?.scrollTo({ top: feed.current.scrollHeight }); }, [msgs]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== inputRef.current) { e.preventDefault(); inputRef.current?.focus(); }
      if (e.key === "Escape" && busy) post({ type: "abort" });
    };
    addEventListener("keydown", k);
    return () => removeEventListener("keydown", k);
  }, [busy]);

  const patch = (id: string, p: Partial<Msg>) => setMsgs((ms) => ms.map((x) => (x.id === id ? { ...x, ...p } : x)));

  // Instant (non-LLM) replies are "typed" for consistency with streamed ones.
  const typeOut = (id: string, full: string) => {
    let i = 0;
    const iv = setInterval(() => {
      i = Math.min(full.length, i + 3);
      patch(id, { text: full.slice(0, i) });
      if (i >= full.length) { clearInterval(iv); patch(id, { streaming: false }); setBusy(null); }
    }, 12);
  };

  const send = async (raw?: string) => {
    const text = (raw ?? input).trim();
    if (!text || busy || !oracle) return;
    setInput("");
    sfx.send();
    const botId = uid();
    setBusy(botId);
    setSession((s) => ({ ...s, queries: s.queries + 1 }));
    setMsgs((ms) => [...ms, { id: uid(), role: "user", model: active, text }]);

    if (active === "nano") {
      setMsgs((ms) => [...ms, { id: botId, role: "bot", model: "nano", kind: "raw", text, streaming: true }]);
      await ensure("nano");
      post({ type: "generate", model: "nano", id: botId, prompt: text, maxTokens: 120, temperature: 0.8 });
      return;
    }

    const plan = oracle.run(text);
    setFocusMsg(botId);
    setTab("trace");
    const base: Msg = { id: botId, role: "bot", model: "batlm", text: "", streaming: true, trace: plan.trace };
    if (plan.kind === "quiz") { setMsgs((ms) => [...ms, { ...base, kind: "quiz" }]); typeOut(botId, `CHALLENGE ▸ ${plan.question}`); return; }
    if (plan.kind === "grade") {
      setMsgs((ms) => [...ms, { ...base, kind: plan.correct ? "grade" : "refuse" }]);
      if (!plan.correct) sfx.deny();
      typeOut(botId, `${plan.correct ? "CORRECT." : "NEGATIVE."} ${plan.answer}`);
      return;
    }
    if (plan.kind === "refuse") { setMsgs((ms) => [...ms, { ...base, kind: "refuse" }]); sfx.deny(); typeOut(botId, plan.answer); return; }

    setMsgs((ms) => [...ms, { ...base, kind: "answer", evidence: plan.evidence }]);
    if (models.batlm.state !== "ready") patch(botId, { status: "Neural core offline — downloading (one-time, cached afterwards)…" });
    await ensure("batlm");
    patch(botId, { status: "Analyzing case files…" });
    post({ type: "generate", model: "batlm", id: botId, prompt: plan.prompt, maxTokens: 120 });
  };

  const focused = useMemo(() => msgs.find((m) => m.id === focusMsg) ?? [...msgs].reverse().find((m) => m.trace), [msgs, focusMsg]);
  const lastStats = [...msgs].reverse().find((m) => m.stats)?.stats;
  const openCite = (msgId: string, chunkId: number) => {
    setFocusMsg(msgId); setTab("trace"); setMobileView("trace"); setFlash(chunkId);
    setTimeout(() => document.getElementById(`ev-${chunkId}`)?.scrollIntoView({ behavior: "smooth", block: "center" }), 120);
    setTimeout(() => setFlash(null), 1600);
  };
  const bm = models.batlm;
  const deviceLabel = bm.device ? `${bm.device.toUpperCase()} · ${bm.dtype}` : "STANDBY";

  const sideTabs: [Tab, string][] = [["trace", "ORACLE TRACE"], ["diag", "DIAGNOSTICS"], ["dossier", "DOSSIER"]];
  const sideView = mobileView === "console" ? tab : mobileView;

  return (
    <>
      <Atmosphere intensity={booted ? 0.8 : 1} />
      <div className="vignette" />
      <div className="overlay-scan" />
      <div className="overlay-grain" />

      <AnimatePresence>{!booted && <Boot onDone={() => { setBooted(true); try { sessionStorage.setItem("batlm.booted", "1"); } catch { /* ignore */ } }} />}</AnimatePresence>

      {booted && (
        <div className="shell">
          <motion.header className="topbar" initial={{ y: -30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}>
            <div className="brand">
              <BatMark />
              <div>
                <div className="brand-name glitch" data-text="BATLM"><b>BAT</b>LM</div>
                <div className="brand-sub hide-sm">BATCOMPUTER // FIELD INTELLIGENCE</div>
              </div>
            </div>
            <div className="spacer" />
            <div className="model-switch-sm">
              {(Object.keys(MODELS) as ModelKey[]).map((k) => (
                <button key={k} className={`iconbtn ${active === k ? "on" : ""}`} onClick={() => setActive(k)}>{k === "batlm" ? "360M" : "NANO"}</button>
              ))}
            </div>
            <span className="chip hide-sm"><span className="dot live" />LINK: ON-DEVICE</span>
            <span className="chip hide-sm"><span className={`dot ${bm.state === "ready" ? "live" : bm.state === "loading" ? "warn" : bm.state === "error" ? "err" : ""}`} />{deviceLabel}</span>
            <button className={`iconbtn hide-sm ${sound ? "on" : ""}`} onClick={() => { sfx.set(!sound); setSound(!sound); }}>SFX {sound ? "ON" : "OFF"}</button>
            <div className="clock hide-sm">{clock}</div>
          </motion.header>

          <div className={`grid ${mobileView !== "console" ? "show-side" : ""}`}>
            {/* ---------- left rail ---------- */}
            <motion.aside className="rail" initial={{ x: -40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: 0.1, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}>
              <section className="panel">
                <div className="panel-head"><span><span className="idx">01</span>NEURAL CORES</span></div>
                <div className="bay">
                  {(Object.keys(MODELS) as ModelKey[]).map((k) => {
                    const m = models[k], p = progressOf(m);
                    return (
                      <button key={k} className={`modelcard ${active === k ? "active" : ""}`} onClick={() => setActive(k)}>
                        <h4>{MODELS[k].name}</h4>
                        <p>{MODELS[k].blurb}</p>
                        <div className="meta">
                          <span>{m.params ?? (k === "batlm" ? "362M" : "1.9M")} PARAMS</span>
                          <span>{MODELS[k].size}</span>
                        </div>
                        <div className="meta">
                          <span style={{ color: m.state === "ready" ? "var(--green)" : m.state === "error" ? "var(--red)" : m.state === "loading" ? "var(--gold)" : undefined }}>
                            ● {m.state === "loading" ? `LOADING ${Math.round(p * 100)}%` : m.state.toUpperCase()}
                          </span>
                          {k === "batlm" && m.state === "idle" && (
                            <span role="button" tabIndex={0} style={{ color: "var(--gold)", textDecoration: "underline" }}
                              onClick={(e) => { e.stopPropagation(); ensure("batlm"); }}
                              onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); ensure("batlm"); } }}>PRELOAD</span>
                          )}
                        </div>
                        {m.state === "loading" && <div className={`bar ${p === 0 ? "indet" : ""}`}><i style={{ width: `${p * 100}%` }} /></div>}
                      </button>
                    );
                  })}
                </div>
              </section>
              <section className="panel">
                <div className="panel-head"><span><span className="idx">02</span>TELEMETRY</span></div>
                <div className="telemetry">
                  <div className="metric"><div className="label">Throughput</div><div className="v">{lastStats && lastStats.ms ? (lastStats.tokens / (lastStats.ms / 1000)).toFixed(1) : "—"}<small>tok/s</small></div></div>
                  <div className="metric"><div className="label">First token</div><div className="v">{lastStats ? Math.round(lastStats.ttftMs) : "—"}<small>ms</small></div></div>
                  <div className="metric"><div className="label">Queries</div><div className="v">{session.queries}</div></div>
                  <div className="metric"><div className="label">Tokens out</div><div className="v">{session.tokens}</div></div>
                </div>
              </section>
              <section className="panel">
                <div className="panel-head"><span><span className="idx">03</span>GOTHAM SCAN</span></div>
                <div className="radar-wrap">
                  <svg className="radar" viewBox="0 0 100 100" aria-hidden>
                    <defs>
                      <radialGradient id="rg"><stop offset="0" stopColor="#f5c518" stopOpacity="0.25" /><stop offset="1" stopColor="#f5c518" stopOpacity="0" /></radialGradient>
                      <linearGradient id="sweep" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stopColor="#f5c518" stopOpacity="0" /><stop offset="1" stopColor="#f5c518" stopOpacity="0.5" /></linearGradient>
                    </defs>
                    {[46, 32, 18].map((r) => <circle key={r} cx="50" cy="50" r={r} fill="none" stroke="rgba(245,197,24,.2)" />)}
                    <line x1="4" y1="50" x2="96" y2="50" stroke="rgba(245,197,24,.12)" />
                    <line x1="50" y1="4" x2="50" y2="96" stroke="rgba(245,197,24,.12)" />
                    <g className="radar-sweep"><path d="M50,50 L96,50 A46,46 0 0,0 82.5,17.5 Z" fill="url(#sweep)" /></g>
                    {[[30, 34], [66, 62], [58, 28], [36, 70]].map(([x, y], i) => (
                      <circle key={i} cx={x} cy={y} r="2" fill="#f5c518"><animate attributeName="opacity" values="1;0.1;1" dur={`${1.6 + i * 0.5}s`} repeatCount="indefinite" /></circle>
                    ))}
                  </svg>
                  <div className="radar-legend">
                    {[["ARKHAM", "CONTAINED", "var(--green)"], ["NARROWS", "ELEVATED", "var(--gold)"], ["DIAMOND", "NOMINAL", "var(--green)"], ["FILES", oracle ? "99 IDX" : "…", "var(--cyan)"]].map(([k, v, c]) => (
                      <div key={k}><span>{k}</span><span style={{ color: c }}>{v}</span></div>
                    ))}
                  </div>
                </div>
              </section>
            </motion.aside>

            {/* ---------- console ---------- */}
            <motion.main className="panel console" initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.18, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}>
              <div className="panel-head">
                <span><span className="idx">&gt;_</span>{active === "batlm" ? "INTERROGATION CONSOLE" : "RAW NEURAL FEED"}</span>
                <span>{MODELS[active].name}</span>
              </div>
              <div className="feed" ref={feed}>
                {msgs.length === 0 ? (
                  <div className="empty">
                    <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.3, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}>
                      <Emblem className="emblem" />
                    </motion.div>
                    <h2 className="glitch" data-text={active === "batlm" ? "ASK THE BATCOMPUTER" : "FEED THE NANO-CORE"}>
                      {active === "batlm" ? "ASK THE BATCOMPUTER" : "FEED THE NANO-CORE"}
                    </h2>
                    <p>
                      {active === "batlm"
                        ? "A 360M-parameter model fine-tuned on Batman's case files, running entirely on your device. Every answer is grounded in retrieved sources you can inspect."
                        : "A 1.9M-parameter GPT trained from scratch on two web pages. It doesn't answer questions. It continues your text."}
                    </p>
                    <div className="suggest">
                      {SUGGEST[active].map((s, i) => (
                        <motion.button key={s} onClick={() => send(s)} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 + i * 0.05 }}>{s}</motion.button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <AnimatePresence initial={false}>
                    {msgs.map((m) => (
                      <motion.div
                        key={m.id}
                        className={`msg ${m.role} ${m.kind === "raw" ? "raw" : ""} ${m.kind === "refuse" ? "refuse" : ""}`}
                        initial={{ opacity: 0, y: 14, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                      >
                        <div className="msg-head">
                          <span className="who">{m.role === "user" ? "DETECTIVE" : m.model === "nano" ? "NANO-CORE" : "BATLM"}</span>
                          {m.role === "bot" && m.trace && <button className="linkbtn" onClick={() => { setFocusMsg(m.id); setTab("trace"); setMobileView("trace"); }}>TRACE</button>}
                        </div>
                        <div className="msg-body">
                          {m.status && !m.text ? <span style={{ color: "var(--ink-3)" }}>{m.status}{m.status.startsWith("Neural") && ` ${Math.round(progressOf(bm) * 100)}%`}</span> : m.text}
                          {m.streaming && <span className="caret" />}
                        </div>
                        {m.evidence && !m.streaming && m.evidence.length > 0 && (
                          <div className="cites">
                            {m.evidence.map((h, i) => (
                              <button key={h.chunk.id} className="cite" onClick={() => openCite(m.id, h.chunk.id)}><b>[{i + 1}]</b>{h.chunk.section}</button>
                            ))}
                          </div>
                        )}
                        {m.stats && (
                          <div className="stats">
                            <span>{m.stats.tokens} TOK</span>
                            <span>{(m.stats.tokens / (m.stats.ms / 1000)).toFixed(1)} TOK/S</span>
                            <span>TTFT {Math.round(m.stats.ttftMs)}MS</span>
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </AnimatePresence>
                )}
              </div>
              <div>
                <div className="inputbar">
                  <span className="prompt-glyph">{active === "batlm" ? "?>" : "~>"}</span>
                  <textarea
                    ref={inputRef} rows={1} value={input} spellCheck={false}
                    placeholder={active === "batlm" ? "Ask about Batman, say 'quiz me', or 'compare X and Y'…" : "Start a sentence and the nano-core will continue it…"}
                    onChange={(e) => { setInput(e.target.value); e.target.style.height = "auto"; e.target.style.height = `${e.target.scrollHeight}px`; }}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                  />
                  {busy ? (
                    <button className="exec stop" onClick={() => post({ type: "abort" })}>ABORT</button>
                  ) : (
                    <button className="exec" disabled={!input.trim() || !oracle} onClick={() => send()}>EXECUTE ⏎</button>
                  )}
                </div>
                <div className="hint">
                  <span><span className="kbd">/</span> focus · <span className="kbd">⏎</span> send · <span className="kbd">esc</span> abort</span>
                  <span className="hide-sm">inference on this device · no data leaves your browser</span>
                </div>
              </div>
            </motion.main>

            {/* ---------- side panel ---------- */}
            <motion.aside className="panel side" initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: 0.26, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}>
              <div className="tabs">
                {sideTabs.map(([k, l]) => (
                  <button key={k} className={`tab ${sideView === k ? "on" : ""}`} onClick={() => { setTab(k); if (mobileView !== "console") setMobileView(k); }}>
                    {l}
                    {sideView === k && <motion.div layoutId="tab-u" className="tab-underline" />}
                  </button>
                ))}
              </div>
              <div className="side-body">
                <AnimatePresence mode="wait">
                  <motion.div key={sideView + (sideView === "trace" ? focused?.id ?? "" : "")} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }}>
                    {sideView === "trace" && <Trace trace={focused?.trace} flash={flash} />}
                    {sideView === "diag" && <Diagnostics report={report} />}
                    {sideView === "dossier" && <Dossier />}
                  </motion.div>
                </AnimatePresence>
              </div>
            </motion.aside>
          </div>

          <nav className="mobile-tabs">
            {([["console", "CONSOLE"], ["trace", "TRACE"], ["diag", "DIAG"], ["dossier", "DOSSIER"]] as const).map(([k, l]) => (
              <button key={k} className={mobileView === k ? "on" : ""} onClick={() => { setMobileView(k); if (k !== "console") setTab(k); }}>{l}</button>
            ))}
          </nav>
        </div>
      )}

      <AnimatePresence>
        {toast && <motion.div className="toast" initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 30, opacity: 0 }}>⚠ {toast}</motion.div>}
      </AnimatePresence>
    </>
  );
}
