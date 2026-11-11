import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { BatMark, Emblem } from "./components/Bat";
import { Composer } from "./components/Composer";
import { Message, type Msg } from "./components/Message";
import { useModel } from "./hooks/useModel";
import { plan, type Knowledge } from "./lib/agent";
import { Index } from "./lib/search";

const SUGGESTIONS = ["Who is the Batman?", "What is the Joker's real identity?", "Who is Alfred?", "Top 5 Batman villains"];

export default function App() {
  const model = useModel();
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [chat, setChat] = useState(0); // bumped by "new chat" so the empty state replays its entrance
  const knowledge = useRef<Promise<Knowledge> | null>(null);
  const feed = useRef<HTMLElement>(null);
  const pinned = useRef(true);       // is the view at the bottom? Only then do new words scroll it
  const run = useRef(0);             // identifies the current answer; callbacks from an older one are ignored
  const live = useRef<{ id: string; generating: boolean; stopped: boolean } | null>(null);
  const buffer = useRef("");         // streamed text waiting for the next animation frame
  const frame = useRef(0);

  const loadKnowledge = useCallback(() => {
    knowledge.current ??= fetch("/data/index.json")
      .then((r) => { if (!r.ok) throw new Error(`could not load the case files (HTTP ${r.status})`); return r.json(); })
      .then((d): Knowledge => ({ index: new Index(d.passages), persona: d.persona, refusal: d.refusal }))
      .catch((e) => { knowledge.current = null; throw e; }); // a failed load can be retried
    return knowledge.current;
  }, []);
  useEffect(() => { loadKnowledge().catch(() => {}); }, [loadKnowledge]);

  const patch = useCallback((id: string, p: Partial<Msg> | ((m: Msg) => Partial<Msg>)) =>
    setMsgs((ms) => ms.map((m) => (m.id === id ? { ...m, ...(typeof p === "function" ? p(m) : p) } : m))), []);

  const send = useCallback(async (text: string) => {
    const q = text.trim();
    if (!q || live.current) return;
    const me = ++run.current, id = `a${me}`;
    const state = { id, generating: false, stopped: false };
    live.current = state;
    pinned.current = true;
    setBusy(true);
    setMsgs((ms) => [...ms, { id: `q${me}`, role: "user", text: q, status: "done" }, { id, role: "bot", text: "", status: "searching", question: q }]);
    const finish = (p: Partial<Msg> | ((m: Msg) => Partial<Msg>)) => {
      if (run.current !== me) return;
      cancelAnimationFrame(frame.current);
      frame.current = 0;
      const rest = buffer.current;
      buffer.current = "";
      patch(id, (m) => { const n = typeof p === "function" ? p(m) : p; return { ...n, text: ((n.text ?? m.text) + (n.status === "error" ? "" : rest)).trim() }; });
      live.current = null;
      setBusy(false);
    };
    try {
      const p = plan(await loadKnowledge(), q);
      if (run.current !== me) return;
      if (p.kind === "refuse") return finish({ text: p.answer, status: "done" });
      patch(id, { sources: p.sources, status: "waiting" });
      await model.load();
      if (run.current !== me) return;
      patch(id, { status: "searching" });
      state.generating = true;
      model.generate(id, p.prompt, {
        onText: (t) => {
          if (run.current !== me) return;
          buffer.current += t;
          frame.current ||= requestAnimationFrame(() => { // one render per frame, not per token
            frame.current = 0;
            const add = buffer.current;
            buffer.current = "";
            patch(id, (m) => ({ text: m.text + add, status: "streaming" }));
          });
        },
        onDone: (stats) => finish({ status: state.stopped ? "stopped" : "done", stats }),
        onError: (message) => finish({ status: "error", text: message }),
      });
    } catch (e) {
      finish({ status: "error", text: String((e as Error)?.message ?? e) });
    }
  }, [loadKnowledge, model.load, model.generate, patch]);

  const stop = useCallback(() => {
    const s = live.current;
    if (!s) return;
    if (s.generating) { s.stopped = true; model.abort(); return; } // the worker replies "done" with what it has
    run.current++; // still searching or loading: drop the answer; the model keeps loading in the background
    patch(s.id, { status: "stopped" });
    live.current = null;
    setBusy(false);
  }, [model.abort, patch]);

  const newChat = useCallback(() => {
    if (live.current?.generating) model.abort();
    run.current++;
    live.current = null;
    buffer.current = "";
    setBusy(false);
    setMsgs([]);
    setChat((c) => c + 1);
  }, [model.abort]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") stop(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stop]);

  useLayoutEffect(() => { // follow the answer as it grows, unless the reader has scrolled up
    const el = feed.current!;
    if (pinned.current) el.scrollTop = el.scrollHeight;
  }, [msgs]);

  const { status, progress, loadedMB, totalMB, device, firstRun } = model.state;
  const pct = Math.round(progress * 100);
  const statusText =
    status === "loading" ? (firstRun ? `Downloading ${pct}%` : "Loading model")
    : status === "ready" ? `On-device · ${device === "webgpu" ? "WebGPU" : "CPU"}`
    : status === "error" ? "Model unavailable" : "On-device";
  const download = status === "loading" && firstRun
    ? { progress, label: totalMB > 1 ? `Downloading model · ${pct}% · ${Math.round(loadedMB)} of ${Math.round(totalMB)} MB · one time only` : "Downloading model · one time only" }
    : null;

  return (
    <div className="app">
      <header className="bar gutter">
        <div className="brand"><BatMark />BatLM</div>
        <div className="status" data-s={status}><span className="dot" /><span>{statusText}</span></div>
        <button className="newchat" aria-label="New chat" title="New chat" onClick={newChat}>
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M10 4H5.5A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16h9a1.5 1.5 0 0 0 1.5-1.5V10" /><path d="M14.6 3.6a1.3 1.3 0 0 1 1.8 1.8L10.5 11.3 8 12l.7-2.5z" />
          </svg>
        </button>
      </header>

      <main className="feed gutter" ref={feed} onScroll={(e) => {
        const el = e.currentTarget;
        pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}>
        <div className="col">
          {msgs.length === 0 ? (
            <section className="empty" key={chat}>
              <Emblem />
              <h1>What do you need to know?</h1>
              <p>Ask anything about the Batman universe. Runs entirely on this device.</p>
              <div className="chips">
                {SUGGESTIONS.map((s) => <button className="chip" key={s} onClick={() => send(s)}>{s}</button>)}
              </div>
            </section>
          ) : (
            msgs.map((m) => <Message key={m.id} msg={m} onRetry={send} />)
          )}
        </div>
      </main>

      <footer className="dock gutter">
        <div className="col">
          <Composer busy={busy} onSend={send} onStop={stop} download={download} />
          <p className="note">BatLM is a small model running in your browser. It can be wrong.</p>
        </div>
      </footer>
    </div>
  );
}
