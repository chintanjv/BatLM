import { memo, useState, type ReactNode } from "react";
import type { Hit } from "../lib/search";
import { BatMark } from "./Bat";

export type Msg = {
  id: string;
  role: "user" | "bot";
  text: string;
  /** waiting: model still loading · searching: about to answer · streaming · done · stopped · error */
  status: "waiting" | "searching" | "streaming" | "done" | "stopped" | "error";
  question?: string;
  sources?: Hit[];
  stats?: { tokens: number; ms: number };
};

/** Words become spans so each one fades in as it arrives; `**bold**` and `1.` lists are the only markup. */
function inline(line: string, k: string): ReactNode[] {
  return line.split("**").map((seg, i) => {
    const words = seg.split(/(\s+)/).map((w, j) => (w.trim() ? <span className="w" key={j}>{w}</span> : w));
    return i % 2 ? <strong key={`${k}b${i}`}>{words}</strong> : <span key={`${k}s${i}`}>{words}</span>;
  });
}

function Rich({ text, caret }: { text: string; caret: boolean }) {
  const lines = text.split("\n").filter((l) => l.trim());
  const blocks: { list: boolean; lines: string[] }[] = [];
  for (const l of lines) {
    const item = /^\s*\d+[.)]\s+/.test(l);
    const last = blocks[blocks.length - 1];
    if (item && last?.list) last.lines.push(l);
    else blocks.push({ list: item, lines: [l] });
  }
  const tail = caret ? <span className="caret" /> : null;
  return (
    <div className="ans">
      {blocks.map((b, i) => {
        const end = i === blocks.length - 1;
        return b.list ? (
          <ol key={i}>
            {b.lines.map((l, j) => <li key={j}>{inline(l.replace(/^\s*\d+[.)]\s+/, ""), `${i}.${j}`)}{end && j === b.lines.length - 1 && tail}</li>)}
          </ol>
        ) : (
          <p key={i}>{inline(b.lines[0], `${i}`)}{end && tail}</p>
        );
      })}
    </div>
  );
}

function Sources({ sources, stats }: { sources: Hit[]; stats?: Msg["stats"] }) {
  const [open, setOpen] = useState(false);
  const speed = stats && stats.ms > 0 && stats.tokens > 0 ? Math.round(stats.tokens / (stats.ms / 1000)) : 0;
  return (
    <>
      <div className="meta">
        {sources.length > 0 && (
          <button className="srcbtn" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            <svg viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M3.5 1.5 7 5 3.5 8.5" /></svg>
            {sources.length} {sources.length === 1 ? "source" : "sources"}
          </button>
        )}
        {speed > 0 && <span>{speed} tokens/s</span>}
      </div>
      {sources.length > 0 && (
        <div className={open ? "src open" : "src"}>
          <div>
            <ul>
              {sources.map(({ passage: p }, i) => (
                <li key={p.id}>
                  <span className="n">{i + 1}</span>
                  <span className="t">
                    {p.title} <span>· {p.section}</span>
                    {p.url && <a href={p.url} target="_blank" rel="noreferrer" tabIndex={open ? 0 : -1}>Wikipedia ↗</a>}
                  </span>
                  <span className="x">{p.text}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </>
  );
}

export const Message = memo(function Message({ msg, onRetry }: { msg: Msg; onRetry(question: string): void }) {
  if (msg.role === "user") return <div className="msg user">{msg.text}</div>;
  const pending = msg.status === "waiting" || msg.status === "searching";
  const active = pending || msg.status === "streaming";
  return (
    <div className={active ? "msg bot busy" : "msg bot"}>
      <div className="av"><BatMark /></div>
      <div className="body">
        {pending && <div className="think">{msg.status === "waiting" ? "Getting the model ready" : "Searching case files"}</div>}
        {msg.status === "error" ? (
          <div className="err">
            Something went wrong: {msg.text}
            <button onClick={() => onRetry(msg.question ?? "")}>Try again</button>
          </div>
        ) : (
          msg.text && <Rich text={msg.text} caret={msg.status === "streaming"} />
        )}
        {msg.status === "stopped" && <div className="stopped">Stopped.</div>}
        {msg.status === "done" && <Sources sources={msg.sources ?? []} stats={msg.stats} />}
      </div>
    </div>
  );
});
