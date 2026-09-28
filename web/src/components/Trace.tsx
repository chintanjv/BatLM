import { AnimatePresence, motion } from "motion/react";
import type { Step } from "../lib/agent";

export function Trace({ trace, flash }: { trace: Step[] | undefined; flash: number | null }) {
  if (!trace) {
    return (
      <div className="side-empty">
        NO ACTIVE INVESTIGATION<br />
        Ask a question — every step the ORACLE takes<br />(plan → retrieve → guard → generate)<br />will be logged here.
      </div>
    );
  }
  const maxScore = Math.max(1, ...trace.flatMap((s) => s.hits?.map((h) => h.score) ?? []));
  let evIdx = 0;
  return (
    <div className="trace">
      <AnimatePresence initial>
        {trace.map((s, i) => (
          <motion.div
            className="step" key={i}
            initial={{ opacity: 0, x: 14 }} animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.09, duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className={`step-node ${s.tool === "batlm.generate" || s.tool === "guardrail" ? "hot" : ""}`} />
            <div className="step-title"><span>{s.tool.toUpperCase()}</span><span>{s.ms ? `${s.ms}ms` : ""}</span></div>
            <div className="step-input">{s.input}</div>
            {s.detail && <div className="step-detail">↳ {s.detail}</div>}
            {s.hits && s.hits.length > 0 && (
              <div className="evidence">
                {s.hits.map((h) => {
                  const n = ++evIdx;
                  return (
                    <motion.div
                      key={h.chunk.id} className={`ev ${flash === h.chunk.id ? "flash" : ""}`} id={`ev-${h.chunk.id}`}
                      initial={{ opacity: 0, rotateX: -60 }} animate={{ opacity: 1, rotateX: 0 }}
                      transition={{ delay: i * 0.09 + n * 0.06, duration: 0.4 }}
                    >
                      <div className="ev-head">
                        <span><b>#{String(h.chunk.id).padStart(3, "0")}</b> {h.chunk.source.toUpperCase()} / {h.chunk.section}</span>
                        <span>BM25 {h.score.toFixed(2)}</span>
                      </div>
                      <div className="scorebar"><i style={{ width: `${(100 * h.score) / maxScore}%` }} /></div>
                      {h.chunk.text}
                      <div style={{ marginTop: 6 }}><a href={h.chunk.url} target="_blank" rel="noreferrer" className="linkbtn">SOURCE ↗</a></div>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
