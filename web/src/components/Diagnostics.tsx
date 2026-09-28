import { motion } from "motion/react";
import { useState } from "react";

type Score = { n: number; accuracy: number; f1: number; false_refusals?: number; mean_latency_s?: number };
export type Report = {
  generated: string;
  n_items: number;
  retrieval: Record<string, number>;
  nano: { params: number; val_loss: number; val_ppl: number; vocab: number; curve: { iter: number; train: number; val: number }[] } | null;
  configs: Record<string, Record<string, Score>>;
};

// Charted configs (3 validated series colors). "ft+rag" (no guardrail) appears in the table view.
const CONFIG_META: Record<string, { label: string; color: string }> = {
  agent: { label: "BatLM + RAG + guardrail (deployed)", color: "var(--s1)" },
  "base+rag": { label: "SmolLM2 base + RAG", color: "var(--s2)" },
  "ft+oracle": { label: "BatLM + gold passage", color: "var(--s3)" },
};
const TABLE_LABEL: Record<string, string> = { agent: "deployed", "base+rag": "base", "ft+rag": "ft, no guard", "ft+oracle": "ft+gold" };
const KINDS: [string, string][] = [["reworded", "Reworded Qs"], ["unseen_fact", "Unseen facts"], ["out_of_scope", "Off-topic refusal"], ["overall", "Overall"]];
const pct = (x: number) => `${Math.round(x * 100)}%`;

type Tip = { x: number; y: number; lines: [string, string][] } | null;

function Tooltip({ tip }: { tip: Tip }) {
  if (!tip) return null;
  return (
    <div className="tip" style={{ left: tip.x, top: tip.y, transform: "translate(-50%, calc(-100% - 10px))" }}>
      {tip.lines.map(([k, v]) => <div key={k}><span className="k">{k}</span> {v}</div>)}
    </div>
  );
}

function AccuracyChart({ configs }: { configs: Report["configs"] }) {
  const [tip, setTip] = useState<Tip>(null);
  const [table, setTable] = useState(false);
  const cfgs = Object.keys(CONFIG_META).filter((c) => configs[c]);
  const W = 340, L = 92, R = 30, barH = 8, gap = 2, groupGap = 14;
  const groupH = cfgs.length * (barH + gap) - gap;
  const H = KINDS.length * (groupH + groupGap);
  const x = (v: number) => L + v * (W - L - R);
  return (
    <div className="chart" style={{ position: "relative" }}>
      <div className="chart-title">Answer accuracy by question type</div>
      <div className="chart-sub">Held-out eval set ({Object.values(configs)[0]?.overall?.n ?? 0} Qs). Correct = all key facts present, no false refusal.</div>
      <div className="legend">
        {cfgs.map((c) => <span key={c}><i style={{ background: CONFIG_META[c].color }} />{CONFIG_META[c].label}</span>)}
      </div>
      {table ? (
        <table className="data">
          <thead><tr><th>TYPE</th>{Object.keys(configs).map((c) => <th key={c}>{TABLE_LABEL[c] ?? c}</th>)}</tr></thead>
          <tbody>{KINDS.map(([k, lab]) => <tr key={k}><td>{lab}</td>{Object.keys(configs).map((c) => <td key={c}>{configs[c][k] ? pct(configs[c][k].accuracy) : "—"}</td>)}</tr>)}</tbody>
        </table>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`} onMouseLeave={() => setTip(null)}>
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={0} y2={H - groupGap} stroke="rgba(255,255,255,0.06)" />
              <text x={x(t)} y={H - 2} textAnchor="middle">{pct(t)}</text>
            </g>
          ))}
          {KINDS.map(([k, lab], gi) => {
            const y0 = gi * (groupH + groupGap);
            return (
              <g key={k}>
                <text x={0} y={y0 + groupH / 2 + 3} style={{ fill: "var(--ink-2)" }}>{lab}</text>
                {cfgs.map((c, ci) => {
                  const s = configs[c][k];
                  if (!s) return null;
                  const y = y0 + ci * (barH + gap);
                  const w = Math.max(2, x(s.accuracy) - L);
                  return (
                    <g key={c}
                      onMouseMove={(e) => {
                        const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                        setTip({ x: ((L + w) / W) * r.width, y: (y / H) * r.height, lines: [[CONFIG_META[c].label, ""], ["accuracy", pct(s.accuracy)], ["token F1", s.f1.toFixed(2)], ["n", String(s.n)]] });
                      }}>
                      <rect x={L} y={y - 3} width={W - L - R} height={barH + 6} fill="transparent" />
                      <motion.rect
                        x={L} y={y} height={barH} rx={2} fill={CONFIG_META[c].color}
                        initial={{ width: 0 }} animate={{ width: w }} transition={{ delay: gi * 0.08 + ci * 0.05, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                      />
                      {c === "agent" && <text x={L + w + 4} y={y + barH - 1} style={{ fill: "var(--ink)" }}>{pct(s.accuracy)}</text>}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      )}
      <button className="linkbtn" onClick={() => setTable((t) => !t)}>{table ? "SHOW CHART" : "SHOW TABLE"}</button>
      <Tooltip tip={tip} />
    </div>
  );
}

function LossChart({ curve }: { curve: NonNullable<Report["nano"]>["curve"] }) {
  const [hover, setHover] = useState<number | null>(null);
  const [table, setTable] = useState(false);
  const W = 340, H = 170, L = 28, R = 40, T = 10, B = 20;
  const maxIt = curve[curve.length - 1].iter;
  const vals = curve.flatMap((p) => [p.train, p.val]);
  const lo = Math.floor(Math.min(...vals)), hi = Math.ceil(Math.max(...vals));
  const x = (it: number) => L + (it / maxIt) * (W - L - R);
  const y = (v: number) => T + ((hi - v) / (hi - lo)) * (H - T - B);
  const line = (k: "train" | "val") => curve.map((p, i) => `${i ? "L" : "M"}${x(p.iter)},${y(p[k])}`).join(" ");
  const best = curve.reduce((a, b) => (b.val < a.val ? b : a));
  const last = curve[curve.length - 1];
  const hp = hover !== null ? curve[hover] : null;
  return (
    <div className="chart" style={{ position: "relative" }}>
      <div className="chart-title">BatLM-nano pretraining loss</div>
      <div className="chart-sub">Cross-entropy by training step. The gap between the lines is overfitting: 40 KB of text is not much.</div>
      <div className="legend">
        <span><i style={{ background: "var(--s1)" }} />Train</span>
        <span><i style={{ background: "var(--s2)" }} />Validation (held-out 10%)</span>
      </div>
      {table ? (
        <table className="data">
          <thead><tr><th>STEP</th><th>TRAIN</th><th>VAL</th></tr></thead>
          <tbody>{curve.map((p) => <tr key={p.iter}><td>{p.iter}</td><td>{p.train.toFixed(3)}</td><td>{p.val.toFixed(3)}</td></tr>)}</tbody>
        </table>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`}
          onMouseMove={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            const it = (((e.clientX - r.left) / r.width) * W - L) / (W - L - R) * maxIt;
            let bi = 0;
            curve.forEach((p, i) => { if (Math.abs(p.iter - it) < Math.abs(curve[bi].iter - it)) bi = i; });
            setHover(bi);
          }}
          onMouseLeave={() => setHover(null)}>
          {[lo, (lo + hi) / 2, hi].map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="rgba(255,255,255,0.06)" />
              <text x={L - 4} y={y(v) + 3} textAnchor="end">{v.toFixed(1)}</text>
            </g>
          ))}
          <text x={L} y={H - 4}>0</text>
          <text x={W - R} y={H - 4} textAnchor="end">{maxIt} steps</text>
          <motion.path d={line("train")} fill="none" stroke="var(--s1)" strokeWidth={2} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.4 }} />
          <motion.path d={line("val")} fill="none" stroke="var(--s2)" strokeWidth={2} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.4, delay: 0.15 }} />
          <text x={W - R + 4} y={y(last.train) + 3} style={{ fill: "var(--ink-2)" }}>train</text>
          <text x={W - R + 4} y={y(last.val) + 3} style={{ fill: "var(--ink-2)" }}>val</text>
          <circle cx={x(best.iter)} cy={y(best.val)} r={4} fill="var(--bg)" stroke="var(--s2)" strokeWidth={2} />
          <text x={x(best.iter) + 7} y={y(best.val) + 14} style={{ fill: "var(--ink-2)" }}>best ckpt · {best.val.toFixed(2)}</text>
          {hp && (
            <g>
              <line x1={x(hp.iter)} x2={x(hp.iter)} y1={T} y2={H - B} stroke="var(--ink-3)" strokeDasharray="2 3" />
              <circle cx={x(hp.iter)} cy={y(hp.train)} r={4} fill="var(--s1)" stroke="var(--bg)" strokeWidth={2} />
              <circle cx={x(hp.iter)} cy={y(hp.val)} r={4} fill="var(--s2)" stroke="var(--bg)" strokeWidth={2} />
            </g>
          )}
        </svg>
      )}
      {hp && !table && (
        <div className="tip" style={{ left: `${(x(hp.iter) / W) * 100}%`, top: 70, transform: "translate(-50%, -100%)" }}>
          <div><span className="k">step</span> {hp.iter}</div>
          <div><span className="k">train</span> {hp.train.toFixed(3)}</div>
          <div><span className="k">val</span> {hp.val.toFixed(3)}</div>
        </div>
      )}
      <button className="linkbtn" onClick={() => setTable((t) => !t)}>{table ? "SHOW CHART" : "SHOW TABLE"}</button>
    </div>
  );
}

export function Diagnostics({ report }: { report: Report | null }) {
  if (!report) return <div className="side-empty">NO EVAL REPORT FOUND<br />Run <code>python -m batlm.evals</code></div>;
  const depCfg = report.configs["agent"] ?? report.configs["ft+rag"];
  const dep = depCfg?.overall;
  const base = report.configs["base+rag"]?.overall;
  const tiles: [string, string, string][] = [
    ["BatLM accuracy", dep ? pct(dep.accuracy) : "—", base ? `vs ${pct(base.accuracy)} before fine-tuning` : ""],
    ["Refuses off-topic", depCfg?.out_of_scope ? pct(depCfg.out_of_scope.accuracy) : "—", base ? `base model: ${pct(report.configs["base+rag"].out_of_scope?.accuracy ?? 0)}` : ""],
    ["False refusals", dep ? String(dep.false_refusals) : "—", `of ${dep ? dep.n - (depCfg.out_of_scope?.n ?? 0) : 0} answerable`],
    ["Nano val perplexity", report.nano ? String(report.nano.val_ppl) : "—", report.nano ? `${(report.nano.params / 1e6).toFixed(2)}M params · vocab ${report.nano.vocab}` : ""],
  ];
  return (
    <div>
      <div className="tiles">
        {tiles.map(([k, v, d], i) => (
          <motion.div className="tile" key={k} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}>
            <div className="label">{k}</div>
            <div className="v">{v}</div>
            <div className="d">{d}</div>
          </motion.div>
        ))}
      </div>
      {Object.keys(report.configs).length > 0 && <AccuracyChart configs={report.configs} />}
      {report.nano && <LossChart curve={report.nano.curve} />}
      <div className="fineprint">
        Retrieval recall@3: {pct(report.retrieval["recall@3"] ?? 0)} · report generated {report.generated} · greedy decoding · CPU.
        Scoring caveat: "correct" checks that key facts appear, so it can reward answers that mention the right names in the wrong roles —
        read the per-question rows in <code>evals/</code>. Reproduce: <code>python -m batlm.evals</code>
      </div>
    </div>
  );
}
