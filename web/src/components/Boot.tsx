import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Emblem } from "./Emblem";

const LINES: [string, string?][] = [
  ["WAYNE ENTERPRISES // APPLIED SCIENCES DIVISION"],
  ["BATCOMPUTER OS v4.27 ........................ ", "ok"],
  ["mounting case files: wikipedia/batman, fandom/batman_wiki ", "ok"],
  ["indexing 99 dossiers · bm25 k1=1.5 b=0.75 ...... ", "ok"],
  ["neural cores: BATLM-360M (lora) · BATLM-NANO (scratch) ", "ok"],
  ["ORACLE protocol armed · guardrail threshold 3.0 ", "ok"],
  ["all inference local to this device. nothing leaves the cave.", "gold"],
];

export function Boot({ onDone }: { onDone: () => void }) {
  const [n, setN] = useState(0);
  const [leaving, setLeaving] = useState(false);
  const ready = n >= LINES.length;

  useEffect(() => {
    if (ready) return;
    const t = setTimeout(() => setN((x) => x + 1), n === 0 ? 1300 : 190 + Math.random() * 160);
    return () => clearTimeout(t);
  }, [n, ready]);

  const leave = () => { if (!leaving) { setLeaving(true); setTimeout(onDone, 650); } };
  useEffect(() => {
    const k = (e: KeyboardEvent) => (e.key === "Enter" || e.key === "Escape" || e.key === " ") && leave();
    addEventListener("keydown", k);
    return () => removeEventListener("keydown", k);
  });

  return (
    <motion.div
      className="boot"
      onClick={leave}
      initial={{ clipPath: "circle(150% at 50% 50%)" }}
      animate={leaving ? { clipPath: "circle(0% at 50% 42%)" } : { clipPath: "circle(150% at 50% 50%)" }}
      transition={{ duration: 0.65, ease: [0.7, 0, 0.2, 1] }}
    >
      <div className="boot-inner">
        <Emblem className="boot-emblem" draw />
        <div className="boot-log" aria-live="polite">
          {LINES.slice(0, n).map(([t, s], i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.18 }}>
              <span style={{ color: "var(--ink-3)" }}>[{(0.137 * (i + 1)).toFixed(3)}] </span>
              {s === "gold" ? <span className="gold">{t}</span> : <>{t}{s && <span className="ok">[ OK ]</span>}</>}
            </motion.div>
          ))}
        </div>
        {ready && <div className="boot-cta">PRESS ENTER TO ENGAGE</div>}
      </div>
      <button className="boot-skip" onClick={leave}>SKIP ›</button>
    </motion.div>
  );
}
