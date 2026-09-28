import { motion } from "motion/react";

// Original bat emblem, symmetric about x=100 (drawn for this project; not an official logo).
export const BAT_PATH =
  "M100,30 L104,26 L107,10 L111,28 Q125,32 140,24 Q165,10 196,8 Q176,22 178,46 Q164,40 152,52 Q142,46 132,62 Q118,58 110,74 L100,92 L90,74 Q82,58 68,62 Q58,46 48,52 Q36,40 22,46 Q24,22 4,8 Q35,10 60,24 Q75,32 89,28 L93,10 L96,26 Z";

export function Emblem({ className, draw = false, glow = true }: { className?: string; draw?: boolean; glow?: boolean }) {
  return (
    <svg className={className} viewBox="-10 -12 220 120" aria-hidden>
      <defs>
        <radialGradient id="emb-oval" cx="50%" cy="45%" r="60%">
          <stop offset="0%" stopColor="#f5c518" stopOpacity="0.95" />
          <stop offset="70%" stopColor="#d9a90a" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#8a6a00" stopOpacity="0.9" />
        </radialGradient>
        <filter id="emb-glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      <motion.ellipse
        cx="100" cy="48" rx="106" ry="58" fill="url(#emb-oval)" stroke="#f5c518" strokeWidth="1.5"
        filter={glow ? "url(#emb-glow)" : undefined}
        initial={draw ? { pathLength: 0, fillOpacity: 0 } : false}
        animate={{ pathLength: 1, fillOpacity: 1 }}
        transition={{ pathLength: { duration: 1.1, ease: [0.22, 1, 0.36, 1] }, fillOpacity: { delay: 1.0, duration: 0.6 } }}
      />
      <motion.path
        d={BAT_PATH} fill="#050607" stroke="#050607" strokeWidth="1"
        initial={draw ? { scale: 0.6, opacity: 0 } : false}
        animate={{ scale: 1, opacity: 1 }}
        style={{ transformOrigin: "100px 48px" }}
        transition={{ delay: draw ? 1.1 : 0, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      />
    </svg>
  );
}

export function BatMark({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 -6 200 106" aria-hidden>
      <path d={BAT_PATH} fill="#f5c518" />
    </svg>
  );
}
