// Original emblem drawn for this project: a bat inside an oval. It is not DC's logo, and must not become a copy of it.
const BAT_PATH =
  "M100,30 L104,26 L107,10 L111,28 Q125,32 140,24 Q165,10 196,8 Q176,22 178,46 Q164,40 152,52 Q142,46 132,62 Q118,58 110,74 L100,92 L90,74 Q82,58 68,62 Q58,46 48,52 Q36,40 22,46 Q24,22 4,8 Q35,10 60,24 Q75,32 89,28 L93,10 L96,26 Z";

function Badge({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="-6 -12 212 124" aria-hidden>
      <ellipse className="oval" cx="100" cy="50" rx="104" ry="60" pathLength={1} />
      <ellipse className="rim" cx="100" cy="50" rx="96" ry="52" />
      <path className="bat" d={BAT_PATH} transform="translate(100 50) scale(0.78) translate(-100 -50)" />
    </svg>
  );
}

/** Small mark: top bar and answer avatar. */
export const BatMark = () => <Badge className="mark" />;

/** Large mark for the empty state: the oval draws itself and fills, then the bat appears (see .emblem in styles.css). */
export const Emblem = () => <Badge className="mark emblem" />;
