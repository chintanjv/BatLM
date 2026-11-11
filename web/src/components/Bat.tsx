// Original bat emblem, symmetric about x=100 (drawn for this project; not an official logo).
const BAT_PATH =
  "M100,30 L104,26 L107,10 L111,28 Q125,32 140,24 Q165,10 196,8 Q176,22 178,46 Q164,40 152,52 Q142,46 132,62 Q118,58 110,74 L100,92 L90,74 Q82,58 68,62 Q58,46 48,52 Q36,40 22,46 Q24,22 4,8 Q35,10 60,24 Q75,32 89,28 L93,10 L96,26 Z";

/** Small solid mark: top bar and answer avatar. */
export function BatMark() {
  return <svg viewBox="0 -6 200 106" aria-hidden><path d={BAT_PATH} /></svg>;
}

/** Large mark for the empty state: the outline draws itself, then fills (see .emblem in styles.css). */
export function Emblem() {
  return <svg className="emblem" viewBox="0 -6 200 106" aria-hidden><path pathLength={1} d={BAT_PATH} /></svg>;
}
