import { useEffect, useMemo, useRef } from "react";
import { BAT_PATH } from "./Emblem";

// Seeded RNG so Gotham's skyline is the same on every visit.
function rng(seed: number) {
  return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
}

function skyline(seed: number, n: number, minH: number, maxH: number) {
  const r = rng(seed);
  let x = 0, d = "M0,400 ";
  const windows: [number, number][] = [];
  while (x < 1600) {
    const w = 30 + r() * (1600 / n);
    const h = minH + r() * (maxH - minH);
    const top = 400 - h;
    d += `L${x},${top} `;
    if (r() > 0.72) { // spire
      d += `L${x + w * 0.45},${top} L${x + w * 0.5},${top - 20 - r() * 50} L${x + w * 0.55},${top} `;
    } else if (r() > 0.6) { // stepped gothic top
      d += `L${x + w * 0.2},${top} L${x + w * 0.2},${top - 14} L${x + w * 0.8},${top - 14} L${x + w * 0.8},${top} `;
    }
    d += `L${x + w},${top} `;
    for (let wy = top + 10; wy < 395; wy += 12)
      for (let wx = x + 5; wx < x + w - 5; wx += 9) if (r() > 0.93) windows.push([wx, wy]);
    x += w;
  }
  return { d: d + "L1600,400 Z", windows };
}

export function Atmosphere({ intensity = 1 }: { intensity?: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const far = useMemo(() => skyline(7, 34, 90, 230), []);
  const near = useMemo(() => skyline(1939, 22, 50, 170), []);

  useEffect(() => {
    const c = canvas.current!;
    const ctx = c.getContext("2d")!;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0, h = 0, raf = 0;
    const dpr = Math.min(devicePixelRatio, 2);
    const mouse = { x: 0.62, y: 0.25, tx: 0.62, ty: 0.25 };
    type Drop = { x: number; y: number; l: number; v: number; o: number };
    let drops: Drop[] = [];
    const resize = () => {
      w = c.clientWidth; h = c.clientHeight;
      c.width = w * dpr; c.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drops = Array.from({ length: Math.round((w * h) / 9000) }, () => ({
        x: Math.random() * w, y: Math.random() * h, l: 8 + Math.random() * 18, v: 7 + Math.random() * 9, o: 0.05 + Math.random() * 0.16,
      }));
    };
    const bat = new Path2D(BAT_PATH);
    const onMove = (e: PointerEvent) => { mouse.tx = e.clientX / w; mouse.ty = Math.min(e.clientY / h, 0.55); };

    const frame = () => {
      ctx.clearRect(0, 0, w, h);
      mouse.x += (mouse.tx - mouse.x) * 0.035;
      mouse.y += (mouse.ty - mouse.y) * 0.035;
      const sx = w * 0.5, sy = h + 20, tx = mouse.x * w, ty = Math.max(mouse.y * h, 60) - 20;
      // searchlight beam
      const ang = Math.atan2(ty - sy, tx - sx), spread = 0.07, len = Math.hypot(tx - sx, ty - sy) + 80;
      const g = ctx.createLinearGradient(sx, sy, tx, ty);
      g.addColorStop(0, `rgba(255,236,170,${0.16 * intensity})`);
      g.addColorStop(1, `rgba(255,236,170,${0.035 * intensity})`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(sx - 10, sy);
      ctx.lineTo(sx + Math.cos(ang - spread) * len, sy + Math.sin(ang - spread) * len);
      ctx.lineTo(sx + Math.cos(ang + spread) * len, sy + Math.sin(ang + spread) * len);
      ctx.lineTo(sx + 10, sy);
      ctx.fill();
      // bat-signal projected on the clouds
      const R = Math.min(w, h) * 0.11;
      ctx.save();
      ctx.translate(tx, ty);
      ctx.rotate((mouse.x - 0.5) * 0.25);
      const halo = ctx.createRadialGradient(0, 0, R * 0.2, 0, 0, R * 1.6);
      halo.addColorStop(0, `rgba(255,232,150,${0.22 * intensity})`);
      halo.addColorStop(0.55, `rgba(255,220,120,${0.1 * intensity})`);
      halo.addColorStop(1, "rgba(255,220,120,0)");
      ctx.fillStyle = halo;
      ctx.beginPath(); ctx.ellipse(0, 0, R * 1.6, R * 1.05, 0, 0, Math.PI * 2); ctx.fill();
      ctx.scale((R * 1.5) / 200, (R * 1.5) / 200);
      ctx.translate(-100, -48);
      ctx.fillStyle = `rgba(0,0,0,${0.5 * intensity})`;
      ctx.fill(bat);
      ctx.restore();
      // rain
      ctx.strokeStyle = "rgba(180,195,215,1)";
      ctx.lineWidth = 1;
      for (const d of drops) {
        ctx.globalAlpha = d.o;
        ctx.beginPath(); ctx.moveTo(d.x, d.y); ctx.lineTo(d.x - d.l * 0.18, d.y + d.l); ctx.stroke();
        if (!reduce) { d.y += d.v; d.x -= d.v * 0.18; }
        if (d.y > h) { d.y = -20; d.x = Math.random() * (w + 100); }
      }
      ctx.globalAlpha = 1;
      if (!reduce) raf = requestAnimationFrame(frame);
    };
    resize();
    addEventListener("resize", resize);
    addEventListener("pointermove", onMove);
    frame();
    return () => { cancelAnimationFrame(raf); removeEventListener("resize", resize); removeEventListener("pointermove", onMove); };
  }, [intensity]);

  return (
    <div className="atmos">
      <div className="fog" />
      <canvas ref={canvas} />
      <svg className="skyline" viewBox="0 0 1600 400" preserveAspectRatio="xMidYMax slice" aria-hidden>
        <defs>
          <linearGradient id="sky-far" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#15181d" /><stop offset="1" stopColor="#0a0c0f" />
          </linearGradient>
        </defs>
        <path d={far.d} fill="url(#sky-far)" />
        {far.windows.map(([x, y], i) => (
          <rect key={i} x={x} y={y} width="3" height="4" fill="#f5c518" opacity={0.18 + (i % 5) * 0.06}>
            {i % 9 === 0 && <animate attributeName="opacity" values="0.4;0.05;0.4" dur={`${4 + (i % 7)}s`} repeatCount="indefinite" />}
          </rect>
        ))}
        <path d={near.d} fill="#050607" />
      </svg>
    </div>
  );
}
