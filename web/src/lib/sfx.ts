// Tiny synthesized UI sounds (no audio files). Off by default; toggled in the top bar.
let ctx: AudioContext | null = null;
let enabled = false;
try { enabled = localStorage.getItem("batlm.sfx") === "1"; } catch { /* storage blocked */ }

export const sfx = {
  get enabled() { return enabled; },
  set(on: boolean) {
    enabled = on;
    try { localStorage.setItem("batlm.sfx", on ? "1" : "0"); } catch { /* ignore */ }
    if (on) blip(880, 0.05);
  },
  send: () => { blip(520, 0.06); setTimeout(() => blip(1040, 0.05), 60); },
  tick: () => blip(1800 + Math.random() * 400, 0.012, 0.02),
  done: () => { blip(660, 0.05); setTimeout(() => blip(990, 0.08), 70); },
  deny: () => blip(180, 0.16, 0.08, "sawtooth"),
};

function blip(freq: number, dur: number, vol = 0.05, type: OscillatorType = "square") {
  if (!enabled) return;
  ctx ??= new AudioContext();
  const o = ctx.createOscillator(), g = ctx.createGain(), t = ctx.currentTime;
  o.type = type;
  o.frequency.setValueAtTime(freq, t);
  g.gain.setValueAtTime(vol, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(ctx.destination);
  o.start(t);
  o.stop(t + dur + 0.02);
}
