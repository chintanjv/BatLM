// Owns the inference worker: loading (with progress), streaming answers, stopping, and GPU-failure recovery.
import { useCallback, useEffect, useRef, useState } from "react";
import type { WorkerIn, WorkerOut } from "../worker";

export type ModelState = {
  status: "idle" | "loading" | "ready" | "error";
  /** 0..1 while loading. */
  progress: number;
  loadedMB: number;
  totalMB: number;
  device?: "webgpu" | "wasm";
  /** True until the model has loaded once in this browser, i.e. the files still have to download. */
  firstRun: boolean;
};
export type Stream = {
  onText(text: string): void;
  onDone(stats: { tokens: number; ms: number }): void;
  onError(message: string): void;
};

const CACHED_KEY = "batlm:model-cached";
const cached = () => { try { return localStorage.getItem(CACHED_KEY) === "1"; } catch { return false; } };

export function useModel() {
  const [state, setState] = useState<ModelState>(() => ({ status: "idle", progress: 0, loadedMB: 0, totalMB: 0, firstRun: !cached() }));
  const worker = useRef<Worker | null>(null);
  const forceWasm = useRef(false);
  const pending = useRef<{ promise: Promise<void>; resolve(): void; reject(e: Error): void } | null>(null);
  const stream = useRef<Stream | null>(null);
  const files = useRef(new Map<string, [number, number]>());
  const frame = useRef(0);

  const post = (m: WorkerIn) => worker.current?.postMessage(m);

  const spawn = useCallback(() => {
    const w = new Worker(new URL("../worker.ts", import.meta.url), { type: "module" });
    w.onmessage = ({ data: m }: MessageEvent<WorkerOut>) => {
      if (m.type === "token") stream.current?.onText(m.text);
      else if (m.type === "done") { const s = stream.current; stream.current = null; s?.onDone(m); }
      else if (m.type === "progress") {
        files.current.set(m.file, [m.loaded, m.total]);
        frame.current ||= requestAnimationFrame(() => { // at most one render per frame, however fast chunks arrive
          frame.current = 0;
          let loaded = 0, total = 0;
          files.current.forEach(([l, t]) => { loaded += l; total += t; });
          setState((s) => (s.status === "loading" ? { ...s, progress: total ? loaded / total : 0, loadedMB: loaded / 1e6, totalMB: total / 1e6 } : s));
        });
      } else if (m.type === "ready") {
        try { localStorage.setItem(CACHED_KEY, "1"); } catch { /* private window: the model simply downloads again next time */ }
        setState((s) => ({ ...s, status: "ready", progress: 1, device: m.device, firstRun: false }));
        pending.current?.resolve();
      } else if (m.type === "gpu-failed") {
        // The GPU path is unusable on this machine: replace the worker with a CPU-only one and load again.
        w.terminate();
        forceWasm.current = true;
        files.current.clear();
        worker.current = spawn();
        post({ type: "load", forceWasm: true });
      } else if (m.loading) {
        setState((s) => ({ ...s, status: "error" }));
        pending.current?.reject(new Error(m.message));
        pending.current = null;
      } else { const s = stream.current; stream.current = null; s?.onError(m.message); }
    };
    return w;
  }, []);

  const load = useCallback((): Promise<void> => {
    if (!pending.current) {
      let resolve!: () => void, reject!: (e: Error) => void;
      const promise = new Promise<void>((res, rej) => { resolve = res; reject = rej; });
      promise.catch(() => {}); // callers handle the failure; this stops an unhandled-rejection warning
      pending.current = { promise, resolve, reject };
      files.current.clear();
      setState((s) => ({ ...s, status: "loading", progress: 0 }));
      post({ type: "load", forceWasm: forceWasm.current });
    }
    return pending.current.promise;
  }, []);

  const generate = useCallback((id: string, prompt: string, s: Stream) => {
    stream.current = s;
    post({ type: "generate", id, prompt, maxTokens: 160 });
  }, []);

  const abort = useCallback(() => post({ type: "abort" }), []);

  useEffect(() => {
    worker.current = spawn();
    if (cached()) void load(); // already downloaded once: get it ready before the first question
    return () => {
      worker.current?.terminate();
      worker.current = null;
      pending.current = null;
      cancelAnimationFrame(frame.current);
      frame.current = 0;
    };
  }, [spawn, load]);

  return { state, load, generate, abort };
}
