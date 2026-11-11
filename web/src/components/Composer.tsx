import { memo, useEffect, useRef, useState } from "react";

type Props = {
  busy: boolean;
  onSend(text: string): void;
  onStop(): void;
  /** Model download in progress: 0..1 plus the line of text shown above the input. */
  download: { progress: number; label: string } | null;
};

// The draft lives here, not in App, so typing never re-renders the conversation.
export const Composer = memo(function Composer({ busy, onSend, onStop, download }: Props) {
  const [value, setValue] = useState("");
  const input = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { // grow with the text, and return to one line when it is cleared
    const el = input.current!;
    el.style.height = "";
    if (value) el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);
  // Focus the input when it is free again, but only with a mouse or trackpad: on a touch screen focusing
  // opens the keyboard over the answer the person is about to read.
  useEffect(() => { if (!busy && matchMedia("(pointer: fine)").matches) input.current?.focus(); }, [busy]);

  const submit = () => {
    if (busy || !value.trim()) return;
    onSend(value);
    setValue("");
  };

  return (
    <div className="composer" data-busy={busy ? 1 : 0}>
      <div className={download ? "track on" : "track"} style={{ transform: `scaleX(${download?.progress ?? 0})` }} />
      <div className={download ? "dl on" : "dl"}><div><span>{download?.label}</span></div></div>
      <textarea
        ref={input} id="question" rows={1} value={value} placeholder="Ask about Gotham…" aria-label="Your question" enterKeyHint="send"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); }
        }}
      />
      <button className="send" aria-label={busy ? "Stop" : "Send"} disabled={!busy && !value.trim()} onClick={busy ? onStop : submit}>
        <svg className="i-go" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M8 13V3M3.5 7.5 8 3l4.5 4.5" /></svg>
        <svg className="i-stop" viewBox="0 0 16 16" fill="currentColor" aria-hidden><rect x="3.5" y="3.5" width="9" height="9" rx="2" /></svg>
      </button>
    </div>
  );
});
