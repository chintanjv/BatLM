const STAGES: [string, string, string][] = [
  ["DATA", "Two pages scraped via the MediaWiki API, cleaned of refs/tables, split into 99 sentence-aligned ~80-word chunks.", "batlm/scrape.py"],
  ["TOKENIZER", "Byte-level BPE written from scratch: 768 tokens learned from the corpus.", "batlm/tokenizer.py"],
  ["PRETRAIN", "BatLM-nano: a 4-layer, 1.9M-param GPT trained from random weights on a laptop CPU.", "batlm/model.py · train_nano.py"],
  ["INSTRUCT", "BatLM: SmolLM2-360M-Instruct + LoRA (r=16) on 129 hand-written, source-grounded Q&A — with retrieved context in every prompt.", "batlm/sft_data.py · train_lora.py"],
  ["RETRIEVE", "BM25 with stemming + a Gotham thesaurus for query expansion. Same code in Python and TypeScript.", "batlm/rag.py · lib/bm25.ts"],
  ["ORACLE", "Agent loop: plan → tools (search, timeline, quiz) → confidence guardrail → generate → trace.", "batlm/agent.py · lib/agent.ts"],
  ["EVALS", "Held-out rewordings, unseen facts, off-topic refusals, retrieval recall, ablations vs base & oracle.", "batlm/evals.py"],
  ["SHIP", "ONNX + 4-bit quantization for BatLM; our own TypeScript transformer engine for nano. Runs in your browser — WebGPU or WASM.", "batlm/export_web.py · worker.ts"],
];

export function Dossier() {
  return (
    <div>
      <div className="label" style={{ marginBottom: 8 }}>How BatLM was built — end to end</div>
      <div className="pipeline">
        {STAGES.map(([t, d, f], i) => (
          <div className="stage" key={t}>
            <div className="n">{String(i + 1).padStart(2, "0")}</div>
            <div><h5>{t}</h5><p>{d}</p><code>{f}</code></div>
          </div>
        ))}
      </div>
      <div className="fineprint">
        Training data: <a href="https://en.wikipedia.org/wiki/Batman" target="_blank" rel="noreferrer">Wikipedia — Batman</a> and{" "}
        <a href="https://batman.fandom.com/wiki/Batman_Wiki" target="_blank" rel="noreferrer">Batman Wiki (Fandom)</a>, both CC BY-SA; derived
        data and model weights are shared under CC BY-SA 4.0. Base model: SmolLM2-360M-Instruct (Apache-2.0).<br /><br />
        BatLM is an unofficial, educational fan project. Batman and related characters are trademarks of DC Comics; this project is not affiliated
        with or endorsed by DC or Warner Bros. Small models make mistakes — verify against the cited sources.
      </div>
    </div>
  );
}
