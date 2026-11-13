// Checks that the browser's search returns exactly what the Python search returns (run by tests/test_pipeline.py).
//   node web/scripts/search-parity.mjs <dir with index.json and expected.json>
// Each case is one conversation turn: the question, the previous turn (or null), and what Python found.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Index } from "../src/lib/search.ts";

const dir = process.argv[2];
const index = new Index(JSON.parse(readFileSync(join(dir, "index.json"), "utf8")).passages);
const cases = JSON.parse(readFileSync(join(dir, "expected.json"), "utf8"));
let bad = 0;
for (const { q, last, named, carried, ids } of cases) {
  const s = index.subjects(q, last);
  const got = index.search(q, 3, s.carried).map((h) => h.passage.id);
  if (got.join() !== ids.join() || JSON.stringify(s.named) !== JSON.stringify(named) || JSON.stringify(s.carried) !== JSON.stringify(carried)) {
    bad++;
    console.log(`MISMATCH ${q}\n  python  ${ids} named ${JSON.stringify(named)} carried ${JSON.stringify(carried)}\n  browser ${got} named ${JSON.stringify(s.named)} carried ${JSON.stringify(s.carried)}`);
  }
}
console.log(bad ? `${bad} mismatches in ${cases.length} cases` : `browser search matches Python on ${cases.length} cases`);
process.exit(bad ? 1 : 0);
