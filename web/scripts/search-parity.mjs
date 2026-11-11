// Checks that the browser's search returns exactly what the Python search returns (run by tests/test_pipeline.py).
//   node web/scripts/search-parity.mjs <dir with index.json and expected.json>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Index } from "../src/lib/search.ts";

const dir = process.argv[2];
const index = new Index(JSON.parse(readFileSync(join(dir, "index.json"), "utf8")).passages);
let bad = 0;
for (const { q, ids, named } of JSON.parse(readFileSync(join(dir, "expected.json"), "utf8"))) {
  const got = index.search(q, 3).map((h) => h.passage.id);
  if (got.join() !== ids.join() || index.mentions(q).length !== named) {
    bad++;
    console.log(`MISMATCH ${q}\n  python ${ids}\n  browser ${got}`);
  }
}
console.log(bad ? `${bad} mismatches` : "browser search matches Python");
process.exit(bad ? 1 : 0);
