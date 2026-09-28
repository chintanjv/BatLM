// After `vite build`: drop the local copy of BatLM-360M from dist/ when production loads it from the HF Hub.
// (public/models/batlm-360m exists only for local dev; shipping it would push ~700 MB to the host.)
import { existsSync, rmSync } from "node:fs";

const id = process.env.VITE_BATLM_MODEL || "chintanjv/BatLM-360M"; // same default as worker.ts for production builds
if (id.includes("/") && existsSync("dist/models/batlm-360m")) {
  rmSync("dist/models/batlm-360m", { recursive: true });
  console.log(`pruned dist/models/batlm-360m (production loads ${id} from the Hugging Face Hub)`);
}
