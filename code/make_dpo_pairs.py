import argparse
import pandas as pd
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, required=True, help="weights CSV from collect_weights.py")
parser.add_argument("--out", type=str, required=True, help="output JSONL path (HF DPO format)")
parser.add_argument("--min_spread", type=float, default=0.1, help="min(max-min) weight spread to keep query")
parser.add_argument("--prompt_template", type=str,
    default="Generate search query reformulations for: {query}",
    help="prompt template, use {query} placeholder")
args = parser.parse_args()

df = pd.read_csv(args.weights)
w_cols = [c for c in df.columns if c.startswith("w_") and c != "w_orig"]
r_cols = [c for c in df.columns if c.startswith("reform_")]

assert len(w_cols) == len(r_cols), "weight/reform column count mismatch"

pairs = []
skipped = 0
for _, row in df.iterrows():
    weights = np.array([row[c] for c in w_cols])
    reforms = [str(row[c]) for c in r_cols]
    spread = weights.max() - weights.min()

    if spread < args.min_spread:
        skipped += 1
        continue

    best_idx = int(np.argmax(weights))
    worst_idx = int(np.argmin(weights))

    pairs.append({
        "prompt": args.prompt_template.format(query=row["orig_query"]),
        "chosen": reforms[best_idx],
        "rejected": reforms[worst_idx],
        "qid": str(row["qid"]),
        "w_chosen": float(weights[best_idx]),
        "w_rejected": float(weights[worst_idx]),
    })

with open(args.out, "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

print(f"Written {len(pairs)} DPO pairs to {args.out} ({skipped} skipped, spread < {args.min_spread})")
print(f"Mean w_chosen: {np.mean([p['w_chosen'] for p in pairs]):.3f}")
print(f"Mean w_rejected: {np.mean([p['w_rejected'] for p in pairs]):.3f}")
