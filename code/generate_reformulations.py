import os
import argparse
import pandas as pd
import ir_datasets
import querygym as qg
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="msmarco-passage/train", help="ir_datasets dataset id")
parser.add_argument("--n_queries", type=int, default=1000, help="number of queries to reformulate")
parser.add_argument("--model", type=str, default="llama3.2:latest")
parser.add_argument("--base_url", type=str, default="http://127.0.0.1:11434/v1")
parser.add_argument("--out", type=str, required=True, help="output CSV path")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

dataset = ir_datasets.load(args.dataset)
queries = list(dataset.queries_iter())

import random
random.seed(args.seed)
random.shuffle(queries)
queries = queries[:args.n_queries]

reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model=args.model,
    llm_config={
        "base_url": args.base_url,
        "api_key": "ollama",
        "temperature": 0.5,
        "max_tokens": 256,
    }
)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

results = []
failed = 0
for q in tqdm(queries, desc="reformulating"):
    try:
        result = reformulator.reformulate(qg.QueryItem(q.query_id, q.text))
        variants = [result.metadata["variant_outputs"][f"variant_{i+1}"]["raw_output"]
                    for i in range(5)]
        reformulated = "--".join(variants)
        results.append({"qid": q.query_id, "query": reformulated})
    except Exception as e:
        failed += 1
        print(f"Failed qid={q.query_id}: {e}")

pd.DataFrame(results).to_csv(args.out, index=False)
print(f"Done. {len(results)} queries written to {args.out} ({failed} failed)")
