import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
import argparse
import pyterrier as pt
import torch
from pyterrier_t5 import MonoT5ReRanker
from reformir import ReformIR

parser = argparse.ArgumentParser()
parser.add_argument("--dl", type=int, required=True, help="19, 20, 21, or 22")
parser.add_argument("--budget", type=int, default=100)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--ce", type=int, default=7, help="number of cross-encoder calls per query")
parser.add_argument("--reformulation_file", type=str, required=True)
parser.add_argument("--weights_out", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

indexref = os.path.join(
    os.environ["PYTERRIER_HOME"],
    "corpora", "msmarco_passage", "index", "terrier_stemmed_reverse"
)
existing_index = pt.IndexFactory.of(indexref)
terrier_ = pt.terrier.Retriever(existing_index, wmodel="BM25", num_results=100)
bm25_cerberus = pt.terrier.Retriever(existing_index, wmodel="BM25", num_results=args.budget)

dataset = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{args.dl}/judged")
scorer = pt.text.get_text(dataset, "text") >> MonoT5ReRanker(verbose=False, batch_size=args.batch)

os.makedirs(os.path.dirname(os.path.abspath(args.weights_out)), exist_ok=True)


class RemoveColon(pt.Transformer):
    def transform(self, topics):
        topics = topics.copy()
        topics["query"] = topics["query"].str.replace(":", " ", regex=False)
        return topics


pipeline = RemoveColon() >> bm25_cerberus >> ReformIR(
    scorer,
    num_results=args.budget,
    terrier_index=existing_index,
    bm25_retriever=terrier_,
    cross_enc_budget=args.ce,
    file_name=args.reformulation_file,
    weights_log_path=args.weights_out,
    verbose=True,
)

pipeline.transform(dataset.get_topics())
print(f"Weights written to {args.weights_out}")
