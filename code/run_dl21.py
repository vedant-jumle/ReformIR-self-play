import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import pyterrier as pt
import pyterrier_alpha as pta
from ir_measures import nDCG, R
#uncomment below line if u face java issues after you download zstd version 1.5.5-2. Note you have to point to correct path of java_tmp folder
#pt.init(jvm_opts=["-Djava.io.tmpdir="+"<java tmp path>", "-Dterrier.zstd.version=1.5.5-2"])


from reformir import ReformIR

import torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

from baselines.genqr_ensemble_baseline import ReformRank

from pyterrier_t5 import MonoT5ReRanker
import torch
import argparse
import pandas as pd
import random
parser = argparse.ArgumentParser()
parser.add_argument("--lk", type=int, default=16, help="the value of k for selecting k neighbourhood graph")
parser.add_argument("--dl", type=int, default=19, help="dl 19 or 20")
parser.add_argument("--budget", type=int, default=100, help="budget c")
parser.add_argument("--batch", type=int, default=16, help="batch size")
parser.add_argument("--ce", type=int, default=7, help="number of cross encoder calls")
parser.add_argument("--s", type=int, default=30, help="top s docs (S) to calculate the RM3 features.")
parser.add_argument("--reformulation_file", type=str, default="", help="path to the file containing generated query reformulations")

parser.add_argument("--verbose", action="store_true", help="if show progress bar.")
args = parser.parse_args()
dataset = pt.get_dataset("irds:msmarco-passage-v2")
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
indexref  =  <path to bm25 index>
existing_index = pt.IndexFactory.of(indexref)
expander_rm3 = pt.rewrite.RM3(indexref, fb_terms=10, fb_docs=10)
terrier_ = pt.terrier.Retriever(existing_index, wmodel="BM25", num_results =100)
reform_bm25 = pt.terrier.Retriever(existing_index, wmodel="BM25", num_results = 100)
retriever = pt.terrier.Retriever(existing_index, wmodel="BM25")
bm25_cerberus = pt.terrier.Retriever(existing_index, wmodel="BM25", num_results=args.budget)
"""
We re-use the existing bm25 and tct-based corpus graphs (introduced in GAR paper) and laff graph (introduced in Quam paper).
"""
scorer = pt.text.get_text(dataset, "text") >> MonoT5ReRanker(verbose=False, batch_size=args.batch)
dataset = pt.get_dataset(f"irds:msmarco-passage-v2/trec-dl-20{args.dl}/judged")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
save_dir=f"runs/reform/dl{args.dl}/"
print(f"number of cross encoder calls: {args.ce}")
#llm_ranker = PointwiseReranker('castorini/rankllama-v1-7b-lora-passage', max_length=4096) #
#llm_reranker = LLMReRanker("castorini/rankllama-v1-7b-lora-passage")
class RemoveColon(pt.Transformer):
    def transform(self, topics):
        topics = topics.copy()
        topics["query"] = topics["query"].str.replace(":", " ", regex=False)
        return topics
qrels = pd.read_csv(f"data/dl21.dedup.qrels", header=None, sep=" ", names=["qid", "Q0", "docno", "label"],
                    dtype={"qid":str, "Q0":str, "docno":str, "label":int})
#llama_ranker = pt.text.get_text(dataset, 'text') >> llm_ranker
result = pt.Experiment(
    [  #RemoveColon() >> retriever % args.budget >> scorer,
       # retriever % args.budget >> llama_ranker
       # retriever % args.budget >> expander_rm3 >> retriever % args.budget >>pt.rewrite.reset() >> scorer,
       # retriever % args.budget >> expander_rm3 >> retriever % args.budget >>pt.rewrite.reset() >> llama_ranker,

       RemoveColon() >> retriever % args.budget >> ReformRank(budget = args.budget, bm25_retriever = reform_bm25,scorer=scorer, batch_size = 16, file_name = args.reformulation_file),
    #    RemoveColon() >> bm25_cerberus >> GenQRPRFORE(scorer, num_results=args.budget, laff_graph = tct_graph, dense_retriever = tct_retriever,
    #                                               terrier_index = existing_index, bm25_retriever = terrier_ ,cross_enc_budget=args.ce ,verbose=args.verbose)        # retriever >> HybridRRF(scorer, num_results=args.budget, df2 = tct_res,verbose=args.verbose),
       RemoveColon() >> bm25_cerberus >> ReformIR(scorer, num_results=args.budget,
                                                terrier_index = existing_index, bm25_retriever = terrier_ ,cross_enc_budget=args.ce ,verbose=args.verbose, file_name = args.reformulation_file)        # retriever >> HybridRRF(scorer, num_results=args.budget, df2 = tct_res,verbose=args.verbose),
        ],
    dataset.get_topics(),
    qrels,
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[

        f"BM25_genqr_ensemble_monot5.c{args.budget}", 
        f"BM25_genqr_ensemble_reformopt_monot5.c{args.budget}" 
      #  f"BM25_genqr_reformopt_final_monot5.c{args.budget}"
        ],
        save_dir=save_dir,
        save_mode="reuse",
        baseline=0,
        correction="bonferroni" # "overwrite" to overwrite the existing results
)
print(result.T)