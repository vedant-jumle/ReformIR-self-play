from collections import Counter
from typing import List
from statistics import mean
import json
import torch
import numpy as np
import querygym as qg
from itertools import chain
import time
from statistics import mean, stdev
from scipy.special import softmax
from joblib import Parallel
import time
import heapq
import scipy
import random
from collections import defaultdict

random.seed(42)
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_adaptive import CorpusGraph
import ir_datasets
import torch








class ReformRank(pt.Transformer):
    def __init__(self, scorer, 
                 bm25_retriever, budget: int = 100, verbose: bool = False, 
                 batch_size : int = 16, file_name="data/genqr_ensemble/dl19.csv"):
        self.scorer = scorer
        self.budget = budget
        self.verbose = verbose
        self.batch_size = batch_size
        self.bm25_retriever = bm25_retriever
        self.reformulation_file = file_name

    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        result_builder = pta.DataFrameBuilder(['qid', 'query', 'docno', 'score', 'rank'])
        groups = list(inp.groupby('query'))
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

       # print("dataset.get_qrels()",inp, dataset.get_qrels())
        #exit(1)
        refor_queries = pd.read_csv(self.reformulation_file)

        # reformulator = qg.create_reformulator(
        #     "genqr_ensemble",
        #     model="gpt-4o-mini",

        #     llm_config={
        #         "api_key": "<api-key>",
        #         "temperature": 0.5,
        #         "num_docs":3
        #     }
        # )       

        for i, (query, initial_results) in enumerate(groups):
            scores = {}
            reformulated_queries = {"qid":[],"query": []}

            qid = initial_results['qid'].iloc[0]


            #print("qrel_groups", qrel_groups)
   

            initial_results = initial_results.sort_values('score', ascending=False)
            


            results = {}
            #result = llm_rewrite.get_gpt3_completions(query,qrel_docs,docstore)
            
            #reformulated_queries["reformed_queries"].append(result)
            results_saved = refor_queries.loc[refor_queries["qid"].astype(str) == str(qid)]
            queries_new = list(results_saved["query"].values)[0] + query #result_1.reformulated # queries_new
#list(results_saved["query"].values)[0] + query #
            reformulated_queries["qid"].append(qid)
            reformulated_queries["query"].append(queries_new)
            retrieved = self.bm25_retriever(pd.DataFrame(reformulated_queries))
            retrieved = dict(iter(retrieved.groupby(by='qid')))
            print("retrieved**",len(list(retrieved[qid]["docno"].values)))


            res_map = [Counter(dict(zip(retrieved[qid].docno, retrieved[qid].score)))]
            while len(scores) < self.budget:
                size = min(self.batch_size, self.budget - len(scores)) # get either the batch size or remaining budget (whichever is smaller)

                # build batch of documents to score in this round
                batch = res_map[0].most_common(size)
                batch = pd.DataFrame(batch, columns=['docno', 'score'])
                batch['qid'] = qid
                batch['query'] = query
                reranked_batch = self.scorer(batch)
                bm25_scores = {} #dict(zip(initial_results["docno"].values,initial_results["score"].values))
            #contexts = [docstore.get(docid).text for docid in list(initial_results["docno"].values)][:5]
                scores.update({k: (s, i) for k, s in zip(reranked_batch.docno, reranked_batch.score)})
                self._drop_docnos_from_counters(batch.docno, res_map)
           # print("result.metadata[questions_json]",result.metadata["questions_json"], result.reformulated)
           # meta_data = result #result.metadata["csqe_sentences"][0].split(".")[:1] #json.loads(result.metadata["questions_json"])
            #query = " ".join(meta_data)
            #print("query*******",queries_new,len(queries_new))
            result['qid'].append(np.full(len(scores), qid))
            result['query'].append(np.full(len(scores), query))
            result['rank'].append(np.arange(len(scores)))
            for did, (score, i) in Counter(scores).most_common():
                result['docno'].append(did)
                result['score'].append(score)
                result['iteration'].append(i) 
        
            print("queries_new",queries_new,len(scores))
            # for rank, (docno, final_score) in enumerate(Counter(results).most_common()):
            #     result_builder.extend({
            #         'qid': qid,
            #         'query': query,
            #         'docno': docno,
            #         'score': final_score,
            #         'rank': rank,
            #     })

        # reformulated_queries = pd.DataFrame(reformulated_queries)
        # reformulated_queries.to_csv("rewritten_queries_2.csv",index=False)
        return pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score'],
            'iteration': result['iteration'],
        })