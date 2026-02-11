

from typing import Optional
import numpy as np
from collections import Counter, defaultdict
import pyterrier as pt
import pandas as pd
import querygym as qg
import time
from statistics import mean, stdev
import ir_datasets
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import normalize
logger = ir_datasets.log.easy()
import torch
from statistics import mean
import scipy



class RRF(pt.Transformer):

    def __init__(self,
        scorer: pt.Transformer,
        num_results: int = 100,
        cross_enc_budget: int = 7,
        batch_size: Optional[int] = None,
        backfill: bool = True,
        verbose: bool = True,
        bm25_retriever = None, file_name="data/genqr_ensemble/dl19.csv"):

        self.scorer = scorer
        self.bm25_retriever = bm25_retriever
        self.num_results = num_results
        self.reformulation_file = file_name

        self.cross_enc_budget = cross_enc_budget
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.backfill = backfill
        self.verbose = verbose
    #     self.reformulator =  qg.create_reformulator(
    #                 "genqr_ensemble",
    #                 model="qwen2.5:0.5b",
    #                 params = {"variant_ids": [
    # "genqr_ensemble.inst1.v1",
    # "genqr_ensemble.inst2.v1",
    # "genqr_ensemble.inst3.v1"],"parallel": True},
    #                 llm_config={
    #                     "base_url": "http://127.0.0.1:11434/v1",  # Ollama's OpenAI-compatible endpoint
    #                     "api_key": "ollama",
    #                     "temperature": 0.5,
    #                     "max_tokens": 200
    #                 }
    #         )

    def min_max_scaling(self,results):
        results = results.copy()
        results["score"] = results.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return results
    
    def normalie(self,results):
        results = results.copy()
        results["score"] = results.groupby('qid')["score"].transform(lambda x: normalize(np.array(x).reshape(1,-1)))
        return results
 
    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]

    def transform(self, df1: pd.DataFrame ) -> pd.DataFrame:
        
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}
        #print("df1",df1)
        #df1 = self.min_max_scaling(df1)
        #print("df1***",df1)
        #df1 = self.normalie(df1)
        refor_queries = pd.read_csv(self.reformulation_file)

        init_results = df1.groupby(by='qid')
        df1 = dict(iter(df1.groupby(by='qid')))
        qids = df1.keys()

        
        reformed_queries = {"qid":[],"query": []}

        if self.verbose:
            qids = logger.pbar(qids, desc='hybrid retrieval based re-ranking', unit='query')

        times = []
        for qid in qids:
            start = time.time()
            reformulated_queries = {"qid":[],"query": []}
            bm25_queries_super_dict = {}
            # for idy1, (qid_group, qrel_group) in enumerate(qrel_groups):
            #     if  qid_group == qid:
            #         print("qid_group",qid_group,qid,qrel_group)
            #         qrel_docs = qrel_group

            # qrel_docs = qrel_docs[qrel_docs["label"].isin([1])]
            # print("qrel_docsqrel_docsqrel_docs",qrel_docs["label"].values)
            #exit(1)
            query = df1[qid]['query'].iloc[0]
            #print("refor",refor_queries)
            results_saved = refor_queries.loc[refor_queries["qid"].astype(str) == str(qid)]
            #print(df1[qid],qid)
            
            # if len(list(results_saved["query"].values))==0:
            #     print("here****###########")
            #     result_1 = reformulator.reformulate(qg.QueryItem(qid, query), contexts = contexts)
            #     queries_new = result_1.metadata["reformulations"]
            # else:
            queries_new = list(results_saved["query"].values)[0].split("--")# #result_1.metadata["reformulations"]
            # else:
            #     queries_new = list(results_saved["query"].values)[0].split("--") #result_1.metadata["reformulations"] # [ (quer +" " + passage + " " + sentences).lower().strip().strip('"').strip("'") for quer, passage, sentences in zip([query] * 5 , result_1.metadata["keqe_passages"], result_1.metadata["csqe_sentences"])]
#list(results_saved["query"].values)[0].split("--") #
            print("queries_new**",queries_new,len(queries_new),len(list(set(queries_new))))
            #exit(1)
            #queries_new = list(results_saved["query"].values)[0].split(",") #results.split("Intention::")[-1].split(",") # list(results_saved["query"].values)[0].split(",") #
            for idk2, queri in enumerate(queries_new):
                reformulated_queries["qid"].append(str(qid)+""+str(idk2))
                reformulated_queries["query"].append(queri)
            reformed_queries["qid"].append(qid)
            reformed_queries["query"].append("--".join(queries_new))
            scores = {}
            lookup_cross_scores = {}
            df2 = self.bm25_retriever.transform(pd.DataFrame(reformulated_queries))
            #df2 = self.min_max_scaling(df2)
            #df2 = self.normalie(df2)
            
            df2 = dict(iter(df2.groupby(by='qid')))
            print("df2",df2)
            new_queries_docs = []
            new_queries_values = []
            result_ks = []
            for idk2, queri in enumerate(queries_new):
                if str(qid)+""+str(idk2) in df2:
                    values_df = [1/(rank+60) for rank in df2[str(qid)+""+str(idk2)]['rank'].values]
                    result2= dict(zip(df2[str(qid)+""+str(idk2)]['docno'].values, values_df))
                    result_ks.append(result2)
                    new_queries_docs.extend(list(result2.keys()))
                # for docidx,scor in zip(df2[str(qid)+""+str(idk2)]['docno'].values,df2[str(qid)+""+str(idk2)]['score'].values):
                #     if docidx not in bm25_queries_super_dict[idk2]:
                #         bm25_queries_super_dict[idk2][docidx] = scor
                #     if docidx not in bm25_scores:
                #         bm25_scores[docidx] = bm25_lambdas[idk2] * scor
                #     else:   
                #         bm25_scores[docidx] += bm25_lambdas[idk2] * scor

            result1 = dict(zip(df1[qid]['docno'].values, df1[qid]['rank'].values)) # initial results {docno: rel score}

            rr_1 = [1/(rank+60) for rank in result1.values()]
            all_docs = set(list(result1.keys()) + new_queries_docs)
            rr_result1 = dict(zip(result1.keys(), rr_1))

            
            #rr_result2 = dict(rr_2)
            hybrid_scores = {}
            for docno in all_docs:
                if docno not in hybrid_scores:
                    hybrid_scores[docno] = rr_result1.get(docno, 0)
                for rr_result2 in result_ks:
                    hybrid_scores[docno]+= rr_result2.get(docno, 0)

            

        

            res_map = [Counter(hybrid_scores)]

            # result1 = Counter(dict(zip(df1[qid].docno, df1[qid].rank))) # initial results {docno: rel score}
            # candidates = list(hybrid_scores.keys())
            # print("candidates",len(candidates))
            iteration=0  
            num_batch = 0 



            while len(scores) < self.num_results and num_batch < self.cross_enc_budget:
                this_res = res_map[0]

                size = min(self.batch_size, self.num_results - len(scores)) 

                batch = this_res.most_common(size)

                batch = pd.DataFrame(batch, columns=['docno', 'score'])
                batch['qid'] = qid
                #batch['qid'] = [qid[0]] * len(batch)
                batch['query'] = query
                    

                # go score the batch of document with the re-ranker
                batch = self.scorer(batch)
                num_batch+=1

                scores.update({k: (s, iteration) for k, s in zip(batch.docno, batch.score)})
                self._drop_docnos_from_counters(batch.docno, res_map)


                iteration+=1   
                times.append(time.time()-start)
            
            result['qid'].append(np.full(len(scores), qid))
            result['query'].append(np.full(len(scores), query))
            result['rank'].append(np.arange(len(scores)))
            for did, (score, i) in Counter(scores).most_common():
                result['docno'].append(did)
                result['score'].append(score)
                result['iteration'].append(i)   
            
            # Backfill unscored items
            print("mean",mean(times),stdev(times))
            if self.backfill and len(scores) < self.num_results:
                last_score = result['score'][-1] if result['score'] else 0.
                count = min(self.num_results - len(scores), len(res_map[0]))
                result['qid'].append(np.full(count, qid))
                result['query'].append(np.full(count, query))
                result['rank'].append(np.arange(len(scores), len(scores) + count))
                for i, (did, score) in enumerate(res_map[0].most_common()):
                    if i >= count:
                        break
                    result['docno'].append(did)
                    result['score'].append(last_score - 1 - i)
                    result['iteration'].append(-1)
    

        print("mean",mean(times),stdev(times))
        return pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score'],
            'iteration': result['iteration'],
        })


