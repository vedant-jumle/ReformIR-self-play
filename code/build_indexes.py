import torch
from pyterrier_dr import FlexIndex, TasB, TctColBert, NumpyIndex
import pyterrier as pt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
from pyterrier_dr import FlexIndex, TasB, TctColBert, NumpyIndex
import pyterrier as pt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### to create bm25 index. The bm25 index will created at path like  "/home/user_name/.pyterrier/corpora/msmarco_passage/index/terrier_stemmed"


bm25 = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')
### to build ms marco v2 bm25 terrier index. The bm25 index will created at path like  "/home/user_name/.pyterrier/corpora/msmarco_passage/index/terrier_stemmed". Alternatively the index can be downloaded as an artifact from HF.





### to create bm25 index. The bm25 index will created at path like  "/home/user_name/.pyterrier/corpora/msmarco_passage/index/terrier_stemmed"


bm25 = pt.terrier.Retriever.from_dataset('msmarcov2_passage', 'terrier_stemmed', wmodel='BM25')