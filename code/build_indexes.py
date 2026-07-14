import os
import pyterrier as pt

# Build BM25 index from scratch with reverse docno lookup enabled.
# Required by RM3 query expansion inside ReformIR.
# Output: $PYTERRIER_HOME/corpora/msmarco_passage/index/terrier_stemmed_reverse/

index_path = os.path.join(
    os.environ["PYTERRIER_HOME"],
    "corpora", "msmarco_passage", "index", "terrier_stemmed_reverse"
)

if not os.path.exists(os.path.join(index_path, "data.properties")):
    print(f"Building index at {index_path} ...")
    dataset = pt.get_dataset("irds:msmarco-passage")
    indexer = pt.IterDictIndexer(
        index_path,
        blocks=True,
        fields=["text"],
        meta={"docno": 26, "text": 4096},
        meta_reverse=["docno"],
    )
    indexref = indexer.index(dataset.get_corpus_iter())
    print(f"Done: {indexref}")
else:
    print(f"Index already exists at {index_path}, skipping build.")
    indexref = pt.IndexRef.of(index_path)

print("Index ref:", indexref)
