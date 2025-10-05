"""
Query a specific collection.
Run:
  python examples/query_example.py
"""
from rag_utils.retrieval import retrieve

if __name__ == "__main__":
    res = retrieve(
        dataset_name="financebench_poc",
        family="hf",
        model="BAAI/bge-m3",
        chunk_size=128,
        query_text="What was Acme Corp revenue in FY2023?",
        top_k=3,
    )
    print("Collection:", res["collection"])
    for i, r in enumerate(res["results"], 1):
        print(f"\n#{i} dist={r['distance']:.4f}")
        print(r["text"])
        print(r["metadata"])
