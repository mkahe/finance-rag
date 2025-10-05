from typing import Dict, Any, List
from .embeddings import load_embeddings
from .chroma_store import get_client, collection_name, query

def retrieve(
    dataset_name: str,
    family: str,
    model: str,
    chunk_size: int,
    query_text: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Simple retrieval against a specific collection (model × chunk size).
    """
    client = get_client()
    emb = load_embeddings(family, model)
    qvec = emb.lc.embed_query(query_text)
    cname = collection_name(dataset_name, emb.name, chunk_size)
    res = query(client, cname, query_embeddings=[qvec], n_results=top_k)
    # Flatten single query result
    out = []
    if res and res.get("documents"):
        for i in range(len(res["documents"][0])):
            out.append({
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
    return {"collection": cname, "results": out}
