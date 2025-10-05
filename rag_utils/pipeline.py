from typing import List, Dict, Any, Sequence
from .embeddings import load_embeddings, EmbeddingBundle
from .chunking import chunk_texts_llamaindex, to_langchain_docs
from .chroma_store import get_client, collection_name, upsert_texts
from .config import settings

def prepare_collections(
    dataset_name: str,
    raw_texts: Sequence[Dict[str, Any]],
    families_and_models: Sequence[Dict[str, str]],
    chunk_sizes: Sequence[int] = (128, 256, 512),
    chunk_overlap: int = 32,
) -> List[str]:
    """
    Builds Chroma collections for each (embedding family/model × chunk size).
    raw_texts: list of {"text": ..., "metadata": {...}}
    families_and_models: e.g. [{"family":"hf","model":"BAAI/bge-m3"}, {"family":"ollama","model":"nomic-embed-text"}]
    Returns the created/updated collection names.
    """
    client = get_client(settings.chroma_persist_dir)
    created = []

    for fm in families_and_models:
        emb: EmbeddingBundle = load_embeddings(fm["family"], fm.get("model"))
        # Pre-embed chunks per chunk size and upsert
        for csz in chunk_sizes:
            chunks = chunk_texts_llamaindex(raw_texts, chunk_size=csz, chunk_overlap=chunk_overlap)
            texts = [c["text"] for c in chunks]
            metas = [c["metadata"] for c in chunks]
            # LangChain interface for batch embedding
            vectors = emb.lc.embed_documents(texts)
            cname = collection_name(dataset_name, emb.name, csz)
            upsert_texts(client, cname, texts, metadatas=metas, embeddings=vectors)
            created.append(cname)

    return created
