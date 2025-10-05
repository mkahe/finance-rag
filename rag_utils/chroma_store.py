from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings as ChromaSettings
import os
import uuid

from .config import settings

def collection_name(dataset: str, embed_name: str, chunk_size: int) -> str:
    safe = embed_name.replace("/", "_").replace(":", "_")
    d = dataset.replace("/", "_")
    return f"{d}__{safe}__c{chunk_size}"

def get_client(persist_dir: Optional[str] = None):
    persist_dir = persist_dir or settings.chroma_persist_dir
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.Client(ChromaSettings(is_persistent=True, allow_reset=True, persist_directory=persist_dir))

def get_or_create_collection(
    client,
    name: str,
    metadata: Optional[dict] = None,
):
    # No embedding function here because we’ll precompute vectors via LC/LI; 
    # That gives you freedom to swap embed pipelines.
    return client.get_or_create_collection(name=name, metadata=metadata)

def upsert_texts(
    client,
    collection_name: str,
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    embeddings: Optional[List[List[float]]] = None,
    ids: Optional[List[str]] = None,
):
    col = get_or_create_collection(client, collection_name)
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    col.upsert(documents=texts, metadatas=metadatas, embeddings=embeddings, ids=ids)
    return col

def query(
    client,
    collection_name: str,
    query_embeddings: List[List[float]],
    n_results: int = 5,
):
    col = get_or_create_collection(client, collection_name)
    return col.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
