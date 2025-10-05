from typing import List, Dict, Any, Iterable
from llama_index.core.node_parser import SentenceSplitter
from langchain_core.documents import Document as LCDocument

def chunk_texts_llamaindex(
    texts: Iterable[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    texts: iterable of {"text": str, "metadata": {...}}
    returns list of {"text": str, "metadata": {...}}
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for item in texts:
        chunks = splitter.split_text(item["text"])
        for i, ch in enumerate(chunks):
            meta = dict(item.get("metadata", {}))
            meta.update({"chunk_index": i, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
            out.append({"text": ch, "metadata": meta})
    return out

def to_langchain_docs(chunks: List[Dict[str, Any]]) -> List[LCDocument]:
    return [LCDocument(page_content=c["text"], metadata=c.get("metadata", {})) for c in chunks]
