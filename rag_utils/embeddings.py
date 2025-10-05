from typing import Optional, Literal, Dict, Any
from langchain_community.embeddings import (
    HuggingFaceEmbeddings, 
    OllamaEmbeddings, 
)
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings  # separate package in langchain-openai, but here we use openai==1.x + langchain wrapper
from llama_index.embeddings.openai import OpenAIEmbedding as LIOpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding as LIHuggingFaceEmbedding
try:
    from llama_index.embeddings.ollama import OllamaEmbedding as LIOllamaEmbedding
except Exception:
    LIOllamaEmbedding = None  # optional

from dataclasses import dataclass

# A small adapter so you can get both LC and LlamaIndex embedding objects from one config.
@dataclass
class EmbeddingBundle:
    name: str
    lc: Any          # LangChain embeddings object
    li: Any          # LlamaIndex embeddings object

def _hf_kwargs_for(model_name: str) -> Dict[str, Any]:
    # sensible defaults for sentence-transformers models
    return dict(
        model_name=model_name,
        # normalize_embeddings=True gives better cosine behavior for many ST models
        encode_kwargs={"normalize_embeddings": True},
    )

def load_embeddings(
    family: Literal["openai", "hf", "ollama"],
    model: Optional[str] = None,
    **kwargs,
) -> EmbeddingBundle:
    """
    Returns a pair of (LangChain embeddings, LlamaIndex embeddings) for the given family/model.
    Examples:
      - family="openai", model="text-embedding-3-large"
      - family="hf", model="sentence-transformers/all-MiniLM-L6-v2"
      - family="hf", model="BAAI/bge-m3"
      - family="hf", model="nomic-ai/nomic-embed-text-v1"
      - family="ollama", model="nomic-embed-text" (served by Ollama)
    """
    if family == "openai":
        model = model or "text-embedding-3-small"
        lc = LCOpenAIEmbeddings(model=model, **kwargs)
        li = LIOpenAIEmbedding(model=model)
        return EmbeddingBundle(name=f"openai__{model}", lc=lc, li=li)

    # if family == "hf":
    #     if not model:
    #         raise ValueError("For family='hf', you must provide a HuggingFace model name.")
    #     hf_args = _hf_kwargs_for(model)
    #     hf_args.update(kwargs or {})
    #     lc = HuggingFaceEmbeddings(**hf_args)
    #     li = LIHuggingFaceEmbedding(model_name=model, embed_batch_size=hf_args.get("encode_kwargs", {}).get("batch_size", 32))
    #     return EmbeddingBundle(name=f"hf__{model.replace('/', '_')}", lc=lc, li=li)

    if family == "ollama":
        model = model or "nomic-embed-text"
        lc = OllamaEmbeddings(model=model, **kwargs)
        if LIOllamaEmbedding is None:
            raise RuntimeError("LlamaIndex OllamaEmbedding not available in your llama-index version.")
        li = LIOllamaEmbedding(model_name=model)
        return EmbeddingBundle(name=f"ollama__{model}", lc=lc, li=li)

    raise ValueError(f"Unknown embedding family: {family}")
