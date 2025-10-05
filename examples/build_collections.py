"""
Build multiple Chroma collections from raw texts using multiple embedding models and chunk sizes.
Run:
  python examples/build_collections.py
"""
from rag_utils.pipeline import prepare_collections

if __name__ == "__main__":
    # Replace with your own loader — keeping it simple here:
    raw_texts = [
        {"text": "Acme Corp reported revenue of $12.3B in FY2023.", "metadata": {"source": "10-K 2023", "ticker": "ACME"}},
        {"text": "Operating margin improved to 18% due to cost controls.", "metadata": {"source": "10-K 2023", "ticker": "ACME"}},
    ]

    families_and_models = [
        # {"family": "hf", "model": "BAAI/bge-m3"},
        # {"family": "hf", "model": "nomic-ai/nomic-embed-text-v1"},
        # If you run Ollama locally with an embedding model:
        {"family": "ollama", "model": "nomic-embed-text"},
        # If you use OpenAI:
        # {"family": "openai", "model": "text-embedding-3-small"},
    ]

    created = prepare_collections(
        dataset_name="financebench_poc",
        raw_texts=raw_texts,
        families_and_models=families_and_models,
        chunk_sizes=(128, 256, 512),
        chunk_overlap=32
    )

    print("Created/updated collections:")
    for c in created:
        print(" -", c)
