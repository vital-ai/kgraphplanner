"""
Factory for LangChain embedding models, driven by WeaviateConfig.
"""

from kgraphplanner.config.agent_config import WeaviateConfig


def get_embeddings(config: WeaviateConfig):
    """Return a LangChain Embeddings instance based on config."""
    provider = config.embedding_provider.lower()
    model = config.embedding_model

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)

    if provider == "cohere":
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model=model)

    raise ValueError(f"Unknown embedding provider: {provider}")
