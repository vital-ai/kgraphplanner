import logging
import json
import asyncio
from typing import Callable, Type

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from kgraphplanner.tool_manager.tool_inf import AbstractTool

logger = logging.getLogger(__name__)

TOOL_NAME = "knowledge_search_tool"


class KnowledgeSearchInput(BaseModel):
    """Input schema for the knowledge search tool."""
    query: str = Field(
        description="The search query to find relevant knowledge base documents"
    )
    collection: str = Field(
        default="",
        description="Optional collection name to search in (defaults to WeaviateConfig.default_collection)"
    )
    k: int = Field(
        default=4,
        description="Number of results to return (default: 4)"
    )
    search_type: str = Field(
        default="",
        description=(
            "Search strategy: 'similarity' (pure vector), 'mmr' (vector + diversity), "
            "'hybrid' (BM25 + vector combined), or 'keyword' (pure BM25 text search). "
            "Defaults to the configured search type if empty."
        )
    )


class KnowledgeSearchTool(AbstractTool):
    """Knowledge base search tool using Weaviate vector store.

    Supports four search strategies:
      - similarity: pure cosine vector search
      - mmr: vector search with diversity re-ranking
      - hybrid: BM25 + vector combined (alpha=0.5)
      - keyword: pure BM25 text search (no embeddings needed)

    Search type defaults to WeaviateConfig.search_type but can be overridden
    per-call via the search_type parameter.
    """

    def __init__(self, config=None, tool_manager=None):
        super().__init__(
            config=config or {},
            tool_manager=tool_manager,
            name=TOOL_NAME,
            description=(
                "Search the knowledge base for relevant information. "
                "Use this tool when you need to look up product details, "
                "company information, policies, procedures, or any other "
                "documented knowledge. Supports vector, hybrid, and keyword search."
            ),
        )
        self._store = None
        self._client = None
        self._wv_config = None

    def _ensure_init(self):
        """Lazy-init the WeaviateVectorStore and native client."""
        if self._store is not None:
            return

        from kgraphplanner.weaviate.client_manager import init_weaviate, get_weaviate_client
        from kgraphplanner.weaviate.embeddings import get_embeddings
        from kgraphplanner.config.agent_config import AgentConfig
        from langchain_weaviate import WeaviateVectorStore

        agent_config = AgentConfig.from_env()
        self._wv_config = agent_config.weaviate

        # Auto-acquire JWT from Keycloak if bearer auth is configured
        jwt_token = None
        if self._wv_config.auth_mode == "bearer":
            from kgraphplanner.weaviate.auth import get_weaviate_jwt
            jwt_token, err = get_weaviate_jwt()
            if err:
                logger.warning(f"JWT auth: {err}")

        init_weaviate(config=self._wv_config, jwt=jwt_token)
        self._client = get_weaviate_client()
        embeddings = get_embeddings(self._wv_config)

        prefix = self._wv_config.collection_prefix
        default_col = (prefix + "xxx" + self._wv_config.default_collection) if prefix else self._wv_config.default_collection
        self._store = WeaviateVectorStore(
            client=self._client,
            index_name=default_col,
            text_key="text",
            embedding=embeddings,
        )

    def _prefixed(self, collection: str) -> str:
        """Apply environment prefix to a collection name."""
        prefix = self._wv_config.collection_prefix
        if prefix and not collection.startswith(prefix + "xxx"):
            return prefix + "xxx" + collection
        return collection

    def _get_store(self, collection: str = ""):
        """Return a WeaviateVectorStore for the given collection."""
        self._ensure_init()
        if collection and collection != self._wv_config.default_collection:
            from kgraphplanner.weaviate.embeddings import get_embeddings
            from langchain_weaviate import WeaviateVectorStore
            embeddings = get_embeddings(self._wv_config)
            return WeaviateVectorStore(
                client=self._client,
                index_name=self._prefixed(collection),
                text_key="text",
                embedding=embeddings,
            )
        return self._store

    def get_tool_schema(self) -> Type[BaseModel]:
        return KnowledgeSearchInput

    def get_tool_function(self) -> Callable:

        @tool(args_schema=KnowledgeSearchInput)
        async def knowledge_search_tool(
            query: str, collection: str = "", k: int = 4, search_type: str = ""
        ) -> str:
            """Search the knowledge base for relevant information.

            Args:
                query: The search query to find relevant documents.
                collection: Optional collection name (defaults to configured default).
                k: Number of results to return (default: 4).
                search_type: 'similarity', 'mmr', 'hybrid', or 'keyword' (default: config value).

            Returns:
                JSON string with search results including text and source metadata.
            """
            effective_type = search_type or (self._wv_config.search_type if self._wv_config else "similarity")
            logger.info(
                f"knowledge_search_tool: query='{query}' collection='{collection}' "
                f"k={k} search_type='{effective_type}'"
            )

            try:
                self._ensure_init()
                store = self._get_store(collection)
                col_name = collection or self._wv_config.default_collection

                # Weaviate client is sync — offload to thread to avoid blocking the loop
                if effective_type == "mmr":
                    fetch_k = self._wv_config.search_fetch_k if self._wv_config else 20
                    docs = await asyncio.to_thread(
                        store.max_marginal_relevance_search, query, k=k, fetch_k=fetch_k,
                    )
                elif effective_type == "hybrid":
                    docs = await asyncio.to_thread(
                        store.similarity_search, query, k=k, alpha=0.5,
                    )
                elif effective_type == "keyword":
                    # Pure BM25 via native Weaviate client (no embeddings)
                    def _bm25():
                        col = self._client.collections.get(col_name)
                        results = col.query.bm25(
                            query=query,
                            limit=k,
                            query_properties=["text"],
                        )
                        return results.objects

                    objects = await asyncio.to_thread(_bm25)
                    results = []
                    for obj in objects:
                        results.append({
                            "text": obj.properties.get("text", ""),
                            "source": obj.properties.get("source", "unknown"),
                            "collection": obj.properties.get("collection", ""),
                        })
                    logger.info(f"knowledge_search_tool: found {len(results)} results (keyword/BM25)")
                    return json.dumps(
                        {"results": results, "count": len(results), "search_type": "keyword"}, indent=2
                    )
                else:
                    # Default: similarity
                    docs = await asyncio.to_thread(store.similarity_search, query, k=k)

                results = []
                for doc in docs:
                    results.append({
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "collection": doc.metadata.get("collection", ""),
                    })

                logger.info(f"knowledge_search_tool: found {len(results)} results ({effective_type})")
                return json.dumps(
                    {"results": results, "count": len(results), "search_type": effective_type}, indent=2
                )

            except Exception as e:
                logger.warning(f"knowledge_search_tool error: {e}")
                return json.dumps({"error": str(e), "results": [], "count": 0})

        return knowledge_search_tool
