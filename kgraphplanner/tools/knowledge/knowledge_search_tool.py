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


class KnowledgeSearchTool(AbstractTool):
    """Knowledge base search tool using Weaviate vector store.

    Performs semantic similarity search against document chunks stored in
    Weaviate.  Embeddings and search parameters are driven by WeaviateConfig.
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
                "documented knowledge."
            ),
        )
        self._store = None
        self._wv_config = None

    def _get_store(self):
        """Lazy-init the WeaviateVectorStore."""
        if self._store is not None:
            return self._store

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
        client = get_weaviate_client()
        embeddings = get_embeddings(self._wv_config)

        self._store = WeaviateVectorStore(
            client=client,
            index_name=self._wv_config.default_collection,
            text_key="text",
            embedding=embeddings,
        )
        return self._store

    def get_tool_schema(self) -> Type[BaseModel]:
        return KnowledgeSearchInput

    def get_tool_function(self) -> Callable:

        @tool(args_schema=KnowledgeSearchInput)
        async def knowledge_search_tool(query: str, collection: str = "", k: int = 4) -> str:
            """Search the knowledge base for relevant information.

            Args:
                query: The search query to find relevant documents.
                collection: Optional collection name (defaults to configured default).
                k: Number of results to return (default: 4).

            Returns:
                JSON string with search results including text and source metadata.
            """
            logger.info(f"knowledge_search_tool: query='{query}' collection='{collection}' k={k}")

            try:
                store = self._get_store()

                # If a specific collection is requested, create a temporary store
                if collection and self._wv_config and collection != self._wv_config.default_collection:
                    from kgraphplanner.weaviate.client_manager import get_weaviate_client
                    from kgraphplanner.weaviate.embeddings import get_embeddings
                    from langchain_weaviate import WeaviateVectorStore

                    client = get_weaviate_client()
                    embeddings = get_embeddings(self._wv_config)
                    store = WeaviateVectorStore(
                        client=client,
                        index_name=collection,
                        text_key="text",
                        embedding=embeddings,
                    )

                search_type = self._wv_config.search_type if self._wv_config else "similarity"

                # Weaviate client is sync — offload to thread to avoid blocking the loop
                if search_type == "mmr":
                    fetch_k = self._wv_config.search_fetch_k if self._wv_config else 20
                    docs = await asyncio.to_thread(
                        store.max_marginal_relevance_search, query, k=k, fetch_k=fetch_k,
                    )
                else:
                    docs = await asyncio.to_thread(store.similarity_search, query, k=k)

                results = []
                for doc in docs:
                    results.append({
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "collection": doc.metadata.get("collection", ""),
                    })

                logger.info(f"knowledge_search_tool: found {len(results)} results")
                return json.dumps({"results": results, "count": len(results)}, indent=2)

            except Exception as e:
                logger.warning(f"knowledge_search_tool error: {e}")
                return json.dumps({"error": str(e), "results": [], "count": 0})

        return knowledge_search_tool
