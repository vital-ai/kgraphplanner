import logging
import json
import asyncio
from typing import Callable, Type

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from kgraphplanner.tool_manager.tool_inf import AbstractTool

logger = logging.getLogger(__name__)

TOOL_NAME = "knowledge_get_document_tool"
SOURCE_DOCS_COLLECTION = "SourceDocuments"


class KnowledgeGetDocumentInput(BaseModel):
    """Input schema for retrieving a full document by source name or keyword search."""
    source: str = Field(
        default="",
        description="Exact source filename to retrieve (e.g. 'guide-chapter-1.md'). If empty, uses query for keyword search."
    )
    query: str = Field(
        default="",
        description="Keyword search query to find documents by content (BM25 text search). Used when source is not specified."
    )
    k: int = Field(
        default=3,
        description="Number of documents to return when using keyword search (default: 3)"
    )


class KnowledgeGetDocumentTool(AbstractTool):
    """Retrieve full documents from Weaviate by source filename or keyword search.

    Supports two modes:
      - Exact lookup: filter by source filename (no embeddings needed).
      - Keyword search: BM25 text search across full document content.
    """

    def __init__(self, config=None, tool_manager=None):
        super().__init__(
            config=config or {},
            tool_manager=tool_manager,
            name=TOOL_NAME,
            description=(
                "Retrieve full documents from the knowledge base. "
                "Provide 'source' for exact filename lookup, or 'query' for "
                "keyword search across document content. Returns complete "
                "document text, not just snippets."
            ),
        )
        self._client = None
        self._wv_config = None

    def _ensure_client(self):
        """Lazy-init the Weaviate client."""
        if self._client is not None:
            return self._client

        from kgraphplanner.weaviate.client_manager import init_weaviate, get_weaviate_client
        from kgraphplanner.config.agent_config import AgentConfig

        agent_config = AgentConfig.from_env()
        self._wv_config = agent_config.weaviate

        jwt_token = None
        if self._wv_config.auth_mode == "bearer":
            from kgraphplanner.weaviate.auth import get_weaviate_jwt
            jwt_token, err = get_weaviate_jwt()
            if err:
                logger.warning(f"JWT auth: {err}")

        init_weaviate(config=self._wv_config, jwt=jwt_token)
        self._client = get_weaviate_client()
        return self._client

    def get_tool_schema(self) -> Type[BaseModel]:
        return KnowledgeGetDocumentInput

    def get_tool_function(self) -> Callable:

        @tool(args_schema=KnowledgeGetDocumentInput)
        async def knowledge_get_document_tool(source: str = "", query: str = "", k: int = 3) -> str:
            """Retrieve full documents by source filename or keyword search.

            Args:
                source: Exact source filename to retrieve.
                query: Keyword search query (BM25) across document content.
                k: Number of results for keyword search (default: 3).

            Returns:
                JSON string with document(s) including full text and metadata.
            """
            logger.info(f"knowledge_get_document_tool: source='{source}' query='{query}' k={k}")

            if not source and not query:
                return json.dumps({"error": "Must provide either 'source' or 'query'", "results": []})

            try:
                client = self._ensure_client()
                from weaviate.classes.query import Filter

                prefix = self._wv_config.collection_prefix or ""
                prefixed_col = (prefix + "xxx" + SOURCE_DOCS_COLLECTION) if prefix else SOURCE_DOCS_COLLECTION
                col = client.collections.get(prefixed_col)

                if source:
                    # Exact lookup by source filename
                    def _fetch_by_source():
                        return col.query.fetch_objects(
                            filters=Filter.by_property("source").equal(source),
                            limit=1,
                        )

                    results = await asyncio.to_thread(_fetch_by_source)

                    if not results.objects:
                        logger.info(f"knowledge_get_document_tool: no document found for source='{source}'")
                        return json.dumps({"results": [], "count": 0, "mode": "source_lookup",
                                           "error": f"No document found with source='{source}'"})

                    docs_out = [self._format_obj(results.objects[0])]

                else:
                    # BM25 keyword search across document text
                    def _bm25_search():
                        return col.query.bm25(
                            query=query,
                            limit=k,
                            query_properties=["text", "source"],
                        )

                    results = await asyncio.to_thread(_bm25_search)

                    if not results.objects:
                        logger.info(f"knowledge_get_document_tool: no results for query='{query}'")
                        return json.dumps({"results": [], "count": 0, "mode": "keyword_search"})

                    docs_out = [self._format_obj(obj) for obj in results.objects]

                logger.info(f"knowledge_get_document_tool: returning {len(docs_out)} document(s)")
                mode = "source_lookup" if source else "keyword_search"
                return json.dumps({"results": docs_out, "count": len(docs_out), "mode": mode}, indent=2)

            except Exception as e:
                logger.warning(f"knowledge_get_document_tool error: {e}")
                return json.dumps({"error": str(e), "results": [], "count": 0})

        return knowledge_get_document_tool

    @staticmethod
    def _format_obj(obj) -> dict:
        """Format a Weaviate object into a result dict."""
        props = obj.properties
        return {
            "source": props.get("source", ""),
            "path": props.get("path", ""),
            "collection": props.get("collection", ""),
            "chunk_count": props.get("chunk_count", 0),
            "text": props.get("text", ""),
        }
