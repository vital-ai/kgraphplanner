from typing import Dict
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_handler import ToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.serper_web_search.models import (
    SerperWebSearchInput, SerperWebSearchOutput, SerperSearchResult,
    SerperKnowledgeGraph, SerperRelatedQuestion
)


class SerperWebSearchToolHandler(ToolHandler):
    """Handler for Serper web search tool operations."""

    def parse_serper_search_response(self, tool_parameters: ToolParameters, response_json: Dict) -> SerperWebSearchOutput:
        """
        Parse Serper web search response from the API.

        Args:
            tool_parameters: The tool parameters for the request
            response_json: The JSON response from the Serper web search API

        Returns:
            SerperWebSearchOutput: Parsed search results
        """
        if 'tool_output' in response_json:
            tool_output = response_json['tool_output']
        else:
            tool_output = response_json

        search_results_data = tool_output.get('results', [])
        total_results = tool_output.get('total_results', 0)
        knowledge_graph_data = tool_output.get('knowledge_graph')
        related_searches_data = tool_output.get('related_searches')
        people_also_ask_data = tool_output.get('people_also_ask', [])
        search_information = tool_output.get('search_information')
        api_error = tool_output.get('api_error')
        api_status_code = tool_output.get('api_status_code')

        # Parse search results
        search_results = []
        for result in search_results_data:
            search_result = SerperSearchResult(
                title=result.get('title', ''),
                link=result.get('link', result.get('url', '')),
                snippet=result.get('snippet'),
                position=result.get('position'),
                displayed_link=result.get('displayed_link', result.get('display_url')),
                source=result.get('source'),
                date=result.get('date'),
                result_type=result.get('result_type', 'organic'),
                price=result.get('price'),
                rating=result.get('rating'),
                rating_count=result.get('rating_count'),
                image_url=result.get('image_url'),
                thumbnail=result.get('thumbnail'),
                address=result.get('address'),
                phone=result.get('phone'),
                cid=result.get('cid'),
            )
            search_results.append(search_result)

        # Parse knowledge graph
        knowledge_graph = None
        if knowledge_graph_data:
            knowledge_graph = SerperKnowledgeGraph(
                title=knowledge_graph_data.get('title'),
                type=knowledge_graph_data.get('type'),
                description=knowledge_graph_data.get('description'),
                attributes=knowledge_graph_data.get('attributes'),
                source=knowledge_graph_data.get('source'),
                header_images=knowledge_graph_data.get('header_images'),
            )

        # Parse people also ask
        people_also_ask = []
        if people_also_ask_data:
            for question_data in people_also_ask_data:
                related_question = SerperRelatedQuestion(
                    question=question_data.get('question', ''),
                    snippet=question_data.get('snippet'),
                    title=question_data.get('title'),
                    link=question_data.get('link'),
                )
                people_also_ask.append(related_question)

        # Extract query from tool parameters
        query = getattr(tool_parameters, 'search_query', 'Unknown query')

        return SerperWebSearchOutput(
            tool="serper_web_search_tool",
            query=query,
            results=search_results,
            total_results=total_results if total_results > 0 else len(search_results),
            knowledge_graph=knowledge_graph,
            related_searches=related_searches_data,
            people_also_ask=people_also_ask if people_also_ask else None,
            search_information=search_information,
            api_error=api_error,
            api_status_code=api_status_code,
        )

    def handle_response(self, tool_parameters: ToolParameters, response_json: dict) -> SerperWebSearchOutput:
        """
        Handle Serper web search tool response and return parsed results.

        Args:
            tool_parameters: The tool parameters for the request
            response_json: The JSON response from the API

        Returns:
            SerperWebSearchOutput: Parsed search results
        """
        return self.parse_serper_search_response(tool_parameters, response_json)

    def handle_tool_request(self, tool_parameters: ToolParameters, response_json: dict) -> SerperWebSearchOutput:
        """
        Handle Serper web search tool request and return parsed results.

        Args:
            tool_parameters: The tool parameters for the request
            response_json: The JSON response from the API

        Returns:
            SerperWebSearchOutput: Parsed search results
        """
        return self.handle_response(tool_parameters, response_json)
