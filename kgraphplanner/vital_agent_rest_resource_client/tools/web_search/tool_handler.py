from typing import Dict
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_handler import ToolHandler
from kgraphplanner.vital_agent_rest_resource_client.tools.tool_parameters import ToolParameters
from kgraphplanner.vital_agent_rest_resource_client.tools.web_search.models import (
    WebSearchInput, WebSearchOutput, WebSearchResult
)


class WebSearchToolHandler(ToolHandler):
    """Handler for web search tool operations."""
    
    def parse_web_search_response(self, tool_parameters: ToolParameters, response_json: Dict) -> WebSearchOutput:
        """
        Parse web search response from the API.
        
        Args:
            response_json: The JSON response from the web search API
            
        Returns:
            WebSearchOutput: Parsed web search results
        """
        # Extract results from tool_output structure or fallback to direct results
        if 'tool_output' in response_json:
            tool_output = response_json['tool_output']
            search_results = tool_output.get('results', [])
            total_results = tool_output.get('total_results', 0)
            knowledge_graph_data = tool_output.get('knowledge_graph')
            related_questions_data = tool_output.get('related_questions', [])
            search_information = tool_output.get('search_information')
        else:
            search_results = response_json.get('results', [])
            total_results = response_json.get('total_results', 0)
            knowledge_graph_data = response_json.get('knowledge_graph')
            related_questions_data = response_json.get('related_questions', [])
            search_information = response_json.get('search_information')
        
        # Parse search results
        web_search_results = []
        for result in search_results:
            web_search_result = WebSearchResult(
                title=result.get('title', ''),
                link=result.get('link', result.get('url', '')),
                snippet=result.get('snippet', ''),
                position=result.get('position', 0),
                displayed_link=result.get('displayed_link', result.get('display_url', '')),
                thumbnail=result.get('thumbnail'),
                source=result.get('source'),
                date=result.get('date'),
                result_type=result.get('result_type', 'organic'),
                # Shopping/local result fields
                price=result.get('price'),
                rating=result.get('rating'),
                reviews=result.get('reviews'),
                address=result.get('address'),
                phone=result.get('phone'),
                # Recipe result fields
                ingredients=result.get('ingredients'),
                total_time=result.get('total_time')
            )
            web_search_results.append(web_search_result)
        
        # Parse knowledge graph
        knowledge_graph = None
        if knowledge_graph_data:
            from kgraphplanner.vital_agent_rest_resource_client.tools.web_search.models import KnowledgeGraph
            knowledge_graph = KnowledgeGraph(
                title=knowledge_graph_data.get('title'),
                type=knowledge_graph_data.get('type'),
                description=knowledge_graph_data.get('description'),
                source=knowledge_graph_data.get('source'),
                header_images=knowledge_graph_data.get('header_images')
            )
        
        # Parse related questions
        related_questions = []
        if related_questions_data:
            from kgraphplanner.vital_agent_rest_resource_client.tools.web_search.models import RelatedQuestion
            for question_data in related_questions_data:
                related_question = RelatedQuestion(
                    question=question_data.get('question', ''),
                    snippet=question_data.get('snippet'),
                    title=question_data.get('title'),
                    link=question_data.get('link')
                )
                related_questions.append(related_question)
        
        # Extract query from tool parameters
        query = getattr(tool_parameters, 'search_query', 'Unknown query')
        
        return WebSearchOutput(
            tool="google_web_search_tool",
            query=query,
            results=web_search_results,
            total_results=total_results if total_results > 0 else len(web_search_results),
            knowledge_graph=knowledge_graph,
            related_questions=related_questions if related_questions else None,
            search_information=search_information
        )
    
    def handle_response(self, tool_parameters: ToolParameters, response_json: dict) -> WebSearchOutput:
        """
        Handle web search tool response and return parsed results.
        
        Args:
            tool_parameters: The tool parameters for the request
            response_json: The JSON response from the API
            
        Returns:
            WebSearchOutput: Parsed web search results
        """
        return self.parse_web_search_response(tool_parameters, response_json)
    
    def handle_tool_request(self, tool_parameters: ToolParameters, response_json: dict) -> WebSearchOutput:
        """
        Handle web search tool request and return parsed results.
        
        Args:
            tool_parameters: The tool parameters for the request
            response_json: The JSON response from the API
            
        Returns:
            WebSearchOutput: Parsed web search results
        """
        return self.handle_response(tool_parameters, response_json)
