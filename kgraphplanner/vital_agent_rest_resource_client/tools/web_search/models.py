from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class WebSearchResult(BaseModel):
    title: str = Field(..., description="Title of the search result")
    link: str = Field(..., description="URL of the search result")
    snippet: Optional[str] = Field(None, description="Snippet/description of the search result")
    position: Optional[int] = Field(None, description="Position in search results")
    displayed_link: Optional[str] = Field(None, description="Displayed link in search results")
    thumbnail: Optional[str] = Field(None, description="Thumbnail image URL if available")
    source: Optional[str] = Field(None, description="Source domain of the result")
    date: Optional[str] = Field(None, description="Publication date if available")
    result_type: str = Field("organic", description="Type of result (organic, news, shopping, recipe, local, etc.)")
    
    # Additional fields for different result types
    price: Optional[str] = Field(None, description="Price for shopping results")
    rating: Optional[float] = Field(None, description="Rating for local/shopping results")
    reviews: Optional[int] = Field(None, description="Number of reviews")
    address: Optional[str] = Field(None, description="Address for local results")
    phone: Optional[str] = Field(None, description="Phone number for local results")
    ingredients: Optional[List[str]] = Field(None, description="Ingredients for recipe results")
    total_time: Optional[str] = Field(None, description="Total time for recipe results")


class KnowledgeGraph(BaseModel):
    title: Optional[str] = Field(None, description="Knowledge graph title")
    type: Optional[str] = Field(None, description="Knowledge graph type")
    description: Optional[str] = Field(None, description="Knowledge graph description")
    source: Optional[dict] = Field(None, description="Source information")
    header_images: Optional[List[dict]] = Field(None, description="Header images")


class RelatedQuestion(BaseModel):
    question: str = Field(..., description="Related question")
    snippet: Optional[str] = Field(None, description="Answer snippet")
    title: Optional[str] = Field(None, description="Source title")
    link: Optional[str] = Field(None, description="Source link")


class WebSearchInput(BaseModel):
    """Input model for Web Search tool"""
    search_query: str = Field(..., description="Search query string", min_length=1)
    num_results: Optional[int] = Field(10, description="Number of results to return", ge=1, le=100)
    location: Optional[str] = Field(None, description="Location for localized search results (e.g., 'Austin,Texas')")
    language: Optional[str] = Field(None, description="Google UI language (e.g., 'en' for English)")
    country: Optional[str] = Field(None, description="Google country code (e.g., 'us')")
    device: Optional[Literal["desktop", "mobile", "tablet"]] = Field("desktop", description="Device type for search")
    safe_search: Optional[Literal["active", "off"]] = Field(None, description="Safe search setting")
    search_type: Optional[Literal["search", "news", "images", "shopping"]] = Field("search", description="Type of search to perform")
    time_period: Optional[Literal["hour", "day", "week", "month", "year"]] = Field(None, description="Time period filter for results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "search_query": "Apple Cider recipes",
                "num_results": 10,
                "location": "Austin,Texas",
                "language": "en",
                "device": "desktop"
            }
        }
    }


class WebSearchOutput(BaseModel):
    """Output model for Web Search tool"""
    tool: Literal["google_web_search_tool"] = Field(..., description="Tool identifier")
    query: str = Field(..., description="The search query that was executed")
    results: List[WebSearchResult] = Field(default_factory=list, description="Web search results")
    total_results: Optional[int] = Field(None, description="Total number of results found")
    knowledge_graph: Optional[KnowledgeGraph] = Field(None, description="Knowledge graph information")
    related_questions: Optional[List[RelatedQuestion]] = Field(None, description="Related questions")
    search_information: Optional[dict] = Field(None, description="Search metadata and information")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tool": "google_web_search_tool",
                "query": "Apple Cider recipes",
                "results": [
                    {
                        "title": "Best Apple Cider Recipe",
                        "link": "https://example.com/apple-cider-recipe",
                        "snippet": "Learn how to make the perfect apple cider at home with this easy recipe...",
                        "position": 1,
                        "displayed_link": "example.com",
                        "result_type": "organic"
                    }
                ],
                "total_results": 1000,
                "knowledge_graph": {
                    "title": "Apple Cider",
                    "type": "Drink",
                    "description": "Apple cider is a beverage made from apples..."
                }
            }
        }
    }
