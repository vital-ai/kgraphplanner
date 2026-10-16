from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal, Dict, Any


class SerperSearchResult(BaseModel):
    title: str = Field(..., description="Title of the search result")
    link: str = Field("", description="URL of the search result")
    snippet: Optional[str] = Field(None, description="Snippet/description of the search result")
    position: Optional[int] = Field(None, description="Position in search results")
    displayed_link: Optional[str] = Field(None, description="Displayed link in search results")
    source: Optional[str] = Field(None, description="Source domain of the result")
    date: Optional[str] = Field(None, description="Publication date if available")
    result_type: str = Field("organic", description="Type of result (organic, news, shopping, image, place)")

    # Shopping fields
    price: Optional[str] = Field(None, description="Price for shopping results")
    rating: Optional[float] = Field(None, description="Rating for shopping/places results")
    rating_count: Optional[int] = Field(None, description="Number of ratings/reviews")

    # Image fields
    image_url: Optional[str] = Field(None, description="Image URL for image results")
    thumbnail: Optional[str] = Field(None, description="Thumbnail image URL if available")
    sitelinks: Optional[List[dict]] = Field(None, description="Sitelinks if present")

    # Places fields
    address: Optional[str] = Field(None, description="Address for place results")
    phone_number: Optional[str] = Field(None, description="Phone number for place results")
    phone: Optional[str] = Field(
        None, exclude=True,  # not serialised: phone_number carries it
        description="Deprecated: same value as phone_number, kept for existing readers")
    website: Optional[str] = Field(None, description="Business website for place results")
    category: Optional[str] = Field(None, description="Business category for place results (e.g. 'RV repair shop')")
    cid: Optional[str] = Field(None, description="Google CID for place results")
    latitude: Optional[float] = Field(None, description="Latitude for place results")
    longitude: Optional[float] = Field(None, description="Longitude for place results")
    price_level: Optional[str] = Field(None, description="Price level for place results (e.g. '$$')")

    @model_validator(mode="after")
    def _phone_alias(self) -> "SerperSearchResult":
        # The service sends phone_number. Before 0.2.2 the client read a
        # `phone` key that never arrived, so `phone` was always None; keep it
        # filled for code written against it.
        if self.phone_number is None and self.phone is not None:
            self.phone_number = self.phone
        elif self.phone is None:
            self.phone = self.phone_number
        return self


class SerperKnowledgeGraph(BaseModel):
    title: Optional[str] = Field(None, description="Knowledge graph title")
    type: Optional[str] = Field(None, description="Knowledge graph type")
    description: Optional[str] = Field(None, description="Knowledge graph description")
    website: Optional[str] = Field(None, description="Knowledge graph website")
    image_url: Optional[str] = Field(None, description="Knowledge graph image URL")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Key-value attributes")
    source: Optional[dict] = Field(None, description="Source information")
    header_images: Optional[List[dict]] = Field(None, description="Header images")


class SerperRelatedQuestion(BaseModel):
    question: str = Field(..., description="Related question")
    snippet: Optional[str] = Field(None, description="Answer snippet")
    title: Optional[str] = Field(None, description="Source title")
    link: Optional[str] = Field(None, description="Source link")


class SerperWebSearchInput(BaseModel):
    """Input model for Serper Web Search tool"""
    search_query: str = Field(..., description="Search query string", min_length=1)
    num_results: Optional[int] = Field(10, description="Number of results to return", ge=1, le=100)
    search_type: Optional[Literal["search", "news", "images", "shopping", "places"]] = Field(
        "search", description="Type of search to perform"
    )
    location: Optional[str] = Field(None, description="Location for localized search results (e.g., 'New York,New York')")
    language: Optional[str] = Field(None, description="Google UI language (e.g., 'en' for English)")
    country: Optional[str] = Field(None, description="Google country code (e.g., 'us')")
    time_period: Optional[Literal["hour", "day", "week", "month", "year"]] = Field(
        None, description="Time period filter for results"
    )
    page: Optional[int] = Field(None, description="Pagination page number", ge=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "search_query": "Python programming tutorials",
                "num_results": 5,
                "search_type": "search"
            }
        }
    }


class SerperWebSearchOutput(BaseModel):
    """Output model for Serper Web Search tool"""
    tool: Literal["serper_web_search_tool"] = Field(..., description="Tool identifier")
    query: str = Field(..., description="The search query that was executed")
    results: List[SerperSearchResult] = Field(default_factory=list, description="Search results")
    total_results: Optional[int] = Field(None, description="Total number of results found")
    knowledge_graph: Optional[SerperKnowledgeGraph] = Field(None, description="Knowledge graph information")
    related_searches: Optional[List[dict]] = Field(None, description="Related search queries (each dict has a 'query' key)")
    people_also_ask: Optional[List[SerperRelatedQuestion]] = Field(None, description="People also ask questions")
    search_information: Optional[dict] = Field(None, description="Search metadata and information")
    api_error: Optional[str] = Field(None, description="API error message if request failed")
    api_status_code: Optional[int] = Field(None, description="API response status code")

    def compact_dump(self) -> dict:
        """Return a compact dict with all None/null fields stripped."""
        data = self.model_dump(exclude_none=True)
        data.pop("search_information", None)
        if "results" in data:
            data["results"] = [
                {k: v for k, v in r.items() if v is not None}
                for r in data["results"]
            ]
        return data

    model_config = {
        "json_schema_extra": {
            "example": {
                "tool": "serper_web_search_tool",
                "query": "Python programming tutorials",
                "results": [
                    {
                        "title": "Learn Python",
                        "link": "https://example.com/python",
                        "snippet": "A comprehensive Python tutorial...",
                        "position": 1,
                        "result_type": "organic"
                    }
                ]
            }
        }
    }
