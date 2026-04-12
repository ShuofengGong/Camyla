import os
from typing import Dict, List, Optional, Union

from camyla.tools.base_tool import BaseTool
from camyla.tools.ssapi.semantic_scholar_client import SemanticScholarClient

class SemanticScholarSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchSemanticScholar",
        description: str = (
            "Search for relevant literature using Semantic Scholar. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        
        # Initialize the client
        try:
            self.client = SemanticScholarClient(
                min_year="2000-01-01",  # Default reasonable start year
                enable_venue_filter=False,  # Disable venue filter for general tool usage
                require_open_access=False    # Allow non-open access papers
            )
        except Exception as e:
            print(f"Warning: Failed to initialize SemanticScholarClient: {e}")
            self.client = None

    def use_tool(self, **kwargs) -> Optional[str]:
        query = kwargs.get("query")
         # Fallback if passed as positional (unlikely given BaseTool usage but safe) or specific key
        if not query and "query" in kwargs:
             query = kwargs["query"]
        
        if not query:
             return "Please provide a search query."

        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query or not self.client:
            return None
            
        try:
            # Use the new client to search
            # Note: The client returns Paper objects, but the tool expects Dicts (or handles them)
            # We'll convert Paper objects to dictionaries that match the expected structure
            papers = self.client.search(query, limit=self.max_results)
            
            if not papers:
                return None
                
            results = []
            for paper in papers:
                # Convert Paper object to dictionary for compatibility
                authors = [{"name": a} for a in paper.authors]
                
                paper_dict = {
                    "title": paper.title,
                    "authors": authors,
                    "venue": paper.venue,
                    "year": paper.year,
                    "abstract": paper.abstract,
                    "citationCount": paper.metadata.get("citation_count", 0),
                    "url": paper.metadata.get("url", ""),
                    "paperId": paper.metadata.get("semantic_scholar_id", "")
                }
                results.append(paper_dict)
                
            return results
            
        except Exception as e:
            print(f"Error searching for papers: {e}")
            return None

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                [author.get("name", "Unknown") for author in paper.get("authors", [])]
            )
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)

# Legacy function for backward compatibility if imported elsewhere
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    tool = SemanticScholarSearchTool(max_results=result_limit)
    return tool.search_for_papers(query)
