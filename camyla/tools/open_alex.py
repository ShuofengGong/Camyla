import os
import requests
import time
from typing import Dict, List, Optional, Union

import backoff
import pyalex
from pyalex import Work, Works

from camyla.tools.base_tool import BaseTool
import json

import openai
import httpx

def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )

class OpenAlexSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchOpenAlex",
        description: str = (
            "Search for relevant literature using OpenAlex. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 20,
        min_year: str = "2022-01-01",
        max_abstract_length: int = 1500,
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
        self.min_year = min_year
        self.max_abstract_length = max_abstract_length
        
        # Configure pyalex with email if available
        mail = os.environ.get("OPENALEX_MAIL_ADDRESS", None)
        if mail is None:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")
        else:
            pyalex.config.email = mail

    def use_tool(self, query: str) -> Optional[str]:
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
    backoff.expo,
    # --- Exception list ---
    (
        requests.exceptions.HTTPError,           # In case code anywhere still uses requests directly
        requests.exceptions.ConnectionError,     # Same as above
        openai.APIConnectionError,               # Catches OpenAI connection errors
        openai.APITimeoutError,                  # Optionally catch OpenAI timeouts
        openai.RateLimitError,                   # Rate-limit errors (retry as appropriate)
        httpx.RequestError,                      # Broader httpx network errors (optional fallback)
        json.JSONDecodeError,
        httpx.RemoteProtocolError
        # Add further httpx / third-party network exceptions as needed,
        # e.g. httpx.RemoteProtocolError, httpx.ConnectError.
        ),
        # ----------------------
        on_backoff=on_backoff,
        max_tries=10,  # Explicitly cap the retry count (optional)
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        # Create the query with filters
        works_query = Works().search(query)
        works_query = works_query.filter(from_publication_date=self.min_year)
        
        # Get the results
        works = works_query.get(per_page=self.max_results)
        
        # Process the results
        papers = [self.extract_info_from_work(work) for work in works]
        
        # Sort papers by citationCount in descending order
        papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
        
        return papers

    def extract_info_from_work(self, work: Work) -> dict:
        # Extract venue information
        venue = "Unknown"
        for location in work["locations"]:
            if location["source"] is not None:
                venue_name = location["source"]["display_name"]
                if venue_name:
                    venue = venue_name
                    break
        
        # Extract title
        title = work["title"]
        
        # Extract and truncate abstract if needed
        abstract = work["abstract"] or ""
        if len(abstract) > self.max_abstract_length:
            print(f"[WARNING] {title=}: {len(abstract)=} is too long! Use first {self.max_abstract_length} chars.")
            abstract = abstract[:self.max_abstract_length]
        
        # Extract authors
        authors_list = [author["author"]["display_name"] for author in work["authorships"]]
        if len(authors_list) < 20:
            authors = ", ".join(authors_list)
        else:
            authors = f"{authors_list[0]} et al."
        
        # Create formatted authors list in the same format as the original tool
        formatted_authors = []
        for author_name in authors_list[:10]:  # Limit to first 10 authors
            formatted_authors.append({"name": author_name})
        
        # Construct paper dict
        paper = {
            "title": title,
            "authors": formatted_authors,  # List of author dicts for compatibility
            "venue": venue,
            "year": work["publication_year"],
            "abstract": abstract,
            "citationCount": work["cited_by_count"],
            "openAccessUrl": work.get("open_access", {}).get("oa_url", ""),
            "doi": work.get("doi", "")
        }
        
        return paper

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                [author.get("name", "Unknown") for author in paper.get("authors", [])]
            )
            
            # Add DOI and open access URL if available
            additional_info = ""
            if paper.get("doi"):
                additional_info += f"\nDOI: {paper.get('doi')}"
            if paper.get("openAccessUrl"):
                additional_info += f"\nOpen Access URL: {paper.get('openAccessUrl')}"
            
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}{additional_info}"""
            )
        return "\n\n".join(paper_strings)


import openai
import httpx

def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )
@backoff.on_exception(
    backoff.expo,
    # --- Exception list ---
    (
        requests.exceptions.HTTPError,           # In case code anywhere still uses requests directly
        requests.exceptions.ConnectionError,     # Same as above
        openai.APIConnectionError,               # Catches OpenAI connection errors
        openai.APITimeoutError,                  # Optionally catch OpenAI timeouts
        openai.RateLimitError,                   # Rate-limit errors (retry as appropriate)
        httpx.RequestError,                      # Broader httpx network errors (optional fallback)
        # Add further httpx / third-party network exceptions as needed,
        # e.g. httpx.RemoteProtocolError, httpx.ConnectError.
    ),
    # ----------------------
    on_backoff=on_backoff,
    max_tries=10,  # Explicitly cap the retry count (optional)
)
def search_for_papers(query, result_limit=20, engine="openalex") -> Union[None, List[Dict]]:
    """
    Search for academic papers using OpenAlex API.
    
    Note: For Semantic Scholar searches, use SemanticScholarClient from 
    camyla.tools.ssapi.semantic_scholar_client instead.
    
    Args:
        query (str): The search query
        result_limit (int): Maximum number of results to return
        engine (str): Must be "openalex" (semanticscholar support removed to avoid duplication)
        
    Returns:
        List of paper dictionaries or None if no results found
    """
    if not query:
        return None
    
    if engine == "semanticscholar":
        raise NotImplementedError(
            "Semantic Scholar support has been removed from this function to avoid code duplication. "
            "Please use SemanticScholarClient from camyla.tools.ssapi.semantic_scholar_client instead."
        )
        
    elif engine == "openalex":
        mail = os.environ.get("OPENALEX_MAIL_ADDRESS", None)
        if mail is None:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")
        else:
            pyalex.config.email = mail

        max_abstract_length = 2500
        
        def extract_info_from_work(work: Work) -> dict:
            # Extract venue information
            venue = "Unknown"
            for location in work["locations"]:
                if location["source"] is not None:
                    venue_name = location["source"]["display_name"]
                    if venue_name:
                        venue = venue_name
                        break
            
            # Extract title and abstract
            title = work["title"]
            abstract = work["abstract"] or ""
            if len(abstract) > max_abstract_length:
                print(f"[WARNING] {title=}: {len(abstract)=} is too long! Use first {max_abstract_length} chars.")
                abstract = abstract[:max_abstract_length]
            
            # Extract authors
            authors_list = [author["author"]["display_name"] for author in work["authorships"]]
            authors = ", ".join(authors_list) if len(authors_list) < 20 else f"{authors_list[0]} et al."
            
            # Construct paper dict
            paper = {
                "title": title,
                "authors": authors,
                "venue": venue,
                "year": work["publication_year"],
                "abstract": abstract,
                "citationCount": work["cited_by_count"],
                "doi": work.get("doi", "")
            }
            return paper
        
        try:
            # Create the query with filters
            works_query = Works().search(query)
            
            min_year = '2022-01-01'
            works_query = works_query.filter(from_publication_date=min_year)
            
            # Get the results
            works = works_query.get(per_page=result_limit)
            
            if not works:
                return None
                
            # Process the results
            papers = [extract_info_from_work(work) for work in works]
            return papers
            
        except Exception as e:
            print(f"Error searching OpenAlex: {e}")
            return None
    else:
        raise NotImplementedError(f"{engine=} not supported!")