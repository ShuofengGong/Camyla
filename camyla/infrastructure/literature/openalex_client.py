import os
import time
import backoff
import requests
import json
import pyalex
from typing import List, Optional, Dict, Any
from pyalex import Works, Work
from .base import BaseLiteratureAPI, Paper

def on_backoff(details):
    """Backoff callback."""
    print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries")

class OpenAlexClient(BaseLiteratureAPI):
    """OpenAlex API client implementation."""

    def __init__(self,
                 min_year: str = "2022-01-01",
                 max_abstract_length: int = 1500):
        """Initialize the OpenAlex client.

        Args:
            min_year: Earliest publication date to include.
            max_abstract_length: Maximum abstract length to keep.
        """
        self.min_year = min_year
        self.max_abstract_length = max_abstract_length

        # Configure pyalex email
        mail = os.environ.get("OPENALEX_MAIL_ADDRESS", None)
        if mail is None:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")
        else:
            pyalex.config.email = mail

    @backoff.on_exception(
        backoff.expo,
        (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            json.JSONDecodeError
        ),
        on_backoff=on_backoff,
        max_tries=10
    )
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Paper]:
        """Search for papers.

        Args:
            query: Search query.
            limit: Maximum number of results to return.
            **kwargs: Additional parameters.

        Returns:
            List[Paper]: List of matching papers.
        """
        if not query:
            return []

        try:
            # Build the query
            works_query = Works().search(query)
            works_query = works_query.filter(from_publication_date=self.min_year)

            # Fetch results
            works = works_query.get(per_page=limit)

            if not works:
                return []

            # Process results
            papers = []
            for work in works:
                paper = self._extract_info_from_work(work)
                papers.append(paper)

            # Sort by citation count
            papers.sort(key=lambda x: x.metadata.get("citation_count", 0), reverse=True)

            return papers

        except Exception as e:
            print(f"Error searching OpenAlex: {e}")
            return []

    def format_citation(self, paper: Paper, style: str = "bibtex") -> str:
        """Format a citation.

        Args:
            paper: Paper object.
            style: Citation format; currently only bibtex is supported.

        Returns:
            str: Formatted citation string.
        """
        if style != "bibtex":
            raise ValueError("Currently only bibtex style is supported")
        return paper.bibtex

    def get_full_text(self, paper_id: str) -> Optional[str]:
        """Retrieve the full text of a paper.

        Args:
            paper_id: Paper ID.

        Returns:
            Optional[str]: The full text, or None if retrieval fails.
        """
        # OpenAlex API does not directly expose full text.
        # If the paper has an open-access URL, we can try to download it.
        try:
            work = Works()[paper_id]
            if work and work.get("open_access", {}).get("oa_url"):
                print(f"Paper available at: {work['open_access']['oa_url']}")
                return None
            return None
        except Exception as e:
            print(f"Error getting full text from OpenAlex: {e}")
            return None

    def _extract_info_from_work(self, work: Work) -> Paper:
        """Extract paper information from a Work object.

        Args:
            work: Work object.

        Returns:
            Paper: Paper object.
        """
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

        # Extract and truncate the abstract
        abstract = work["abstract"] or ""
        if len(abstract) > self.max_abstract_length:
            print(f"[WARNING] {title=}: {len(abstract)=} is too long! Use first {self.max_abstract_length} chars.")
            abstract = abstract[:self.max_abstract_length]

        # Extract authors
        authors_list = [author["author"]["display_name"] for author in work["authorships"]]

        # Generate citation key
        citation_key = f"openalex_{work['id'].split('/')[-1]}_{work['publication_year']}"

        # Build Paper object
        paper = Paper(
            title=title,
            authors=authors_list[:10],  # Limit to the first 10 authors
            year=work["publication_year"],
            venue=venue,
            abstract=abstract,
            citation_key=citation_key,
            bibtex=self._generate_bibtex(work),
            metadata={
                "openalex_id": work["id"],
                "citation_count": work["cited_by_count"],
                "open_access_url": work.get("open_access", {}).get("oa_url", ""),
                "doi": work.get("doi", "")
            }
        )

        return paper

    def _generate_bibtex(self, work: Work) -> str:
        """Generate a BibTeX citation.

        Args:
            work: Work object.

        Returns:
            str: BibTeX-formatted citation string.
        """
        # Extract authors
        authors = " and ".join([
            author["author"]["display_name"]
            for author in work["authorships"][:10]  # Limit to the first 10 authors
        ])

        # Extract venue
        venue = "Unknown"
        for location in work["locations"]:
            if location["source"] is not None:
                venue_name = location["source"]["display_name"]
                if venue_name:
                    venue = venue_name
                    break

        # Generate citation key
        citation_key = f"openalex_{work['id'].split('/')[-1]}_{work['publication_year']}"

        bibtex = f"""@article{{{citation_key},
    title={{{work['title']}}},
    author={{{authors}}},
    journal={{{venue}}},
    year={{{work['publication_year']}}},
    doi={{{work.get('doi', '')}}},
    url={{{work.get('open_access', {}).get('oa_url', '')}}},
    abstract={{{work['abstract'] or ''}}}
}}"""
        return bibtex
