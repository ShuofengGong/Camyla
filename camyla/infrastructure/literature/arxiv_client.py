import os
import time
import arxiv
import urllib.error
from pathlib import Path
from typing import List, Optional, Dict, Any
from pypdf import PdfReader
from .base import BaseLiteratureAPI, Paper

class ArxivClient(BaseLiteratureAPI):
    """ArXiv API client implementation."""

    def __init__(self,
                 max_retries: int = 3,
                 page_wait_seconds: float = 3.0,
                 download_retry_delay_base: int = 3,
                 categories: Optional[List[str]] = None):
        """Initialize the ArXiv client.

        Args:
            max_retries: Maximum number of retries.
            page_wait_seconds: Delay between page requests.
            download_retry_delay_base: Base delay for download retries.
            categories: List of arXiv categories to restrict searches to (e.g. ['cs.CV', 'cs.AI', 'eess.IV']).
        """
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=page_wait_seconds,
            num_retries=max_retries
        )
        self.max_retries = max_retries
        self.download_retry_delay_base = download_retry_delay_base
        self.categories = categories or ['cs.CV', 'cs.AI', 'eess.IV']  # Default to CV/AI/IV domains

    def _process_query(self, query: str) -> str:
        """Process the query string, ensuring it does not exceed the maximum length."""
        MAX_QUERY_LENGTH = 300

        if len(query) <= MAX_QUERY_LENGTH:
            return query

        words = query.split()
        processed_query = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break

        return ' '.join(processed_query)

    def search(self, query: str, limit: int = 10, **kwargs) -> List[Paper]:
        """Search for papers.

        Args:
            query: Search query.
            limit: Maximum number of results to return.
            **kwargs: Additional parameters.

        Returns:
            List[Paper]: List of matching papers.
        """
        processed_query = self._process_query(query)

        # Build a query that restricts to the configured categories
        if self.categories:
            category_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
            full_query = f"({category_query}) AND abs:{processed_query}"
        else:
            full_query = f"abs:{processed_query}"

        retry_count = 0

        while retry_count < self.max_retries:
            try:
                search = arxiv.Search(
                    query=full_query,
                    max_results=limit,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                papers = []
                for result in self.client.results(search):
                    paper_id = result.pdf_url.split("/")[-1]
                    pub_date = str(result.published).split(" ")[0]

                    # Build Paper object
                    paper = Paper(
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        year=result.published.year,
                        venue="arXiv",
                        abstract=result.summary,
                        citation_key=f"arxiv_{paper_id}_{result.published.year}",
                        bibtex=self._generate_bibtex(result),
                        metadata={
                            "arxiv_id": paper_id,
                            "publication_date": pub_date,
                            "pdf_url": result.pdf_url,
                            "primary_category": result.primary_category,
                            "categories": result.categories,
                            "links": [link.href for link in result.links],
                            "doi": result.doi
                        }
                    )
                    papers.append(paper)

                time.sleep(1.0)  # Inter-request delay
                return papers

            except Exception as e:
                print(f"Error in search for query '{query}' (Attempt {retry_count+1}/{self.max_retries}): {e}")
                retry_count += 1
                if retry_count < self.max_retries:
                    time.sleep(2 * retry_count)  # Exponential backoff
                    continue
                else:
                    print(f"All retries failed for search with query '{query}'.")
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
        pdf_text = ""
        download_filename = f"downloaded-paper_{paper_id.replace('/', '_').replace(':', '_')}.pdf"

        try:
            # Fetch the paper object
            search = arxiv.Search(id_list=[paper_id])
            paper = next(self.client.results(search))
        except Exception as e:
            print(f"Error fetching paper metadata for '{paper_id}': {e}")
            return None

        # Download and process the PDF
        for attempt in range(self.max_retries):
            try:
                # Download the PDF
                paper.download_pdf(filename=download_filename)

                # Read the PDF
                reader = PdfReader(download_filename)

                # Extract text
                for page_number, page in enumerate(reader.pages, start=1):
                    try:
                        text = page.extract_text()
                        if text is None:
                            text = ""
                    except Exception as e:
                        print(f"Error extracting text from page {page_number}: {e}")
                        text = f"[Could not extract text from page {page_number}]"

                    pdf_text += f"--- Page {page_number} ---\n"
                    pdf_text += text
                    pdf_text += "\n"

                break  # Exit on success

            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    delay = self.download_retry_delay_base * (attempt + 1)
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    # Clean up potentially corrupted file
                    if os.path.exists(download_filename):
                        try:
                            os.remove(download_filename)
                        except OSError as e_rm:
                            print(f"Warning: Could not remove temporary file: {e_rm}")
                else:
                    print(f"All {self.max_retries} attempts failed.")
                    return None

        # Clean up the downloaded file
        if os.path.exists(download_filename):
            try:
                os.remove(download_filename)
            except OSError as e:
                print(f"Warning: Could not remove temporary file: {e}")

        time.sleep(1.0)  # Inter-request delay
        return pdf_text

    def _generate_bibtex(self, result: arxiv.Result) -> str:
        """Generate a BibTeX citation.

        Args:
            result: arxiv.Result object.

        Returns:
            str: BibTeX-formatted citation string.
        """
        paper_id = result.pdf_url.split("/")[-1]
        authors = " and ".join([author.name for author in result.authors])

        bibtex = f"""@article{{arxiv_{paper_id}_{result.published.year},
    title={{{result.title}}},
    author={{{authors}}},
    journal={{arXiv preprint arXiv:{paper_id}}},
    year={{{result.published.year}}},
    url={{{result.pdf_url}}},
    abstract={{{result.summary}}}
}}"""
        return bibtex
