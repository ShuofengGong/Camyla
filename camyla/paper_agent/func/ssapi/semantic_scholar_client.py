import os
import time
import backoff
import requests
from typing import List, Optional, Dict, Any
from pathlib import Path
from pypdf import PdfReader
from fake_useragent import UserAgent
from .base import BaseLiteratureAPI, Paper

def on_backoff(details):
    """Backoff callback."""
    print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries")

# Default list of top-tier venues
DEFAULT_VENUE_FILTER = [
    "IEEE International Conference on Computer Vision",
    "AAAI Conference on Artificial Intelligence",
    "IEEE International Conference on Multimedia and Expo",
    "International Conference on Pattern Recognition",
    "Annual Meeting of the Association for Computational Linguistics",
    "International Conference on Intelligent Computing",
    "International Joint Conference on Neural Networks",
    "European Conference on Computer Vision",
    "International Conference on Machine Learning",
    "International Conference on Learning Representations",
    "International Conference on Information Photonics",
    "Computer Vision and Pattern Recognition",
    "Conference on Empirical Methods in Natural Language Processing",
    "European Conference on Artificial Intelligence",
    "British Machine Vision Conference",
    "International Conference on Neural Information Processing",
    "International Joint Conference on Artificial Intelligence",
    "International Conference on Artificial Neural Networks",
    "Pacific Rim International Conference on Artificial Intelligence",
    "Neural Information Processing Systems",
    "International Conference on 3D Vision",
    "ACM Multimedia",
    "ACM Multimedia Asia",
    "Interspeech",
    "Conference on Multimedia Modeling",
    "IEEE International Conference on Acoustics, Speech, and Signal Processing",
    "Chinese Conference on Pattern Recognition and Computer Vision",
    "International Conference on Medical Image Computing and Computer-Assisted Intervention",
    "IEEE International Conference on Bioinformatics and Biomedicine",
    "Asian Conference on Machine Learning",
    "Asian Conference on Computer Vision",
]

# Venue abbreviation aliases
VENUE_ALIASES = {
    "ICCV": "IEEE International Conference on Computer Vision",
    "CVPR": "Computer Vision and Pattern Recognition",
    "ECCV": "European Conference on Computer Vision",
    "NeurIPS": "Neural Information Processing Systems",
    "NIPS": "Neural Information Processing Systems",
    "ICML": "International Conference on Machine Learning",
    "ICLR": "International Conference on Learning Representations",
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "IJCAI": "International Joint Conference on Artificial Intelligence",
    "ACL": "Annual Meeting of the Association for Computational Linguistics",
    "EMNLP": "Conference on Empirical Methods in Natural Language Processing",
    "ICASSP": "IEEE International Conference on Acoustics, Speech, and Signal Processing",
    "MICCAI": "International Conference on Medical Image Computing and Computer-Assisted Intervention",
}

class SemanticScholarClient(BaseLiteratureAPI):
    """Semantic Scholar API client implementation."""

    def __init__(self,
                 min_year: str = "2023-01-01",
                 fields_of_study: str = "Computer Science",
                 venue_filter: Optional[List[str]] = None,
                 enable_venue_filter: bool = True,
                 require_open_access: bool = True):
        """Initialize the Semantic Scholar client.

        Args:
            min_year: Earliest publication date (YYYY-MM-DD).
            fields_of_study: Fields-of-study filter, defaults to "Computer Science".
            venue_filter: Custom venue allow-list; uses the default list when None.
            enable_venue_filter: Whether to enable venue filtering. Defaults to True.
            require_open_access: Whether to restrict to open-access papers. Defaults to True.
        """
        from camyla.model_config import get_api_key
        self.api_key = get_api_key("s2")
        if not self.api_key:
            print("Warning: S2_API_KEY not found. Using unauthenticated access (lower rate limits).")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"X-API-KEY": self.api_key} if self.api_key else {}

        # Filter parameters
        self.min_year = min_year
        self.min_year_filter = min_year.split('-')[0] + "-"  # Convert to API format "2023-"
        self.fields_of_study = fields_of_study
        self.venue_filter = venue_filter if venue_filter is not None else DEFAULT_VENUE_FILTER
        self.enable_venue_filter = enable_venue_filter
        self.require_open_access = require_open_access

    def _match_venue(self, paper_venue: str) -> bool:
        """Check whether a paper's venue is in the allow-list.

        Args:
            paper_venue: The paper's venue string.

        Returns:
            Whether it matches.
        """
        if not paper_venue or not self.venue_filter:
            return False

        paper_venue_lower = paper_venue.lower().strip()

        # 1. Exact match against the venue allow-list
        for allowed_venue in self.venue_filter:
            allowed_venue_lower = allowed_venue.lower().strip()
            if allowed_venue_lower == paper_venue_lower:
                return True

        # 2. Check the alias table
        for alias, full_name in VENUE_ALIASES.items():
            if alias.lower() == paper_venue_lower:
                if full_name in self.venue_filter:
                    return True

        return False

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
        max_tries=5
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
            # When venue filter is on, fetch extra results to filter client-side
            api_limit = limit * 10 if self.enable_venue_filter else limit
            api_limit = min(api_limit, 100)  # API caps at 100

            # Build request parameters
            params = {
                "query": query,
                "limit": api_limit,
                "fields": "paperId,title,abstract,year,authors,isOpenAccess,openAccessPdf,citationCount,url,fieldsOfStudy,venue,publicationVenue,journal",
                "year": self.min_year_filter,  # Year filter
            }

            # Only add the open-access filter when required
            if self.require_open_access:
                params["openAccessPdf"] = ""  # Restrict to open-access papers

            # Fields-of-study filter
            if self.fields_of_study:
                params["fieldsOfStudy"] = self.fields_of_study

            # Send API request
            response = requests.get(
                f"{self.base_url}/paper/search",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()

            # Parse results
            results = response.json()
            total = results.get("total", 0)
            if total == 0:
                return []

            raw_papers = results.get("data", [])

            # Client-side venue filter when enabled
            if self.enable_venue_filter and self.venue_filter:
                filtered_data = []
                for data in raw_papers:
                    # Paper venue
                    paper_venue = data.get("venue", "")

                    # Fall back to publicationVenue when venue is missing
                    if not paper_venue:
                        pub_venue = data.get("publicationVenue", {})
                        if isinstance(pub_venue, dict):
                            paper_venue = pub_venue.get("name", "")

                    # Check whether it matches any allowed venue
                    if self._match_venue(paper_venue):
                        filtered_data.append(data)
                        if len(filtered_data) >= limit:
                            break

                raw_papers = filtered_data

            # Materialize paper data
            papers = []
            for data in raw_papers:
                paper = self._create_paper_from_data(data)
                papers.append(paper)

            # Sort by citation count
            papers.sort(key=lambda x: x.metadata.get("citation_count", 0), reverse=True)

            time.sleep(1.0)  # Inter-request delay
            return papers

        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
            raise
        except Exception as e:
            print(f"Error searching Semantic Scholar: {e}")
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
            paper_id: Paper ID (Semantic Scholar ID).

        Returns:
            Optional[str]: The full text, or None if retrieval fails.
        """
        pdf_text = ""
        download_filename = f"downloaded-s2-paper_{paper_id}.pdf"
        max_retries = 3

        # UserAgent for PDF downloads
        ua = UserAgent()

        try:
            # Fetch paper metadata (with retries)
            data = None
            for attempt in range(max_retries):
                try:
                    print(f"  Fetching paper metadata (attempt {attempt + 1}/{max_retries})...")
                    response = requests.get(
                        f"{self.base_url}/paper/{paper_id}",
                        headers=self.headers,
                        params={"fields": "openAccessPdf,url,title"},
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    break  # Exit on success
                except (requests.exceptions.HTTPError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.SSLError,
                        requests.exceptions.Timeout) as e:
                    print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        delay = 3 * (attempt + 1)
                        print(f"  Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"  All {max_retries} attempts failed to fetch metadata.")
                        return None

            if data is None:
                return None

            # Resolve the open-access PDF URL
            pdf_url = None
            if data.get("openAccessPdf", {}).get("url"):
                pdf_url = data["openAccessPdf"]["url"]

            if not pdf_url:
                print(f"No open access PDF available for paper {paper_id}")
                return None

            print(f"Paper available at: {pdf_url}")

            # Download and process the PDF
            for attempt in range(max_retries):
                try:
                    # Download the PDF
                    print(f"  Downloading PDF (attempt {attempt + 1}/{max_retries})...")
                    headers = {'User-Agent': ua.random}
                    pdf_response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
                    pdf_response.raise_for_status()

                    # Save the PDF
                    with open(download_filename, 'wb') as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Read the PDF
                    print(f"  Extracting text from PDF...")
                    reader = PdfReader(download_filename)

                    # Extract text
                    for page_number, page in enumerate(reader.pages, start=1):
                        try:
                            text = page.extract_text()
                            if text is None:
                                text = ""
                        except Exception as e:
                            print(f"  Error extracting text from page {page_number}: {e}")
                            text = f"[Could not extract text from page {page_number}]"

                        pdf_text += f"--- Page {page_number} ---\n"
                        pdf_text += text
                        pdf_text += "\n"

                    print(f"  Successfully extracted {len(pdf_text)} characters from {len(reader.pages)} pages")
                    break  # Exit on success

                except Exception as e:
                    print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        delay = 3 * (attempt + 1)
                        print(f"  Retrying in {delay}s...")
                        time.sleep(delay)
                        # Clean up potentially corrupted file
                        if os.path.exists(download_filename):
                            try:
                                os.remove(download_filename)
                            except OSError as e_rm:
                                print(f"  Warning: Could not remove temporary file: {e_rm}")
                    else:
                        print(f"  All {max_retries} attempts failed.")
                        return None

        except Exception as e:
            print(f"Error getting full text from Semantic Scholar: {e}")
            return None

        finally:
            # Clean up downloaded file
            if os.path.exists(download_filename):
                try:
                    os.remove(download_filename)
                except OSError as e:
                    print(f"Warning: Could not remove temporary file: {e}")

        time.sleep(1.0)  # Inter-request delay
        return pdf_text if pdf_text else None

    def _generate_citation_key(self, data: Dict) -> str:
        """Generate an academic-style citation key (FirstAuthorLastName + Year).

        Args:
            data: Paper data.

        Returns:
            str: Citation key (e.g. Ronneberger2015, Chen2023a).
        """
        # Extract first author's surname
        authors = data.get("authors", [])
        if authors and len(authors) > 0:
            first_author_name = authors[0].get("name", "Unknown")
            # Assume the surname is the last whitespace-delimited word
            name_parts = first_author_name.split()
            last_name = name_parts[-1] if name_parts else "Unknown"
        else:
            last_name = "Unknown"

        # Sanitize (strip non-alphabetic characters)
        import re
        last_name = re.sub(r'[^a-zA-Z]', '', last_name)

        # Year
        year = data.get("year", "")
        if not year:
            year = "unknown"

        # Base citation key
        citation_key = f"{last_name}{year}"

        # Append a suffix from the paper ID to disambiguate duplicates
        paper_id = data.get("paperId", "")
        if paper_id:
            paper_id = paper_id.split("/")[-1]
            # Use the last 4 chars of the hash as a suffix (helps with same author + year)
            suffix = paper_id[-4:] if len(paper_id) >= 4 else paper_id
            citation_key = f"{last_name}{year}_{suffix}"

        return citation_key

    def _create_paper_from_data(self, data: Dict) -> Paper:
        """Build a Paper object from API response data.

        Args:
            data: API response data.

        Returns:
            Paper: Paper object.
        """
        # Authors
        authors = [
            author.get("name", "Unknown")
            for author in data.get("authors", [])
        ]

        # Citation key (academic format: FirstAuthorLastName + Year)
        citation_key = self._generate_citation_key(data)

        # Venue
        venue = data.get("venue", "")
        if not venue:
            pub_venue = data.get("publicationVenue", {})
            if isinstance(pub_venue, dict):
                venue = pub_venue.get("name", "Unknown Venue")
        if not venue:
            venue = "Unknown Venue"

        # Open-access URL
        open_access_url = ""
        open_access_pdf = data.get("openAccessPdf", {})
        if isinstance(open_access_pdf, dict):
            open_access_url = open_access_pdf.get("url", "")

        # paper_id for metadata
        paper_id = data.get("paperId", "")
        if paper_id:
            paper_id = paper_id.split("/")[-1]

        # Build Paper object
        paper = Paper(
            title=data.get("title", "Unknown Title"),
            authors=authors[:10],  # Limit to the first 10 authors
            year=data.get("year", 0),
            venue=venue,
            abstract=data.get("abstract", ""),
            citation_key=citation_key,
            bibtex=self._generate_bibtex(data),
            metadata={
                "semantic_scholar_id": paper_id,
                "citation_count": data.get("citationCount", 0),
                "url": data.get("url", ""),
                "open_access_url": open_access_url,
                "is_open_access": data.get("isOpenAccess", False),
                "fields_of_study": data.get("fieldsOfStudy", []),
            }
        )

        return paper

    def _generate_bibtex(self, data: Dict) -> str:
        """Generate a BibTeX citation.

        Args:
            data: Paper data.

        Returns:
            str: BibTeX-formatted citation string.
        """
        # Authors
        authors = " and ".join([
            author.get("name", "Unknown")
            for author in data.get("authors", [])[:10]  # Limit to the first 10 authors
        ])

        # Citation key using the academic format
        citation_key = self._generate_citation_key(data)

        bibtex = f"""@article{{{citation_key},
    title={{{data.get('title', 'Unknown Title')}}},
    author={{{authors}}},
    journal={{{data.get('venue', 'Unknown Venue')}}},
    year={{{data.get('year', '')}}},
    abstract={{{data.get('abstract', '')}}}
}}"""
        return bibtex
