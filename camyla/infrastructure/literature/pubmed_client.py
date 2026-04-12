import os
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus
from .base import BaseLiteratureAPI, Paper


class PubMedClient(BaseLiteratureAPI):
    """PubMed API client implementation using the NCBI E-utilities."""

    def __init__(self,
                 email: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_retries: int = 3,
                 request_delay: float = 1.0):
        """Initialize the PubMed client.

        Args:
            email: User email (required by NCBI).
            api_key: NCBI API key (optional, raises rate limits).
            max_retries: Maximum number of retries.
            request_delay: Delay between requests (seconds).
        """
        from camyla.model_config import get_api_key
        self.email = email or os.getenv("NCBI_EMAIL")
        self.api_key = api_key or get_api_key("ncbi")
        self.max_retries = max_retries
        self.request_delay = request_delay

        if not self.email:
            raise ValueError(
                "Email is required for NCBI E-utilities. "
                "Please provide email parameter or set NCBI_EMAIL environment variable."
            )

        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.session = requests.Session()

        # Default parameters
        self.default_params = {
            "email": self.email,
            "tool": "PubMedClient"
        }

        if self.api_key:
            self.default_params["api_key"] = self.api_key

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[requests.Response]:
        """Send an HTTP request.

        Args:
            endpoint: API endpoint.
            params: Request parameters.

        Returns:
            Optional[requests.Response]: Response object, or None on failure.
        """
        url = f"{self.base_url}/{endpoint}"
        request_params = {**self.default_params, **params}

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=request_params, timeout=30)
                response.raise_for_status()

                # Inter-request delay
                time.sleep(self.request_delay)
                return response

            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"All {self.max_retries} attempts failed")
                    return None

    def search(self, query: str, limit: int = 10, **kwargs) -> List[Paper]:
        """Search for papers.

        Args:
            query: Search query.
            limit: Maximum number of results to return.
            **kwargs: Additional parameters:
                - sort: Sort order ('relevance', 'pub_date', 'author', 'journal').
                - mindate: Minimum date (YYYY/MM/DD).
                - maxdate: Maximum date (YYYY/MM/DD).
                - field: Search field ('title', 'author', 'abstract', 'all').

        Returns:
            List[Paper]: List of matching papers.
        """
        if not query:
            return []

        try:
            # Step 1: use ESearch to fetch the list of PMIDs
            pmids = self._search_pmids(query, limit, **kwargs)
            if not pmids:
                return []

            # Step 2: use EFetch to retrieve detailed information
            papers = self._fetch_paper_details(pmids)

            return papers

        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []

    def _search_pmids(self, query: str, limit: int, **kwargs) -> List[str]:
        """Search for a list of PMIDs.

        Args:
            query: Search query.
            limit: Maximum number of results.
            **kwargs: Additional search parameters.

        Returns:
            List[str]: List of PMIDs.
        """
        # Build search parameters
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(limit),
            "retmode": "xml",
            "sort": kwargs.get("sort", "relevance")
        }

        # Date filters
        if "mindate" in kwargs:
            search_params["mindate"] = kwargs["mindate"]
        if "maxdate" in kwargs:
            search_params["maxdate"] = kwargs["maxdate"]

        # Field restriction
        field = kwargs.get("field")
        if field and field != "all":
            if field == "title":
                search_params["term"] = f"{query}[Title]"
            elif field == "author":
                search_params["term"] = f"{query}[Author]"
            elif field == "abstract":
                search_params["term"] = f"{query}[Abstract]"

        response = self._make_request("esearch.fcgi", search_params)
        if not response:
            return []

        try:
            root = ET.fromstring(response.text)
            pmids = []

            for id_elem in root.findall(".//Id"):
                if id_elem.text:
                    pmids.append(id_elem.text)

            return pmids

        except ET.ParseError as e:
            print(f"Error parsing search results: {e}")
            return []

    def _fetch_paper_details(self, pmids: List[str]) -> List[Paper]:
        """Fetch detailed paper information.

        Args:
            pmids: List of PMIDs.

        Returns:
            List[Paper]: List of papers.
        """
        if not pmids:
            return []

        # Batch to avoid overly long URLs
        batch_size = 200
        all_papers = []

        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            batch_papers = self._fetch_batch_details(batch_pmids)
            all_papers.extend(batch_papers)

        return all_papers

    def _fetch_batch_details(self, pmids: List[str]) -> List[Paper]:
        """Fetch details for a batch of papers.

        Args:
            pmids: List of PMIDs.

        Returns:
            List[Paper]: List of papers.
        """
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract"
        }

        response = self._make_request("efetch.fcgi", fetch_params)
        if not response:
            return []

        try:
            root = ET.fromstring(response.text)
            papers = []

            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)

            return papers

        except ET.ParseError as e:
            print(f"Error parsing fetch results: {e}")
            return []

    def _parse_article(self, article_elem: ET.Element) -> Optional[Paper]:
        """Parse a single article.

        Args:
            article_elem: Article XML element.

        Returns:
            Optional[Paper]: Parsed paper object.
        """
        try:
            # PMID
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "Unknown Title"

            # Authors
            authors = []
            for author in article_elem.findall(".//Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                if last_name is not None:
                    name = last_name.text
                    if first_name is not None:
                        name = f"{first_name.text} {name}"
                    authors.append(name)

            # Abstract
            abstract_parts = []
            for abstract in article_elem.findall(".//AbstractText"):
                if abstract.text:
                    abstract_parts.append(abstract.text)
            abstract = " ".join(abstract_parts) if abstract_parts else ""

            # Journal info
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

            # Publication year
            year = 0
            pub_date = article_elem.find(".//PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None:
                    try:
                        year = int(year_elem.text)
                    except ValueError:
                        year = 0

            # DOI
            doi = ""
            for article_id in article_elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break

            # Citation key
            citation_key = f"pubmed_{pmid}_{year}"

            # Build Paper object
            paper = Paper(
                title=title,
                authors=authors[:10],  # Limit author count
                year=year,
                venue=journal,
                abstract=abstract,
                citation_key=citation_key,
                bibtex=self._generate_bibtex(pmid, title, authors, year, journal, doi),
                metadata={
                    "pmid": pmid,
                    "doi": doi,
                    "journal": journal,
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "source": "PubMed"
                }
            )

            return paper

        except Exception as e:
            print(f"Error parsing article: {e}")
            return None

    def _generate_bibtex(self, pmid: str, title: str, authors: List[str],
                        year: int, journal: str, doi: str) -> str:
        """Generate a BibTeX citation.

        Args:
            pmid: PubMed ID.
            title: Paper title.
            authors: List of authors.
            year: Publication year.
            journal: Journal name.
            doi: DOI.

        Returns:
            str: BibTeX-formatted citation string.
        """
        # Author formatting
        if authors:
            author_str = " and ".join(authors[:10])  # Limit author count
        else:
            author_str = "Unknown Author"

        # Build BibTeX
        bibtex_parts = [
            f"@article{{pubmed_{pmid}_{year},",
            f"    title={{{title}}},",
            f"    author={{{author_str}}},",
            f"    journal={{{journal}}},",
            f"    year={{{year}}},",
            f"    pmid={{{pmid}}},",
            f"    url={{https://pubmed.ncbi.nlm.nih.gov/{pmid}/}}"
        ]

        if doi:
            bibtex_parts.append(f"    doi={{{doi}}},")

        bibtex_parts.append("}")

        return "\n".join(bibtex_parts)

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
            paper_id: Paper ID (PMID).

        Returns:
            Optional[str]: PubMed does not provide full text directly;
            returns abstract and metadata instead.
        """
        try:
            # PubMed does not serve full text directly, but we can fetch rich metadata and the abstract.
            papers = self._fetch_paper_details([paper_id])
            if papers:
                paper = papers[0]
                full_info = f"""Title: {paper.title}

Authors: {', '.join(paper.authors)}

Journal: {paper.venue}

Year: {paper.year}

Abstract:
{paper.abstract}

PMID: {paper.metadata.get('pmid', '')}
DOI: {paper.metadata.get('doi', 'N/A')}
URL: {paper.metadata.get('pubmed_url', '')}"""
                return full_info.strip()
            return None

        except Exception as e:
            print(f"Error getting full text for PMID {paper_id}: {e}")
            return None
