"""
Multi-source literature search.

Integrates four literature databases — ArXiv, OpenAlex, PubMed, and Semantic Scholar —
with deduplication, open-access filtering, and time-range filtering.
"""
import os
import random
import logging
import time
from typing import List, Optional, Dict, Any
from .base import Paper
from .arxiv_client import ArxivClient
from .openalex_client import OpenAlexClient
from .pubmed_client import PubMedClient
# Import SemanticScholarClient from the correct location in tools/ssapi
from camyla.tools.ssapi.semantic_scholar_client import SemanticScholarClient

logger = logging.getLogger(__name__)


class MultiSourceLiteratureSearch:
    """Multi-source literature search integrating ArXiv, OpenAlex, PubMed, and Semantic Scholar."""

    def __init__(self,
                 min_year: str = "2023-01-01",
                 phase1_min_year: str = "2023-01-01",
                 enable_randomization: bool = True,
                 use_arxiv: bool = True,
                 use_openalex: bool = True,
                 use_pubmed: bool = True,
                 use_semantic_scholar: bool = False,
                 filter_open_access: bool = True):
        """Initialize the multi-source search.

        Args:
            min_year: Earliest publication date (YYYY-MM-DD).
            phase1_min_year: Year used in Phase 1 challenge-discovery mode (YYYY-MM-DD).
            enable_randomization: Whether to apply randomization.
            use_arxiv: Whether to use ArXiv.
            use_openalex: Whether to use OpenAlex.
            use_pubmed: Whether to use PubMed.
            use_semantic_scholar: Whether to use Semantic Scholar.
            filter_open_access: Whether to keep only open-access papers.
        """
        self.min_year = min_year
        self.min_year_int = int(min_year.split('-')[0])
        self.phase1_min_year = phase1_min_year
        self.phase1_min_year_int = int(phase1_min_year.split('-')[0])
        self.enable_randomization = enable_randomization
        self.filter_open_access = filter_open_access

        # Initialize each search client
        self.clients = {}

        if use_arxiv:
            try:
                self.clients['arxiv'] = ArxivClient()
                logger.info("✓ ArXiv client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ArXiv client: {e}")

        if use_openalex:
            try:
                self.clients['openalex'] = OpenAlexClient(min_year=min_year)
                logger.info("✓ OpenAlex client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAlex client: {e}")

        if use_pubmed:
            try:
                email = os.getenv("NCBI_EMAIL")
                if not email:
                    logger.warning("NCBI_EMAIL environment variable not set; skipping PubMed")
                else:
                    self.clients['pubmed'] = PubMedClient(email=email)
                    logger.info("✓ PubMed client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PubMed client: {e}")

        if use_semantic_scholar:
            try:
                from camyla.model_config import get_api_key
                s2_api_key = get_api_key("s2")
                if not s2_api_key:
                    logger.warning("S2_API_KEY not set (config or env); skipping Semantic Scholar")
                else:
                    self.clients['semantic_scholar'] = SemanticScholarClient(
                        min_year=min_year,
                        phase1_min_year=phase1_min_year,
                        fields_of_study="Computer Science",
                        enable_venue_filter=True
                    )
                    logger.info("✓ Semantic Scholar client initialized (with top-venue filter)")
            except Exception as e:
                logger.warning(f"Failed to initialize Semantic Scholar client: {e}")

        if not self.clients:
            raise ValueError("At least one search source must be enabled")

        print(f"    🔍 Multi-source search initialized: {list(self.clients.keys())}")
        print(f"    📅 Time range: after {self.min_year_int} (Phase 1: after {self.phase1_min_year_int})")
        print(f"    🔓 Open-access filter: {'enabled' if filter_open_access else 'disabled'}")
        print(f"    🎲 Randomization: {'enabled' if enable_randomization else 'disabled'}")

    def find_papers_by_str(self, query: str, N: int = 20, disable_venue_filter: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Search for papers across multiple sources.

        Args:
            query: Search query.
            N: Target number of papers to return.
            disable_venue_filter: Whether to enable Phase 1 challenge-discovery mode:
                - disables top-venue filter,
                - disables fields-of-study filter (Semantic Scholar),
                - uses a more permissive year range (configured phase1 year).

        Returns:
            Optional[List[Dict]]: Structured paper list; None on failure.
        """
        phase1_mode = disable_venue_filter  # Keep legacy parameter name support

        print(f"\n    🔍 Multi-source search: {query}")
        print(f"    🎯 Target count: {N}")
        if phase1_mode:
            print(f"    ⚠️ Phase 1 mode: top-venue/fields-of-study filters disabled, year=phase1 min")

        # Temporarily disable venue filter by saving and overriding the Semantic Scholar client state
        saved_venue_filter_state = None
        if phase1_mode and 'semantic_scholar' in self.clients:
            ss_client = self.clients['semantic_scholar']
            saved_venue_filter_state = ss_client.enable_venue_filter
            ss_client.enable_venue_filter = False

        # Use the configured phase1 year filter in Phase 1 mode
        saved_min_year_int = self.min_year_int
        if phase1_mode:
            self.min_year_int = self.phase1_min_year_int  # Use configured Phase 1 year

        try:
            all_papers = []

            # Search each source
            for source_name, client in self.clients.items():
                try:
                    print(f"\n    📚 Searching {source_name.upper()}...")

                    # Adjust search parameters per source
                    search_params = self._prepare_search_params(source_name, query, N)

                    # Propagate Phase 1 mode to Semantic Scholar
                    if source_name == 'semantic_scholar' and phase1_mode:
                        search_params['phase1_mode'] = True

                    # Run the search
                    papers = client.search(**search_params)

                    if papers:
                        # Filter by year
                        papers = [p for p in papers if p.year >= self.min_year_int]

                        # Tag with source
                        for paper in papers:
                            paper.metadata['source'] = source_name

                        all_papers.extend(papers)
                        print(f"       ✓ Found {len(papers)} papers (after {self.min_year_int})")
                    else:
                        print(f"       ✗ No papers found")

                except Exception as e:
                    print(f"       ❌ {source_name.upper()} search failed: {e}")
                    logger.error(f"{source_name} search failed: {e}")

            if not all_papers:
                print(f"\n    ❌ No papers found from any source")
                return None

            print(f"\n    📊 Search summary:")
            print(f"       • Total found: {len(all_papers)} papers")

            # Tally by source
            source_counts = {}
            for paper in all_papers:
                source = paper.metadata.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1

            for source, count in source_counts.items():
                print(f"       • {source.upper()}: {count} papers")

            # Open-access filter
            if self.filter_open_access:
                all_papers = self._filter_open_access(all_papers)
                print(f"       • After open-access filter: {len(all_papers)} papers")

            # Deduplicate
            all_papers = self._deduplicate_papers(all_papers)
            print(f"       • After deduplication: {len(all_papers)} papers")

            # Sort by citation count (when available)
            all_papers.sort(
                key=lambda p: p.metadata.get('citation_count', 0),
                reverse=True
            )

            # Apply randomization
            if self.enable_randomization and len(all_papers) > N:
                all_papers = self._apply_randomization(all_papers, N)
            elif len(all_papers) > N:
                # Even with randomization disabled, shuffle for diversity
                random.shuffle(all_papers)
                all_papers = all_papers[:N]
                print(f"       • Shuffled and truncated to top {N}")

            # Convert to structured format
            structured_papers = self._convert_to_structured_format(all_papers)

            print(f"\n    ✅ Final return: {len(structured_papers)} papers")
            return structured_papers

        finally:
            # Restore the Semantic Scholar client's venue filter state
            if saved_venue_filter_state is not None and 'semantic_scholar' in self.clients:
                self.clients['semantic_scholar'].enable_venue_filter = saved_venue_filter_state
            # Restore the year setting
            self.min_year_int = saved_min_year_int

    def _prepare_search_params(self, source_name: str, query: str, N: int) -> Dict[str, Any]:
        """Prepare search parameters.

        Args:
            source_name: Source name.
            query: Search query.
            N: Target count.

        Returns:
            Dict: Search parameters.
        """
        # Base parameters
        params = {
            'query': query,
            'limit': N * 2  # Over-fetch by 2x to ensure enough papers
        }

        # PubMed special handling
        if source_name == 'pubmed':
            # Convert YYYY-MM-DD to YYYY/MM/DD
            mindate = self.min_year.replace('-', '/')
            params['mindate'] = mindate

            # Add open-access filter if enabled
            if self.filter_open_access:
                params['query'] = f"{query} AND free full text[filter]"

        return params

    def _filter_open_access(self, papers: List[Paper]) -> List[Paper]:
        """Filter for open-access papers.

        Args:
            papers: Paper list.

        Returns:
            List[Paper]: Filtered paper list.
        """
        open_access_papers = []

        for paper in papers:
            source = paper.metadata.get('source', '')

            if source == 'arxiv':
                # ArXiv is fully open-access
                open_access_papers.append(paper)
            elif source == 'openalex':
                # Check for an open-access URL
                oa_url = paper.metadata.get('open_access_url', '') or ''
                if oa_url.strip():
                    open_access_papers.append(paper)
            elif source == 'pubmed':
                # Results from PubMed's free-full-text filter are all open-access;
                # alternatively, DOIs pointing to open-access journals could be checked.
                open_access_papers.append(paper)
            else:
                # Unknown source — keep conservatively
                open_access_papers.append(paper)

        return open_access_papers

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Deduplicate a list of papers.

        Priority: DOI > ArXiv ID > (title, first author).

        Args:
            papers: Paper list.

        Returns:
            List[Paper]: Deduplicated paper list.
        """
        seen = set()
        unique_papers = []
        duplicate_count = 0

        for paper in papers:
            # Try multiple dedup keys
            keys = []

            # 1. DOI (most reliable)
            doi = paper.metadata.get('doi', '') or ''
            doi = doi.lower().strip() if doi else ''
            if doi and doi.startswith('http'):
                # Normalize DOI URL
                doi = doi.split('doi.org/')[-1] if 'doi.org/' in doi else doi
            if doi:
                keys.append(('doi', doi))

            # 2. ArXiv ID
            arxiv_id = paper.metadata.get('arxiv_id', '') or ''
            arxiv_id = arxiv_id.strip() if arxiv_id else ''
            if arxiv_id:
                # Strip version suffix
                arxiv_id = arxiv_id.split('v')[0]
                keys.append(('arxiv', arxiv_id))

            # 3. PMID
            pmid = paper.metadata.get('pmid', '') or ''
            pmid = pmid.strip() if pmid else ''
            if pmid:
                keys.append(('pmid', pmid))

            # 4. OpenAlex ID
            openalex_id = paper.metadata.get('openalex_id', '') or ''
            openalex_id = openalex_id.strip() if openalex_id else ''
            if openalex_id:
                keys.append(('openalex', openalex_id))

            # 5. Title + first author (last resort)
            title_key = paper.title.lower().strip()
            author_key = paper.authors[0].lower().strip() if paper.authors else ''
            keys.append(('title_author', (title_key, author_key)))

            # Check if already seen
            is_duplicate = False
            for key in keys:
                if key in seen:
                    is_duplicate = True
                    duplicate_count += 1
                    break

            if not is_duplicate:
                # Record all keys
                for key in keys:
                    seen.add(key)
                unique_papers.append(paper)

        if duplicate_count > 0:
            logger.info(f"Dedup: removed {duplicate_count} duplicate papers")

        return unique_papers

    def _apply_randomization(self, papers: List[Paper], target_count: int) -> List[Paper]:
        """Apply randomization.

        Args:
            papers: Paper list.
            target_count: Target count.

        Returns:
            List[Paper]: Randomized paper list.
        """
        original_count = len(papers)

        print(f"       • Randomization: selecting {target_count} out of {original_count}")

        # Random selection
        selected_papers = random.sample(papers, target_count)

        # Random shuffle
        random.shuffle(selected_papers)

        return selected_papers

    def _convert_to_structured_format(self, papers: List[Paper]) -> List[Dict[str, Any]]:
        """Convert Paper objects to structured dicts (legacy-compatible interface).

        Args:
            papers: List of Paper objects.

        Returns:
            List[Dict]: Structured paper list.
        """
        structured_papers = []

        for paper in papers:
            # Extract ID
            paper_id = self._extract_paper_id(paper)

            # Extract URL
            url = self._extract_url(paper)

            # Build structured entry
            structured = {
                'id': paper_id,
                'title': paper.title,
                'abstract': paper.abstract,
                'publication_date': f"{paper.year}-01-01",  # Simplified date format
                'authors': paper.authors,
                'url': url,
                'source': paper.metadata.get('source', 'unknown'),
                'citation_count': paper.metadata.get('citation_count', 0),
                'venue': paper.venue,
                'year': paper.year
            }

            structured_papers.append(structured)

        return structured_papers

    def _extract_paper_id(self, paper: Paper) -> str:
        """Extract a paper ID.

        Args:
            paper: Paper object.

        Returns:
            str: Paper ID.
        """
        source = paper.metadata.get('source', '')

        if source == 'arxiv':
            return paper.metadata.get('arxiv_id', 'unknown')
        elif source == 'pubmed':
            return paper.metadata.get('pmid', 'unknown')
        elif source == 'openalex':
            openalex_id = paper.metadata.get('openalex_id', '')
            # Extract the ID (drop URL prefix)
            if openalex_id:
                return openalex_id.split('/')[-1]
            return 'unknown'
        elif source == 'semantic_scholar':
            return paper.metadata.get('semantic_scholar_id', 'unknown')
        else:
            return 'unknown'

    def _extract_url(self, paper: Paper) -> str:
        """Extract a paper URL.

        Args:
            paper: Paper object.

        Returns:
            str: Paper URL.
        """
        source = paper.metadata.get('source', '')

        if source == 'arxiv':
            return paper.metadata.get('pdf_url', '')
        elif source == 'pubmed':
            return paper.metadata.get('pubmed_url', '')
        elif source == 'openalex':
            # Prefer an open-access URL
            oa_url = paper.metadata.get('open_access_url', '')
            if oa_url:
                return oa_url
            # Otherwise use DOI
            doi = paper.metadata.get('doi', '')
            if doi:
                return doi if doi.startswith('http') else f"https://doi.org/{doi}"
            return ''
        elif source == 'semantic_scholar':
            # Prefer an open-access URL
            oa_url = paper.metadata.get('open_access_url', '')
            if oa_url:
                return oa_url
            # Otherwise use the paper page URL
            return paper.metadata.get('url', '')
        else:
            return ''

    def retrieve_full_paper_text(self, paper_id: str, source: str = None, MAX_LEN: int = 90000) -> Optional[str]:
        """Retrieve the full text of a paper.

        Args:
            paper_id: Paper ID.
            source: Source name (arxiv/pubmed/openalex); auto-detected when None.
            MAX_LEN: Maximum text length.

        Returns:
            Optional[str]: Full text, or None on failure.
        """
        # Auto-detect source
        if source is None:
            source = self._detect_source_from_id(paper_id)

        if source not in self.clients:
            logger.warning(f"Source {source} is not available")
            return None

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = self.clients[source]
                full_text = client.get_full_text(paper_id)

                if full_text and len(full_text) > MAX_LEN:
                    full_text = full_text[:MAX_LEN]

                return full_text

            except Exception as e:
                logger.error(f"Full-text fetch failed ({source}:{paper_id}), attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    delay = 3 * (attempt + 1)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    return None

        return None

    def _detect_source_from_id(self, paper_id: str) -> str:
        """Detect a source from an ID.

        Args:
            paper_id: Paper ID.

        Returns:
            str: Source name.
        """
        # ArXiv ID format: YYMM.NNNNN or arch-ive/YYMMNNN
        if '.' in paper_id and paper_id[0].isdigit():
            return 'arxiv'

        # PubMed ID: all digits
        if paper_id.isdigit():
            return 'pubmed'

        # OpenAlex ID: starts with W
        if paper_id.startswith('W'):
            return 'openalex'

        # Semantic Scholar ID: 40-char hex string
        if len(paper_id) == 40 and all(c in '0123456789abcdefABCDEF' for c in paper_id):
            return 'semantic_scholar'

        # Default: try arxiv
        return 'arxiv'
