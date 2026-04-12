from .base import BaseLiteratureAPI, Paper
from .arxiv_client import ArxivClient
from .openalex_client import OpenAlexClient
from .pubmed_client import PubMedClient
from .multi_source_search import MultiSourceLiteratureSearch
# Import SemanticScholarClient from the correct location in tools/ssapi
from camyla.tools.ssapi.semantic_scholar_client import SemanticScholarClient

__all__ = [
    'BaseLiteratureAPI',
    'Paper',
    'ArxivClient',
    'OpenAlexClient',
    'SemanticScholarClient',
    'PubMedClient',
    'MultiSourceLiteratureSearch'
]
