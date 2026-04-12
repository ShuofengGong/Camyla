from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class Paper:
    """Paper data class."""
    title: str
    authors: List[str]
    year: int
    venue: str
    abstract: str
    citation_key: str
    bibtex: str
    metadata: Dict[str, Any]

class BaseLiteratureAPI(ABC):
    """Base literature API interface."""
    
    @abstractmethod
    def search(self,
              query: str,
              limit: int = 10,
              **kwargs) -> List[Paper]:
        """Search for papers."""
        pass
    
    @abstractmethod
    def format_citation(self, paper: Paper, style: str = "bibtex") -> str:
        """Format a citation."""
        pass
    
    @abstractmethod
    def get_full_text(self, paper_id: str) -> Optional[str]:
        """Retrieve the full text of a paper."""
        pass 