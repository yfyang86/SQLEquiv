# sql_equivalence/representations/base.py
"""Base class for query representations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json

class QueryRepresentation(ABC):
    """Abstract base class for different query representations."""
    
    def __init__(self, parsed_query: 'ParsedQuery'):
        """
        Initialize query representation.
        
        Args:
            parsed_query: Parsed query object from parser module
        """
        self.parsed_query = parsed_query
        self._built = False
        self._representation = None
    
    @abstractmethod
    def build(self) -> None:
        """Build the representation from parsed query."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert representation to dictionary format."""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert representation to string format."""
        pass
    
    @abstractmethod
    def visualize(self, output_path: Optional[str] = None) -> Any:
        """
        Visualize the representation.
        
        Args:
            output_path: Optional path to save visualization
            
        Returns:
            Visualization object or path to saved file
        """
        pass
    
    def to_json(self) -> str:
        """Convert representation to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def is_built(self) -> bool:
        """Check if representation has been built."""
        return self._built
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(built={self._built})"