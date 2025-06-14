# sql_equivalence/equivalence/base.py
"""Base classes for equivalence checking."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

class EquivalenceType(Enum):
    """Types of equivalence."""
    EXACT = "exact"                    # Exact structural match
    SEMANTIC = "semantic"              # Semantically equivalent
    APPROXIMATE = "approximate"        # Approximately equivalent
    NOT_EQUIVALENT = "not_equivalent"  # Not equivalent

@dataclass
class EquivalenceResult:
    """Result of equivalence check."""
    is_equivalent: bool
    equivalence_type: EquivalenceType
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    proof_steps: List[str] = field(default_factory=list)
    
    def add_proof_step(self, step: str) -> None:
        """Add a proof step."""
        self.proof_steps.append(step)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_equivalent': self.is_equivalent,
            'equivalence_type': self.equivalence_type.value,
            'confidence': self.confidence,
            'details': self.details,
            'execution_time': self.execution_time,
            'proof_steps': self.proof_steps
        }

class EquivalenceChecker(ABC):
    """Abstract base class for equivalence checkers."""
    
    def __init__(self):
        self.rules = []
        self.config = {}
    
    @abstractmethod
    def check_equivalence(self, query1: Any, query2: Any) -> EquivalenceResult:
        """
        Check if two queries are equivalent.
        
        Args:
            query1: First query representation
            query2: Second query representation
            
        Returns:
            EquivalenceResult object
        """
        pass
    
    @abstractmethod
    def compute_similarity(self, query1: Any, query2: Any) -> float:
        """
        Compute similarity score between two queries.
        
        Args:
            query1: First query representation
            query2: Second query representation
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        pass
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set configuration parameters."""
        self.config.update(config)
    
    def add_rule(self, rule: Any) -> None:
        """Add a transformation or equivalence rule."""
        self.rules.append(rule)
    
    def clear_rules(self) -> None:
        """Clear all rules."""
        self.rules.clear()