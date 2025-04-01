"""BaseAnalyst class."""

from abc import ABCMeta, abstractmethod
from typing import List, Optional


class BaseAnalyst(metaclass=ABCMeta):
    """BaseAnalyst class."""

    def __init__(self):
        """Initializes BaseAnalyst objects."""

    @abstractmethod
    def query_analyst(self, message: str, history: Optional[List[str]] = None) -> str:
        """Queries the analyst."""
        raise NotImplementedError
