"""Base agent class for all agents in the multi-agent system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Attributes:
        name: Agent name
        logger: Logger instance
        config: Agent configuration
    """

    def __init__(self, name: str, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize base agent.

        Args:
            name: Agent name
            config: Agent configuration dictionary
            logger: Logger instance
        """
        self.name = name
        self.config = config or {}
        self.logger = logger or logging.getLogger(f"agent.{name}")

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and return results.

        Args:
            input_data: Input data to process

        Returns:
            Processed output data
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input data to validate

        Returns:
            True if valid, False otherwise
        """
        return input_data is not None

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information.

        Returns:
            Dictionary containing agent information
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
