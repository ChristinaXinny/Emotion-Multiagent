"""Tests for agent modules."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.base_agent import BaseAgent
from src.agents.agent_a_perception import PerceptionAgent
from src.agents.agent_c_coordinator import CoordinatorAgent


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = BaseAgent(name="TestAgent", config={"key": "value"})
        assert agent.name == "TestAgent"
        assert agent.config == {"key": "value"}

    def test_validate_input(self):
        """Test input validation."""
        agent = BaseAgent(name="TestAgent")
        assert agent.validate_input("test") == True
        assert agent.validate_input(None) == False
        assert agent.validate_input("") == False

    def test_get_info(self):
        """Test getting agent info."""
        agent = BaseAgent(name="TestAgent", config={"key": "value"})
        info = agent.get_info()
        assert info["name"] == "TestAgent"
        assert "type" in info
        assert info["config"] == {"key": "value"}


class TestPerceptionAgent:
    """Tests for PerceptionAgent class."""

    @pytest.fixture
    def perception_agent(self):
        """Create a PerceptionAgent instance for testing."""
        # Use a smaller model for testing
        config = {
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "device": "cpu"
        }
        return PerceptionAgent(config=config)

    def test_initialization(self, perception_agent):
        """Test agent initialization."""
        assert perception_agent.name == "PerceptionAgent"
        assert perception_agent.model_name == "distilbert-base-uncased-finetuned-sst-2-english"

    def test_validate_input(self, perception_agent):
        """Test input validation."""
        assert perception_agent.validate_input("Test text") == True
        assert perception_agent.validate_input("") == False
        assert perception_agent.validate_input(None) == False
        assert perception_agent.validate_input(["Text 1", "Text 2"]) == True
        assert perception_agent.validate_input({"text": "Test"}) == True


class TestCoordinatorAgent:
    """Tests for CoordinatorAgent class."""

    def test_initialization_without_api_key(self):
        """Test that CoordinatorAgent requires API key for InferenceAgent."""
        # This should fail without ANTHROPIC_API_KEY
        with pytest.raises(ValueError, match="API key"):
            coordinator = CoordinatorAgent(config={})


def test_project_structure():
    """Test that project structure is correct."""
    project_root = Path(__file__).parent.parent

    # Check key directories exist
    assert (project_root / "src").exists()
    assert (project_root / "src" / "agents").exists()
    assert (project_root / "src" / "data").exists()
    assert (project_root / "src" / "features").exists()
    assert (project_root / "src" / "evaluation").exists()
    assert (project_root / "config").exists()
    assert (project_root / "data").exists()

    # Check key files exist
    assert (project_root / "main.py").exists()
    assert (project_root / "requirements.txt").exists()
    assert (project_root / "config" / "config.yaml").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
