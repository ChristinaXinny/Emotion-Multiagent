"""Coordinator Agent (Agent C) for orchestrating multi-agent workflow."""

from typing import Dict, Any, List
import logging
from .base_agent import BaseAgent
from .agent_a_perception import PerceptionAgent
from .agent_b_inference import InferenceAgent


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent responsible for orchestrating the multi-agent workflow.

    Coordinates between PerceptionAgent and InferenceAgent to produce final sentiment analysis.
    """

    def __init__(self, name: str = "CoordinatorAgent", config: Dict[str, Any] = None, logger = None):
        """
        Initialize Coordinator Agent.

        Args:
            name: Agent name
            config: Configuration dictionary containing sub-agent configs
            logger: Logger instance
        """
        super().__init__(name, config, logger)

        # Initialize sub-agents
        self.perception_agent = None
        self.inference_agent = None

        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize sub-agents."""
        try:
            # Initialize PerceptionAgent
            perception_config = self.config.get("perception_agent", {})
            self.perception_agent = PerceptionAgent(
                config=perception_config,
                logger=self.logger
            )

            # Initialize InferenceAgent
            inference_config = self.config.get("inference_agent", {})
            self.inference_agent = InferenceAgent(
                config=inference_config,
                logger=self.logger
            )

            self.logger.info("All sub-agents initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize sub-agents: {str(e)}")
            raise

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input through multi-agent pipeline.

        Args:
            input_data: Input text(s) or data

        Returns:
            Combined results from all agents
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")

        self.logger.info(f"Starting multi-agent processing for input")

        try:
            # Stage 1: Perception
            self.logger.info("Stage 1: Perception (FinBERT)")
            perception_result = self.perception_agent.process(input_data)

            # Stage 2: Inference
            self.logger.info("Stage 2: Inference (LLM)")
            inference_result = self.inference_agent.process(perception_result)

            # Stage 3: Coordination and Final Output
            self.logger.info("Stage 3: Coordination")
            final_result = self._coordinate_results(perception_result, inference_result)

            return final_result

        except Exception as e:
            self.logger.error(f"Error in multi-agent processing: {str(e)}")
            raise

    def _coordinate_results(self, perception_result: Dict[str, Any], inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate and combine results from sub-agents.

        Args:
            perception_result: Result from PerceptionAgent
            inference_result: Result from InferenceAgent

        Returns:
            Coordinated final result
        """
        # Combine results
        final_result = {
            "text": perception_result.get("text", ""),
            "perception": {
                "sentiment": perception_result.get("sentiment", ""),
                "confidence": perception_result.get("confidence", 0.0),
                "scores": perception_result.get("scores", {})
            },
            "inference": {
                "reasoning": inference_result.get("reasoning", ""),
                "final_sentiment": inference_result.get("final_sentiment", ""),
                "confidence": inference_result.get("confidence", 0.0),
                "factors": inference_result.get("factors", [])
            },
            "final_assessment": self._make_final_assessment(perception_result, inference_result),
            "agent": self.name
        }

        return final_result

    def _make_final_assessment(self, perception_result: Dict[str, Any], inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final sentiment assessment by combining both agents' outputs.

        Args:
            perception_result: Result from PerceptionAgent
            inference_result: Result from InferenceAgent

        Returns:
            Final assessment dictionary
        """
        # Get sentiments
        perception_sentiment = perception_result.get("sentiment", "").lower()
        inference_sentiment = inference_result.get("final_sentiment", "").lower()

        # Get confidences
        perception_confidence = perception_result.get("confidence", 0.0)
        inference_confidence = inference_result.get("confidence", 0.0)

        # Simple combination strategy (can be enhanced)
        if perception_sentiment == inference_sentiment:
            # Agents agree
            final_sentiment = perception_sentiment
            final_confidence = (perception_confidence + inference_confidence) / 2
            agreement = True
        else:
            # Agents disagree, prefer inference with higher confidence
            if inference_confidence > perception_confidence:
                final_sentiment = inference_sentiment
                final_confidence = inference_confidence
            else:
                final_sentiment = perception_sentiment
                final_confidence = perception_confidence
            agreement = False

        return {
            "sentiment": final_sentiment,
            "confidence": final_confidence,
            "agreement": agreement,
            "strategy": "weighted_confidence"
        }

    def batch_process(self, input_list: List[Any]) -> List[Dict[str, Any]]:
        """
        Process multiple inputs through multi-agent pipeline.

        Args:
            input_list: List of inputs

        Returns:
            List of coordinated results
        """
        results = []

        for i, item in enumerate(input_list):
            self.logger.info(f"Processing item {i+1}/{len(input_list)}")
            try:
                result = self.process(item)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process item {i+1}: {str(e)}")
                results.append({
                    "error": str(e),
                    "agent": self.name
                })

        return results

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, str):
            return len(input_data.strip()) > 0

        if isinstance(input_data, dict):
            return "text" in input_data and len(input_data["text"].strip()) > 0

        if isinstance(input_data, list):
            return len(input_data) > 0

        return False

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the multi-agent workflow."""
        return {
            "coordinator": self.get_info(),
            "perception_agent": self.perception_agent.get_info() if self.perception_agent else None,
            "inference_agent": self.inference_agent.get_info() if self.inference_agent else None,
            "workflow_stages": ["Perception", "Inference", "Coordination"]
        }
