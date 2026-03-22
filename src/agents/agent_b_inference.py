"""Inference Agent (Agent B) using LLM for reasoning and inference."""

from typing import Dict, Any, List
import os
from anthropic import Anthropic
from .base_agent import BaseAgent


class InferenceAgent(BaseAgent):
    """
    Inference Agent responsible for reasoning and context-based inference.

    Uses LLM (Claude) to provide deeper analysis and inference on sentiment data.
    """

    def __init__(self, name: str = "InferenceAgent", config: Dict[str, Any] = None, logger = None):
        """
        Initialize Inference Agent with LLM client.

        Args:
            name: Agent name
            config: Configuration dictionary containing:
                - api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
                - model: Model name (default: "claude-3-5-sonnet-20241022")
                - max_tokens: Maximum tokens in response
                - temperature: Sampling temperature
            logger: Logger instance
        """
        super().__init__(name, config, logger)

        self.api_key = self.config.get("api_key", os.getenv("ANTHROPIC_API_KEY"))
        self.model = self.config.get("model", "claude-3-5-sonnet-20241022")
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.3)

        if not self.api_key:
            raise ValueError("Anthropic API key not provided in config or ANTHROPIC_API_KEY environment variable")

        # Initialize client
        self.client = Anthropic(api_key=self.api_key)
        self.logger.info(f"InferenceAgent initialized with model: {self.model}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process sentiment data and provide inference.

        Args:
            input_data: Dictionary containing:
                - text: Original text
                - sentiment: Sentiment from PerceptionAgent
                - scores: Sentiment scores
                - context: Additional context (optional)

        Returns:
            Dictionary with inference results including:
                - reasoning: Chain of thought reasoning
                - final_sentiment: Final sentiment assessment
                - confidence: Confidence in assessment
                - factors: Key factors influencing sentiment
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")

        # Construct prompt
        prompt = self._construct_prompt(input_data)

        try:
            # Call LLM
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Parse response
            result = self._parse_response(response, input_data)

            return result

        except Exception as e:
            self.logger.error(f"Error during LLM inference: {str(e)}")
            raise

    def _construct_prompt(self, input_data: Dict[str, Any]) -> str:
        """Construct prompt for LLM."""
        text = input_data.get("text", "")
        sentiment = input_data.get("sentiment", "unknown")
        scores = input_data.get("scores", {})
        context = input_data.get("context", {})

        prompt = f"""You are a financial sentiment analysis expert. Analyze the following financial news text and provide your inference.

Text: {text}

Initial Sentiment Analysis (from FinBERT):
- Sentiment: {sentiment}
- Scores: {scores}

Additional Context:
{context if context else "No additional context provided"}

Please provide:
1. Your reasoning process
2. Final sentiment assessment (positive/negative/neutral)
3. Confidence level (0-1)
4. Key factors influencing your assessment

Format your response as:
Reasoning: <your reasoning>
Final Sentiment: <positive/negative/neutral>
Confidence: <0-1>
Factors: <comma-separated list of key factors>
"""

        return prompt

    def _parse_response(self, response: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response and extract structured data."""
        content = response.content[0].text

        # Parse structured response (simple implementation)
        result = {
            "text": input_data.get("text", ""),
            "original_sentiment": input_data.get("sentiment", ""),
            "reasoning": "",
            "final_sentiment": "",
            "confidence": 0.0,
            "factors": [],
            "agent": self.name,
            "raw_response": content
        }

        # Extract information from response
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Reasoning:"):
                result["reasoning"] = line.replace("Reasoning:", "").strip()
            elif line.startswith("Final Sentiment:"):
                result["final_sentiment"] = line.replace("Final Sentiment:", "").strip().lower()
            elif line.startswith("Confidence:"):
                try:
                    result["confidence"] = float(line.replace("Confidence:", "").strip())
                except:
                    pass
            elif line.startswith("Factors:"):
                factors_str = line.replace("Factors:", "").strip()
                result["factors"] = [f.strip() for f in factors_str.split(',')]

        return result

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            return False

        required_fields = ["text", "sentiment", "scores"]
        return all(field in input_data for field in required_fields)

    def batch_process(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple inputs.

        Args:
            input_list: List of input dictionaries

        Returns:
            List of inference results
        """
        results = []
        for item in input_list:
            try:
                result = self.process(item)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process item: {str(e)}")
                results.append({
                    "error": str(e),
                    "agent": self.name
                })

        return results
