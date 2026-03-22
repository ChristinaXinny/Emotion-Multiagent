"""Perception Agent (Agent A) using FinBERT for sentiment analysis."""

from typing import Dict, Any, List, Union
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from .base_agent import BaseAgent


class PerceptionAgent(BaseAgent):
    """
    Perception Agent responsible for extracting sentiment from financial text.

    Uses FinBERT (Financial BERT) model for sentiment classification.
    """

    def __init__(self, name: str = "PerceptionAgent", config: Dict[str, Any] = None, logger = None):
        """
        Initialize Perception Agent with FinBERT model.

        Args:
            name: Agent name
            config: Configuration dictionary containing:
                - model_name: HuggingFace model name (default: "ProsusAI/finbert")
                - device: Device to run model on ("cpu" or "cuda")
                - batch_size: Batch size for inference
            logger: Logger instance
        """
        super().__init__(name, config, logger)

        self.model_name = self.config.get("model_name", "ProsusAI/finbert")
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config.get("batch_size", 8)

        self.logger.info(f"Initializing PerceptionAgent with model: {self.model_name}")

        # Initialize model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load FinBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )

            self.logger.info(f"Model loaded successfully on device: {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def process(self, input_data: Union[str, List[str], Dict[str, Any]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process input text and extract sentiment.

        Args:
            input_data: Input text(s) or dict containing 'text' field

        Returns:
            Sentiment analysis results with scores for positive, negative, neutral
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")

        # Handle different input formats
        if isinstance(input_data, str):
            texts = [input_data]
            single_input = True
        elif isinstance(input_data, list):
            texts = input_data
            single_input = False
        elif isinstance(input_data, dict):
            texts = [input_data.get("text", "")]
            single_input = True
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Run inference
        try:
            results = self.pipeline(texts)

            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                sentiment_scores = {item['label']: item['score'] for item in result}

                # Determine dominant sentiment
                dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])

                formatted_results.append({
                    "text": texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],
                    "sentiment": dominant_sentiment[0],
                    "confidence": dominant_sentiment[1],
                    "scores": sentiment_scores,
                    "agent": self.name
                })

            return formatted_results[0] if single_input else formatted_results

        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            raise

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if input_data is None:
            return False

        if isinstance(input_data, str):
            return len(input_data.strip()) > 0

        if isinstance(input_data, list):
            return len(input_data) > 0 and all(isinstance(t, str) and len(t.strip()) > 0 for t in input_data)

        if isinstance(input_data, dict):
            return "text" in input_data and len(input_data["text"].strip()) > 0

        return False

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batches.

        Args:
            texts: List of input texts

        Returns:
            List of sentiment analysis results
        """
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self.process(batch)
            results.extend(batch_results)

        return results
