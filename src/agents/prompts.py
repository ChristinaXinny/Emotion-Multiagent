"""Prompt templates for InferenceAgent."""

from typing import Dict, List

# System prompt for financial sentiment analysis
FINANCIAL_SENTIMENT_SYSTEM_PROMPT = """You are an expert financial analyst specializing in sentiment analysis of financial news and market data. Your task is to:

1. Analyze financial text for sentiment indicators
2. Consider contextual factors that might influence sentiment
3. Provide reasoning for your sentiment assessment
4. Identify key factors driving the sentiment

Key aspects to consider:
- Financial terminology and their connotations
- Market sentiment indicators (bullish/bearish language)
- Company-specific vs. market-wide sentiment
- Temporal aspects (short-term vs. long-term impacts)
- Risk and opportunity signals
"""

# Template for single text analysis
SENTIMENT_ANALYSIS_TEMPLATE = """Analyze the following financial news text and provide a comprehensive sentiment assessment.

Text: {text}

Initial Model Analysis:
- Predicted Sentiment: {sentiment}
- Confidence: {confidence:.3f}
- Score Breakdown: {scores}

Please provide:
1. Your analysis and reasoning
2. Final sentiment classification (positive/negative/neutral)
3. Confidence level (0.0 to 1.0)
4. Key factors influencing this sentiment (3-5 factors)

Format your response as:
---
Analysis: <detailed reasoning>
Final Sentiment: <positive/negative/neutral>
Confidence: <0.0-1.0>
Key Factors: <factor1>, <factor2>, <factor3>, ...
---
"""

# Template for batch analysis with context
BATCH_SENTIMENT_TEMPLATE = """Analyze the following set of financial news texts and provide sentiment assessments for each.

Context: {context}

Number of texts: {num_texts}

{texts_data}

For each text, provide:
1. Analysis
2. Final Sentiment
3. Confidence
4. Key Factors

Format your response with clear separators for each text.
"""

# Template for comparative analysis
COMPARATIVE_ANALYSIS_TEMPLATE = """Compare and analyze the sentiment across these related financial texts.

Relationship: {relationship}

Texts:
{texts_pairs}

Provide:
1. Individual sentiment analysis for each text
2. Comparative analysis (how sentiments relate or contrast)
3. Overall combined sentiment assessment
4. Notable patterns or discrepancies
"""


def get_sentiment_analysis_prompt(text: str, sentiment: str, confidence: float, scores: Dict[str, float]) -> str:
    """
    Generate prompt for sentiment analysis.

    Args:
        text: Input text
        sentiment: Initial sentiment prediction
        confidence: Confidence score
        scores: Sentiment score breakdown

    Returns:
        Formatted prompt string
    """
    scores_str = ", ".join([f"{k}: {v:.3f}" for k, v in scores.items()])

    return SENTIMENT_ANALYSIS_TEMPLATE.format(
        text=text,
        sentiment=sentiment,
        confidence=confidence,
        scores=scores_str
    )


def get_batch_analysis_prompt(texts_data: List[Dict], context: str = "") -> str:
    """
    Generate prompt for batch sentiment analysis.

    Args:
        texts_data: List of dictionaries containing text and initial analysis
        context: Additional context information

    Returns:
        Formatted prompt string
    """
    texts_formatted = ""
    for i, item in enumerate(texts_data, 1):
        texts_formatted += f"""
Text {i}:
{item.get('text', '')}
Initial Sentiment: {item.get('sentiment', 'unknown')}
---
"""

    return BATCH_SENTIMENT_TEMPLATE.format(
        context=context or "No additional context",
        num_texts=len(texts_data),
        texts_data=texts_formatted
    )
