# Emotion Multi-Agent System

A multi-agent system for financial sentiment analysis combining FinBERT (perception) and Large Language Models (inference).

## Architecture

The system uses a three-agent architecture:

### Agent A: Perception Agent (FinBERT)
- Extracts initial sentiment from financial text
- Uses the FinBERT model (ProsusAI/finbert)
- Provides sentiment scores: positive, neutral, negative

### Agent B: Inference Agent (LLM)
- Performs reasoning and context-based inference
- Uses Claude (Anthropic API)
- Provides analysis, confidence, and key factors

### Agent C: Coordinator Agent
- Orchestrates the multi-agent workflow
- Combines outputs from both agents
- Makes final sentiment assessment

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd emotion_multiagent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Project Structure

```
emotion_multiagent/
├── data/
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed data
│   └── external/                 # External dictionaries/models
├── src/
│   ├── agents/                   # Multi-agent modules
│   │   ├── base_agent.py         # Base agent class
│   │   ├── agent_a_perception.py # FinBERT agent
│   │   ├── agent_b_inference.py  # LLM agent
│   │   ├── agent_c_coordinator.py # Coordinator agent
│   │   └── prompts.py            # Prompt templates
│   ├── data/                     # Data handling
│   │   ├── collector.py          # Data collection
│   │   ├── preprocessor.py       # Data preprocessing
│   │   └── stock_mapper.py       # News-stock mapping
│   ├── features/                 # Feature engineering
│   │   └── sentiment_features.py # Sentiment features
│   ├── evaluation/               # Evaluation metrics
│   │   └── metrics.py            # Metrics calculator
│   └── utils/                    # Utilities
│       ├── config.py             # Configuration
│       ├── logger.py             # Logging
│       └── retry.py              # Retry mechanism
├── config/
│   └── config.yaml               # Configuration file
├── outputs/
│   ├── features/                 # Generated features
│   └── logs/                     # Log files
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test files
├── main.py                       # Main entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Usage

### Interactive Mode

Run the system in interactive mode for single text analysis:

```bash
python main.py --mode interactive
```

### Pipeline Mode

Run batch processing on a CSV file:

```bash
python main.py --mode pipeline --input data/raw/financial_news.csv
```

### Custom Configuration

Use a custom configuration file:

```bash
python main.py --config config/custom_config.yaml
```

## Configuration

Edit `config/config.yaml` to customize:

- **Model settings**: FinBERT model, LLM model choice
- **Data processing**: Text length limits, split ratios
- **Features**: Rolling windows, momentum periods
- **Output**: Log and feature directories

## Features

### Sentiment Features
- Daily sentiment scores
- Rolling averages (3, 7, 14, 30 days)
- Momentum indicators
- Volatility metrics
- Extreme sentiment flags

### Evaluation Metrics
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Inter-agent agreement
- Cohen's Kappa

## Example Output

```
Processing record 1/10
------------------------------------------------------------
ANALYSIS RESULTS
------------------------------------------------------------
Text: Apple reports strong quarterly earnings, stock surges 5%

Final Assessment:
  Sentiment:  positive
  Confidence: 0.875
  Agreement:  True

Perception (FinBERT):
  Sentiment: positive
  Confidence: 0.985

Inference (LLM):
  Reasoning: Strong earnings report and stock surge indicate positive sentiment...
  Sentiment: positive
  Factors: earnings growth, stock performance, company strength
```

## Development

### Running Tests

```bash
pytest tests/
```

### Jupyter Notebooks

Explore data and prototype agents:

```bash
jupyter notebook notebooks/
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- Anthropic API key

## License

[Your License Here]

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

[Your Contact Information]
