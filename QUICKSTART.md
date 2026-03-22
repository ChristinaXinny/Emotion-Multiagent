# Quick Start Guide

## Setup Instructions

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Anthropic API key
# ANTHROPIC_API_KEY=your_actual_api_key_here
```

### 3. Run the System

**Interactive Mode** (analyze single texts):
```bash
python main.py --mode interactive
```

**Pipeline Mode** (process CSV file):
```bash
python main.py --mode pipeline --input data/raw/your_data.csv
```

## Project Structure Overview

```
emotion_multiagent/
├── src/
│   ├── agents/           # Multi-agent implementation
│   ├── data/             # Data collection & preprocessing
│   ├── features/         # Feature engineering
│   ├── evaluation/       # Metrics & evaluation
│   └── utils/            # Utilities (logging, config, retry)
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Test suite
└── main.py              # Entry point
```

## Key Components

### Three-Agent Architecture

1. **PerceptionAgent** (Agent A)
   - Uses FinBERT for initial sentiment extraction
   - Model: `ProsusAI/finbert`

2. **InferenceAgent** (Agent B)
   - Uses Claude for reasoning and inference
   - Requires: Anthropic API key

3. **CoordinatorAgent** (Agent C)
   - Orchestrates the multi-agent workflow
   - Combines outputs from both agents

## Next Steps

1. Prepare your data in CSV format with a `text` column
2. Place it in `data/raw/`
3. Run the pipeline: `python main.py --mode pipeline --input data/raw/your_file.csv`
4. Check `outputs/features/` for generated sentiment features

## Development

- Explore notebooks in `notebooks/` directory
- Run tests: `pytest tests/`
- Customize settings in `config/config.yaml`

## Troubleshooting

**Issue**: Model download fails
- **Solution**: Ensure you have internet connection and sufficient disk space

**Issue**: Anthropic API error
- **Solution**: Verify your API key in `.env` file

**Issue**: CUDA out of memory
- **Solution**: Set `device: "cpu"` in `config/config.yaml`

For more details, see [README.md](README.md)
