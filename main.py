#!/usr/bin/env python3
"""
Emotion Multi-Agent System - Main Entry Point

Financial sentiment analysis using a multi-agent architecture:
- Agent A (Perception): FinBERT for initial sentiment extraction
- Agent B (Inference): LLM for reasoning and inference
- Agent C (Coordinator): Orchestrates the multi-agent workflow
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.agents.agent_c_coordinator import CoordinatorAgent
from src.features.sentiment_features import SentimentFeatureBuilder
from src.evaluation.metrics import MetricsCalculator


def run_pipeline(config_path: str = None, input_file: str = None):
    """
    Run the complete multi-agent pipeline.

    Args:
        config_path: Path to configuration file
        input_file: Path to input data file
    """
    # Setup
    config = load_config(config_path)
    logger = setup_logger(
        name="emotion_multiagent",
        log_dir=config['output']['logs_dir'],
        level=logging.INFO
    )

    logger.info("Starting Emotion Multi-Agent System Pipeline")
    logger.info(f"Configuration loaded from: {config_path or 'config/config.yaml'}")

    try:
        # Step 1: Load data
        logger.info("=" * 60)
        logger.info("Step 1: Loading data")
        logger.info("=" * 60)

        collector = DataCollector(config=config['data'], logger=logger)

        if input_file:
            df = collector.load_from_csv(input_file)
        else:
            logger.warning("No input file provided. Using sample data...")
            # Create sample data
            import pandas as pd
            df = pd.DataFrame({
                'text': [
                    "Apple reports strong quarterly earnings, stock surges 5%",
                    "Market volatility increases as investors worry about inflation",
                    "Tech stocks rally on positive economic data"
                ],
                'date': ['2024-01-01', '2024-01-02', '2024-01-03']
            })

        logger.info(f"Loaded {len(df)} records")

        # Step 2: Preprocess data
        logger.info("=" * 60)
        logger.info("Step 2: Preprocessing data")
        logger.info("=" * 60)

        preprocessor = DataPreprocessor(config=config['data'], logger=logger)
        df_processed = preprocessor.preprocess_dataframe(df)

        # Step 3: Run multi-agent sentiment analysis
        logger.info("=" * 60)
        logger.info("Step 3: Multi-agent sentiment analysis")
        logger.info("=" * 60)

        coordinator = CoordinatorAgent(config=config['agents'], logger=logger)

        # Process a sample of data
        sample_size = min(10, len(df_processed))
        df_sample = df_processed.head(sample_size)

        results = []
        for idx, row in df_sample.iterrows():
            logger.info(f"Processing record {idx + 1}/{sample_size}")
            result = coordinator.process(row['text'])
            results.append(result)

        # Step 4: Build features
        logger.info("=" * 60)
        logger.info("Step 4: Building sentiment features")
        logger.info("=" * 60)

        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                'text': r['text'],
                'sentiment': r['final_assessment']['sentiment'],
                'confidence': r['final_assessment']['confidence']
            }
            for r in results
        ])

        results_df['date'] = df_sample['date'].values[:len(results_df)]

        feature_builder = SentimentFeatureBuilder(config=config['features'], logger=logger)
        features_df = feature_builder.build_all_features(
            results_df,
            date_column='date',
            sentiment_column='sentiment'
        )

        # Save features
        output_path = Path(config['output']['features_dir']) / 'sentiment_features.csv'
        feature_builder.save_features(features_df, str(output_path))

        # Step 5: Evaluation (if ground truth available)
        logger.info("=" * 60)
        logger.info("Step 5: Pipeline complete")
        logger.info("=" * 60)

        logger.info(f"✓ Processed {len(results)} records")
        logger.info(f"✓ Generated features for {len(features_df)} days")
        logger.info(f"✓ Features saved to {output_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


def run_interactive():
    """Run interactive mode for single text analysis."""
    import pandas as pd

    # Setup
    config = load_config()
    logger = setup_logger(name="emotion_multiagent")

    print("\n" + "=" * 60)
    print("Emotion Multi-Agent System - Interactive Mode")
    print("=" * 60 + "\n")

    # Initialize coordinator
    coordinator = CoordinatorAgent(config=config['agents'], logger=logger)

    while True:
        print("\nEnter financial text to analyze (or 'quit' to exit):")
        text = input("> ")

        if text.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not text.strip():
            continue

        try:
            print("\nProcessing...")
            result = coordinator.process(text)

            print("\n" + "-" * 60)
            print("ANALYSIS RESULTS")
            print("-" * 60)
            print(f"\nText: {result['text'][:100]}...")
            print(f"\nFinal Assessment:")
            print(f"  Sentiment:  {result['final_assessment']['sentiment']}")
            print(f"  Confidence: {result['final_assessment']['confidence']:.3f}")
            print(f"  Agreement:  {result['final_assessment']['agreement']}")
            print(f"\nPerception (FinBERT):")
            print(f"  Sentiment: {result['perception']['sentiment']}")
            print(f"  Confidence: {result['perception']['confidence']:.3f}")
            print(f"\nInference (LLM):")
            print(f"  Reasoning: {result['inference']['reasoning'][:200]}...")
            print(f"  Sentiment: {result['inference']['final_sentiment']}")
            print(f"  Factors: {', '.join(result['inference']['factors'][:3])}")

        except Exception as e:
            print(f"\nError: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Emotion Multi-Agent System for Financial Sentiment Analysis"
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['pipeline', 'interactive'],
        default='interactive',
        help='Run mode: pipeline (batch) or interactive (single text)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input CSV file (for pipeline mode)'
    )

    args = parser.parse_args()

    if args.mode == 'pipeline':
        run_pipeline(config_path=args.config, input_file=args.input)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
