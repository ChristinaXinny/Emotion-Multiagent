"""Data collection module for financial news data."""

import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging


class DataCollector:
    """
    Collect financial news data from various sources.

    Supports:
    - CSV files
    - API sources (to be implemented)
    - Database sources (to be implemented)
    """

    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize DataCollector.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

    def load_from_csv(self, file_path: str, text_column: str = "text",
                      date_column: str = "date", **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            text_column: Name of text column
            date_column: Name of date column
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with loaded data
        """
        try:
            self.logger.info(f"Loading data from {file_path}")

            df = pd.read_csv(file_path, **kwargs)

            # Validate required columns
            required_columns = [text_column]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            self.logger.info(f"Loaded {len(df)} records from CSV")

            return df

        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise

    def load_from_multiple_sources(self, file_paths: List[str],
                                   text_column: str = "text") -> pd.DataFrame:
        """
        Load and combine data from multiple CSV files.

        Args:
            file_paths: List of CSV file paths
            text_column: Name of text column

        Returns:
            Combined DataFrame
        """
        dataframes = []

        for file_path in file_paths:
            try:
                df = self.load_from_csv(file_path, text_column=text_column)
                df['source_file'] = Path(file_path).name
                dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {str(e)}")

        if not dataframes:
            raise ValueError("No data loaded from any source")

        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Combined {len(combined_df)} records from {len(dataframes)} sources")

        return combined_df

    def save_to_csv(self, data: pd.DataFrame, file_path: str,
                    index: bool = False) -> None:
        """
        Save data to CSV file.

        Args:
            data: DataFrame to save
            file_path: Output file path
            index: Whether to save index
        """
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            data.to_csv(file_path, index=index)
            self.logger.info(f"Saved {len(data)} records to {file_path}")

        except Exception as e:
            self.logger.error(f"Error saving to CSV: {str(e)}")
            raise

    def sample_data(self, data: pd.DataFrame, n: int = None,
                    fraction: float = None, random_state: int = 42) -> pd.DataFrame:
        """
        Sample data from DataFrame.

        Args:
            data: Input DataFrame
            n: Number of samples (if specified, takes precedence over fraction)
            fraction: Fraction of data to sample
            random_state: Random seed

        Returns:
            Sampled DataFrame
        """
        if n:
            return data.sample(n=min(n, len(data)), random_state=random_state)
        elif fraction:
            return data.sample(frac=min(fraction, 1.0), random_state=random_state)
        else:
            return data
