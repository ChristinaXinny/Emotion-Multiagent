"""Data preprocessing module for cleaning and preparing text data."""

import pandas as pd
import re
from typing import List, Dict, Any, Optional
import logging


class DataPreprocessor:
    """
    Preprocess financial text data.

    Handles:
    - Text cleaning
    - Deduplication
    - Language filtering
    - Quality checks
    """

    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize DataPreprocessor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Preprocessing settings
        self.min_length = self.config.get("min_text_length", 50)
        self.max_length = self.config.get("max_text_length", 5000)

    def clean_text(self, text: str) -> str:
        """
        Clean individual text string.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-\'\"()]', ' ', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def preprocess_dataframe(self, df: pd.DataFrame,
                             text_column: str = "text") -> pd.DataFrame:
        """
        Preprocess DataFrame containing text data.

        Args:
            df: Input DataFrame
            text_column: Name of text column

        Returns:
            Preprocessed DataFrame
        """
        self.logger.info(f"Preprocessing {len(df)} records")

        df_processed = df.copy()

        # Clean text
        df_processed[text_column] = df_processed[text_column].apply(self.clean_text)

        # Remove empty texts
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed[text_column].str.len() > 0]
        removed_empty = initial_count - len(df_processed)

        # Filter by length
        df_processed = df_processed[
            (df_processed[text_column].str.len() >= self.min_length) &
            (df_processed[text_column].str.len() <= self.max_length)
        ]
        removed_by_length = initial_count - removed_empty - len(df_processed)

        # Remove duplicates
        initial_count_before_dedup = len(df_processed)
        df_processed = df_processed.drop_duplicates(subset=[text_column])
        removed_duplicates = initial_count_before_dedup - len(df_processed)

        self.logger.info(f"Preprocessing complete:")
        self.logger.info(f"  - Removed empty texts: {removed_empty}")
        self.logger.info(f"  - Removed by length: {removed_by_length}")
        self.logger.info(f"  - Removed duplicates: {removed_duplicates}")
        self.logger.info(f"  - Final records: {len(df_processed)}")

        return df_processed.reset_index(drop=True)

    def filter_by_keywords(self, df: pd.DataFrame, text_column: str = "text",
                          keywords: List[str] = None,
                          require_all: bool = False) -> pd.DataFrame:
        """
        Filter DataFrame by keyword presence.

        Args:
            df: Input DataFrame
            text_column: Name of text column
            keywords: List of keywords to filter by
            require_all: If True, require all keywords; if False, require any

        Returns:
            Filtered DataFrame
        """
        if not keywords:
            return df

        keywords_lower = [k.lower() for k in keywords]

        if require_all:
            mask = df[text_column].apply(
                lambda x: all(kw in x.lower() for kw in keywords_lower)
            )
        else:
            mask = df[text_column].apply(
                lambda x: any(kw in x.lower() for kw in keywords_lower)
            )

        filtered_df = df[mask]
        self.logger.info(f"Filtered to {len(filtered_df)} records containing keywords")

        return filtered_df.reset_index(drop=True)

    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7,
                   val_ratio: float = 0.15, test_ratio: float = 0.15,
                   random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train, validation, and test sets.

        Args:
            df: Input DataFrame
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Validate ratios
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        # Shuffle
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Calculate split points
        n = len(df_shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = df_shuffled.iloc[:train_end]
        val_df = df_shuffled.iloc[train_end:val_end]
        test_df = df_shuffled.iloc[val_end:]

        self.logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df
