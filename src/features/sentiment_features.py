"""Sentiment feature building module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging


class SentimentFeatureBuilder:
    """
    Build sentiment features for financial analysis.

    Features:
    - Daily sentiment scores
    - Rolling averages
    - Sentiment momentum
    - Volatility metrics
    """

    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize SentimentFeatureBuilder.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

    def build_sentiment_scores(self, df: pd.DataFrame,
                               sentiment_column: str = "sentiment",
                               confidence_column: str = "confidence",
                               date_column: str = "date") -> pd.DataFrame:
        """
        Build numeric sentiment scores from categorical labels.

        Args:
            df: Input DataFrame with sentiment labels
            sentiment_column: Name of sentiment column
            confidence_column: Name of confidence column
            date_column: Name of date column

        Returns:
            DataFrame with numeric sentiment scores added
        """
        df_result = df.copy()

        # Convert sentiment to numeric score
        sentiment_map = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }

        df_result['sentiment_score'] = df_result[sentiment_column].map(sentiment_map)

        # Weight by confidence
        if confidence_column in df_result.columns:
            df_result['weighted_sentiment'] = df_result['sentiment_score'] * df_result[confidence_column]
        else:
            df_result['weighted_sentiment'] = df_result['sentiment_score']

        self.logger.info(f"Built sentiment scores for {len(df_result)} records")

        return df_result

    def build_daily_features(self, df: pd.DataFrame,
                            date_column: str = "date",
                            sentiment_column: str = "weighted_sentiment") -> pd.DataFrame:
        """
        Aggregate sentiment features by date.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            sentiment_column: Name of sentiment column

        Returns:
            DataFrame with daily aggregated features
        """
        # Ensure date column is datetime
        df_result = df.copy()
        df_result[date_column] = pd.to_datetime(df_result[date_column])

        # Group by date and calculate features
        daily_features = df_result.groupby(df_result[date_column].dt.date).agg({
            sentiment_column: ['mean', 'std', 'count', 'min', 'max']
        }).reset_index()

        # Flatten column names
        daily_features.columns = ['date', 'sentiment_mean', 'sentiment_std',
                                  'article_count', 'sentiment_min', 'sentiment_max']

        # Calculate additional features
        daily_features['sentiment_range'] = daily_features['sentiment_max'] - daily_features['sentiment_min']

        # Fill NaN in std for single-article days
        daily_features['sentiment_std'] = daily_features['sentiment_std'].fillna(0)

        self.logger.info(f"Built daily features for {len(daily_features)} days")

        return daily_features

    def build_rolling_features(self, df: pd.DataFrame,
                               windows: List[int] = [3, 7, 14, 30],
                               sentiment_column: str = "sentiment_mean") -> pd.DataFrame:
        """
        Build rolling window features.

        Args:
            df: Input daily DataFrame
            windows: List of window sizes in days
            sentiment_column: Name of sentiment column

        Returns:
            DataFrame with rolling features added
        """
        df_result = df.copy()

        for window in windows:
            df_result[f'sentiment_ma_{window}'] = df_result[sentiment_column].rolling(
                window=window, min_periods=1
            ).mean()

            df_result[f'sentiment_std_{window}'] = df_result[sentiment_column].rolling(
                window=window, minperiods=1
            ).std().fillna(0)

        self.logger.info(f"Built rolling features for windows: {windows}")

        return df_result

    def build_momentum_features(self, df: pd.DataFrame,
                                sentiment_column: str = "sentiment_mean") -> pd.DataFrame:
        """
        Build sentiment momentum features.

        Args:
            df: Input daily DataFrame
            sentiment_column: Name of sentiment column

        Returns:
            DataFrame with momentum features added
        """
        df_result = df.copy()

        # Day-over-day change
        df_result['sentiment_change'] = df_result[sentiment_column].diff()
        df_result['sentiment_change_pct'] = df_result[sentiment_column].pct_change()

        # Momentum indicators
        for periods in [1, 3, 7]:
            df_result[f'momentum_{periods}d'] = df_result[sentiment_column] - \
                df_result[sentiment_column].shift(periods)

        self.logger.info("Built momentum features")

        return df_result

    def build_volatility_features(self, df: pd.DataFrame,
                                  sentiment_column: str = "sentiment_mean") -> pd.DataFrame:
        """
        Build sentiment volatility features.

        Args:
            df: Input daily DataFrame
            sentiment_column: Name of sentiment column

        Returns:
            DataFrame with volatility features added
        """
        df_result = df.copy()

        # Rolling volatility (std)
        for window in [7, 14, 30]:
            df_result[f'volatility_{window}d'] = df_result[sentiment_column].rolling(
                window=window, minperiods=1
            ).std().fillna(0)

        # Historical volatility percentile
        df_result['volatility_percentile'] = df_result['volatility_14d'].rank(pct=True)

        self.logger.info("Built volatility features")

        return df_result

    def build_extreme_features(self, df: pd.DataFrame,
                               sentiment_column: str = "sentiment_mean") -> pd.DataFrame:
        """
        Build extreme sentiment indicators.

        Args:
            df: Input daily DataFrame
            sentiment_column: Name of sentiment column

        Returns:
            DataFrame with extreme sentiment features added
        """
        df_result = df.copy()

        # Percentile ranks
        df_result['sentiment_percentile'] = df_result[sentiment_column].rank(pct=True)

        # Extreme indicators (top/bottom 10%)
        df_result['extreme_positive'] = (df_result['sentiment_percentile'] >= 0.9).astype(int)
        df_result['extreme_negative'] = (df_result['sentiment_percentile'] <= 0.1).astype(int)

        # Z-score
        mean_sentiment = df_result[sentiment_column].mean()
        std_sentiment = df_result[sentiment_column].std()
        df_result['sentiment_zscore'] = (df_result[sentiment_column] - mean_sentiment) / std_sentiment

        self.logger.info("Built extreme sentiment features")

        return df_result

    def build_all_features(self, df: pd.DataFrame,
                          date_column: str = "date",
                          sentiment_column: str = "sentiment") -> pd.DataFrame:
        """
        Build complete feature set.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            sentiment_column: Name of sentiment column

        Returns:
            DataFrame with all features built
        """
        # Step 1: Convert to numeric scores
        df_features = self.build_sentiment_scores(df, sentiment_column=sentiment_column, date_column=date_column)

        # Step 2: Aggregate by date
        df_daily = self.build_daily_features(df_features, date_column=date_column)

        # Step 3: Add rolling features
        df_daily = self.build_rolling_features(df_daily)

        # Step 4: Add momentum features
        df_daily = self.build_momentum_features(df_daily)

        # Step 5: Add volatility features
        df_daily = self.build_volatility_features(df_daily)

        # Step 6: Add extreme features
        df_daily = self.build_extreme_features(df_daily)

        self.logger.info(f"Built complete feature set with {len(df_daily.columns)} features")

        return df_daily

    def save_features(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Save features to CSV.

        Args:
            df: DataFrame with features
            file_path: Output file path
        """
        try:
            df.to_csv(file_path, index=False)
            self.logger.info(f"Saved features to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving features: {str(e)}")
            raise
