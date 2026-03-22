"""News to stock mapper module."""

import pandas as pd
from typing import Dict, List, Optional, Set
import logging
import re


class StockMapper:
    """
    Map financial news to relevant stocks.

    Features:
    - Extract stock tickers from text
    - Match company names to tickers
    - Build news-stock relationships
    """

    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """
        Initialize StockMapper.

        Args:
            config: Configuration dictionary containing:
                - ticker_map: Dictionary mapping company names to tickers
                - common_tickers: Set of common stock tickers
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Load ticker mappings
        self.ticker_map = self.config.get("ticker_map", {})
        self.common_tickers = set(self.config.get("common_tickers", []))

        # Pattern for matching stock tickers (e.g., AAPL, MSFT)
        self.ticker_pattern = re.compile(r'\b[A-Z]{2,5}\b')

    def extract_tickers_from_text(self, text: str) -> Set[str]:
        """
        Extract potential stock tickers from text.

        Args:
            text: Input text

        Returns:
            Set of potential tickers
        """
        # Find all capital letter sequences
        potential_tickers = set(self.ticker_pattern.findall(text))

        # Filter to common tickers if available
        if self.common_tickers:
            return potential_tickers & self.common_tickers

        return potential_tickers

    def extract_company_names(self, text: str) -> List[str]:
        """
        Extract company names from text using ticker map.

        Args:
            text: Input text

        Returns:
            List of company names found
        """
        found_companies = []

        for company, ticker in self.ticker_map.items():
            # Check if company name appears in text
            if company.lower() in text.lower():
                found_companies.append(company)
            # Also check for ticker
            elif ticker in text:
                found_companies.append(company)

        return found_companies

    def map_news_to_stocks(self, df: pd.DataFrame,
                           text_column: str = "text") -> pd.DataFrame:
        """
        Map news articles to related stocks.

        Args:
            df: Input DataFrame
            text_column: Name of text column

        Returns:
            DataFrame with stock mappings added
        """
        self.logger.info(f"Mapping stocks for {len(df)} news articles")

        df_result = df.copy()

        # Extract tickers and companies
        tickers_list = []
        companies_list = []

        for text in df_result[text_column]:
            tickers = self.extract_tickers_from_text(text)
            companies = self.extract_company_names(text)

            tickers_list.append(list(tickers))
            companies_list.append(companies)

        # Add columns
        df_result['tickers'] = tickers_list
        df_result['companies'] = companies_list
        df_result['has_stock_reference'] = df_result['tickers'].apply(len) > 0

        # Log statistics
        with_stocks = df_result['has_stock_reference'].sum()
        self.logger.info(f"Found stock references in {with_stocks}/{len(df)} articles")

        return df_result

    def build_stock_sentiment_features(self, df: pd.DataFrame,
                                       sentiment_column: str = "sentiment") -> pd.DataFrame:
        """
        Build sentiment features grouped by stock.

        Args:
            df: DataFrame with sentiment and ticker mappings
            sentiment_column: Name of sentiment column

        Returns:
            DataFrame with sentiment features by stock
        """
        # Expand tickers to create stock-level rows
        stock_rows = []

        for _, row in df.iterrows():
            for ticker in row.get('tickers', []):
                stock_rows.append({
                    'ticker': ticker,
                    'sentiment': row.get(sentiment_column, ''),
                    'confidence': row.get('confidence', 0.0),
                    'date': row.get('date', None)
                })

        if not stock_rows:
            return pd.DataFrame()

        stock_df = pd.DataFrame(stock_rows)

        # Aggregate by ticker
        features = stock_df.groupby('ticker').agg({
            'sentiment': lambda x: x.value_counts().to_dict(),
            'confidence': 'mean',
            'date': 'count'
        }).rename(columns={'date': 'mention_count'})

        return features

    def load_ticker_map(self, file_path: str) -> None:
        """
        Load ticker mappings from file.

        Args:
            file_path: Path to file with company-ticker mappings
        """
        # Implementation depends on file format
        # This is a placeholder for future implementation
        pass

    def get_ticker_stats(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get statistics about ticker mentions.

        Args:
            df: DataFrame with ticker mappings

        Returns:
            Dictionary with ticker statistics
        """
        all_tickers = []
        for tickers in df.get('tickers', []):
            all_tickers.extend(tickers)

        if not all_tickers:
            return {}

        from collections import Counter
        ticker_counts = Counter(all_tickers)

        return dict(ticker_counts.most_common(20))
