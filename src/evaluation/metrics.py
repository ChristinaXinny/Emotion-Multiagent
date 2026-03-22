"""Evaluation metrics module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import logging


class MetricsCalculator:
    """
    Calculate evaluation metrics for sentiment analysis.

    Metrics:
    - Classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix
    - Agreement metrics
    - Correlation metrics
    """

    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """
        Initialize MetricsCalculator.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Label mapping
        self.label_map = {
            'positive': 2,
            'neutral': 1,
            'negative': 0
        }

    def calculate_classification_metrics(self,
                                         y_true: List[str],
                                         y_pred: List[str],
                                         average: str = 'weighted') -> Dict[str, Any]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method ('micro', 'macro', 'weighted')

        Returns:
            Dictionary of metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(support) if not np.isnan(support) else 0
        }

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        labels = ['negative', 'neutral', 'positive']
        for i, label in enumerate(labels):
            if i < len(precision_per_class):
                metrics[f'{label}_precision'] = float(precision_per_class[i])
                metrics[f'{label}_recall'] = float(recall_per_class[i])
                metrics[f'{label}_f1'] = float(f1_per_class[i])

        return metrics

    def calculate_confusion_matrix(self,
                                   y_true: List[str],
                                   y_pred: List[str]) -> Dict[str, Any]:
        """
        Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with confusion matrix data
        """
        cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])

        return {
            'confusion_matrix': cm.tolist(),
            'true_negative': int(cm[0, 0]),
            'false_neutral': int(cm[0, 1]),
            'false_positive': int(cm[0, 2]),
            'false_negative': int(cm[2, 0]),
            'false_neutral_from_negative': int(cm[2, 1]),
            'true_positive': int(cm[2, 2])
        }

    def calculate_agreement_metrics(self,
                                     predictions_a: List[str],
                                     predictions_b: List[str]) -> Dict[str, Any]:
        """
        Calculate agreement metrics between two sets of predictions.

        Args:
            predictions_a: First set of predictions
            predictions_b: Second set of predictions

        Returns:
            Dictionary of agreement metrics
        """
        if len(predictions_a) != len(predictions_b):
            raise ValueError("Predictions must have same length")

        # Exact agreement
        exact_agreement = sum(1 for a, b in zip(predictions_a, predictions_b) if a == b)
        agreement_rate = exact_agreement / len(predictions_a)

        # Cohen's Kappa (using sklearn)
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(predictions_a, predictions_b)

        return {
            'exact_agreement_count': int(exact_agreement),
            'exact_agreement_rate': float(agreement_rate),
            'cohens_kappa': float(kappa),
            'disagreement_count': int(len(predictions_a) - exact_agreement)
        }

    def calculate_correlation_metrics(self,
                                      scores_a: List[float],
                                      scores_b: List[float]) -> Dict[str, Any]:
        """
        Calculate correlation metrics between two score sets.

        Args:
            scores_a: First set of scores
            scores_b: Second set of scores

        Returns:
            Dictionary of correlation metrics
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")

        # Pearson correlation
        pearson = np.corrcoef(scores_a, scores_b)[0, 1]

        # Spearman correlation
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(scores_a, scores_b)

        return {
            'pearson_correlation': float(pearson) if not np.isnan(pearson) else 0.0,
            'spearman_correlation': float(spearman) if not np.isnan(spearman) else 0.0
        }

    def evaluate_agent_performance(self,
                                   results_df: pd.DataFrame,
                                   true_column: str = "true_sentiment",
                                   pred_column: str = "predicted_sentiment") -> Dict[str, Any]:
        """
        Comprehensive evaluation of agent performance.

        Args:
            results_df: DataFrame with true and predicted labels
            true_column: Name of true label column
            pred_column: Name of predicted label column

        Returns:
            Dictionary with comprehensive metrics
        """
        y_true = results_df[true_column].tolist()
        y_pred = results_df[pred_column].tolist()

        # Classification metrics
        class_metrics = self.calculate_classification_metrics(y_true, y_pred)

        # Confusion matrix
        conf_matrix = self.calculate_confusion_matrix(y_true, y_pred)

        # Combine all metrics
        all_metrics = {
            **class_metrics,
            **conf_matrix
        }

        # Add sample count
        all_metrics['sample_count'] = len(results_df)

        return all_metrics

    def compare_agents(self,
                       results_df: pd.DataFrame,
                       agent_columns: List[str]) -> Dict[str, Any]:
        """
        Compare performance across multiple agents.

        Args:
            results_df: DataFrame with predictions from multiple agents
            agent_columns: List of column names for each agent's predictions
            true_column: Name of true label column

        Returns:
            Dictionary comparing agent performance
        """
        true_column = "true_sentiment"

        comparison = {}

        # Evaluate each agent
        for agent_col in agent_columns:
            if agent_col in results_df.columns:
                metrics = self.evaluate_agent_performance(
                    results_df,
                    true_column=true_column,
                    pred_column=agent_col
                )
                comparison[agent_col] = metrics

        # Pairwise agreement
        agreement_data = {}
        for i, agent_a in enumerate(agent_columns):
            for agent_b in agent_columns[i+1:]:
                if agent_a in results_df.columns and agent_b in results_df.columns:
                    agreement = self.calculate_agreement_metrics(
                        results_df[agent_a].tolist(),
                        results_df[agent_b].tolist()
                    )
                    agreement_data[f"{agent_a}_vs_{agent_b}"] = agreement

        comparison['pairwise_agreement'] = agreement_data

        return comparison

    def generate_report(self, metrics: Dict[str, Any], title: str = "Sentiment Analysis Report") -> str:
        """
        Generate human-readable metrics report.

        Args:
            metrics: Dictionary of metrics
            title: Report title

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            f"{title}",
            "=" * 60,
            ""
        ]

        # Overall metrics
        if 'accuracy' in metrics:
            report_lines.extend([
                "Overall Performance:",
                f"  Accuracy:  {metrics['accuracy']:.4f}",
                f"  Precision: {metrics['precision']:.4f}",
                f"  Recall:    {metrics['recall']:.4f}",
                f"  F1 Score:  {metrics['f1_score']:.4f}",
                ""
            ])

        # Per-class metrics
        for sentiment in ['positive', 'neutral', 'negative']:
            if f'{sentiment}_f1' in metrics:
                report_lines.extend([
                    f"{sentiment.capitalize()} Class:",
                    f"  Precision: {metrics[f'{sentiment}_precision']:.4f}",
                    f"  Recall:    {metrics[f'{sentiment}_recall']:.4f}",
                    f"  F1:        {metrics[f'{sentiment}_f1']:.4f}",
                    ""
                ])

        # Agreement metrics if available
        if 'exact_agreement_rate' in metrics:
            report_lines.extend([
                "Agreement Metrics:",
                f"  Agreement Rate: {metrics['exact_agreement_rate']:.4f}",
                f"  Cohen's Kappa:  {metrics['cohens_kappa']:.4f}",
                ""
            ])

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def save_metrics(self, metrics: Dict[str, Any], file_path: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
            file_path: Output file path
        """
        import json
        from pathlib import Path

        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            self.logger.info(f"Saved metrics to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise
