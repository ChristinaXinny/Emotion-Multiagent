"""Retry mechanism with exponential backoff."""

import time
import logging
from functools import wraps
from typing import Callable, Any, Type, Tuple


def retry_with_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: logging.Logger = None
) -> Callable:
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Multiplier for backoff time after each retry
        exceptions: Tuple of exceptions to catch
        logger: Logger instance for logging retry attempts

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            backoff = initial_backoff

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1

                    if retries > max_retries:
                        if logger:
                            logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise

                    if logger:
                        logger.warning(
                            f"Attempt {retries}/{max_retries} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {backoff:.1f}s..."
                        )

                    time.sleep(backoff)
                    backoff *= backoff_factor

        return wrapper
    return decorator
