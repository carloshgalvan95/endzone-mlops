"""
Data loading utilities for endzone-mlops.

Provides helper functions for downloading, caching, and validating data sources.
"""

import hashlib
from pathlib import Path

import pandas as pd


def download_with_cache(
    url: str,
    cache_path: Path,
    force_refresh: bool = False,
) -> Path:
    """
    Download a file with local caching.

    Args:
        url: URL to download from
        cache_path: Local path to cache the file
        force_refresh: If True, re-download even if cached

    Returns:
        Path to cached file

    Example:
        >>> cache_dir = Path("./data/cache")
        >>> file_path = download_with_cache(
        ...     "https://example.com/data.parquet",
        ...     cache_dir / "data.parquet"
        ... )
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        return cache_path

    # pandas.read_parquet can read directly from URLs
    # For actual download, use requests or urllib in production
    # For now, pandas handles it
    df = pd.read_parquet(url)
    df.to_parquet(cache_path)

    return cache_path


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file for integrity checking.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        Hexadecimal hash string

    Example:
        >>> hash_val = compute_file_hash(Path("data.parquet"))
        >>> print(f"SHA256: {hash_val}")
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: list[str],
    df_name: str = "DataFrame",
) -> None:
    """
    Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name for error messages

    Raises:
        ValueError: If required columns are missing

    Example:
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> validate_dataframe_schema(df, ["a", "b"], "test_data")
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def load_parquet_safe(
    path_or_url: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Safely load Parquet file with error handling.

    Args:
        path_or_url: Local path or URL to Parquet file
        columns: Optional list of columns to load (for efficiency)

    Returns:
        DataFrame

    Raises:
        IOError: If file cannot be loaded

    Example:
        >>> df = load_parquet_safe("data/games.parquet", columns=["game_id", "season"])
    """
    try:
        df = pd.read_parquet(path_or_url, columns=columns)
        return df
    except Exception as e:
        raise OSError(f"Failed to load Parquet from {path_or_url}: {e}") from e
