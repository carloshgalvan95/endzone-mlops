"""
Load raw nflverse data into Databricks Free Edition.

Downloads nflverse datasets (games, play-by-play, weekly depth charts) from
official GitHub releases, uploads to Unity Catalog Volumes, and creates
bronze Delta tables.

Usage:
    python -m endzone.ingest.load_raw_nflverse --seasons 2022 2023 2024
    python -m endzone.ingest.load_raw_nflverse --seasons 2022 2023 --dataset pbp

License: nflverse data is CC BY 4.0
Source: https://github.com/nflverse/nflverse-data
"""

import sys

from endzone.ingest.cli import main

if __name__ == "__main__":
    sys.exit(main())
