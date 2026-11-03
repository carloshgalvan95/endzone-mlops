"""
Data ingestion module for loading raw NFL data into Databricks.

This module contains scripts for:
- Downloading nflverse datasets (games, play-by-play)
- Uploading to Unity Catalog Volumes
- Creating bronze Delta tables in workspace.nfl_raw schema
"""
