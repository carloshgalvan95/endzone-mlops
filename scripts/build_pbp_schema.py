#!/usr/bin/env python3
"""
Build the fixed pbp target schema from real nflverse data.

Downloads representative seasons (1999, 2012, 2025 at minimum) and unifies
their schemas using wider-type rules, then saves as JSON for use in ingestion.

Usage:
    python scripts/build_pbp_schema.py [--seasons 1999 2012 2025]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

NFLVERSE_PBP_URL_PATTERN = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
)


def download_season(season: int) -> pa.Table:
    """Download one season's pbp parquet."""
    url = NFLVERSE_PBP_URL_PATTERN.format(season=season)
    print(f"Downloading season {season} from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pq.read_table(pa.BufferReader(response.content))


def unify_schemas(tables: list[pa.Table]) -> pa.Schema:
    """
    Create unified schema from multiple pyarrow Tables.

    Uses wider-type rules: prefers string over all, float64 over int, int64 over int32.
    """
    column_types: dict[str, set[pa.DataType]] = defaultdict(set)

    for table in tables:
        for field in table.schema:
            column_types[field.name].add(field.type)

    unified_fields = []
    for col_name in sorted(column_types.keys()):
        types = column_types[col_name]

        if len(types) == 1:
            chosen_type = list(types)[0]
        else:
            if pa.string() in types or pa.large_string() in types:
                chosen_type = pa.string()
            elif any(pa.types.is_floating(t) for t in types):
                chosen_type = pa.float64()
            elif any(pa.types.is_integer(t) for t in types):
                chosen_type = pa.int64()
            else:
                chosen_type = pa.string()

        unified_fields.append(pa.field(col_name, chosen_type))

    return pa.schema(unified_fields)


def schema_to_json(schema: pa.Schema) -> dict:
    """Convert pyarrow schema to JSON-serializable dict."""
    return {field.name: str(field.type) for field in schema}


def main() -> int:
    """Build and save pbp schema."""
    parser = argparse.ArgumentParser(description="Build pbp target schema from real data")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[1999, 2012, 2025],
        help="Seasons to sample for schema (default: 1999 2012 2025)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/ingest/schemas/pbp_schema.json"),
        help="Output path for schema JSON",
    )

    args = parser.parse_args()

    print(f"Building pbp schema from seasons: {args.seasons}")

    tables = []
    for season in args.seasons:
        try:
            table = download_season(season)
            print(f"  Season {season}: {len(table.schema)} columns")
            tables.append(table)
        except Exception as e:
            print(f"  Failed to download season {season}: {e}")
            return 1

    if not tables:
        print("No tables downloaded, cannot build schema")
        return 1

    unified_schema = unify_schemas(tables)
    print(f"\nUnified schema: {len(unified_schema)} columns")

    schema_dict = schema_to_json(unified_schema)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(schema_dict, f, indent=2)

    print(f"Saved schema to {args.output}")
    print(f"\nFirst 10 columns: {list(schema_dict.keys())[:10]}")
    print(f"Last 10 columns: {list(schema_dict.keys())[-10:]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
