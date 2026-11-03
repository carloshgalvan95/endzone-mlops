"""Fixed play-by-play Arrow schema and the cast onto that schema."""

import json
from pathlib import Path

import pyarrow as pa

from endzone.utils.logging import setup_logger

logger = setup_logger(__name__)

_ARROW_TYPES: dict[str, pa.DataType] = {
    "int64": pa.int64(),
    "int32": pa.int32(),
    "float64": pa.float64(),
    "float32": pa.float32(),
    "string": pa.string(),
    "bool": pa.bool_(),
    "double": pa.float64(),
}


def _parse_arrow_type(type_str: str) -> pa.DataType:
    """Parse an Arrow type name from the schema JSON."""
    return _ARROW_TYPES.get(type_str.strip(), pa.string())


def load_target_schema() -> pa.Schema:
    """Load the fixed pbp target schema from JSON.

    Returns:
        Target pyarrow Schema for pbp data.
    """
    schema_path = Path(__file__).parent / "schemas" / "pbp_schema.json"
    with open(schema_path) as handle:
        schema_dict = json.load(handle)

    fields = []
    for col_name, type_str in schema_dict.items():
        fields.append(pa.field(col_name, _parse_arrow_type(type_str)))
    return pa.schema(fields)


def conform_table_to_schema(table: pa.Table, target_schema: pa.Schema, season: int) -> pa.Table:
    """Conform a pyarrow Table to a target schema.

    Adds missing columns as null, removes extra columns, and casts types to match target.
    Raises error on cast failures instead of silently nulling.

    Args:
        table: Input pyarrow Table.
        target_schema: Target schema to conform to.
        season: Season number (for error messages).

    Returns:
        Conformed pyarrow Table.

    Raises:
        ValueError: If a column type cannot be cast to the target type.
    """
    arrays = []
    names = []

    extra_columns = set(table.column_names) - {field.name for field in target_schema}
    if extra_columns:
        logger.warning(
            f"Season {season}: dropping {len(extra_columns)} columns not in target schema: "
            f"{sorted(list(extra_columns)[:10])}" + (" ..." if len(extra_columns) > 10 else "")
        )

    for field in target_schema:
        if field.name in table.column_names:
            col = table.column(field.name)
            if col.type != field.type:
                try:
                    col = col.cast(field.type, safe=False)
                except pa.ArrowInvalid as exc:
                    raise ValueError(
                        f"Season {season}: cannot cast column '{field.name}' "
                        f"from {col.type} to {field.type}: {exc}"
                    ) from exc
            arrays.append(col)
        else:
            arrays.append(pa.nulls(len(table), type=field.type))
        names.append(field.name)

    return pa.table(arrays, names=names)
