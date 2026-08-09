"""Schema validation for the unified feature-store tables.

Hand-rolled rather than pandera/pydantic: the checks needed are simple
boolean-mask predicates over a DataFrame, and the pytest suite required by
ADR-002 asserts these same properties anyway — a validation library would be
redundant with tests that must exist regardless. Collects every violation
before raising, rather than failing on the first, for a useful error message.
"""
import numpy as np
import pandas as pd

from .schema import FieldSpec


class SchemaValidationError(Exception):
    pass


def _is_null(series: pd.Series) -> pd.Series:
    """True where a value is null (None/NaN/NaT).

    A container (list/array) is never null, even when empty — an empty
    history/entities list is a legitimate value ("this user has no prior
    clicks"), not a missing field. Conflating the two would make every
    no-history user look like a schema violation.
    """
    def is_null_value(v):
        if v is None:
            return True
        if isinstance(v, (list, tuple, np.ndarray)):
            return False
        try:
            return bool(pd.isna(v))
        except (TypeError, ValueError):
            return False

    return series.map(is_null_value)


def validate_table(df: pd.DataFrame, schema: dict[str, FieldSpec], dataset: str) -> None:
    violations: list[str] = []

    for field, spec in schema.items():
        if field not in df.columns:
            violations.append(f"missing column: '{field}'")
            continue

        col = df[field]

        if spec.mandatory:
            n_null = int(_is_null(col).sum())
            if n_null:
                violations.append(
                    f"mandatory field '{field}' has {n_null} null value(s)"
                )

        if spec.dataset_only is not None and spec.dataset_only != dataset:
            n_populated = int((~_is_null(col)).sum())
            if n_populated:
                violations.append(
                    f"field '{field}' is marked dataset_only='{spec.dataset_only}' "
                    f"but has {n_populated} non-null value(s) in a '{dataset}' table"
                )

    if violations:
        raise SchemaValidationError(
            f"Schema validation failed for dataset='{dataset}' "
            f"({len(violations)} violation(s)):\n  - " + "\n  - ".join(violations)
        )
