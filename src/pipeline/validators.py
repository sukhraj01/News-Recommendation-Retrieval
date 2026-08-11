"""Schema validation for the unified feature-store tables.

Hand-rolled rather than pandera/pydantic: the checks needed are simple
boolean-mask predicates over a DataFrame, and the pytest suite required by
ADR-002 asserts these same properties anyway — a validation library would be
redundant with tests that must exist regardless. Collects every violation
before raising, rather than failing on the first, for a useful error message.
"""
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

    `series.isna()` (pandas' own vectorized, C-level null check) already has
    exactly this semantics — verified directly:
    `pd.Series([None, [], [1,2], nan, NaT, "x"]).isna()` returns `[True,
    False, False, True, True, False]`. The previous implementation
    reimplemented this by hand via `series.map(a_python_function)`, a
    per-row Python call that was measured to be the actual bottleneck at
    MINDlarge scale (~81M rows): a real build ran for over an hour with no
    forward progress, and sampling the stuck process showed it spending
    essentially all of that time inside pandas' `map_infer_mask` internals
    for this one call. Same category of naive-loop-doesn't-scale bug as
    ADR-006's BM25 `get_scores()` fix and this session's
    `_explode_impressions` fix — found the same way, by sampling the actual
    stuck process rather than guessing."""
    return series.isna()


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
