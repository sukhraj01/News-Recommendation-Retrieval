import pandas as pd
import pytest

from src.pipeline.schema import FieldSpec
from src.pipeline.validators import SchemaValidationError, validate_table

SCHEMA = {
    "id": FieldSpec(mandatory=True),
    "value": FieldSpec(mandatory=True),
    "mind_only_field": FieldSpec(mandatory=False, dataset_only="mind"),
    "shared_optional": FieldSpec(mandatory=False),
}


def _valid_df():
    return pd.DataFrame({
        "id": ["a", "b"],
        "value": [1, 2],
        "mind_only_field": [None, None],
        "shared_optional": [None, "x"],
    })


def test_valid_dataframe_passes():
    validate_table(_valid_df(), SCHEMA, dataset="ebnerd")


def test_missing_mandatory_field_raises():
    df = _valid_df()
    df.loc[0, "value"] = None
    with pytest.raises(SchemaValidationError, match="mandatory field 'value'"):
        validate_table(df, SCHEMA, dataset="ebnerd")


def test_dataset_only_field_populated_for_wrong_dataset_raises():
    df = _valid_df()
    df.loc[0, "mind_only_field"] = "should not be here"
    with pytest.raises(SchemaValidationError, match="dataset_only='mind'"):
        validate_table(df, SCHEMA, dataset="ebnerd")


def test_dataset_only_field_allowed_for_correct_dataset():
    df = _valid_df()
    df.loc[0, "mind_only_field"] = "fine here"
    validate_table(df, SCHEMA, dataset="mind")


def test_multiple_violations_all_reported():
    df = _valid_df()
    df.loc[0, "value"] = None
    df.loc[0, "mind_only_field"] = "bad"
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_table(df, SCHEMA, dataset="ebnerd")
    message = str(exc_info.value)
    assert "mandatory field 'value'" in message
    assert "dataset_only='mind'" in message


def test_missing_column_reported():
    df = _valid_df().drop(columns=["value"])
    with pytest.raises(SchemaValidationError, match="missing column: 'value'"):
        validate_table(df, SCHEMA, dataset="ebnerd")
