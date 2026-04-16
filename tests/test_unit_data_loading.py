"""
Unit tests for the data loading stage of the HeartLink pipeline.

Validates Requirements 1.1, 1.2, 1.4.
"""

import pandas as pd
import pytest

from src.pipeline import EXPECTED_COLUMNS, load_data


class TestLoadDataSuccess:
    """Tests for successful CSV loading with correct shape and columns."""

    def test_loads_dataframe_with_correct_columns(self):
        """Requirement 1.2 — DataFrame contains exactly the 16 expected columns."""
        df = load_data("data/heart_disease_uci.csv")
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_loads_dataframe_with_expected_shape(self):
        """Requirement 1.1 — DataFrame has 920 rows and 16 columns."""
        df = load_data("data/heart_disease_uci.csv")
        assert df.shape == (920, 16)

    def test_returns_pandas_dataframe(self):
        """Requirement 1.1 — load_data returns a pandas DataFrame."""
        df = load_data("data/heart_disease_uci.csv")
        assert isinstance(df, pd.DataFrame)


class TestLoadDataFileNotFound:
    """Tests for FileNotFoundError on missing file."""

    def test_raises_file_not_found_for_missing_path(self):
        """Requirement 1.4 — FileNotFoundError raised when CSV is absent."""
        with pytest.raises(FileNotFoundError, match="not_a_real_file.csv"):
            load_data("not_a_real_file.csv")


class TestLoadDataColumnValidation:
    """Tests for ValueError on incorrect column schema."""

    def test_raises_value_error_on_wrong_columns(self, tmp_path):
        """Requirement 1.2 — ValueError raised when columns don't match."""
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Column mismatch"):
            load_data(str(bad_csv))

    def test_raises_value_error_on_extra_column(self, tmp_path):
        """Requirement 1.2 — ValueError raised when an extra column is present."""
        bad_csv = tmp_path / "extra.csv"
        data = {col: [0] for col in EXPECTED_COLUMNS}
        data["extra_col"] = [0]
        pd.DataFrame(data).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Column mismatch"):
            load_data(str(bad_csv))

    def test_raises_value_error_on_missing_column(self, tmp_path):
        """Requirement 1.2 — ValueError raised when a column is missing."""
        bad_csv = tmp_path / "missing.csv"
        cols = EXPECTED_COLUMNS[:-1]  # drop 'num'
        data = {col: [0] for col in cols}
        pd.DataFrame(data).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Column mismatch"):
            load_data(str(bad_csv))
