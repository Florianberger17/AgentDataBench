from pathlib import Path

import pytest

from agentdatabench.domain.dataset import Dataset


def test_dataset_sniffs_comma_delimiter(pkg1_root):
    dataset = Dataset(pkg1_root / "data" / "dataset.csv")
    assert list(dataset.df.columns) == [
        "CustNo",
        "CompanyName",
        "StreetAddress",
        "HouseNo",
        "ZipCode",
        "Town",
        "CountryCode",
        "LastOrderDate",
        "CustomerType",
    ]


def test_dataset_sniffs_semicolon_delimiter(pkg1_root):
    dataset = Dataset(pkg1_root / "ground_truth" / "ground_truth.csv")
    assert list(dataset.df.columns) == [
        "CustomerID",
        "Name",
        "Street",
        "City",
        "PostalCode",
        "Country",
        "LastBusinessActivityDate",
        "Status",
    ]


def test_dataset_df_is_cached(pkg1_root):
    dataset = Dataset(pkg1_root / "data" / "dataset.csv")
    assert dataset.df is dataset.df


def test_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Dataset(Path("/nonexistent/dataset.csv"))
