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


def test_dataset_reads_bom_and_bomless_utf8_identically(tmp_path):
    content = "name,city\nDörr GmbH,Göttingen\n"
    with_bom = tmp_path / "with_bom.csv"
    with_bom.write_bytes(content.encode("utf-8-sig"))
    without_bom = tmp_path / "without_bom.csv"
    without_bom.write_bytes(content.encode("utf-8"))

    df_bom = Dataset(with_bom).df
    df_plain = Dataset(without_bom).df

    assert list(df_bom.columns) == ["name", "city"]
    assert df_bom.equals(df_plain)
    assert df_bom.loc[0, "city"] == "Göttingen"
