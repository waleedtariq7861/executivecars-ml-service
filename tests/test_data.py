import pandas as pd

from app.data import clean_dataframe, parse_number


def test_pakistani_price_parsing():
    assert parse_number("PKR 45 Lacs") == 4_500_000
    assert parse_number("1.2 Crore") == 12_000_000


def test_cleaning_removes_duplicates_and_invalid_rows():
    valid = {"make": "Toyota", "model": "Corolla", "year": 2020, "engine": 1300, "mileage": 45_000, "price": 4_500_000}
    frame, report = clean_dataframe(pd.DataFrame([valid, valid, {**valid, "year": 1970}]))
    assert len(frame) == 1
    assert report["duplicateRecords"] == 1
    assert report["rejectionReasons"]["invalid_model_year"] == 1


def test_cleaning_does_not_retain_personal_seller_fields():
    frame, _ = clean_dataframe(pd.DataFrame([{
        "make": "Honda", "model": "City", "year": 2021, "mileage": 30_000,
        "price": 4_000_000, "sellerPhone": "+92 300 0000000", "sellerName": "Private Person",
    }]))
    assert "sellerPhone" not in frame.columns
    assert "sellerName" not in frame.columns
