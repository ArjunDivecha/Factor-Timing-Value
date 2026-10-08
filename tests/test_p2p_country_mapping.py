"""Tests for p2p_country_mapping.py"""

import pandas as pd
import pytest

from p2p_country_mapping import (
    load_p2p_country_frame_from_excel,
    load_ticker_country_pairs,
    validate_p2p_excel_column_order,
)


ASSET_LIST = "AssetList.xlsx"


def test_asset_list_has_34_tickers():
    pairs = load_ticker_country_pairs(ASSET_LIST)
    assert len(pairs) == 34
    assert pairs[0][0] == "EWS"
    assert pairs[7][0] == "QQQ"
    assert pairs[7][1] == "NASDAQ"


def test_map_by_ticker_header_not_position():
    pairs = load_ticker_country_pairs(ASSET_LIST)
    # Deliberately scrambled column order; mapping must still be correct.
    scrambled = ["Date", "SPY", "QQQ", "EWS"]
    rows = {
        "Date": pd.to_datetime(["2000-03-01"]),
        "SPY": [0.1],
        "QQQ": [0.2],
        "EWS": [0.3],
    }
    raw = pd.DataFrame(rows)
    out = load_p2p_country_frame_from_excel(raw, ASSET_LIST)
    assert out.loc[0, "U.S."] == 0.1
    assert out.loc[0, "NASDAQ"] == 0.2
    assert out.loc[0, "Singapore"] == 0.3


def test_validate_column_order_passes_canonical():
    pairs = load_ticker_country_pairs(ASSET_LIST)
    cols = ["Date"] + [t for t, _ in pairs]
    raw = pd.DataFrame({c: [] for c in cols})
    validate_p2p_excel_column_order(raw, ASSET_LIST)


def test_validate_column_order_fails_on_swap():
    pairs = load_ticker_country_pairs(ASSET_LIST)
    tickers = [t for t, _ in pairs]
    tickers[7], tickers[8] = tickers[8], tickers[7]  # QQQ <-> SPY
    raw = pd.DataFrame({c: [] for c in ["Date"] + tickers})
    with pytest.raises(ValueError, match="column order mismatch"):
        validate_p2p_excel_column_order(raw, ASSET_LIST)
