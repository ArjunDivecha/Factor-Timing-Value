"""
P2P ticker ↔ country mapping for the T2 pipeline (Step Zero / Step One).

INPUT FILES (read by callers):
    AssetList.xlsx — sheet 'Yahoo': ETF tickers in Bloomberg country order
    P2P_Country_Historical_Scores.xlsx — Date + ticker columns (Step Zero output)

OUTPUT:
    No direct file I/O here; returns DataFrames for Step One / validation.

VERSION: 1.0
LAST UPDATED: 2026-10-08
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Must match Step One Create T2Master.py country order (after the date column).
T2_COUNTRY_NAMES: List[str] = [
    "Singapore",
    "Australia",
    "Canada",
    "Germany",
    "Japan",
    "Switzerland",
    "U.K.",
    "NASDAQ",
    "U.S.",
    "France",
    "Netherlands",
    "Sweden",
    "Italy",
    "ChinaA",
    "Chile",
    "Indonesia",
    "Philippines",
    "Poland",
    "US SmallCap",
    "Malaysia",
    "Taiwan",
    "Mexico",
    "Korea",
    "Brazil",
    "South Africa",
    "Denmark",
    "India",
    "ChinaH",
    "Hong Kong",
    "Thailand",
    "Turkey",
    "Spain",
    "Vietnam",
    "Saudi Arabia",
]

T2_COUNTRY_NAMES_WITH_DATE: List[str] = ["Country"] + T2_COUNTRY_NAMES

_DATE_COLUMN_ALIASES = frozenset(
    {"date", "country", "unnamed: 0", "unnamed:0", "as of", "asof"}
)


def _normalize_ticker_label(label: object) -> str:
    """'EWS Equity' / 'ews' -> 'EWS'."""
    text = str(label).strip().upper()
    text = re.sub(r"\s+EQUITY$", "", text)
    return text


def load_ticker_country_pairs(
    asset_list_path: str | Path,
    sheet_name: str = "Yahoo",
) -> List[Tuple[str, str]]:
    """
    Load (ticker, country) pairs in canonical T2 order from AssetList.xlsx.
    """
    path = Path(asset_list_path)
    if not path.is_file():
        raise FileNotFoundError(f"AssetList not found: {path.resolve()}")

    tickers = [
        _normalize_ticker_label(t)
        for t in pd.read_excel(path, sheet_name=sheet_name).iloc[:, 0]
        if pd.notna(t)
    ]
    if len(tickers) != len(T2_COUNTRY_NAMES):
        raise ValueError(
            f"Expected {len(T2_COUNTRY_NAMES)} tickers on sheet '{sheet_name}' "
            f"in {path.name}, found {len(tickers)}"
        )
    return list(zip(tickers, T2_COUNTRY_NAMES))


def load_ticker_to_country(
    asset_list_path: str | Path,
    sheet_name: str = "Yahoo",
) -> Dict[str, str]:
    return dict(load_ticker_country_pairs(asset_list_path, sheet_name=sheet_name))


def _detect_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() in _DATE_COLUMN_ALIASES:
            return col
    return df.columns[0]


def p2p_raw_ticker_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map normalized ticker -> original column name in a raw P2P Excel frame.
    """
    date_col = _detect_date_column(df)
    mapping: Dict[str, str] = {}
    for col in df.columns:
        if col == date_col:
            continue
        ticker = _normalize_ticker_label(col)
        if ticker in mapping:
            raise ValueError(
                f"Duplicate P2P ticker column '{ticker}' "
                f"({mapping[ticker]!r} and {col!r})"
            )
        mapping[ticker] = col
    return mapping


def load_p2p_country_frame_from_excel(
    p2p_raw: pd.DataFrame,
    asset_list_path: str | Path,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Convert Step Zero P2P output (Date + ETF tickers) to T2 Master layout:
    columns ['Country', <34 country names in order>].
    """
    log = logger or logging.getLogger(__name__)
    pairs = load_ticker_country_pairs(asset_list_path)
    ticker_to_col = p2p_raw_ticker_columns(p2p_raw)
    date_col = _detect_date_column(p2p_raw)

    out = pd.DataFrame()
    out["Country"] = pd.to_datetime(p2p_raw[date_col], errors="coerce")
    missing_tickers: List[str] = []

    for ticker, country in pairs:
        col = ticker_to_col.get(ticker)
        if col is None:
            missing_tickers.append(ticker)
            out[country] = pd.NA
            continue
        out[country] = pd.to_numeric(p2p_raw[col], errors="coerce")

    out = out.dropna(subset=["Country"]).reset_index(drop=True)

    if missing_tickers:
        log.warning(
            "P2P file missing ticker column(s) %s — filled with NaN for those countries",
            ", ".join(missing_tickers),
        )

    extra = set(ticker_to_col) - {t for t, _ in pairs}
    if extra:
        log.warning(
            "P2P file has extra ticker column(s) not in AssetList (ignored): %s",
            ", ".join(sorted(extra)),
        )

    expected = T2_COUNTRY_NAMES_WITH_DATE
    if list(out.columns) != expected:
        raise ValueError(
            "P2P country frame column order mismatch after mapping; "
            f"got {list(out.columns)[:5]}..."
        )

    log.info(
        "Mapped P2P scores by ETF ticker header → %d countries (%d rows)",
        len(T2_COUNTRY_NAMES),
        len(out),
    )
    return out


def validate_p2p_excel_column_order(
    p2p_raw: pd.DataFrame,
    asset_list_path: str | Path,
) -> None:
    """
    Raise if ticker columns are not exactly AssetList order (Step Zero contract).
    """
    pairs = load_ticker_country_pairs(asset_list_path)
    expected_tickers = [t for t, _ in pairs]
    date_col = _detect_date_column(p2p_raw)
    file_tickers = [
        _normalize_ticker_label(c) for c in p2p_raw.columns if c != date_col
    ]
    if file_tickers != expected_tickers:
        first_mismatch = next(
            (
                (i, exp, got)
                for i, (exp, got) in enumerate(zip(expected_tickers, file_tickers))
                if exp != got
            ),
            None,
        )
        if first_mismatch:
            i, exp, got = first_mismatch
            raise ValueError(
                f"P2P Excel ticker column order mismatch at index {i}: "
                f"expected {exp}, found {got}. "
                f"Re-run Step Zero or fix column order to match AssetList Yahoo sheet."
            )
        raise ValueError(
            f"P2P Excel has {len(file_tickers)} ticker columns; "
            f"expected {len(expected_tickers)} in AssetList order."
        )


def ordered_ticker_columns(tickers: Sequence[str]) -> List[str]:
    """Canonical ticker column order for Step Zero Excel output."""
    return [_normalize_ticker_label(t) for t in tickers]
