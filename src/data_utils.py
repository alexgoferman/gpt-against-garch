"""Load the 49 Industry Portfolios (daily) for modelling.

The raw Ken French CSV is parsed only once, by preprocess.py, which writes
data/processed/returns_{vw,ew}.csv (% returns). This module reads those files, so every
model (random walk, GARCH, Chronos) sees exactly the same data and the same split dates.

Usage:
    from data_utils import load_french49, split
    vw = load_french49("value")          # decimal returns, DatetimeIndex, 49 columns
    train, val, test = split(vw)
"""
import pandas as pd

import preprocess
from preprocess import OUT, SPLITS

TRAIN_END = SPLITS["train"][1]   # "2010-12-31"
VAL_START, VAL_END = SPLITS["val"]
TEST_START = SPLITS["test"][0]

_KEY = {"value": "vw", "equal": "ew"}


def load_french49(table: str = "value") -> pd.DataFrame:
    """Return daily returns (decimal, not %) indexed by date, 49 industry columns, NaN = missing.

    table: "value" (value-weighted) or "equal" (equal-weighted).
    Runs preprocess.py first if the processed file does not exist yet.
    """
    path = OUT / f"returns_{_KEY[table]}.csv"
    if not path.exists():
        preprocess.main()
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return df / 100.0


def split(df: pd.DataFrame):
    """Time split: train <=2010, val 2011-2015, test >=2016 (dates from preprocess.SPLITS)."""
    return df.loc[:TRAIN_END], df.loc[VAL_START:VAL_END], df.loc[TEST_START:]
