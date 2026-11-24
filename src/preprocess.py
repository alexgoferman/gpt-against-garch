"""Parse the Kenneth French 49 Industry Portfolios (Daily) CSV.

The raw file holds two stacked tables (value-weighted, then equal-weighted)
with free-text headers in between. This script splits them, converts the
missing-value codes (-99.99, -999) to NaN, and writes one wide CSV per
weighting scheme (rows = dates, columns = industries, values = daily % returns),
plus a per-industry summary that includes the train/val/test row counts.
"""
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "49_Industry_Portfolios_Daily.csv"
OUT = ROOT / "data" / "processed"

SECTIONS = {
    "vw": "Average Value Weighted Returns -- Daily",
    "ew": "Average Equal Weighted Returns -- Daily",
}
MISSING = [-99.99, -999.0]

SPLITS = {  # inclusive date ranges
    "train": ("1926-01-01", "2010-12-31"),
    "val": ("2011-01-01", "2015-12-31"),
    "test": ("2016-01-01", None),
}


def read_section(lines, title):
    start = next(i for i, l in enumerate(lines) if l.strip() == title) + 1
    block = [lines[start]]  # header row
    for l in lines[start + 1:]:
        if not l.strip() or not l[:8].isdigit():
            break
        block.append(l)
    df = pd.read_csv(StringIO("".join(block)), index_col=0, skipinitialspace=True)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.index.name = "date"
    df.columns = df.columns.str.strip()
    return df.astype(float).replace(MISSING, np.nan)


def split_of(dates):
    s = pd.Series("", index=dates)
    for name, (lo, hi) in SPLITS.items():
        s[(dates >= lo) & ((dates <= hi) if hi else True)] = name
    return s


def main():
    lines = RAW.read_text().splitlines(keepends=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for key, title in SECTIONS.items():
        df = read_section(lines, title)
        df.to_csv(OUT / f"returns_{key}.csv")
        split = split_of(df.index)
        summary = pd.DataFrame({
            "first_valid": df.apply(lambda c: c.first_valid_index()),
            "n_missing": df.isna().sum(),
            **{f"n_{s}": df[split == s].notna().sum() for s in SPLITS},
            "mean_pct": df.mean(),
            "std_pct": df.std(),
        })
        summary.to_csv(OUT / f"summary_{key}.csv")
        print(f"[{key}] {df.shape[0]} days x {df.shape[1]} industries, "
              f"{df.index.min().date()} -> {df.index.max().date()}, "
              f"missing cells: {int(df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
