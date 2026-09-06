"""
Shared utilities for the ECON5129 computer labs.

Adam Smith Business School, University of Glasgow.

The module is deliberately dependency-light: numpy, pandas, matplotlib and
scipy only, all of which are preinstalled on Google Colab and in Anaconda.
It is downloaded automatically by the bootstrap cell at the top of each lab
notebook, so it works identically on Colab and on a local machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "REPO_RAW",
    "COLORS",
    "set_style",
    "load_fredmd",
    "fredmd_transform",
    "load_usrec",
    "standardise",
    "oos_r2",
    "mse",
    "rmse",
    "expanding_window_splits",
    "diebold_mariano",
]

REPO_RAW = "https://raw.githubusercontent.com/korobilis/ECON5129-labs/main"
DATA_URL = f"{REPO_RAW}/data"

# Palette matched to the lecture slides.
COLORS = [
    "#00274d",  # darkblue
    "#4a90c2",  # lightblue
    "#c0392b",  # red
    "#e0a300",  # gold
    "#2e7d32",  # green
    "#6a4c93",  # purple
    "#7f8c8d",  # grey
]


def set_style() -> None:
    """Apply the course plotting defaults."""
    mpl.rcParams.update(
        {
            "figure.figsize": (8.0, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "axes.prop_cycle": mpl.cycler(color=COLORS),
        }
    )


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_fredmd(source: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the FRED-MD monthly panel.

    Parameters
    ----------
    source
        Path or URL of the FRED-MD csv file. Defaults to the copy stored in
        the course repository.

    Returns
    -------
    data
        Monthly observations indexed by a `DatetimeIndex` set to the first
        day of each month.
    codes
        Transformation code for each series, as published with the vintage.
    """
    if source is None:
        source = f"{DATA_URL}/FREDMD.csv"

    raw = pd.read_csv(source)
    codes = raw.iloc[0, 1:].astype(int)
    codes.name = "transform"

    data = raw.iloc[1:].copy()
    data.index = pd.to_datetime(data["sasdate"], format="%m/%d/%Y")
    data.index.name = "date"
    data = data.drop(columns="sasdate").astype(float)
    data = data.dropna(axis=0, how="all")

    return data, codes


def fredmd_transform(data: pd.DataFrame, codes: pd.Series) -> pd.DataFrame:
    """Apply the standard FRED-MD stationarity transformations.

    Codes follow McCracken and Ng (2016):
    1 level, 2 first difference, 3 second difference, 4 log,
    5 first difference of logs, 6 second difference of logs,
    7 first difference of the growth rate.
    """
    out = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)

    for col in data.columns:
        x = data[col]
        code = int(codes[col])
        if code == 1:
            z = x
        elif code == 2:
            z = x.diff()
        elif code == 3:
            z = x.diff().diff()
        elif code == 4:
            z = np.log(x)
        elif code == 5:
            z = np.log(x).diff()
        elif code == 6:
            z = np.log(x).diff().diff()
        elif code == 7:
            z = (x / x.shift(1) - 1.0).diff()
        else:
            raise ValueError(f"Unknown transformation code {code} for {col}.")
        out[col] = z

    return out


def load_usrec(source: str | None = None) -> pd.Series:
    """Load the NBER recession indicator, monthly, 1 during recessions."""
    if source is None:
        source = f"{DATA_URL}/USREC.csv"

    raw = pd.read_csv(source)
    rec = pd.Series(
        raw["USREC"].to_numpy(dtype=float),
        index=pd.to_datetime(raw["observation_date"]),
        name="USREC",
    )
    rec.index.name = "date"
    return rec


# ----------------------------------------------------------------------
# Estimation and evaluation helpers
# ----------------------------------------------------------------------

def standardise(
    X_train: np.ndarray, X_test: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Standardise using training-sample moments only.

    Returns the standardised training matrix, the standardised test matrix
    (or None), and the means and standard deviations used.
    """
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0, ddof=1)
    sd = np.where(sd < 1e-12, 1.0, sd)

    Z_train = (X_train - mu) / sd
    Z_test = None if X_test is None else (X_test - mu) / sd
    return Z_train, Z_test, mu, sd


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def oos_r2(
    y_true: np.ndarray, y_pred: np.ndarray, benchmark: np.ndarray | None = None
) -> float:
    """Out-of-sample R-squared relative to a benchmark forecast.

    The benchmark defaults to the mean of the evaluation sample. A positive
    value means the model beats the benchmark.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if benchmark is None:
        benchmark = np.full_like(y_true, y_true.mean())
    else:
        benchmark = np.asarray(benchmark, dtype=float)

    sse_model = np.sum((y_true - y_pred) ** 2)
    sse_bench = np.sum((y_true - benchmark) ** 2)
    return float(1.0 - sse_model / sse_bench)


def expanding_window_splits(n: int, n_init: int, step: int = 1):
    """Yield (train_index, test_index) pairs for expanding-window evaluation."""
    for end in range(n_init, n, step):
        yield np.arange(end), np.arange(end, min(end + step, n))


def diebold_mariano(
    e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Compares forecast errors `e1` and `e2`. A negative statistic favours the
    first model. Returns the statistic and its two-sided p-value using the
    Harvey, Leybourne and Newbold small-sample correction.
    """
    from scipy import stats

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = d.size
    d_bar = d.mean()

    gamma = [np.sum((d[k:] - d_bar) * (d[: n - k] - d_bar)) / n for k in range(h)]
    var_d = (gamma[0] + 2.0 * sum(gamma[1:])) / n

    stat = d_bar / np.sqrt(var_d)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= correction
    pval = 2.0 * (1.0 - stats.t.cdf(np.abs(stat), df=n - 1))
    return float(stat), float(pval)
