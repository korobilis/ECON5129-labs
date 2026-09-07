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
    "build_forecast_dataset",
    "simulate_sparse_regression",
    "simulate_asset_panel",
    "load_fomc_meetings",
    "tokenise",
    "STOPWORDS",
    "TONE_DICTIONARY",
    "NeuralNetwork",
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


# ----------------------------------------------------------------------
# Forecasting design matrices
# ----------------------------------------------------------------------

def build_forecast_dataset(
    data: pd.DataFrame,
    codes: pd.Series,
    target: str = "INDPRO",
    horizon: int = 1,
    n_lags: int = 3,
    start: str = "1960-01",
    end: str = "2019-12",
    scale: float | None = None,
    winsorise: bool = True,
):
    """Construct a direct h-step-ahead forecasting dataset from FRED-MD.

    The target is the transformed target series led by `horizon` months. The
    predictors are lags 1 to `n_lags` of every series that is fully observed
    over the requested window. Extreme observations are clipped at ten
    interquartile ranges from the series median, following McCracken and Ng
    (2016), unless `winsorise` is set to False.

    Returns a dictionary with the design matrix `X`, the target `y`, the
    prediction dates, and the predictor names.
    """
    transformed = fredmd_transform(data, codes).loc[start:end]
    complete = transformed.columns[transformed.notna().all(axis=0)]
    panel = transformed[complete] * 1.0

    if winsorise:
        median = panel.median()
        iqr = panel.quantile(0.75) - panel.quantile(0.25)
        bound = 10.0 * iqr.replace(0.0, np.nan).fillna(panel.std())
        panel = panel.clip(lower=median - bound, upper=median + bound, axis=1)

    blocks, names = [], []
    for lag in range(1, n_lags + 1):
        blocks.append(panel.shift(lag))
        names.extend([f"{c}.L{lag}" for c in panel.columns])

    X_df = pd.concat(blocks, axis=1)
    X_df.columns = names
    if scale is None:
        scale = 100.0 if int(codes[target]) in (4, 5, 6, 7) else 1.0
    y_series = scale * panel[target].shift(-(horizon - 1))

    frame = pd.concat([y_series.rename("__target__"), X_df], axis=1).dropna()

    return {
        "X": frame.drop(columns="__target__").to_numpy(),
        "y": frame["__target__"].to_numpy(),
        "dates": frame.index,
        "names": list(frame.columns[1:]),
        "target": target,
        "horizon": horizon,
        "scale": scale,
    }


# ----------------------------------------------------------------------
# Simulated data generating processes
# ----------------------------------------------------------------------

def simulate_sparse_regression(
    n: int = 200,
    p: int = 400,
    n_active: int = 10,
    rho: float = 0.5,
    signal_to_noise: float = 3.0,
    seed: int | None = None,
):
    """Sparse linear model with equicorrelated predictors.

    Returns the design matrix, the response, and the true coefficient vector.
    """
    rng = np.random.default_rng(seed)

    factor = rng.normal(size=(n, 1))
    idio = rng.normal(size=(n, p))
    X = np.sqrt(rho) * factor + np.sqrt(1.0 - rho) * idio

    beta = np.zeros(p)
    active = rng.choice(p, size=n_active, replace=False)
    beta[active] = rng.choice([-1.0, 1.0], size=n_active) * rng.uniform(0.5, 2.0, n_active)

    signal = X @ beta
    sigma = np.std(signal) / np.sqrt(signal_to_noise)
    y = signal + sigma * rng.normal(size=n)

    return X, y, beta


def simulate_asset_panel(
    n_stocks: int = 200,
    n_periods: int = 240,
    seed: int | None = None,
):
    """Panel of stock returns with a convex dependence on momentum.

    Expected returns depend nonlinearly on two characteristics: a convex
    function of momentum and an interaction between size and value. Linear
    models cannot represent either.
    """
    rng = np.random.default_rng(seed)

    size = rng.normal(size=(n_periods, n_stocks))
    value = rng.normal(size=(n_periods, n_stocks))
    momentum = rng.normal(size=(n_periods, n_stocks))

    expected = 0.5 * momentum ** 2 - 0.3 * size * value + 0.2 * value
    noise = 2.0 * rng.normal(size=(n_periods, n_stocks))
    returns = expected + noise

    characteristics = np.stack([size, value, momentum], axis=-1)
    return characteristics, returns, expected


# ----------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself
no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what
when where which while who whom why will with would you your yours yourself
yourselves also would think well going get one two three like really much may
might must shall since upon whether
""".split())

TONE_DICTIONARY = {
    "positive": frozenset("""
    strong strength strengthen strengthening robust solid improve improved
    improving improvement expansion expanding growth gain gains favourable
    favorable optimistic confidence recovery recovering healthy resilient
    accelerate accelerating boost encouraging stable stability
    """.split()),
    "negative": frozenset("""
    weak weakness weaken weakening decline declining deteriorate deteriorating
    recession slowdown slowing contraction downturn adverse concern concerns
    concerned risk risks uncertain uncertainty fragile sluggish disappointing
    stress strain unemployment losses shortfall
    """.split()),
    "hawkish": frozenset("""
    tighten tightening tighter restraint restrictive inflation inflationary
    overheating pressures firm firming raise raising increase higher vigilance
    price stability restraining resolve
    """.split()),
    "dovish": frozenset("""
    ease easing accommodate accommodative accommodation stimulus support
    supportive lower lowering cut cutting reduce reducing patient patience
    slack unemployment weakness sluggish
    """.split()),
}


def tokenise(text: str, drop_stopwords: bool = True, min_length: int = 3) -> list[str]:
    """Lowercase, strip punctuation and split a document into word tokens."""
    import re

    words = re.findall(r"[a-z]+", str(text).lower())
    words = [w for w in words if len(w) >= min_length]
    if drop_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return words


def load_fomc_meetings(source: str | None = None) -> pd.DataFrame:
    """Load the meeting-level FOMC transcript data.

    The file is produced once by `scripts/build_fomc_data.ipynb` and stored in
    the course repository. Columns are the meeting date, the chair, counts of
    speakers, utterances and words, and the full meeting text.
    """
    if source is None:
        source = f"{DATA_URL}/fomc_meetings.csv.gz"

    frame = pd.read_csv(source, compression="infer")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------
# Neural network
# ----------------------------------------------------------------------

class NeuralNetwork:
    """Fully connected feedforward network with ReLU hidden units, trained by Adam.

    Supports one or more outputs, weight decay, dropout and early stopping.
    Written in numpy so that it runs anywhere; it is the estimator built from
    first principles in Lab 7.
    """

    def __init__(self, layer_sizes, seed=0, weight_decay=0.0, dropout=0.0):
        gen = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.W.append(gen.normal(scale=np.sqrt(2.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))
        self.n_outputs = layer_sizes[-1]
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.generator = gen

    @staticmethod
    def _relu(a):
        return np.maximum(a, 0.0)

    def _as_matrix(self, y):
        y = np.asarray(y, dtype=float)
        return y[:, None] if y.ndim == 1 else y

    def forward(self, X, training=False):
        activations, pre_activations, masks = [X], [], []
        A = X
        for layer, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            pre_activations.append(Z)
            if layer < len(self.W) - 1:
                A = self._relu(Z)
                if training and self.dropout > 0:
                    mask = (self.generator.random(A.shape) > self.dropout) / (1 - self.dropout)
                    A = A * mask
                    masks.append(mask)
                else:
                    masks.append(None)
            else:
                A = Z
            activations.append(A)
        return A, activations, pre_activations, masks

    def gradients(self, X, y):
        Y = self._as_matrix(y)
        out, activations, pre_activations, masks = self.forward(X, training=True)
        n = len(Y)

        grads_W = [np.zeros_like(W) for W in self.W]
        grads_b = [np.zeros_like(b) for b in self.b]

        delta = (2.0 / n) * (out - Y)
        for layer in reversed(range(len(self.W))):
            grads_W[layer] = activations[layer].T @ delta + 2 * self.weight_decay * self.W[layer]
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = delta @ self.W[layer].T
                if masks[layer - 1] is not None:
                    delta = delta * masks[layer - 1]
                delta = delta * (pre_activations[layer - 1] > 0)
        return grads_W, grads_b

    def loss(self, X, y):
        Y = self._as_matrix(y)
        out, *_ = self.forward(X, training=False)
        penalty = self.weight_decay * sum(np.sum(W ** 2) for W in self.W)
        return float(np.mean((out - Y) ** 2) + penalty)

    def predict(self, X):
        out, *_ = self.forward(X, training=False)
        return out.ravel() if self.n_outputs == 1 else out

    def fit(self, X, y, n_epochs=300, batch_size=32, learning_rate=0.01,
            X_val=None, y_val=None, patience=None, verbose=False):
        Y = self._as_matrix(y)
        mW = [np.zeros_like(W) for W in self.W]
        vW = [np.zeros_like(W) for W in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        history = {"train": [], "validation": []}
        best_loss, best_state, wait = np.inf, None, 0

        for epoch in range(n_epochs):
            order = self.generator.permutation(len(Y))
            for start in range(0, len(Y), batch_size):
                idx = order[start:start + batch_size]
                grads_W, grads_b = self.gradients(X[idx], Y[idx])
                step += 1
                for layer in range(len(self.W)):
                    mW[layer] = beta1 * mW[layer] + (1 - beta1) * grads_W[layer]
                    vW[layer] = beta2 * vW[layer] + (1 - beta2) * grads_W[layer] ** 2
                    mb[layer] = beta1 * mb[layer] + (1 - beta1) * grads_b[layer]
                    vb[layer] = beta2 * vb[layer] + (1 - beta2) * grads_b[layer] ** 2

                    self.W[layer] -= learning_rate * (mW[layer] / (1 - beta1 ** step)) / (
                        np.sqrt(vW[layer] / (1 - beta2 ** step)) + eps)
                    self.b[layer] -= learning_rate * (mb[layer] / (1 - beta1 ** step)) / (
                        np.sqrt(vb[layer] / (1 - beta2 ** step)) + eps)

            history["train"].append(self.loss(X, Y))
            if X_val is not None:
                val_loss = self.loss(X_val, y_val)
                history["validation"].append(val_loss)
                if patience is not None:
                    if val_loss < best_loss - 1e-8:
                        best_loss, wait = val_loss, 0
                        best_state = ([W.copy() for W in self.W], [b.copy() for b in self.b])
                    else:
                        wait += 1
                        if wait >= patience:
                            self.W, self.b = best_state
                            history["stopped_at"] = epoch + 1
                            break
        return history
