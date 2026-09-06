# Getting set up

There are two ways to run the labs. Choose one before the first session.

## Option 1: Google Colab, recommended

Colab runs the notebook in your browser on Google's machines. Nothing is installed, everything needed is already available, and it works identically on Windows, macOS and the University lab machines.

1. Open any lab from this site using the rocket icon in the top right, then choose **Colab**.
2. Sign in with a Google account.
3. **Immediately choose `File` &rarr; `Save a copy in Drive`.** This is the single most important step. Until you do it you are working on a temporary copy, and everything you write disappears when the tab closes.
4. Work through the notebook. Run a cell with `Shift + Enter`.

A Colab session disconnects after a period of inactivity and the variables are lost, although your code is not. If that happens, choose `Runtime` &rarr; `Run all` to rebuild the state.

## Option 2: Anaconda, on a lab machine or your own laptop

Anaconda is installed on the University lab computers and is a free download for personal machines.

1. Download the notebook using the download icon at the top of the lab page.
2. Open Anaconda Navigator and launch JupyterLab.
3. Navigate to the notebook and open it.

Labs 1 to 6 need only what Anaconda already provides. Labs 7 and 8 use PyTorch and Lab 10 uses the `transformers` library, neither of which ships with Anaconda. If you cannot install packages on a lab machine, use Colab for those four sessions.

To create a dedicated environment on your own machine:

```bash
conda env create -f environment.yml
conda activate econ5129
jupyter lab
```

## What the notebooks assume

Some prior exposure to Python. If you have not used it before, or have not used it recently, work through the QuantEcon introduction before the first lab: [Python Programming for Economics and Finance](https://python-programming.quantecon.org/intro.html). The first three sections are enough.

Every lab begins with a cell that downloads the shared helper module `econ5129_utils.py` and imports the libraries. Run it first. It is the only cell that needs an internet connection, along with the cells that load data.

## Getting help

Ask during the lab. The sessions are staffed for two hours and the fastest route to an answer is to turn round and ask. Outside the sessions, post on the course Moodle forum so that the answer reaches everyone.
