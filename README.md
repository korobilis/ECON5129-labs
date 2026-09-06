# ECON5129 Computer Labs

Python labs for ECON5129 Statistical Machine Learning, Adam Smith Business School, University of Glasgow.

**Site:** https://korobilis.github.io/ECON5129-labs

## Repository layout

```
master/            authored notebooks, the only files that are edited by hand
labs/              generated student notebooks, outputs cleared
solutions/         generated solution notebooks
data/              frozen datasets
econ5129_utils.py  shared helper module, downloaded by each notebook at runtime
scripts/           build tooling
intro.md           book landing page
_config.yml        Jupyter Book configuration
_toc.yml           book table of contents
```

## Rebuilding

```bash
conda activate econ5129
python scripts/build_labs.py       # regenerate labs/ and solutions/ from master/
jupyter-book build .               # build the site into _build/html
ghp-import -n -p -f _build/html    # publish to GitHub Pages
```

Full instructions, including first-time setup, are in `PUBLISHING.md`.
