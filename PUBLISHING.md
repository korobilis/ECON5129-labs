# Setting up and publishing the lab site

This is a one-time setup followed by a four-command routine you will repeat for each lab. Everything is free and nothing here requires administrator rights beyond installing Anaconda and Git.

The finished site will live at **https://korobilis.github.io/ECON5129-labs**.

---

## Part 1: One-time setup

### Step 1. Install Git

Check whether you already have it. Open a terminal (macOS: Terminal; Windows: Anaconda Prompt) and type:

```bash
git --version
```

If that prints a version number, skip ahead. If not, install from [git-scm.com/downloads](https://git-scm.com/downloads), accepting the defaults.

Tell Git who you are, once:

```bash
git config --global user.name "Dimitris Korobilis"
git config --global user.email "your.email@glasgow.ac.uk"
```

### Step 2. Create the build environment

From the folder containing this repository:

```bash
conda env create -f environment.yml
conda activate econ5129
```

This gives you Python 3.11, the scientific stack, Jupyter Book and the publishing tool. Activate this environment every time you work on the labs.

Verify:

```bash
jupyter-book --version
ghp-import --version
```

### Step 3. Create the GitHub repository

1. Go to [github.com/new](https://github.com/new).
2. Repository name: `ECON5129-labs`.
3. Visibility: **Public**. This matters. The Colab buttons and the raw data links only work for public repositories, and GitHub Pages requires it on a free account.
4. Do **not** tick "Add a README file", "Add .gitignore" or "Choose a license". The repository must start empty, because the files already exist locally.
5. Click *Create repository*.

### Step 4. Push the files

From inside the repository folder on your machine:

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: lab infrastructure and Lab 1"
git remote add origin https://github.com/korobilis/ECON5129-labs.git
git push -u origin main
```

The first `git push` asks for credentials. GitHub no longer accepts your account password here. When prompted for a password, paste a **personal access token** instead:

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens), choose *Generate new token (classic)*.
2. Give it a name, set an expiry, tick the `repo` scope.
3. Copy the token and paste it as the password. Your machine will remember it.

If you would rather avoid the command line entirely, install [GitHub Desktop](https://desktop.github.com), choose *Add local repository*, point it at this folder, and use *Publish repository*. It handles authentication for you. The routine in Part 2 still needs a terminal for the build commands.

### Step 5. Build the site locally

```bash
jupyter-book build .
```

This writes the site into `_build/html`. Open `_build/html/index.html` in a browser and check that it looks right. Building locally means that when something goes wrong you see the error immediately, on your own screen.

### Step 6. Publish

```bash
ghp-import -n -p -f _build/html
```

This copies the built site to a branch called `gh-pages` and pushes it. The `-n` flag adds the file that stops GitHub trying to process the site as a blog, which is a common and confusing failure.

### Step 7. Switch on GitHub Pages

1. Go to `https://github.com/korobilis/ECON5129-labs/settings/pages`.
2. Under *Build and deployment*, set Source to **Deploy from a branch**.
3. Set Branch to **gh-pages** and folder to **/ (root)**. Save.

Wait a minute or two, then visit https://korobilis.github.io/ECON5129-labs. You only ever do this step once.

---

## Part 2: The routine for each lab

Four commands, from the repository folder with the `econ5129` environment active.

```bash
python scripts/build_labs.py       # 1. regenerate student and solution notebooks
jupyter-book build .               # 2. build the site
ghp-import -n -p -f _build/html    # 3. publish the site
git add . && git commit -m "Lab 2" && git push    # 4. save the source
```

Step 4 is separate from step 3 on purpose. Step 3 publishes the *website*; step 4 saves the *source files* that produced it. Both matter, and forgetting the second is the usual way people lose work.

### Where to make edits

Only ever edit the notebooks in `master/`. The contents of `labs/` and `solutions/` are generated and will be silently overwritten every time you run `build_labs.py`.

### How exercises are marked up

In a master notebook, a code cell that contains the answer to an exercise carries the cell tag `solution`. To see or change tags in JupyterLab, open the right-hand sidebar and choose the property inspector, the gear icon.

Within a tagged cell, any line ending in `#@keep` is scaffolding that the students should be given. Everything else is stripped out and replaced by a single `# your code here` placeholder. So this master cell:

```python
def rbf_basis(x, centres, width):  #@keep
    """Gaussian radial basis design matrix."""  #@keep
    Z = np.exp(-0.5 * ((x[:, None] - centres[None, :]) / width) ** 2)
    return np.column_stack([np.ones(len(x)), Z])
```

becomes this in the student notebook:

```python
def rbf_basis(x, centres, width):
    """Gaussian radial basis design matrix."""
    # your code here
```

while the solution notebook keeps everything, with the markers removed.

### Outputs, figures and tables

The site does not execute notebooks when it builds. It displays whatever outputs are saved inside the file. So:

- **Solution notebooks** should be run before publishing, so that the site shows the figures and tables. Open `solutions/lab01_regression_solutions.ipynb` in JupyterLab, choose `Run` &rarr; `Restart Kernel and Run All Cells`, save, then build.
- **Student notebooks** are published deliberately without outputs. Students generate their own.

For Labs 7 to 10, which need PyTorch and `transformers`, run the solution notebook in Colab instead, then download it with `File` &rarr; `Download` &rarr; `Download .ipynb` and replace the file in `solutions/` before building.

### Withholding solutions until after the session

The repository is public, so anything you commit is visible. To release solutions only after the lab has run:

1. Before the session, comment out that lab's line in the *Solutions* part of `_toc.yml`, and do not commit the file in `solutions/`.
2. After the session, uncomment the line, commit the solution notebook, and run the routine again.

A blunter alternative, if you prefer not to think about it each week, is to publish solutions immediately and simply tell students not to look until they have tried. Many courses do this and it works about as well.

---

## Troubleshooting

**The site builds but the page is blank or unstyled.** You published without the `-n` flag. Re-run `ghp-import -n -p -f _build/html`.

**Changes do not appear on the site.** GitHub Pages caches aggressively. Wait two minutes and hard-refresh the browser. If it still looks stale, confirm the `gh-pages` branch updated on GitHub.

**`jupyter-book build .` fails with a cryptic Sphinx error.** Delete `_build` and try again. Stale caches cause most of these.

**A notebook does not appear in the site.** Every page must be listed in `_toc.yml`, because `only_build_toc_files` is on. Add it there.

**Colab opens but the first cell fails.** The repository must be public and the branch in `scripts/build_labs.py` must match your default branch name, which is `main`.

**A page renders LaTeX as raw text.** Use `$ ... $` for inline maths and `$$ ... $$` on their own lines for display maths. The `\(` and `\[` delimiters are not enabled.
