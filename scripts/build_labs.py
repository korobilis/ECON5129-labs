#!/usr/bin/env python3
"""Generate student and solution notebooks from the masters in `master/`.

Each lab is authored once, in `master/`. Code cells that contain the answer to
an exercise carry the cell tag `solution`. Within such a cell, any line ending
in the marker `#@keep` is scaffolding that students should be given; every
other line is removed and replaced by a single `# your code here` placeholder.

Running this script writes two files for each master notebook:

    labs/<name>.ipynb                  student version, outputs cleared
    solutions/<name>_solutions.ipynb   full version, markers removed

Usage:

    python scripts/build_labs.py            build every master notebook
    python scripts/build_labs.py lab01      build one, matched by prefix
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER_DIR = ROOT / "master"
LAB_DIR = ROOT / "labs"
SOLUTION_DIR = ROOT / "solutions"

REPO = "korobilis/ECON5129-labs"
BRANCH = "main"
BADGE_TOKEN = "{{COLAB_BADGE}}"
KEEP_MARKER = "#@keep"
PLACEHOLDER = "# your code here"


def colab_badge(relative_path: str) -> str:
    url = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{relative_path}"
    return f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def cell_source(cell: dict) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def set_source(cell: dict, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def clear_outputs(cell: dict) -> None:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def strip_markers(text: str) -> str:
    return re.sub(r"[ \t]*" + re.escape(KEEP_MARKER) + r"\b", "", text)


def make_stub(text: str) -> str:
    """Keep only the marked scaffolding lines, with one placeholder comment."""
    kept, inserted = [], False
    for line in text.split("\n"):
        if KEEP_MARKER in line:
            kept.append(strip_markers(line))
        elif not inserted and line.strip():
            indent = " " * (len(line) - len(line.lstrip()))
            opens_block = bool(kept) and kept[-1].rstrip().endswith(":")
            kept.append(f"{indent}{PLACEHOLDER}")
            if opens_block:
                kept.append(f"{indent}pass")
            inserted = True
    if not inserted:
        kept.append(PLACEHOLDER)
    return "\n".join(kept)


def build(master_path: Path) -> None:
    notebook = json.loads(master_path.read_text())
    name = master_path.stem

    student = json.loads(json.dumps(notebook))
    solution = json.loads(json.dumps(notebook))

    for cell in student["cells"]:
        clear_outputs(cell)
        text = cell_source(cell)
        if "solution" in cell.get("metadata", {}).get("tags", []):
            set_source(cell, make_stub(text))
        else:
            set_source(cell, text.replace(BADGE_TOKEN, colab_badge(f"labs/{name}.ipynb")))

    for cell in solution["cells"]:
        text = strip_markers(cell_source(cell))
        set_source(
            cell,
            text.replace(BADGE_TOKEN, colab_badge(f"solutions/{name}_solutions.ipynb")),
        )

    LAB_DIR.mkdir(exist_ok=True)
    SOLUTION_DIR.mkdir(exist_ok=True)

    student_path = LAB_DIR / f"{name}.ipynb"
    solution_path = SOLUTION_DIR / f"{name}_solutions.ipynb"
    student_path.write_text(json.dumps(student, indent=1, ensure_ascii=False))
    solution_path.write_text(json.dumps(solution, indent=1, ensure_ascii=False))

    print(f"{master_path.name} -> {student_path.relative_to(ROOT)}, {solution_path.relative_to(ROOT)}")


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    masters = sorted(p for p in MASTER_DIR.glob("*.ipynb") if p.name.startswith(pattern))
    if not masters:
        raise SystemExit(f"No master notebooks matching '{pattern}' in {MASTER_DIR}.")
    for path in masters:
        build(path)


if __name__ == "__main__":
    main()
