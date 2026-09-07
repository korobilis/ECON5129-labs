#!/usr/bin/env python3
"""Execute the solution notebooks in place and report the outcome.

Running a notebook stores its figures and tables inside the file, which is
what the website displays, and simultaneously tests that every cell still
works. Run this before building the site.

Usage:

    python scripts/run_solutions.py            run every solution notebook
    python scripts/run_solutions.py lab03      run one, matched by prefix
    python scripts/run_solutions.py --labs     run the student notebooks too

Notebooks that need PyTorch or transformers are skipped automatically when
those libraries are absent; run those in Colab and replace the file.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SECONDS = 1800


def execute(path: Path) -> tuple[bool, float, str]:
    """Run one notebook in place, returning success, elapsed time and any error."""
    start = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={TIMEOUT_SECONDS}",
            str(path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        return True, elapsed, ""

    message = result.stderr.strip().splitlines()
    tail = "\n".join(message[-6:]) if message else "unknown error"
    return False, elapsed, tail


def main() -> None:
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    include_labs = "--labs" in sys.argv
    pattern = arguments[0] if arguments else ""

    targets = sorted((ROOT / "solutions").glob("*.ipynb"))
    if include_labs:
        targets += sorted((ROOT / "labs").glob("*.ipynb"))
    targets = [p for p in targets if p.name.startswith(pattern) or not pattern]

    if not targets:
        raise SystemExit(f"No notebooks matching '{pattern}'.")

    failures = []
    for path in targets:
        print(f"running {path.relative_to(ROOT)} ... ", end="", flush=True)
        ok, elapsed, error = execute(path)
        if ok:
            print(f"ok ({elapsed:.0f}s)")
        else:
            print(f"FAILED ({elapsed:.0f}s)")
            print(f"  {error}")
            failures.append(path.name)

    print()
    print(f"{len(targets) - len(failures)} of {len(targets)} notebooks executed cleanly")
    if failures:
        print("failed: " + ", ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
