#!/usr/bin/env python3
"""
py_to_nb.py — Convert a nb_to_py-generated .py file back to a Jupyter notebook.

Recognises the three comment styles produced by nb_to_py:

  H1  → # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #							 Header 1							  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  H2  →  # Header 2
         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  H3  →  # # Header Text

Cell separator (no heading between two code cells):
        # - ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -

Everything else is a code cell.  Consecutive comment lines that don't
match a heading pattern are treated as a markdown paragraph cell.

Usage:
  python py_to_nb.py script.py              # writes script.ipynb
  python py_to_nb.py script.py -o out.ipynb
  python py_to_nb.py script.py --stdout
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── detection helpers ─────────────────────────────────────────────────────────

def is_border(line: str) -> bool:
    """True for lines like  # #  # #  # #  …  (H1/H2 border)."""
    stripped = line.strip()
    # Must start with '#' and consist only of '#' and spaces
    if not stripped.startswith("#"):
        return False
    inner = stripped[1:]          # everything after the leading #
    # Allow only spaces and '#' characters, and must have several '#'
    if re.fullmatch(r"[ #]+", inner) and inner.count("#") >= 3:
        return True
    return False


def is_separator(line: str) -> bool:
    """True for the ─ ─ ─  separator between code cells."""
    return "- --" in line and line.strip().startswith("#")


def strip_comment(line: str) -> str:
    """Remove leading '# ' or '#\t' prefix from a comment line."""
    if line.startswith("# #"):
        return line[3:]
    if line.startswith("# "):
        return line[2:]
    if line.startswith("#\t"):
        return line[1:].strip()
    if line.startswith("#"):
        return line[1:].strip()
    return line


def extract_h1_title(title_line: str) -> str:
    """
    Pull the text out of:   #\t\t\t\t\t\t\t Title text \t\t\t\t\t\t\t  #
    """
    s = title_line.strip()
    # Remove leading # and trailing #
    if s.startswith("#"):
        s = s[1:]
    if s.endswith("#"):
        s = s[:-1]
    return s.strip()


# ── parser ────────────────────────────────────────────────────────────────────

def parse_cells(source: str) -> list[dict]:
    """
    Walk the source line by line and emit a list of
    {'type': 'markdown'|'code', 'source': str} dicts.
    """
    lines = source.splitlines()
    cells = []
    i = 0
    pending_code: list[str] = []

    def flush_code():
        if pending_code:
            src = "\n".join(pending_code).strip()
            if src:
                cells.append({"type": "code", "source": src})
            pending_code.clear()

    while i < len(lines):
        line = lines[i]

        # ── separator: just a cell boundary, no markdown output ──────────────
        if is_separator(line):
            flush_code()
            i += 1
            continue

        # ── H1: border / title / border ───────────────────────────────────────
        if is_border(line):
            flush_code()
            # peek ahead for title line + closing border
            if i + 2 < len(lines) and is_border(lines[i + 2]):
                title = extract_h1_title(lines[i + 1])
                cells.append({"type": "markdown", "source": f"# {title}"})
                i += 3
                continue
            # lone border with no matching pair → treat as code comment
            pending_code.append(line)
            i += 1
            continue

        # ── H2: text line followed immediately by a border ───────────────────
        if (line.startswith("#") and not is_border(line) and not is_separator(line)
                and i + 1 < len(lines) and is_border(lines[i + 1])):
            flush_code()
            title = strip_comment(line)
            cells.append({"type": "markdown", "source": f"## {title}"})
            i += 2          # skip the border line
            continue

        # ── comment line (potential H3 or markdown paragraph) ────────────────
        if line.startswith("# #") and not is_border(line) and not is_separator(line):
            flush_code()
            # Collect a run of comment lines (none of which are borders/separators)
            comment_block: list[str] = []
            while (i < len(lines)
                   and lines[i].startswith("# #")
                   and not is_border(lines[i])
                   and not is_separator(lines[i])):
                comment_block.append(strip_comment(lines[i]))
                i += 1

            # If exactly one line it's H3; multiple lines are a paragraph
            if len(comment_block) == 1:
                md = f"### {comment_block[0]}"
            else:
                md = "\n".join(comment_block)
            cells.append({"type": "markdown", "source": md})
            continue

        # ── everything else is code ───────────────────────────────────────────
        pending_code.append(line)
        i += 1

    flush_code()
    return cells


# ── notebook builder ──────────────────────────────────────────────────────────

def make_notebook(cells: list[dict]) -> dict:
    nb_cells = []
    for c in cells:
        src_lines = c["source"].splitlines(keepends=True)
        # last line should not have a trailing newline in the Jupyter format
        if src_lines and src_lines[-1].endswith("\n"):
            src_lines[-1] = src_lines[-1][:-1]

        if c["type"] == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": src_lines,
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src_lines,
            })

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": nb_cells,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a nb_to_py-generated .py file back to a Jupyter notebook."
    )
    parser.add_argument("script", help="Path to the .py file")
    parser.add_argument("-o", "--output", help="Output .ipynb path (default: same name as script)")
    parser.add_argument("--stdout", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()

    py_path = Path(args.script)
    if not py_path.exists():
        sys.exit(f"Error: file not found: {py_path}")

    source = py_path.read_text(encoding="utf-8")
    cells = parse_cells(source)
    nb = make_notebook(cells)
    nb_json = json.dumps(nb, indent=1, ensure_ascii=False)

    if args.stdout:
        print(nb_json)
    else:
        out_path = Path(args.output) if args.output else py_path.with_suffix(".ipynb")
        out_path.write_text(nb_json + "\n", encoding="utf-8")
        print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()