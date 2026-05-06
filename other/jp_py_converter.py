#!/usr/bin/env python3
"""
nb_to_py.py — Convert a Jupyter notebook to a plain Python file.

Header comment styles (auto-detected from markdown heading level):
  H1  → # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #							 Header 1							  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  H2  →  # Header 2
         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  H3  →  # # Header Text

Cell separator (no heading between two code cells):

Usage:
  python nb_to_py.py notebook.ipynb              # writes notebook.py
  python nb_to_py.py notebook.ipynb -o out.py    # custom output path
  python nb_to_py.py notebook.ipynb --stdout     # print to stdout
"""

import argparse
import json
import re
import sys
from pathlib import Path

# formattng constants
WIDTH = 35          # total line width for the hash borders
SEPARATOR = "# " + "─ ─ " * 18 + "─ ─"   # between cells with no heading

def _border(width: int = WIDTH) -> str:
    return "#" + " #" * (width - 2)

def format_h1(text: str) -> str:
    b = _border()
    return f"{b}\n#\t\t\t\t\t\t\t {text}\t\t\t\t\t\t\t  #\n{b}"

def format_h2(text: str) -> str:
    b = _border()
    return f"# {text}\n{b}"


def format_h3(text: str) -> str:
    return f"# # {text}"


def markdown_cell_to_comments(source: str) -> list[str]:
    """
    Convert a markdown cell to a list of comment blocks.
    Each heading becomes a styled block; plain paragraphs become # lines.
    Returns a list of strings (one per logical block).
    """
    blocks = []
    current_para: list[str] = []

    def flush_para():
        if current_para:
            blocks.append("\n".join(f"# {line}" for line in current_para))
            current_para.clear()

    for line in source.splitlines():
        h1 = re.match(r"^#\s+(.*)", line)
        h2 = re.match(r"^##\s+(.*)", line)
        h3 = re.match(r"^###\s+(.*)", line)

        if h3:
            flush_para()
            blocks.append(format_h3(h3.group(1).strip()))
        elif h2:
            flush_para()
            blocks.append(format_h2(h2.group(1).strip()))
        elif h1:
            flush_para()
            blocks.append(format_h1(h1.group(1).strip()))
        elif line.strip() == "":
            flush_para()
        else:
            current_para.append(line)

    flush_para()
    return blocks


def notebook_to_python(nb: dict) -> str:
    """Convert a parsed notebook dict to a Python source string."""
    cells = nb.get("cells", [])
    output_parts: list[str] = []
    prev_was_code = False

    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown":
            blocks = markdown_cell_to_comments(source)
            if blocks:
                output_parts.extend(blocks)
                prev_was_code = False

        elif cell_type == "code":
            if not source.strip():
                continue  # skip empty code cells
            if prev_was_code:
                output_parts.append(SEPARATOR)
            output_parts.append(source.rstrip())
            prev_was_code = True

        # raw cells: include as block comments
        elif cell_type == "raw":
            if source.strip():
                raw_lines = "\n".join(f"# {l}" for l in source.splitlines())
                output_parts.append(raw_lines)
                prev_was_code = False

    return "\n\n".join(output_parts) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Jupyter notebook (.ipynb) to a Python (.py) file."
    )
    parser.add_argument("--notebook", help="Path to the .ipynb file")
    parser.add_argument("-o", "--output", help="Output .py file path (default: same name as notebook)")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of writing a file")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    if not nb_path.exists():
        sys.exit(f"Error: file not found: {nb_path}")
    if nb_path.suffix.lower() != ".ipynb":
        print(f"Warning: expected .ipynb extension, got '{nb_path.suffix}'", file=sys.stderr)

    with nb_path.open(encoding="utf-8") as f:
        nb = json.load(f)

    py_source = notebook_to_python(nb)

    if args.stdout:
        print(py_source, end="")
    else:
        out_path = Path(args.output) if args.output else nb_path.with_suffix(".py")
        out_path.write_text(py_source, encoding="utf-8")
        print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()