#!/usr/bin/env python3
"""Small generated-code canary for DSpark fast-mode smoke tests."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


DECL_RE = re.compile(
    r"^\s*(?:let|const|var)\s+([A-Za-z_$][\w$]*)\s*="
    r"|^\s*this\.([A-Za-z_$][\w$]*)\s*="
    r"|^\s*([A-Za-z_$][\w$]*)\s*="
)
HEX_COLOR_RE = re.compile(r"#([0-9A-Fa-f]+)\b")
CSS_LENGTH_RE = re.compile(
    r":\s*(-?\d+(?:\.\d+)?)\s*(?:;|$)"
)


def iter_paths(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for arg in paths:
        p = Path(arg)
        if p.is_dir():
            candidates = [
                p / "out.txt",
                p / "output.txt",
                *sorted(p.glob("*.out")),
            ]
            outputs_dir = p / "outputs"
            if outputs_dir.is_dir():
                candidates.extend(sorted(outputs_dir.glob("*.txt")))
            out.extend(c for c in candidates if c.is_file())
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"not found: {p}")
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def max_repeated_line_run(lines: list[str]) -> int:
    best = cur = 0
    prev: str | None = None
    for line in lines:
        if line == prev:
            cur += 1
        else:
            cur = 1
            prev = line
        best = max(best, cur)
    return best


def max_ngram_count(tokens: list[str], n: int) -> int:
    if n <= 0 or len(tokens) < n:
        return 0
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return max(grams.values(), default=0)


def has_chained_assignment(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith("<"):
        return False
    if not (
        re.match(r"^(?:let|const|var)\s+[A-Za-z_$][\w$]*\s*=", stripped)
        or re.match(r"^this\.[A-Za-z_$][\w$]*\s*=", stripped)
        or re.match(r"^[A-Za-z_$][\w$]*\s*=", stripped)
    ):
        return False
    if line.count("=") < 2:
        return False
    scrubbed = re.sub(r"===|!==|==|!=|<=|>=|=>", "", line)
    return scrubbed.count("=") >= 2


def declaration_name(line: str) -> str | None:
    m = DECL_RE.match(line)
    if not m:
        return None
    name = next((g for g in m.groups() if g), None)
    if not name:
        return None
    if line.lstrip().startswith(("if ", "for ", "while ", "switch ")):
        return None
    return name


def malformed_html_line(line: str) -> bool:
    stripped = line.strip()
    if re.match(r"^</[A-Za-z][A-Za-z0-9-]*$", stripped):
        return True
    if re.match(r"^<[A-Za-z][^>]*$", stripped) and not stripped.endswith(("/>", ">")):
        return True
    return False


def malformed_css_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(("/*", "*", "//")):
        return False
    for match in HEX_COLOR_RE.finditer(stripped):
        n = len(match.group(1))
        if n not in (3, 4, 6, 8):
            return True
    if CSS_LENGTH_RE.search(stripped):
        prop = stripped.split(":", 1)[0].strip()
        if prop not in {"font-weight", "line-height", "opacity", "z-index", "flex", "order"}:
            return True
    return False


def duplicate_css_selectors(lines: list[str]) -> list[tuple[str, int]]:
    selectors: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.endswith("{"):
            continue
        selector = stripped[:-1].strip()
        if not selector or selector.startswith(("@", "/*")):
            continue
        if selector in {"from", "to"}:
            continue
        selectors.append(selector)
    counts = Counter(selectors)
    return [(name, n) for name, n in counts.most_common() if n > 1]


def analyze(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    names = [name for line in lines if (name := declaration_name(line))]
    name_counts = Counter(names)
    dup_names = [(name, n) for name, n in name_counts.most_common() if n > 1]
    malformed = [line for line in lines if has_chained_assignment(line)]
    malformed_html = [line for line in lines if malformed_html_line(line)]
    malformed_css = [line for line in lines if malformed_css_line(line)]
    dup_selectors = duplicate_css_selectors(lines)
    tokens = text.split()
    max_dup_decl = max((n for _, n in dup_names), default=0)
    max_dup_selector = max((n for _, n in dup_selectors), default=0)
    suspect = int(
        max_repeated_line_run(lines) >= 6
        or max_dup_decl >= 4
        or len(malformed) > 0
        or len(malformed_html) > 0
        or len(malformed_css) > 0
        or max_dup_selector >= 3
        or max_ngram_count(tokens, 8) >= 8
    )
    return {
        "path": str(path),
        "chars": str(len(text)),
        "lines": str(len(lines)),
        "unique_lines": str(len(set(lines))),
        "max_repeated_line_run": str(max_repeated_line_run(lines)),
        "max_duplicate_declaration_count": str(max_dup_decl),
        "duplicate_declaration_names": ",".join(f"{name}:{n}" for name, n in dup_names[:12]),
        "max_duplicate_css_selector_count": str(max_dup_selector),
        "duplicate_css_selectors": ",".join(f"{name}:{n}" for name, n in dup_selectors[:12]),
        "malformed_assignment_lines": str(len(malformed)),
        "malformed_html_lines": str(len(malformed_html)),
        "malformed_css_lines": str(len(malformed_css)),
        "max_4gram_count": str(max_ngram_count(tokens, 4)),
        "max_8gram_count": str(max_ngram_count(tokens, 8)),
        "suspect": str(suspect),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="generated files or run directories")
    args = ap.parse_args()

    fields = [
        "path",
        "chars",
        "lines",
        "unique_lines",
        "max_repeated_line_run",
        "max_duplicate_declaration_count",
        "duplicate_declaration_names",
        "max_duplicate_css_selector_count",
        "duplicate_css_selectors",
        "malformed_assignment_lines",
        "malformed_html_lines",
        "malformed_css_lines",
        "max_4gram_count",
        "max_8gram_count",
        "suspect",
    ]
    print("\t".join(fields))
    for path in iter_paths(args.paths):
        row = analyze(path)
        print("\t".join(row[field] for field in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
