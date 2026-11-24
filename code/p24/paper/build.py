"""Build the paper number macros (S5 item 5.1).

    python -m p24.paper.build [--root R] [--out-dir D] [--extra-registry CSV ...]

Writes  <out-dir>/numbers.tex  (one \\newcommand per number) and  <out-dir>/numbers_manifest.csv
(macro, value, fmt, source_path, selector, source_sha16, script, used_in).
Default out-dir: paper/paper.  Output is deterministic (sorted by macro, no timestamps).
CPU only, standard library only (no torch / numpy).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from p24.paper import registry as reg_mod
from p24.paper.registry import Reg

DEFAULT_ROOT = Path(__file__).resolve().parents[3]          # <project>/code/p24/paper/build.py -> <project>
DEFAULT_OUT = "paper/paper"
MANIFEST_COLUMNS = ["macro", "value", "fmt", "source_path", "selector", "source_sha16", "script", "used_in"]
AGGS = ("min", "max", "mean", "median", "sum", "count")
MACRO_RE = re.compile(r"^N[A-Z][A-Za-z]*$")


class NumberError(Exception):
    """Selector / format / source problem; message names the macro when raised from compute_entry."""


# ----------------------------------------------------------------------------- sources
def sha16(path: Path) -> str:
    """First 16 hex chars of the SHA-256 of the file bytes (project convention)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_source(path: Path):
    suf = path.suffix.lower()
    if suf == ".json":
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    if suf == ".csv":
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    raise NumberError(f"unsupported source type {suf!r}: {path}")


def _split_agg(selector: str):
    if "#" in selector:
        head, _, tail = selector.rpartition("#")
        if tail == "len" or tail in AGGS:
            return head, tail
    return selector, None


def select_json(obj, selector: str):
    path, op = _split_agg(selector)
    cur = obj
    for tok in path.split(".") if path else []:
        if isinstance(cur, list):
            try:
                cur = cur[int(tok)]
            except (ValueError, IndexError):
                raise NumberError(f"list index {tok!r} invalid in selector {selector!r}")
        elif isinstance(cur, dict):
            if tok not in cur:
                raise NumberError(f"key {tok!r} not found in selector {selector!r}")
            cur = cur[tok]
        else:
            raise NumberError(f"cannot descend into {type(cur).__name__} at {tok!r} in selector {selector!r}")
    if op == "len":
        if not isinstance(cur, (list, dict, str)):
            raise NumberError(f"#len on non-container in selector {selector!r}")
        return len(cur)
    if op is not None:
        raise NumberError(f"aggregation #{op} is CSV-only (selector {selector!r})")
    return cur


def _num(cell: str, what: str) -> float:
    if cell is None or str(cell).strip() == "":
        raise NumberError(f"empty cell in {what}")
    try:
        v = float(cell)
    except ValueError:
        raise NumberError(f"non-numeric cell {cell!r} in {what}")
    if not math.isfinite(v):
        raise NumberError(f"non-finite cell {cell!r} in {what}")
    return v


def select_csv(rows: List[Dict[str, str]], selector: str):
    body, agg = _split_agg(selector)
    col, _, filt = body.partition("?")
    if not col:
        raise NumberError(f"missing column in selector {selector!r}")
    if rows and col not in rows[0]:
        raise NumberError(f"column {col!r} not in CSV header (selector {selector!r})")
    conds = []
    for part in [p for p in filt.split("&") if p]:
        k, eq, v = part.partition("=")
        if not eq:
            raise NumberError(f"bad filter {part!r} in selector {selector!r}")
        if rows and k not in rows[0]:
            raise NumberError(f"filter column {k!r} not in CSV header (selector {selector!r})")
        conds.append((k, v))
    hit = [r for r in rows if all(r.get(k) == v for k, v in conds)]
    if agg is None:
        if len(hit) != 1:
            raise NumberError(f"selector {selector!r} matched {len(hit)} rows (need exactly 1)")
        cell = hit[0][col]
        try:
            return _num(cell, selector)
        except NumberError:
            if str(cell).strip() == "":
                raise
            return cell  # non-numeric string cell (use fmt 's')
    if agg == "len":
        raise NumberError("#len is JSON-only")
    if not hit:
        raise NumberError(f"selector {selector!r} matched 0 rows")
    if agg == "count":
        return len(hit)
    vals = [_num(r[col], selector) for r in hit]
    if agg == "min":
        return min(vals)
    if agg == "max":
        return max(vals)
    if agg == "sum":
        return math.fsum(vals)
    if agg == "mean":
        return math.fsum(vals) / len(vals)
    return statistics.median(vals)


def select(data, selector: str, source_suffix: str):
    return select_json(data, selector) if source_suffix == ".json" else select_csv(data, selector)


# ----------------------------------------------------------------------------- formatting
_FMT_RE = re.compile(r"^(?:([*/])([0-9.eE+-]+):)?(.+)$")


def format_value(value, fmt: str) -> str:
    """Format a selected value.  See registry.py docstring for the fmt grammar."""
    if fmt == "s":
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise NumberError(f"fmt 's' needs a string or integer value, got {value!r}")
        return str(value)
    m = _FMT_RE.match(fmt)
    if not m:
        raise NumberError(f"bad fmt {fmt!r}")
    op, k, spec = m.groups()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumberError(f"fmt {fmt!r} needs a number, got {value!r}")
    v = float(value)
    if op:
        kk = float(k)
        v = v * kk if op == "*" else v / kk
    if not math.isfinite(v):
        raise NumberError(f"non-finite value for fmt {fmt!r}")
    if spec.endswith("d"):
        if abs(v - round(v)) > 1e-9 * max(1.0, abs(v)):
            raise NumberError(f"fmt {fmt!r} needs an integer-valued number, got {v!r}")
        out = format(int(round(v)), spec)
    else:
        out = format(v, spec)
    if re.fullmatch(r"[-+]?-0(\.0+)?", out):      # negative zero after rounding -> plain zero
        out = out.replace("-", "", 1)
    return out


_TEX_ESC = {"\\": r"\textbackslash{}", "#": r"\#", "$": r"\$", "%": r"\%", "&": r"\&", "_": r"\_", "{": r"\{", "}": r"\}"}
_MINUS = r"$-$"


def tex_body(value: str) -> str:
    """Escape a value for use as a macro body; a leading ASCII minus becomes a typeset math minus."""
    s = "".join(_TEX_ESC.get(ch, ch) for ch in value)
    if s.startswith("-"):
        s = _MINUS + s[1:]
    return s


def tex_unbody(body: str) -> str:
    """Inverse of tex_body (used by check.py for criterion (a))."""
    s = body
    if s.startswith(_MINUS):
        s = "-" + s[len(_MINUS):]
    s = re.sub(r"\\([#$%&_{}])", r"\1", s)
    return s.replace(r"\textbackslash{}", "\\")


# ----------------------------------------------------------------------------- build
def validate_registry(registry: Sequence[Reg]) -> None:
    seen = set()
    for r in registry:
        if not MACRO_RE.match(r.macro):
            raise NumberError(f"macro name {r.macro!r} must match {MACRO_RE.pattern} (letters only, prefix N+Upper)")
        if r.macro in seen:
            raise NumberError(f"duplicate macro {r.macro!r}")
        seen.add(r.macro)


def compute_entry(root: Path, r: Reg, cache: Dict[Path, object]) -> Dict[str, str]:
    src = Path(root) / r.source_path
    if not src.is_file():
        raise NumberError(f"{r.macro}: source file missing: {r.source_path}")
    try:
        if src not in cache:
            cache[src] = read_source(src)
        value = select(cache[src], r.selector, src.suffix.lower())
        text = format_value(value, r.fmt)
    except NumberError as exc:
        raise NumberError(f"{r.macro}: {exc}") from None
    return {"macro": r.macro, "value": text, "fmt": r.fmt, "source_path": r.source_path, "selector": r.selector,
            "source_sha16": sha16(src), "script": r.script, "used_in": r.used_in}


def load_registry(root: Path, extra_files: Sequence[Path] = (), include_default_extra: bool = True) -> List[Reg]:
    regs = list(reg_mod.REGISTRY)
    extras = list(extra_files)
    if include_default_extra:
        d = Path(root) / reg_mod.EXTRA_DIR
        if d.is_dir():
            extras += sorted(d.glob("*.csv"))
    for p in extras:
        regs += reg_mod.load_extra(Path(p))
    return regs


def compute_entries(root: Path, registry: Sequence[Reg]) -> List[Dict[str, str]]:
    validate_registry(registry)
    cache: Dict[Path, object] = {}
    return sorted((compute_entry(root, r, cache) for r in registry), key=lambda e: e["macro"])


def render_numbers_tex(entries: Sequence[Dict[str, str]]) -> str:
    lines = ["% AUTO-GENERATED by `python -m p24.paper.build` -- do not edit by hand.",
             "% One macro per number; sources, selectors and SHA16 are in numbers_manifest.csv."]
    for e in entries:
        lines.append("\\newcommand{\\%s}{%s}" % (e["macro"], tex_body(e["value"])))
    return "\n".join(lines) + "\n"


def render_manifest_csv(entries: Sequence[Dict[str, str]]) -> str:
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(MANIFEST_COLUMNS)
    for e in entries:
        w.writerow([e[c] for c in MANIFEST_COLUMNS])
    return buf.getvalue()


def build(root: Path = DEFAULT_ROOT, out_dir: Optional[Path] = None, registry: Optional[Sequence[Reg]] = None,
          extra_files: Sequence[Path] = (), include_default_extra: bool = True) -> Dict[str, object]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / DEFAULT_OUT
    regs = list(registry) if registry is not None else load_registry(root, extra_files, include_default_extra)
    entries = compute_entries(root, regs)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path, man_path = out_dir / "numbers.tex", out_dir / "numbers_manifest.csv"
    tex_path.write_text(render_numbers_tex(entries), encoding="utf-8")
    man_path.write_text(render_manifest_csv(entries), encoding="utf-8")
    return {"n": len(entries), "numbers_tex": tex_path, "manifest": man_path,
            "numbers_sha16": sha16(tex_path), "manifest_sha16": sha16(man_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="project root (default: derived from this file)")
    ap.add_argument("--out-dir", default=None, help=f"output dir (default <root>/{DEFAULT_OUT})")
    ap.add_argument("--extra-registry", action="append", default=[], help="extra registry CSV (repeatable)")
    ap.add_argument("--no-default-extra", action="store_true", help=f"do not auto-load <root>/{reg_mod.EXTRA_DIR}/*.csv")
    a = ap.parse_args(argv)
    try:
        res = build(Path(a.root), Path(a.out_dir) if a.out_dir else None, None,
                    [Path(p) for p in a.extra_registry], not a.no_default_extra)
    except NumberError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"built {res['n']} macros")
    print(f"  {res['numbers_tex']}  sha16={res['numbers_sha16']}")
    print(f"  {res['manifest']}  sha16={res['manifest_sha16']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
