"""Check paper numbers against their frozen sources (S5 item 5.1, criteria of goals/S5_detailed_goals_v1.md item 21).

    python -m p24.paper.check [--root R] [--paper-dir D] [--tex main.tex ...] [--whitelist W] [--sections IV,V]

  (a) recomputed value == macro body in numbers.tex (character for character) and == manifest value; mismatches = 0
  (b) SHA16 of every source file == manifest source_sha16; mismatches = 0
  (c) "result-style" numbers hard-coded in the abstract, sections IV and V, figure/table environments (incl. \\input
      tables) that do not go through a macro, minus whitelist, = 0.  Result-style: decimal, %, x-times, number+unit
      (s/ms/GB/MB), CI brackets, integers with >=3 digits.  Auto-excluded: letter/#/_ prefixed identifiers (H4a, B0, W2,
      #17, pytorch#203030, SD1.5, RTX-4090), section/figure/table/equation numbers, 4-digit years 1990-2039, version
      strings (x.y.z or 'torch 2.9'), TeX dimensions (1.9in, 0.9\\columnwidth) and arguments of \\cite/\\ref/\\label/
      \\input/\\includegraphics/\\setlength/... ; whitelist <= 30 entries, each with a reason.
  (d) manifest macros never referenced in the tex files: WARNING only.
  (e) (extra safety, not in the goal text) tex references a \\N<Upper>... macro that is not in the manifest/numbers.tex:
      counted as a failure because it would not compile.

Exit code 0 iff (a)=(b)=(c)=(e)=0 and the whitelist is well-formed.  CPU only, standard library only.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from p24.paper import build as bld

DEFAULT_ROOT = bld.DEFAULT_ROOT
DEFAULT_PAPER_DIR = "paper/paper"
WHITELIST_MAX = 30
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
TEX_BUILTIN_N = {"NeedsTeXFormat", "NewDocumentCommand", "NewExpandableDocumentCommand", "NewDocumentEnvironment",
                 "NewCommandCopy", "NewEnviron", "NewTotalDocumentCommand"}


@dataclass
class Hit:
    file: str
    line: int
    kind: str
    text: str     # matched text as normalised
    num: str      # numeric core (for whitelist matching)

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: [{self.kind}] {self.text}"


@dataclass
class CheckResult:
    a_mismatches: List[str] = field(default_factory=list)
    b_mismatches: List[str] = field(default_factory=list)
    c_hits: List[Hit] = field(default_factory=list)            # non-whitelisted result-style numbers
    c_whitelisted: List[Hit] = field(default_factory=list)
    d_unused_macros: List[str] = field(default_factory=list)   # warning only
    e_undefined_macros: List[str] = field(default_factory=list)
    f_placeholder_macros: List[str] = field(default_factory=list)   # S5 P0-4: \N...TBD placeholders (blocks not frozen yet); warning only
    whitelist_errors: List[str] = field(default_factory=list)
    whitelist_unused: List[str] = field(default_factory=list)
    n_macros: int = 0
    n_whitelist: int = 0
    scanned_lines: int = 0

    @property
    def ok(self) -> bool:
        return not (self.a_mismatches or self.b_mismatches or self.c_hits or self.e_undefined_macros
                    or self.whitelist_errors)


# ----------------------------------------------------------------------------- manifest / numbers.tex
_DEF_RE = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def read_numbers_tex(path: Path) -> Dict[str, str]:
    macros: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("%") or not line.strip():
            continue
        m = _DEF_RE.match(line)
        if not m:
            raise ValueError(f"unparsable line in {path.name}: {line!r}")
        macros[m.group(1)] = m.group(2)
    return macros


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and list(rows[0].keys()) != bld.MANIFEST_COLUMNS:
        raise ValueError(f"manifest columns {list(rows[0].keys())} != {bld.MANIFEST_COLUMNS}")
    return rows


def check_a_b(root: Path, manifest: List[Dict[str, str]], macros: Dict[str, str], res: CheckResult) -> None:
    cache: Dict[Path, object] = {}
    man_names = {r["macro"] for r in manifest}
    for name in sorted(set(macros) - man_names):
        res.a_mismatches.append(f"{name}: defined in numbers.tex but not in manifest")
    for r in manifest:
        name = r["macro"]
        src = Path(root) / r["source_path"]
        # (b) SHA16
        if not src.is_file():
            res.b_mismatches.append(f"{name}: source missing {r['source_path']}")
        else:
            now = bld.sha16(src)
            if now != r["source_sha16"]:
                res.b_mismatches.append(f"{name}: SHA16 {now} != manifest {r['source_sha16']} ({r['source_path']})")
        # (a) recomputed value vs manifest and macro
        try:
            reg = bld.Reg(name, r["source_path"], r["selector"], r["fmt"], r["used_in"], r["script"])
            recomputed = bld.compute_entry(root, reg, cache)["value"]
        except bld.NumberError as exc:
            res.a_mismatches.append(f"{name}: cannot recompute ({exc})")
            continue
        if recomputed != r["value"]:
            res.a_mismatches.append(f"{name}: recomputed {recomputed!r} != manifest value {r['value']!r}")
        if name not in macros:
            res.a_mismatches.append(f"{name}: in manifest but not defined in numbers.tex")
        else:
            body = bld.tex_unbody(macros[name])
            if body != recomputed:
                res.a_mismatches.append(f"{name}: numbers.tex has {macros[name]!r}, recomputed {recomputed!r}")


# ----------------------------------------------------------------------------- whitelist
def read_whitelist(path: Optional[Path]) -> Tuple[Dict[str, str], List[str]]:
    wl: Dict[str, str] = {}
    errs: List[str] = []
    if path is None or not Path(path).is_file():
        return wl, errs
    for i, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tok, sep, reason = line.partition("|")
        tok, reason = tok.strip(), reason.strip()
        if not sep or not tok or not reason:
            errs.append(f"{Path(path).name}:{i}: entry needs 'token | reason' (reason must be non-empty)")
            continue
        if tok in wl:
            errs.append(f"{Path(path).name}:{i}: duplicate token {tok!r}")
        wl[tok] = reason
    if len(wl) > WHITELIST_MAX:
        errs.append(f"{Path(path).name}: {len(wl)} entries > limit {WHITELIST_MAX}")
    return wl, errs


# ----------------------------------------------------------------------------- tex scanning
_COMMENT_RE = re.compile(r"(?<!\\)%.*$")
_SKIP_LINE_RE = re.compile(r"^\s*\\(?:newcommand|renewcommand|providecommand|def|newif|setlength|usepackage|"
                           r"documentclass|geometry|bibliographystyle|bibliography)\b")
_ARG_STRIP_RE = re.compile(
    r"\\(?:cite[a-z]*|ref|eqref|pageref|cref|Cref|label|input|include|includegraphics|url|href|bibitem|"
    r"vspace\*?|hspace\*?|vskip|hskip|columnsep|tabcolsep|arraystretch|scalebox|rotatebox|raisebox)"
    r"\*?\s*(?:\[[^\]]*\])?\s*(?:\{[^{}]*\})?"
    r"|\\(?:setlength|addtolength|setcounter|renewcommand|newcommand|providecommand)\*?\s*"
    r"(?:\{\\?[A-Za-z@]+\}|\\[A-Za-z@]+)\s*(?:\{[^{}]*\})?"
    r"|\\resizebox\*?\s*\{[^{}]*\}\s*\{[^{}]*\}")
_DIM_RE = re.compile(r"\d*\.?\d+\s*(?:in|pt|em|ex|cm|mm|bp|pc)\b|\d*\.?\d+\s*\\(?:columnwidth|textwidth|linewidth|"
                     r"textheight|hsize|vsize)")
_NUM = r"[-+]\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?"
_DEC = r"[-+]\s?\d+\.\d+|\d+\.\d+"
_PRE = r"(?<![A-Za-z#_\d.])(?<![A-Za-z]-)"
_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("ci", re.compile(r"(?<![A-Za-z\d])\[\s*(?:" + _NUM + r")\s*%?\s*[,;]\s*(?:" + _NUM + r")\s*%?\s*\]")),
    ("unit", re.compile(_PRE + r"(?:" + _NUM + r")\s?(?:ms|s|GB|MB|GiB|MiB)\b")),
    ("pct", re.compile(_PRE + r"(?:" + _NUM + r")\s?%")),
    ("times", re.compile(_PRE + r"(?:" + _NUM + r")\s?×(?!\s?\d)")),
    ("decimal", re.compile(_PRE + r"(?:" + _DEC + r")(?!\.\d)")),
    ("count", re.compile(r"(?<![A-Za-z#_\d.,])(?<![A-Za-z]-)(?:\d{1,3}(?:,\d{3})+|\d{3,})(?!\d|,\d|\.\d)")),
]
_CONTEXT_EXCL = re.compile(r"(?<![A-Za-z])(?:Section|Sec\.?|Sections|§|Fig\.?|Figs\.?|Figure|Tab\.?|Table|Eq\.?|Eqs\.?|Algorithm|Alg\.?|"
                           r"Appendix|Theorem|Lemma|torch|PyTorch|diffusers|CUDA|cuDNN|Python|Triton|version|ver\.?|v)\s*$")
_NUMCORE_RE = re.compile(r"[-+]?\s?\d[\d,]*(?:\.\d+)?")
_EVENT_RE = re.compile(r"\\(begin|end)\{(abstract|figure\*?|table\*?)\}|\\(section)\{")
_MACRO_USE_RE = re.compile(r"\\(N[A-Z][A-Za-z]*)")


def _normalise(line: str) -> str:
    s = _COMMENT_RE.sub("", line)
    s = _ARG_STRIP_RE.sub(" ", s)
    s = _DIM_RE.sub(" ", s)
    s = (s.replace("{,}", ",").replace("\\times", "×").replace("\\%", "%").replace("\\textminus", "-")
          .replace("−", "-").replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ").replace("~", " ")
          .replace("$", "").replace("{", "").replace("}", ""))
    return s


def scan_line(norm: str) -> List[Tuple[str, str, str]]:
    """Return [(kind, text, num)] result-style numbers found in one normalised line."""
    taken: List[Tuple[int, int]] = []
    out: List[Tuple[str, str, str]] = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(norm):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            text = m.group(0).strip()
            before = norm[:s]
            if _CONTEXT_EXCL.search(before):
                continue
            if kind == "ci" and "." not in text and "%" not in text:
                continue                      # integer-only bracket pair: a citation list, not a CI
            if kind == "count":
                digits = text.replace(",", "")
                if len(digits) == 4 and "," not in text and 1990 <= int(digits) <= 2039:
                    continue                  # year
            taken.append((s, e))
            core = _NUMCORE_RE.search(text)
            num = text if kind == "ci" else (core.group(0).replace(" ", "") if core else text)
            out.append((kind, text, num))
    return out


def _roman_or_int(tok: str) -> int:
    tok = tok.strip().upper()
    return ROMAN[tok] if tok in ROMAN else int(tok)


def _scan(path: Path, scan_sections: Set[int], root: Path, forced: bool, seen: Set[Tuple[Path, bool, int]],
          section0: int) -> Tuple[List[Hit], int, Set[str], int]:
    """Scan one tex file (recursing into \\input).  Returns (hits, scanned line count, macros used, final section no.)."""
    path = Path(path).resolve()
    key = (path, forced, section0)
    if key in seen:
        return [], 0, set(), section0
    seen.add(key)
    hits: List[Hit] = []
    n_scanned = 0
    used: Set[str] = set()
    in_abstract, float_depth, section = False, 0, section0
    try:
        label = str(path.relative_to(Path(root).resolve()))
    except ValueError:
        label = str(path)
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        nocomment = _COMMENT_RE.sub("", raw)
        used.update(_MACRO_USE_RE.findall(nocomment))
        active_start = forced or in_abstract or float_depth > 0
        begin_seen = False
        for m in _EVENT_RE.finditer(nocomment):
            kind, env, sec = m.group(1), m.group(2), m.group(3)
            if sec:
                section += 1
            elif env == "abstract":
                in_abstract = kind == "begin"
                begin_seen = begin_seen or in_abstract
            elif kind == "begin":
                float_depth += 1
                begin_seen = True
            else:
                float_depth = max(0, float_depth - 1)
        # the text of a \section line belongs to the NEW section, hence `section` is read after the events
        active_here = active_start or begin_seen or section in scan_sections
        if _SKIP_LINE_RE.match(nocomment):
            continue
        if active_here:
            n_scanned += 1
            for kind, text, num in scan_line(_normalise(raw)):
                hits.append(Hit(label, lineno, kind, text, num))
        for inp in re.findall(r"\\input\{([^}]+)\}", nocomment):
            if Path(inp).name in ("numbers.tex", "numbers"):
                continue
            cand = path.parent / inp
            cand = cand if cand.suffix else cand.with_suffix(".tex")
            if cand.is_file():
                # float/abstract context is inherited (whole child scanned); section context continues via section0
                h2, n2, u2, section = _scan(cand, scan_sections, root, active_start or begin_seen, seen, section)
                hits += h2
                n_scanned += n2
                used |= u2
    return hits, n_scanned, used, section


def scan_tex_file(path: Path, scan_sections: Set[int], root: Path, forced: bool = False,
                  seen: Optional[Set[Tuple[Path, bool, int]]] = None) -> Tuple[List[Hit], int, Set[str]]:
    """Scan one tex file. Returns (hits, scanned line count, macro names referenced anywhere in the file)."""
    hits, n, used, _ = _scan(path, scan_sections, root, forced, seen if seen is not None else set(), 0)
    return hits, n, used


# ----------------------------------------------------------------------------- driver
def run_check(root: Path = DEFAULT_ROOT, paper_dir: Optional[Path] = None, tex_files: Sequence[Path] = (),
              whitelist: Optional[Path] = None, sections: Sequence[int] = (4, 5)) -> CheckResult:
    root = Path(root)
    paper_dir = Path(paper_dir) if paper_dir else root / DEFAULT_PAPER_DIR
    res = CheckResult()
    manifest = read_manifest(paper_dir / "numbers_manifest.csv")
    macros = read_numbers_tex(paper_dir / "numbers.tex")
    res.n_macros = len(manifest)
    check_a_b(root, manifest, macros, res)

    wl_path = Path(whitelist) if whitelist else paper_dir / "number_whitelist.txt"
    wl, res.whitelist_errors = read_whitelist(wl_path)
    res.n_whitelist = len(wl)

    tex_list = [Path(t) for t in tex_files] or [paper_dir / "main.tex"]
    used_all: Set[str] = set()
    used_wl: Set[str] = set()
    seen: Set[Tuple[Path, bool, int]] = set()
    for t in tex_list:
        hits, n, used = scan_tex_file(t, set(sections), root, False, seen)
        res.scanned_lines += n
        used_all |= used
        for h in hits:
            key = h.num if h.num in wl else (h.text if h.text in wl else None)
            if key is not None:
                res.c_whitelisted.append(h)
                used_wl.add(key)
            else:
                res.c_hits.append(h)
    res.whitelist_unused = sorted(set(wl) - used_wl)
    names = {r["macro"] for r in manifest}
    res.d_unused_macros = sorted(names - used_all)
    undefined = sorted(m for m in used_all - names - TEX_BUILTIN_N if m not in macros)
    res.f_placeholder_macros = [m for m in undefined if m.endswith("TBD")]        # placeholders: defined in the tex via \providecommand
    res.e_undefined_macros = [m for m in undefined if not m.endswith("TBD")]
    return res


def format_report(res: CheckResult, limit: int = 25) -> str:
    L = [f"macros in manifest: {res.n_macros}; scanned tex lines (abstract/sections/floats): {res.scanned_lines}",
         f"(a) value mismatches        : {len(res.a_mismatches)}",
         f"(b) SHA16 mismatches        : {len(res.b_mismatches)}",
         f"(c) result-style numbers not via macro (after whitelist {res.n_whitelist}/{WHITELIST_MAX}): {len(res.c_hits)}"
         f"   [whitelisted occurrences: {len(res.c_whitelisted)}]",
         f"(d) manifest macros unused in tex (warning only): {len(res.d_unused_macros)}",
         f"(e) tex references undefined number macros: {len(res.e_undefined_macros)}",
         f"(f) placeholder macros ending in TBD (warning only; must be 0 at submission): {len(res.f_placeholder_macros)}"]
    for title, items in (("(a)", res.a_mismatches), ("(b)", res.b_mismatches), ("(c)", res.c_hits),
                         ("(e)", res.e_undefined_macros), ("whitelist problems", res.whitelist_errors)):
        for it in list(items)[:limit]:
            L.append(f"  {title} {it}")
        if len(items) > limit:
            L.append(f"  {title} ... {len(items) - limit} more")
    if res.d_unused_macros:
        L.append("  (d) unused: " + ", ".join(res.d_unused_macros[:limit]) + (" ..." if len(res.d_unused_macros) > limit else ""))
    if res.f_placeholder_macros:
        L.append("  (f) placeholders: " + ", ".join(res.f_placeholder_macros[:limit]))
    if res.whitelist_unused:
        L.append("  note: whitelist entries never matched: " + ", ".join(res.whitelist_unused))
    L.append("RESULT: " + ("PASS" if res.ok else "FAIL"))
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--paper-dir", default=None, help=f"dir with numbers.tex, numbers_manifest.csv (default <root>/{DEFAULT_PAPER_DIR})")
    ap.add_argument("--tex", action="append", default=[], help="tex file to scan (repeatable; default <paper-dir>/main.tex)")
    ap.add_argument("--whitelist", default=None, help="default <paper-dir>/number_whitelist.txt")
    ap.add_argument("--sections", default="IV,V", help="section numbers scanned for (c) besides abstract and floats")
    a = ap.parse_args(argv)
    secs = [_roman_or_int(t) for t in a.sections.split(",") if t.strip()]
    try:
        res = run_check(Path(a.root), Path(a.paper_dir) if a.paper_dir else None, [Path(t) for t in a.tex],
                        Path(a.whitelist) if a.whitelist else None, secs)
    except (OSError, ValueError) as exc:
        print(f"CHECK ERROR: {exc}", file=sys.stderr)
        return 2
    print(format_report(res))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
