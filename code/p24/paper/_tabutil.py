"""Small helpers shared by the table scripts (S5 P0-4).  CPU only, standard library only."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, List, Sequence

from p24.paper.registry import EXTRA_COLUMNS, Reg


def registry_csv_text(regs: Iterable[Reg]) -> str:
    """Text of a registry_extra CSV (columns macro,source_path,selector,fmt,used_in,script), rows in the given order, '\\n' line ends."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EXTRA_COLUMNS)
    for r in regs:
        w.writerow([r.macro, r.source_path, r.selector, r.fmt, r.used_in, r.script])
    return buf.getvalue()


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def macros_in(tex: str) -> List[str]:
    import re
    return sorted(set(re.findall(r"\\(N[A-Z][A-Za-z]*)", tex)))
