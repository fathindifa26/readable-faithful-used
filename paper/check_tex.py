#!/usr/bin/env python
"""Cek struktural LaTeX tanpa perlu pdflatex (biar bisa dijalankan di lokal).

Yang dicek:
  1. \\begin{X} vs \\end{X} berimbang per file
  2. kurung kurawal berimbang per file (abaikan \\{ \\} dan comment)
  3. \\ref/\\citep menunjuk ke \\label/entri bib yang ada
  4. \\includegraphics menunjuk ke file yang ada di figures/
  5. sisa \\todo / \\pending (ringkasan, bukan error)

Jalankan:  python paper/check_tex.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = [ROOT / "main.tex"] + sorted((ROOT / "sections").glob("*.tex"))
BIB = ROOT / "references.bib"
FIGDIR = ROOT / "figures"

COMMENT = re.compile(r"(?<!\\)%.*$", re.M)
problems, notes = [], []


def strip(t):
    return COMMENT.sub("", t)


labels, refs, cites, figs = set(), [], [], []
for f in FILES:
    if f.name == "00_title_options.tex":
        continue  # catatan kerja, tidak di-include
    raw = f.read_text(encoding="utf-8")
    t = strip(raw)

    envs = {}
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", t):
        envs.setdefault(name, [0, 0])[0 if kind == "begin" else 1] += 1
    for name, (b, e) in envs.items():
        if b != e:
            problems.append(f"{f.name}: environment '{name}' begin={b} end={e}")

    depth = 0
    for m in re.finditer(r"(?<!\\)[{}]", t):
        depth += 1 if m.group() == "{" else -1
        if depth < 0:
            problems.append(f"{f.name}: '}}' berlebih di posisi {m.start()}")
            break
    if depth > 0:
        problems.append(f"{f.name}: {depth} '{{' tidak ditutup")

    labels |= set(re.findall(r"\\label\{([^}]+)\}", t))
    refs += [(f.name, r) for r in re.findall(r"\\(?:eq)?ref\{([^}]+)\}", t)]
    for grp in re.findall(r"\\cite[a-z]*\{([^}]+)\}", t):
        cites += [(f.name, c.strip()) for c in grp.split(",")]
    figs += [(f.name, g) for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t)]

    n_todo = len(re.findall(r"\\todo\{", t))
    n_pend = len(re.findall(r"\\pending\{", t))
    if n_todo or n_pend:
        notes.append(f"{f.name}: {n_todo} todo, {n_pend} pending")

for fname, r in refs:
    if r not in labels:
        problems.append(f"{fname}: \\ref{{{r}}} tidak punya \\label")

bibkeys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
for fname, c in cites:
    if c not in bibkeys:
        problems.append(f"{fname}: \\cite{{{c}}} tidak ada di references.bib")

for fname, g in figs:
    if not any((FIGDIR / (g + ext)).exists() for ext in ("", ".pdf", ".png")):
        problems.append(f"{fname}: gambar '{g}' tidak ditemukan di figures/")

print("label:", len(labels), "| ref:", len(refs), "| sitasi unik:",
      len(set(c for _, c in cites)), "| figur:", len(figs))
for n in notes:
    print("  sisa kerjaan -", n)
unused = bibkeys - set(c for _, c in cites)
if unused:
    print("  bib tidak terpakai:", ", ".join(sorted(unused)))

if problems:
    print("\nMASALAH:")
    for p in problems:
        print("  x", p)
    sys.exit(1)
print("\nstruktur OK")
