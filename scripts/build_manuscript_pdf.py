#!/usr/bin/env python3
"""Build the figure-embedded manuscript without editing canonical sources.

Requires Pandoc and XeLaTeX, but no third-party Python packages.
Relative --output paths are resolved from the repository root.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    "1": Path("figures/fig01/fig01_main.pdf"),
    "2": Path("figures/fig02/fig02_main.pdf"),
    "3": Path("figures/fig03/fig03_main.pdf"),
    "4": Path("figures/fig04/fig04_main.pdf"),
    "5": Path("figures/fig05/fig05_main.pdf"),
    "6": Path("figures/fig06/fig06_main.pdf"),
    "S1": Path("figures/supplementary/figS1/figS1_main.pdf"),
    "S2": Path("figures/supplementary/figS2/figS2_main.pdf"),
    "S3": Path("figures/supplementary/figS3/figS3_main.pdf"),
    "S4": Path("figures/supplementary/figS4/figS4_main.pdf"),
}
FIGURES.update({f"S{i}": Path(f"figures/supplementary/figS{i}/figS{i}_main.pdf") for i in range(5,11)})
MAIN = tuple(str(i) for i in range(1, 7))
SUPPLEMENTARY = tuple(f"S{i}" for i in range(1, 11))

HEADER = r"""\usepackage{graphicx}
\usepackage{needspace}
\usepackage{sectsty}
\allsectionsfont{\raggedright}
\usepackage{newunicodechar}
\usepackage{seqsplit}
\setlength{\emergencystretch}{5em}
\sloppy
\pagestyle{plain}
\microtypesetup{protrusion=false}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{φ}{\ensuremath{\phi}}
\newunicodechar{π}{\ensuremath{\pi}}
\newunicodechar{√}{\ensuremath{\surd}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{ŷ}{\ensuremath{\hat{y}}}
\newunicodechar{ȳ}{\ensuremath{\bar{y}}}
\newunicodechar{²}{\textsuperscript{2}}
\newunicodechar{³}{\textsuperscript{3}}
\newunicodechar{¹}{\textsuperscript{1}}
\newunicodechar{⁷}{\textsuperscript{7}}
\newunicodechar{⁸}{\textsuperscript{8}}
\newunicodechar{⁹}{\textsuperscript{9}}
\newunicodechar{⁻}{\textsuperscript{\ensuremath{-}}}
\newunicodechar{ⁿ}{\textsuperscript{n}}
"""
FILTER = r"""function Table(el)
  local presets = {
    [3] = {0.60, 0.17, 0.23},
    [5] = {0.20, 0.21, 0.17, 0.20, 0.22},
    [6] = {0.25, 0.19, 0.14, 0.14, 0.14, 0.14},
    [7] = {0.17, 0.23, 0.12, 0.12, 0.12, 0.12, 0.12}
  }
  local widths = presets[#el.colspecs]
  if widths then
    for i, width in ipairs(widths) do el.colspecs[i] = {pandoc.AlignLeft, width} end
  end
  return el
end
-- Keep section headings with the following paragraph.
function Header(el)
  local lines = el.level <= 2 and 8 or 4
  return {pandoc.RawBlock('latex', '\\Needspace{' .. lines .. '\\baselineskip}'), el}
end
-- Allow long inline code paths/calls to wrap without changing their text.
function Code(el)
  if #el.text <= 8 then return nil end
  local latex = pandoc.write(pandoc.Pandoc({pandoc.Plain({el})}), 'latex')
  local content = latex:match('^\\texttt{(.*)}%s*$')
  if content then
    return pandoc.RawInline('latex', '\\texttt{\\seqsplit{' .. content .. '}}')
  end
end
"""


def split_sections(text, level):
    """Return preamble and exact heading/body chunks at one Markdown level."""
    pattern = re.compile(r"^" + "#" * level + r" ([^\n]+)\n", re.MULTILINE)
    matches = list(pattern.finditer(text))
    chunks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append((match.group(1), text[match.start():end], text[match.end():end]))
    return text[:matches[0].start()] if matches else text, chunks


def figure_sections(text, expected, supplementary=False):
    preamble, sections = split_sections(text, 3)
    prefix = "Supplementary Figure" if supplementary else "Figure"
    found = {}
    for title, chunk, body in sections:
        match = re.match(r"^" + prefix + r" (S?\d+)\. .+", title)
        if not match or match[1] not in expected:
            raise ValueError(f"Unexpected heading in {prefix} section: {title}")
        key = match[1]
        if key in found:
            raise ValueError(f"Duplicate heading: {prefix} {key}")
        if not body.strip():
            raise ValueError(f"Missing text: {prefix} {key}")
        found[key] = (title, chunk, body)
    for key in expected:
        if key not in found:
            raise ValueError(f"Missing heading/legend: {prefix} {key}")
    if tuple(found) != expected:
        raise ValueError(f"{prefix} headings must be in order: {', '.join(expected)}")
    if preamble.strip():
        raise ValueError(f"Unexpected text before first {prefix} heading")
    return found


def figure_block(key, title, body, start_page=True):
    # Relative to the repository root, used as the TeX working directory.
    # Direct includegraphics avoids floating figures and generated captions.
    path = FIGURES[key].as_posix()
    # Reserve more room for the longest formal legends.
    height = "0.58" if key == "S4" else "0.72" if key.startswith("S") else "0.60"
    return (
        ("\n\\clearpage\n\n" if start_page else "\n")
        + f"### {title}\n\n"
        "\\begin{center}\n"
        f"\\includegraphics[width=0.95\\textwidth,height={height}\\textheight,"
        f"keepaspectratio]{{\\detokenize{{{path}}}}}\n"
        "\\end{center}\n\n" + body.strip() + "\n\n\\clearpage\n\n"
    )


def supplementary_tables_markdown(root):
    """Embed concise summary tables and point to complete machine-readable CSVs."""
    import csv
    directory=root/'figures/supplementary/tables'
    required=['TableS1_source_contracts.csv','TableS2_model_unit_estimates.csv','TableS3_crossswap_SS.csv','TableS4_reproduction_inventory.csv']
    if not directory.is_dir():
        return ''
    def read(name):
        path=directory/name
        if not path.is_file():raise ValueError('Missing supplementary table: '+name)
        with path.open(newline='') as stream:return list(csv.DictReader(stream))
    def read_from(directory,name):
        path=directory/name
        if not path.is_file():raise ValueError('Missing supplementary table: '+str(path))
        with path.open(newline='') as stream:return list(csv.DictReader(stream))
    def tab(rows, columns, reflow_compact=False):
        def cell(value):
            value=str(value or '').replace('|','/').replace('\n',' ')
            if reflow_compact:
                # Table S1's source CSV uses compact metadata tokens. Add display-only
                # word boundaries so fixed-width PDF columns can wrap without changing
                # the source values or their interpretation.
                value=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])|(?<=[a-z])(?=[A-Z])',' ',value)
                value=re.sub(r'(?<=[;:])(?=\S)',' ',value)
                for joined,separated in (
                    ('versusdense','versus dense'),
                    ('vsphysical','vs physical'),
                    ('physicalhorizontal','physical horizontal'),
                    ('historicalgray','historical gray'),
                    ('holdoutdesignation','holdout designation'),
                    ('notprospectiveconfirmatory','not prospective confirmatory'),
                    ('oneobservationperpair','one observation per pair'),
                    ('randomdraws','random draws'),
                    ('kspacing','k spacing'),
                    ('conditionaldescriptive','conditional descriptive'),
                ):
                    value=value.replace(joined,separated)
                for compact,display in (
                    ('A 3','A3'),('T 4','T4'),('T 5','T5'),('C 48','C48'),
                    ('8 field','8-field'),('10 k spacing','10k spacing'),
                    ('0–100 k','0–100k'),
                ):
                    value=value.replace(compact,display)
            return value
        return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(cell(r.get(c,'')) for c in columns)+' |\n' for r in rows)+'\n'
    output=['\n\\clearpage\n\n## Supplementary numerical tables\n\n\\begingroup\n\\small\n\n']
    output.append('### Table S1. Cohorts and measurement conditions\n\n'+tab(read(required[0]),['analysis','cohort_units','renderer_sign','support','boundary'],reflow_compact=True))
    estimates=read(required[1]); sources=sorted({r['source_file'] for r in estimates})
    output.append('\n\\clearpage\n\n### Table S2. Summary of model-level analyses\n\nPrimary estimates are reported in the main text and figure legends. Complete model- and seed-level estimates, intervals, and secondary endpoints remain available in `figures/supplementary/tables/TableS2_model_unit_estimates.csv` and the companion individual-estimate CSVs. Source records include `official50_performance_primary_relations` and `comparator_summary`; repeated conditions remain grouped within models.\n\n'+tab([{'analysis source':Path(v).name,'records':str(sum(r['source_file']==v for r in estimates))} for v in sources],['analysis source','records']))
    output.append('\n\\clearpage\n\n### Table S3. Crossed network/decoder summary\n\nFractions are descriptive; residual combines interaction-like departures and inseparable error. Complete estimates and direction records remain in the companion CSV files.\n\n')
    rows=[{'group/window':r['group']+' / '+r['window'],'metric':r['metric'],'effect':r['effect'],'fraction':r['fraction'],'95% interval':(r.get('ci_low','')+' to '+r.get('ci_high','')).strip()} for r in read(required[2])]
    output.append(tab(rows,['group/window','metric','effect','fraction','95% interval']))
    inventory=read(required[3])
    output.append('\n\\clearpage\n\n### Table S4. Reproduction inventory\n\nComplete relative paths, SHA-256 values, and reproduction routes are provided in `figures/supplementary/tables/TableS4_reproduction_inventory.csv` ('+str(len(inventory))+' entries).\n')
    biological=root/'data/precomputed/supplementary/biological_constraints/tables'
    output.append('\n\\clearpage\n\n### Table S5A. T4c phenotype associations with MIDD response\n\nOfficial pretrained cohort; phase observations are summarized within model. Entries are cluster means/proportions with 95% model-bootstrap intervals; omnibus p values are from the saved fixed-size label-permutation tests.\n\n'+tab(read_from(biological,'TableS5A_t4c_phenotype.csv'),['Endpoint','Cluster 0 (n=31)','Cluster 1 (n=7)','Cluster 2 (n=12)','Effect / p (raw / Holm)']))
    output.append('\n\\clearpage\n\n### Table S5B. T4/T5 activity and ON/OFF source-current summaries\n\nOfficial pretrained cohort (N=50); phases are repeated within model. P values are adjusted within the stated primary or secondary analysis family. P3 sign agreement is descriptive/non-confirmatory; the OFF_A association is secondary.\n\n'+tab(read_from(biological,'TableS5B_activity_and_current.csv'),['Analysis / endpoint','Estimate/statistic','95% CI','Raw p','Adjusted p','Status']))
    phase=read_from(root/'data/precomputed/fig04','official50_phase_reliability.csv')[0]
    composition=read_from(root/'data/precomputed/fig03','composition_bootstrap_summary.csv')
    lolo=next(r for r in composition if r['source']=='LOLO' and r['metric']=='cv_r2')
    directions=read_from(root/'data/precomputed/fig06','crossswap_direction_accuracy.csv')
    matched=[r for r in directions if r['matched'].lower()=='true']
    mismatched=[r for r in directions if r['matched'].lower()!='true']
    n_matched=sum(r['both_correct'].lower()=='true' for r in matched)
    n_mismatched=sum(r['both_correct'].lower()=='true' for r in mismatched)
    key_rows=[
        {'Measure':'Absolute-agreement ICC(A,1)','Condition / cohort':'Fig. 4B; official-50, 12 phases','Reported value':f"{float(phase['icc_absolute']):.4f} (95% CI {float(phase['ci_low']):.4f}–{float(phase['ci_high']):.4f})",'Source key':'phase reliability'},
        {'Measure':'Models changing sign across phase','Condition / cohort':'Fig. 4A; official-50, C=48','Reported value':f"{int(phase['sign_changing_models'])}/{int(phase['n_models'])}",'Source key':'phase reliability'},
        {'Measure':'Conditional LOLO mean R²','Condition / cohort':'Fig. 3D; centered eight-field','Reported value':f"{float(lolo['mean']):.4f} (95% CI {float(lolo['ci_low']):.4f}–{float(lolo['ci_high']):.4f})",'Source key':'LOLO composition'},
        {'Measure':'Both physical directions correct','Condition / cohort':'Fig. 6E; matched / mismatched','Reported value':f"{n_matched}/{len(matched)}; {n_mismatched}/{len(mismatched)}",'Source key':'cross-swap directions'},
    ]
    learning=read_from(root/'data/precomputed/fig05','learning_stability_sign_tendency_100k.csv')
    for row in learning:
        condition={'3MIDD_048':'C=48','3MIDD_207':'C=207','4MIDD_085_170':'4MIDD'}[row['group']]
        raw_p=float(row['two_sided_binomial_p'])
        adjusted=min(1.0,3.0*raw_p)
        key_rows.append({'Measure':'100k sign tendency','Condition / cohort':f"Fig. 5E; {condition}",'Reported value':f"{int(row['reported_sign_count'])}/{int(row['n_seeds'])} {row['reported_direction']}; p={raw_p:.4g} (Bonferroni {adjusted:.4g})",'Source key':'learning sign counts'})
    output.append('\n\\clearpage\n\n### Table S6. Key numerical results and source files\n\nLearning-seed p values are two-sided exact binomial tests at 100,000 steps; Bonferroni correction is across the three listed conditions. Source CSV paths are relative to `data/precomputed/`:\n\n- Phase reliability: `fig04/official50_phase_reliability.csv`\n- LOLO composition: `fig03/composition_bootstrap_summary.csv`\n- Cross-swap directions: `fig06/crossswap_direction_accuracy.csv`\n- Learning sign counts: `fig05/learning_stability_sign_tendency_100k.csv`\n\n'+tab(key_rows,['Measure','Condition / cohort','Reported value','Source key']))
    return ''.join(output)+'\n\\endgroup\n'


def prepare_markdown(source, root):
    for key, path in FIGURES.items():
        if not (root / path).is_file():
            raise ValueError(f"Missing figure: Figure {key} ({path})")
    preamble, sections = split_sections(source, 2)
    by_title = {}
    for title, chunk, body in sections:
        if title in by_title:
            raise ValueError(f"Duplicate section: {title}")
        by_title[title] = body
    for title in ("Results", "Figure legends", "Supplementary figure legends"):
        if title not in by_title:
            raise ValueError(f"Missing section: {title}")
    # Support legacy Figure headings and the six scientific Results subsections.
    _, result_chunks = split_sections(by_title["Results"], 3)
    scientific = not any(re.match(r"Figure \d+\.", title) for title, _, _ in result_chunks)
    if scientific:
        if len(result_chunks) != 6 or any(not body.strip() for _, _, body in result_chunks):
            raise ValueError("Scientific Results require six nonempty subsections")
        if len({title for title, _, _ in result_chunks}) != 6:
            raise ValueError("Duplicate heading in scientific Results")
        placement = ("1", "2", "3", "4", "5", "6")
        result_blocks = [(key, chunk) for key, (_, chunk, _) in zip(placement, result_chunks)]
    else:
        results = figure_sections(by_title["Results"], MAIN)
        result_blocks = [(key, results[key][1]) for key in MAIN]
    legends = figure_sections(by_title["Figure legends"], MAIN)
    supplements = figure_sections(
        by_title["Supplementary figure legends"], SUPPLEMENTARY, True
    )
    output = [preamble]
    for title, chunk, body in sections:
        if title == "Figure legends":
            continue  # Formal legend text is moved, not copied.
        if title == "Results":
            output.append("## Results\n\n")
            for key, result_chunk in result_blocks:
                output.append(result_chunk.rstrip() + "\n\n")
                if key is not None:
                    output.append(figure_block(key, legends[key][0], legends[key][2]))
        elif title == "Supplementary figure legends":
            output.append("\n\\clearpage\n\n## Supplementary figure legends\n\n")
            for key in SUPPLEMENTARY:
                output.append(figure_block(key, supplements[key][0], supplements[key][2], start_page=key != "S1"))
        else:
            output.append(chunk)
    return "".join(output) + supplementary_tables_markdown(root)


def find_program(name):
    found = shutil.which(name)
    if found:
        return found
    # macOS GUI sessions may not inherit Homebrew or MacTeX's PATH.
    for directory in ("/opt/homebrew/bin", "/usr/local/bin", "/Library/TeX/texbin", "/opt/anaconda3/bin"):
        found = shutil.which(name, path=directory)
        if found:
            return found
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("manuscript/flyvis_midd_manuscript.pdf"))
    parser.add_argument("--prepare-only", action="store_true",
                        help="Generate intermediate Markdown without PDF dependencies")
    args = parser.parse_args(argv)
    source = ROOT / "manuscript/manuscript.md"
    build = ROOT / "manuscript/_build"
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    if output in {(ROOT / image).resolve() for image in FIGURES.values()}:
        parser.error("--output cannot overwrite a source figure")
    try:
        # Prepare even if PDF dependencies are absent, as a reviewable fallback.
        markdown = prepare_markdown(source.read_text(encoding="utf-8"), ROOT)
        build.mkdir(parents=True, exist_ok=True)
        intermediate = build / "manuscript_with_figures.md"
        intermediate.write_text(markdown, encoding="utf-8")
        header = build / "layout.tex"
        header.write_text(HEADER, encoding="utf-8")
        lua_filter = build / "layout.lua"
        lua_filter.write_text(FILTER, encoding="utf-8")
        print(f"Prepared all {len(FIGURES)} figures and formal legends: {intermediate}")
        if args.prepare_only:
            return 0
        pandoc, xelatex = find_program("pandoc"), find_program("xelatex")
        missing = []
        if not pandoc:
            missing.append("pandoc was not found. Install with: brew install pandoc")
        if not xelatex:
            missing.append("xelatex was not found. Install with: brew install --cask mactex-no-gui")
        if missing:
            raise ValueError("\n".join(missing) + "\nNo software was installed automatically.")
        # Separate conversion/compilation retains diagnostics in _build and
        # makes XeLaTeX's working directory and temporary-output location explicit.
        tex = build / "manuscript_with_figures.tex"
        command = [pandoc, str(intermediate), "--standalone",
                   "--from=markdown+tex_math_single_backslash", "--to=latex",
                   "--citeproc", "--bibliography", str(ROOT / "manuscript/references.bib"),
                   "--pdf-engine=xelatex", "--include-in-header", str(header),
                   "--lua-filter", str(lua_filter),
                   "--variable=papersize:a4", "--variable=geometry:margin=22mm",
                   "--variable=fontsize:11pt", "--variable=linestretch:1.1",
                   "--output", str(tex)]
        with (build / "pandoc.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [xelatex, "-interaction=nonstopmode", "-halt-on-error",
                   "-file-line-error", f"-output-directory={build}", str(tex)]
        # Second pass resolves PDF bookmarks and page references.
        with (build / "xelatex_output.log").open("w", encoding="utf-8") as log:
            for _ in range(2):
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        latex_log = (build / "manuscript_with_figures.log").read_text(encoding="utf-8", errors="replace")
        if "Missing character:" in latex_log:
            raise ValueError("Missing font glyphs; inspect manuscript/_build/manuscript_with_figures.log")
        compiled = build / "manuscript_with_figures.pdf"
        if not compiled.is_file() or compiled.stat().st_size == 0:
            raise ValueError("XeLaTeX did not produce a nonempty PDF")
        output.parent.mkdir(parents=True, exist_ok=True)
        if compiled.resolve() != output:
            shutil.copyfile(compiled, output)
        print(f"Built: {output} ({output.stat().st_size:,} bytes)")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"PDF build failed: {error}\nDiagnostics: {build}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
