#!/usr/bin/env python3
"""
Format and rank promptfoo cold call evaluation results.

Usage (from promptfoo_sim/):
  python show_results.py
  python show_results.py results/latest_eval_coldcall.json
"""

import io
import json
import os
import re
import sys
from pathlib import Path

# Force UTF-8 on Windows so Rich box-drawing characters don't crash cp1252
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from openai import OpenAI
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

DEFAULT_RESULTS = Path(__file__).parent / "results" / "latest_eval_coldcall.json"

_RUBRIC_LABELS = {
    "NATURAL SPEECH": "Natural Speech",
    "CONTEXTUAL RESPONSIVENESS": "Contextual Responsiveness",
    "HEDGING AND TENTATIVENESS": "Hedging & Tentativeness",
    "BACKCHANNEL AND ACKNOWLEDGMENT BEHAVIOR": "Backchannel & Ack.",
}


def _normalize(text: str) -> str:
    # Build chars via chr() so source stays ASCII-safe.
    rsq = chr(0x2019)
    lsq = chr(0x2018)
    ldq = chr(0x201C)
    rdq = chr(0x201D)
    emdash = chr(0x2014)
    endash = chr(0x2013)
    mj = {c: c.encode("utf-8").decode("cp1252", errors="replace")
          for c in [rsq, lsq, ldq, rdq, emdash, endash]}
    return (
        text
        .replace(mj[rsq], "'").replace(mj[lsq], "'")
        .replace(mj[ldq], chr(34)).replace(mj[rdq], chr(34))
        .replace(mj[emdash], "--").replace(mj[endash], "-")
        .replace(rsq, "'").replace(lsq, "'")
        .replace(ldq, chr(34)).replace(rdq, chr(34))
        .replace(emdash, "--").replace(endash, "-")
    )


def _rubric_name(value: str) -> str:
    m = re.search(r"Score the (.+?) of this", value)
    if m:
        return _RUBRIC_LABELS.get(m.group(1).strip(), m.group(1).title())
    return "Rubric"


def _score_color(score: float) -> str:
    if score >= 8:
        return "green"
    if score >= 6:
        return "yellow"
    return "red"


def _rank_label(rank: int) -> str:
    return {1: "[bold gold1]#1[/bold gold1]",
            2: "[bold grey70]#2[/bold grey70]",
            3: "[bold dark_orange]#3[/bold dark_orange]"}.get(rank, f"#{rank}")


def _build_text_report(ranked: list[tuple], analysis: str, generated_at: str) -> str:
    lines = [
        "COLD CALL SIMULATION -- METHOD RANKINGS",
        f"Generated: {generated_at}",
        "",
        "SCORES",
        "-" * 80,
    ]
    rubric_names = [r["name"] for r in ranked[0][1]["rubrics"]] if ranked else []
    header = f"{'Method':<22}" + "".join(f"{n:<28}" for n in rubric_names) + f"{'Avg':>6}"
    lines.append(header)
    lines.append("-" * 80)
    for rank, (label, info) in enumerate(ranked, 1):
        scores = "".join(f"{r['score']}/10{'':<22}" for r in info["rubrics"])
        lines.append(f"#{rank} {label:<19}{scores}{info['avg']:>6.2f}")
    lines += ["", "RANKING ANALYSIS", "-" * 80, ""]
    lines.append(analysis if analysis else "(no analysis generated)")
    lines += ["", "", "JUDGE FEEDBACK BY METHOD", "-" * 80]
    for rank, (label, info) in enumerate(ranked, 1):
        lines += ["", f"#{rank} {label.upper()}  (avg {info['avg']:.2f}/10)", ""]
        for rubric in info["rubrics"]:
            lines.append(f"  {rubric['score']:>4}/10  {rubric['name']}")
            lines.append(f"    {rubric['reason']}")
            lines.append("")
    return "\n".join(lines)


def _generate_qualitative_analysis(ranked: list[tuple], api_key: str) -> str:
    sections = []
    for rank, (label, info) in enumerate(ranked, 1):
        rubric_lines = []
        for r in info["rubrics"]:
            rubric_lines.append(
                "  " + r["name"] + ": " + str(r["score"]) + "/10\n"
                "    Judge observed: " + r["reason"]
            )
        header = "#" + str(rank) + " " + label + " (avg " + f"{info['avg']:.2f}" + "/10)"
        sections.append(header + "\n" + "\n".join(rubric_lines))

    prompt = "\n\n".join([
        (
            "You are analyzing results from a cold call simulation evaluation. "
            "Five AI prompting methods were tested on how well they simulate a real person "
            "receiving a cold call. Each was scored on four rubrics by a judge LLM that read "
            "the actual transcripts."
        ),
        "Results (best to worst):\n\n" + "\n\n".join(sections),
        (
            "Provide a qualitative ranking breakdown. For each method in order, write 2-3 sentences:\n"
            "- WHY it received the scores it did, grounded in what the judge actually observed\n"
            "- What specifically it did well or poorly compared to the method ranked above it\n\n"
            "Then write an OVERALL COMPARISON paragraph (3-4 sentences) on what behavioral "
            "differences separate the top from the bottom, and what that implies about which "
            "prompting strategies produce the most human-like cold call responses.\n\n"
            "Format exactly as:\n"
            "#1 [method name]\n"
            "[explanation]\n\n"
            "#2 [method name]\n"
            "[explanation]\n\n"
            "...etc, then:\n\n"
            "OVERALL COMPARISON\n"
            "[paragraph]"
        ),
    ])

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 120
    console = Console(force_terminal=True, legacy_windows=False, width=term_width)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    evals = data["results"]["results"]

    # --- Parse results by method ---
    methods: dict[str, dict] = {}
    for entry in evals:
        label = entry["provider"]["label"]
        components = entry["gradingResult"]["componentResults"]
        rubrics = [
            {
                "name": _rubric_name(cr["assertion"]["value"]),
                "score": cr["score"],
                "reason": _normalize(cr["reason"]),
            }
            for cr in components
        ]
        avg = sum(r["score"] for r in rubrics) / len(rubrics) if rubrics else 0
        methods[label] = {"rubrics": rubrics, "avg": avg}

    ranked = sorted(methods.items(), key=lambda x: x[1]["avg"], reverse=True)
    rubric_names = [r["name"] for r in ranked[0][1]["rubrics"]] if ranked else []

    # --- Ranking table ---
    console.print()
    table = Table(
        title="[bold]Cold Call Simulation -- Method Rankings[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold white",
    )
    table.add_column("Rank", width=6, justify="center")
    table.add_column("Method", style="cyan", width=22)
    for name in rubric_names:
        table.add_column(name, justify="center", width=14)
    table.add_column("Avg", justify="center", style="bold", width=8)

    for rank, (label, info) in enumerate(ranked, 1):
        score_cells = []
        for r in info["rubrics"]:
            c = _score_color(r["score"])
            score_cells.append(f"[{c}]{r['score']}/10[/{c}]")
        avg_c = _score_color(info["avg"])
        table.add_row(
            _rank_label(rank),
            label,
            *score_cells,
            f"[{avg_c}]{info['avg']:.2f}[/{avg_c}]",
        )

    console.print(table)

    # --- Ranking analysis (LLM-generated) ---
    import datetime
    api_key = os.environ.get("OPENAI_API_KEY", "")
    analysis = ""
    console.print()
    console.print(Rule("[bold]Ranking Analysis[/bold]"))
    console.print()
    if api_key:
        console.print("  [dim]Generating qualitative analysis...[/dim]")
        analysis = _normalize(_generate_qualitative_analysis(ranked, api_key))
        # Save report
        out_path = path.parent / (path.stem + "_analysis.txt")
        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        out_path.write_text(
            _build_text_report(ranked, analysis, generated_at), encoding="utf-8"
        )
        console.print(f"  [dim]Saved to {out_path}[/dim]")
        console.print()
        for line in analysis.splitlines():
            stripped = line.strip()
            if re.match(r"^#\d+", stripped):
                parts = stripped.split(" ", 1)
                rank_tag = parts[0]
                rest = parts[1] if len(parts) > 1 else ""
                console.print(f"  [bold]{rank_tag}[/bold] [bold cyan]{rest}[/bold cyan]")
            elif stripped.upper() == "OVERALL COMPARISON":
                console.print()
                console.print("  [bold]Overall Comparison[/bold]")
            elif stripped:
                console.print(f"    {stripped}")
            else:
                console.print()
    else:
        console.print("  [yellow]OPENAI_API_KEY not set -- skipping qualitative analysis.[/yellow]")
    console.print()

    # --- Per-method judge feedback ---
    console.print(Rule("[bold]Judge Feedback by Method[/bold]"))
    for rank, (label, info) in enumerate(ranked, 1):
        console.print()
        avg_c = _score_color(info["avg"])
        console.print(
            f"  {_rank_label(rank)}  [bold cyan]{label.upper()}[/bold cyan]"
            f"  --  avg [{avg_c}]{info['avg']:.2f}[/{avg_c}]/10"
        )
        console.print()
        for rubric in info["rubrics"]:
            sc = _score_color(rubric["score"])
            console.print(
                f"    [{sc}]{rubric['score']:>4}/10[/{sc}]  [bold]{rubric['name']}[/bold]"
            )
            console.print(f"            {rubric['reason']}")
            console.print()

    console.print(Rule())


if __name__ == "__main__":
    main()
