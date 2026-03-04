#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.common.config import load_config
from src.query.answer import answer_question
from src.storage.db import Database


def _report_dataframe(payload: dict) -> pd.DataFrame:
    model_cols = list(payload.get("model_columns", []))
    rows = []
    for r in payload.get("rows", []):
        probs = r.get("model_win_probabilities", {}) or {}
        out = {
            "Team": r.get("team") or "-",
            "Next Opponent": r.get("next_opponent") or "-",
            "Date": r.get("next_game_date_utc") or "-",
            "Home/Away": r.get("home_or_away") or "-",
        }
        for m in model_cols:
            p = probs.get(m)
            out[m] = "-" if p is None else f"{float(p):.1%}"
        rows.append(out)
    return pd.DataFrame(rows)


def _dynamic_col_widths(df: pd.DataFrame) -> list[float]:
    if df.empty:
        return []

    def max_len(col: str) -> int:
        vals = [len(str(v)) for v in df[col].tolist()]
        return max([len(col)] + vals)

    weights = []
    for col in df.columns:
        length = max_len(col)
        if col == "Team":
            weight = max(6.0, min(11.0, float(length + 2)))
        elif col == "Next Opponent":
            weight = max(10.0, min(16.0, float(length + 2)))
        elif col == "Date":
            weight = 10.0
        elif col == "Home/Away":
            weight = 8.0
        else:
            weight = max(7.0, min(14.0, float(length + 1)))
        weights.append(weight)

    total = sum(weights) or 1.0
    return [w / total for w in weights]


def _save_pdf(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        raise RuntimeError("No report rows available to export.")

    n_rows, n_cols = df.shape
    fig_w = max(17.0, min(40.0, 8.0 + 1.35 * n_cols))
    fig_h = max(8.5, min(20.0, 2.5 + 0.28 * (n_rows + 1)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_widths = _dynamic_col_widths(df)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        colWidths=col_widths,
        cellLoc="center",
        loc="upper left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.16)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#ECECEC")
            cell.set_text_props(weight="bold")
        if c in (0, 1, 2, 3) and r > 0:
            cell.set_text_props(ha="left")

    ax.set_title(title, fontsize=11, pad=14)
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export shareable single-page PDF report from local forecast DB")
    parser.add_argument("--config", default="configs/nba.yaml")
    parser.add_argument("--league", default=None)
    parser.add_argument("--question", default="Give me the report of all teams in a table.")
    parser.add_argument("--pdf-output", required=True)
    parser.add_argument("--md-output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    league = str(args.league or cfg.data.league).upper()

    db = Database(cfg.paths.db_path)
    answer, payload = answer_question(db, args.question, default_league=league)

    out_pdf = Path(args.pdf_output)
    out_md = Path(args.md_output)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(answer + "\n", encoding="utf-8")

    df = _report_dataframe(payload)
    as_of = payload.get("as_of_utc", "unknown")
    title = f"{league} All-Teams Report (Ensemble Desc) | As of {as_of}"
    _save_pdf(df, title, out_pdf)

    print(out_md)
    print(out_pdf)


if __name__ == "__main__":
    main()
