#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from xhtml2pdf import pisa

from src.common.config import load_config
from src.query.answer import answer_question
from src.storage.db import Database


MODEL_DISPLAY_LABELS = {
    "ensemble": "Ensemble",
    "elo_baseline": "Elo",
    "glm_logit": "GLM",
    "dynamic_rating": "Dyn Rating",
    "gbdt": "GBDT",
    "rf": "RF",
    "two_stage": "Two Stage",
    "goals_poisson": "Goals Pois",
    "simulation_first": "Sim",
    "bayes_bt_state_space": "Bayes BT",
    "bayes_goals": "Bayes Goals",
    "nn_mlp": "NN",
}


def _display_label(column: str) -> str:
    return MODEL_DISPLAY_LABELS.get(column, column)


def _report_dataframe(payload: dict) -> pd.DataFrame:
    model_cols = list(payload.get("model_columns", []))
    rows = []
    for r in payload.get("rows", []):
        probs = r.get("model_win_probabilities", {}) or {}
        team = r.get("team") or "-"
        opponent = r.get("next_opponent") or "-"
        home_or_away = r.get("home_or_away")
        if str(home_or_away or "").strip().lower() != "home":
            continue
        home_team = r.get("home_team")
        away_team = r.get("away_team")
        if not home_team or not away_team:
            if home_or_away == "Home":
                home_team = team
                away_team = opponent
            elif home_or_away == "Away":
                home_team = opponent
                away_team = team
            else:
                home_team = home_team or "-"
                away_team = away_team or "-"
        out = {
            "Home Team": home_team or "-",
            "Away Team": away_team or "-",
            "Date": r.get("next_game_date_utc") or "-",
        }
        for m in model_cols:
            p = probs.get(m)
            if p is None:
                out[m] = "-"
                continue
            out[m] = f"{float(p):.1%}"
        rows.append(out)
    return pd.DataFrame(rows)


def _dynamic_col_width_percentages(df: pd.DataFrame) -> list[float]:
    if df.empty:
        return []

    weights: list[float] = []
    for col in df.columns:
        header_label = _display_label(col)
        max_len = max([len(header_label)] + [len(str(v)) for v in df[col].tolist()])
        if col == "Home Team":
            w = max(7.0, min(12.0, float(max_len + 1)))
        elif col == "Away Team":
            w = max(7.0, min(12.0, float(max_len + 1)))
        elif col == "Date":
            w = 8.0
        else:
            w = max(5.0, min(8.0, float(max_len + 1)))
        weights.append(w)

    total = sum(weights) or 1.0
    return [100.0 * w / total for w in weights]


def _build_html(
    df: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    model_trust_notes: dict[str, str] | None = None,
) -> str:
    if df.empty:
        raise RuntimeError("No report rows available to export.")

    widths = _dynamic_col_width_percentages(df)
    trust_notes = model_trust_notes or {}
    colgroup = "".join([f'<col style="width:{w:.4f}%" />' for w in widths])
    header_cells = "".join(
        [f"<th>{html.escape(_display_label(str(c)))}</th>" for c in df.columns]
    )

    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for idx, col in enumerate(df.columns):
            val = html.escape(str(row[col]))
            cell_class = "left" if idx < 3 else "num"
            cells.append(f'<td class="{cell_class}">{val}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    explanation_items = []
    for col in df.columns[3:]:
        note = trust_notes.get(str(col))
        if not note:
            continue
        short = _display_label(str(col))
        explanation_items.append(
            f'<div class="note-item"><b>{html.escape(short)}:</b> {html.escape(note)}</div>'
        )

    notes_rows = []
    if explanation_items:
        n_left = (len(explanation_items) + 1) // 2
        left_items = explanation_items[:n_left]
        right_items = explanation_items[n_left:]
        for idx in range(n_left):
            left = left_items[idx]
            right = right_items[idx] if idx < len(right_items) else ""
            notes_rows.append(f"<tr><td>{left}</td><td>{right}</td></tr>")
    notes_html = ""
    if notes_rows:
        notes_html = (
            '<div class="notes-title">Model explanations</div>'
            '<table class="notes-table"><tbody>'
            + "".join(notes_rows)
            + "</tbody></table>"
        )

    html_doc = f"""
<html>
  <head>
    <meta charset=\"utf-8\" />
    <style>
      @page {{ size: A3 landscape; margin: 8mm; }}
      body {{ font-family: Helvetica, Arial, sans-serif; font-size: 8px; color: #111; }}
      h1 {{ font-size: 15px; margin: 0 0 2px 0; }}
      .subtitle {{ font-size: 12px; margin: 0 0 6px 0; color: #333; }}
      table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
      th, td {{ border: 1px solid #222; padding: 2px 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
      th {{ background: #efefef; font-weight: 700; text-align: center; }}
      td.left {{ text-align: left; }}
      td.num {{ text-align: right; }}
      .notes-title {{ margin-top: 4px; font-size: 7px; font-weight: 700; }}
      .notes-table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
      .notes-table td {{ border: none; vertical-align: top; width: 50%; padding: 1px 4px 1px 0; font-size: 6px; line-height: 1.1; }}
      .note-item {{ margin: 0; }}
    </style>
  </head>
  <body>
    <h1>{html.escape(title)}</h1>
    <div class="subtitle">{html.escape(subtitle or "")}</div>
    <table>
      <colgroup>{colgroup}</colgroup>
      <thead><tr>{header_cells}</tr></thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
    {notes_html}
  </body>
</html>
"""
    return html_doc


def _save_pdf_from_html(html_doc: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        result = pisa.CreatePDF(src=html_doc, dest=f, encoding="utf-8")
    if result.err:
        raise RuntimeError("xhtml2pdf failed to generate the PDF output.")


def _format_as_of_label(as_of: str | None, timezone_name: str) -> str:
    raw = str(as_of or "").strip()
    if not raw:
        return "unknown"

    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = ZoneInfo("UTC")
        local_dt = dt.astimezone(tz)
        return (
            f"{local_dt.strftime('%b')} {local_dt.day}, {local_dt.year} "
            f"{local_dt.strftime('%I').lstrip('0') or '0'}:{local_dt.strftime('%M')} "
            f"{local_dt.strftime('%p')} {local_dt.strftime('%Z')}"
        )
    except Exception:
        return raw


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
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(answer + "\n", encoding="utf-8")

    df = _report_dataframe(payload)
    as_of = payload.get("as_of_utc", "unknown")
    as_of_label = _format_as_of_label(as_of=str(as_of), timezone_name=cfg.project.timezone)
    title = f"{league} All-Teams Report | As of {as_of_label}"
    html_doc = _build_html(
        df,
        title,
        subtitle="Win chance of next game by team",
        model_trust_notes=payload.get("model_trust_notes", {}),
    )
    _save_pdf_from_html(html_doc, out_pdf)

    print(out_md)
    print(out_pdf)


if __name__ == "__main__":
    main()
