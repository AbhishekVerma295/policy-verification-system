"""
The "Evaluation results" tab of the Streamlit app.

Read-only: it shows saved runs from eval/results/ (written by
scripts/evaluate.py) and never runs the system. A full run of the final test
takes about 22 minutes, so re-running it from a web page would be the wrong
tool; this tab is for looking at what was measured.

Each run has two files:
  run-<id>.summary.json   the scores (and, for most runs, the same output
                          re-scored with verification off)
  run-<id>.jsonl          one line per question: the question, the outcome,
                          and the system's full answer
The expected answers come from the labelled question files in eval/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "eval" / "results"
LABEL_FILES = [ROOT / "eval" / n for n in ("final_test.jsonl", "adversarial.jsonl", "final_heldout.jsonl")]

# The run reported as the final result (README, report, paper).
FINAL_RUN = "run-20261120-165959"

# Outcome codes in plain words, in the order they are listed.
OUTCOMES = {
    "correct": "Correct",
    "correct_abstention": "Correctly declined",
    "false_abstention": "Wrongly declined",
    "incomplete": "Incomplete",
    "wrong_citation": "Wrong citation",
    "wrong_value": "Wrong value",
    "answered_unanswerable": "Answered, should have declined",
}
CATEGORIES = {
    "numeric": "Numeric, table, calculation",
    "unanswerable": "Unanswerable",
    "false_premise": "False premise",
    "negation_exception": "Yes/no with exception",
    "cross_policy": "Cross-policy",
}

# Read by hand after the final run (PROJECT_REPORT.md, section 6.1). The
# automatic scorer cannot judge these, so the finding is shown next to them.
HAND_NOTES = {
    "fb-12": "Read by hand: every row of the score table was listed, all verified and true, "
    "including the correct one. The scorer flags the other rows' values. Over-long, not wrong.",
    "fb-13": "Read by hand: every row of the score table was listed, all verified and true, "
    "including the correct one. The scorer flags the other rows' values. Over-long, not wrong.",
    "fb-23": "Read by hand: the answer says the policy gives no number of days and the receipt "
    "is handed in at check-in, which is the correct near-miss answer. Scored as 'answered' "
    "only because it did not decline outright.",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def list_runs() -> list[dict]:
    """Saved runs that have both a summary and per-question results, newest first."""
    runs = []
    for summary_file in RESULTS.glob("run-*.summary.json"):
        run_id = summary_file.name.removesuffix(".summary.json")
        if not (RESULTS / f"{run_id}.jsonl").exists():
            continue
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        meta, summary = data.get("meta", {}), data.get("summary", {})
        if summary.get("items", 0) < 10:  # smoke tests, not evaluations
            continue
        stamp = run_id.split("-")[1]
        runs.append({
            "id": run_id,
            "date": f"{stamp[6:8]}/{stamp[4:6]}/{stamp[:4]}",
            "model": meta.get("config", {}).get("llm_model", "?"),
            "split": meta.get("split") or "all",
            "items": summary.get("items", 0),
            "meta": meta,
            "summary": summary,
        })
    return sorted(runs, key=lambda r: r["id"], reverse=True)


@st.cache_data(show_spinner=False)
def load_questions(run_id: str) -> list[dict]:
    return _jsonl(RESULTS / f"{run_id}.jsonl")


@st.cache_data(show_spinner=False)
def load_labels() -> dict[str, dict]:
    labels = {}
    for path in LABEL_FILES:
        if path.exists():
            for row in _jsonl(path):
                labels.setdefault(row["id"], row)
    return labels


def run_label(run: dict) -> str:
    if run["id"] == FINAL_RUN:
        return f"Final test · {run['date']} · {run['model']} · {run['items']} questions"
    kind = {"tuning": "tuning (used for decisions)", "heldout": "held-out"}.get(run["split"], run["split"])
    return f"{run['date']} · {run['model']} · {kind} · {run['items']} questions"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


def _ci(ci: list | None) -> str:
    return f"95% CI {ci[0] * 100:.0f}–{ci[1] * 100:.0f}%" if ci else ""


def draw_headline(run: dict) -> None:
    s = run["summary"]
    ab = s.get("abstention", {})
    lat = s.get("latency", {}).get("total", {})
    unans = ab.get("unanswerable_items", 0)
    ans = ab.get("answerable_items", 0)
    with st.container(border=True):
        m = st.columns(5)
        m[0].metric("Accuracy", _pct(s.get("accuracy")), help=_ci(s.get("accuracy_ci")))
        m[1].metric(
            "Hallucination", _pct(s.get("hallucination_rate")),
            help="Counted by the automatic scorer. " + _ci(s.get("hallucination_ci")),
        )
        m[2].metric(
            "Unanswerable declined",
            f"{round(ab.get('correct_abstention_rate', 0) * unans)} / {unans}",
        )
        m[3].metric(
            "Answerable declined",
            f"{round(ab.get('false_abstention_rate', 0) * ans)} / {ans}",
        )
        m[4].metric(
            "Median time", f"{lat.get('median_ms', 0) / 1000:.1f} s" if lat else "–",
            help=f"p95 {lat.get('p95_ms', 0) / 1000:.1f} s" if lat else None,
        )
    if run["id"] == FINAL_RUN:
        st.info(
            "**Read by hand: 0 of 99 answers showed a false fact as verified** (95% CI 0–3.7%). "
            "The 3 hallucinations counted by the automatic scorer are scoring artefacts; "
            "they are marked in the question list below.",
            icon=":material/fact_check:",
        )
    if run["split"] == "tuning":
        st.caption(
            "Tuning questions were used to make design decisions, so their scores are "
            "optimistic. Final results come from the final test only."
        )


def draw_on_off(run: dict) -> None:
    s = run["summary"]
    off = s.get("baseline_without_verification")
    if not off:
        return
    on_ab, off_ab = s.get("abstention", {}), off.get("abstention", {})
    st.markdown("##### Verification on vs off")
    st.caption("The same model output, scored twice: as shown, and with every drafted claim kept.")
    table = pd.DataFrame(
        {
            "Verification on": [
                _pct(s.get("accuracy")), _pct(s.get("hallucination_rate")),
                _pct(on_ab.get("correct_abstention_rate")), _pct(on_ab.get("false_abstention_rate")),
                _pct(s.get("citation_accuracy")),
            ],
            "Verification off": [
                _pct(off.get("accuracy")), _pct(off.get("hallucination_rate")),
                _pct(off_ab.get("correct_abstention_rate")), _pct(off_ab.get("false_abstention_rate")),
                _pct(off.get("citation_accuracy")),
            ],
        },
        index=["Accuracy", "Hallucination", "Unanswerable declined",
               "Answerable declined", "Citation accuracy"],
    )
    st.dataframe(table, width="stretch")


def draw_breakdown(run: dict) -> None:
    s = run["summary"]
    left, right = st.columns(2)
    with left:
        st.markdown("##### By question type")
        rows = [
            {"Type": CATEGORIES.get(k, k), "Correct": f"{v['passed']} / {v['items']}",
             "Score": v["accuracy"] * 100}
            for k, v in sorted(s.get("by_category", {}).items(), key=lambda kv: -kv[1]["accuracy"])
        ]
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={"Score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=100.0, format="%.0f%%")},
        )
    with right:
        st.markdown("##### Outcomes")
        counts = s.get("outcomes", {})
        rows = [{"Outcome": label, "Questions": counts.get(code, 0)}
                for code, label in OUTCOMES.items() if counts.get(code)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def draw_question(row: dict, label: dict | None) -> None:
    answer = row.get("answer") or {}
    with st.container(border=True):
        st.markdown(f"**{row['id']}** · {CATEGORIES.get(row['category'], row['category'])} · "
                    f"**{OUTCOMES.get(row['outcome'], row['outcome'])}**")
        st.markdown(f"> {row['question']}")
        note = (label or {}).get("note")
        if note:
            st.caption("Expected answer (answer key)")
            st.markdown(note)
        if row["id"] in HAND_NOTES and row["split"] != "tuning":
            st.info(HAND_NOTES[row["id"]], icon=":material/fact_check:")

        st.caption("What the system said")
        kept = answer.get("claims_kept") or []
        if kept:
            for c in kept:
                cites = ", ".join(c["claim"].get("citation_ids") or [])
                st.markdown(f"- {c['claim']['text']}  \n  <small><code>{cites}</code></small>",
                            unsafe_allow_html=True)
        else:
            st.markdown(f"_Declined._ {answer.get('reason') or ''}")
        removed = answer.get("claims_removed") or []
        if removed:
            st.caption(f"{len(removed)} drafted statement(s) removed by the checks")
        problems = []
        if row.get("required_missing"):
            problems.append("missing: " + ", ".join(row["required_missing"]))
        if row.get("forbidden_found"):
            problems.append("known-wrong value: " + ", ".join(row["forbidden_found"]))
        if problems:
            st.caption("Scorer: " + "; ".join(problems))


def draw_question_browser(run: dict) -> None:
    st.markdown("##### Every question")
    rows = load_questions(run["id"])
    labels = load_labels()
    c1, c2 = st.columns(2)
    outcome_pick = c1.multiselect(
        "Outcome", list(OUTCOMES.values()), placeholder="All outcomes", key=f"oc-{run['id']}"
    )
    type_pick = c2.multiselect(
        "Question type", list(CATEGORIES.values()), placeholder="All types", key=f"ty-{run['id']}"
    )
    shown = [
        r for r in rows
        if (not outcome_pick or OUTCOMES.get(r["outcome"]) in outcome_pick)
        and (not type_pick or CATEGORIES.get(r["category"]) in type_pick)
    ]
    table = pd.DataFrame([
        {"ID": r["id"], "Type": CATEGORIES.get(r["category"], r["category"]),
         "Outcome": OUTCOMES.get(r["outcome"], r["outcome"]), "Question": r["question"]}
        for r in shown
    ])
    st.caption(f"{len(shown)} of {len(rows)} questions. Select a row to see the answer.")
    if table.empty:
        return
    event = st.dataframe(
        table, hide_index=True, width="stretch", height=360,
        on_select="rerun", selection_mode="single-row", key=f"qs-{run['id']}",
        column_config={"Question": st.column_config.TextColumn(width="large")},
    )
    picked = event.selection.rows if event and event.selection else []
    if picked:
        row = shown[picked[0]]
        draw_question(row, labels.get(row["id"]))


def draw_evaluation_tab() -> None:
    runs = list_runs()
    if not runs:
        st.info("No saved evaluation runs in eval/results/. Run scripts/evaluate.py first.")
        return
    ids = [r["id"] for r in runs]
    default = ids.index(FINAL_RUN) if FINAL_RUN in ids else 0
    run = runs[st.selectbox(
        "Run", range(len(runs)), index=default, format_func=lambda i: run_label(runs[i]),
        help="Saved runs of scripts/evaluate.py. The final test is the reported result.",
    )]
    meta = run["meta"]
    st.caption(
        f"`{run['id']}` · model {run['model']} · "
        f"{meta.get('wall_seconds', 0) / 60:.0f} min for {run['items']} questions · "
        "results read from eval/results/, not re-run"
    )
    draw_headline(run)
    draw_on_off(run)
    draw_breakdown(run)
    draw_question_browser(run)
