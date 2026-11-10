"""
The Streamlit interface (Phase 7).

    streamlit run app.py

Requires an index (python scripts/build_index.py) and a running Ollama.

Everything shown here comes from pipeline.answer_question(), the same entry
point the CLI and the evaluation harness use, so the UI cannot drift from what
was measured. This file only decides how an `Answer` is drawn.

Three things the page is built to make visible:

  1. every claim's verdict, with the policy text it was checked against
  2. claims the system generated and then REMOVED - shown, never hidden
  3. abstention as a first-class result, not an error message
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from policyverify.config import get_config  # noqa: E402
from policyverify.evaluate import load_results, summarise_results  # noqa: E402
from policyverify.generate import GenerationError  # noqa: E402
from policyverify.indexing import IndexMismatchError, VectorStore  # noqa: E402
from policyverify.llm import LLMError, get_llm  # noqa: E402
from policyverify.pipeline import answer_question  # noqa: E402
from policyverify.retrieve import retrieve  # noqa: E402
from policyverify.schema import Answer, ClaimVerdict, VerdictStatus  # noqa: E402
from policyverify.verify import NLIChecker  # noqa: E402

st.set_page_config(page_title="Policy Verification System", page_icon="✅", layout="wide")

# Verdict -> (label, accent colour, translucent background). The backgrounds
# are translucent so the cards read correctly in both light and dark themes.
STYLES = {
    VerdictStatus.SUPPORTED: ("SUPPORTED", "#1a9c4b", "rgba(26, 156, 75, 0.10)"),
    VerdictStatus.REFUTED: ("REFUTED", "#d93025", "rgba(217, 48, 37, 0.10)"),
    VerdictStatus.NEUTRAL: ("UNVERIFIED", "#d98a00", "rgba(217, 138, 0, 0.10)"),
}

# What the page says while each pipeline stage runs.
STAGE_MESSAGES = {
    "retrieve": "1/4  Searching the policy documents for relevant passages ...",
    "generate": "2/4  The language model is writing claims from those passages "
    "(the slow step: 1-2 minutes without a GPU) ...",
    "verify": "3/4  Checking every claim against the policy text it cites ...",
    "decide": "4/4  Keeping verified claims, removing the rest ...",
}

EXAMPLES = [
    ("Answerable", "What is the minimum attendance required to sit the end semester examination?"),
    ("Answerable", "How many days in advance must a parent inform the hostel office before staying?"),
    ("Numeric trap", "How much attendance condonation can be granted at most?"),
    ("False premise", "Why does the university require 90% attendance in every course?"),
    ("Not in the corpus", "What is the campus wifi password?"),
    ("Not in the corpus", "What is the placement package for computer science graduates?"),
]

st.markdown(
    """
    <style>
      .claim-card { border-left: 5px solid; border-radius: 6px; padding: 0.7rem 1rem;
                    margin: 0.6rem 0 0.2rem 0; }
      .claim-head { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; }
      .claim-text { font-size: 1.05rem; margin: 0.25rem 0; }
      .claim-why  { font-size: 0.85rem; opacity: 0.8; }
      .cite { font-family: monospace; font-size: 0.8rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Loading - once per server process, not once per question
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_system():
    """The store, the fact-checker and the language model client.

    Cached so the embedding and NLI weights are read once. Without this every
    question would spend longer loading models than answering.
    """
    config = get_config()
    store = VectorStore(config)
    store.check_ready()
    _ = store.embedder.model  # force the load now, behind the spinner
    nli = NLIChecker(config)
    _ = nli.pipeline
    return config, store, nli, get_llm(config)


@st.cache_resource(show_spinner=False)
def warm_up_llm() -> str | None:
    """Load the language model into memory. Returns an error message, or None.

    The first request to Ollama pays the model load. Paying it here means the
    first real question does not.
    """
    _, _, _, llm = load_system()
    try:
        llm.generate('Reply with exactly this JSON: {"ok": true}')
    except LLMError as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Drawing an Answer
# ---------------------------------------------------------------------------


def evidence_for(verdict: ClaimVerdict, store: VectorStore) -> list:
    """The policy text this claim was checked against."""
    chunks = []
    for citation in verdict.claim.citation_ids:
        try:
            chunks.extend(store.get_by_citation(citation))
        except Exception:  # noqa: BLE001 - evidence is display-only; never fail the page
            continue
    if verdict.evidence_chunk_ids:
        wanted = set(verdict.evidence_chunk_ids)
        chunks = [c for c in chunks if c.chunk_id in wanted] or chunks
    return chunks


def draw_claim(verdict: ClaimVerdict, store: VectorStore, key: str) -> None:
    label, colour, background = STYLES[verdict.status]
    citations = "".join(
        f'<div class="cite">&rarr; {html.escape(c)}</div>' for c in verdict.claim.citation_ids
    ) or '<div class="cite">&rarr; (no citation given)</div>'

    st.markdown(
        f"""
        <div class="claim-card" style="border-color: {colour}; background: {background};">
          <div class="claim-head" style="color: {colour};">
            {label} &middot; confidence {verdict.score:.2f}
          </div>
          <div class="claim-text">{html.escape(verdict.claim.text)}</div>
          {citations}
          <div class="claim-why">{html.escape(verdict.explanation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Evidence and checks", key=key):
        checks = verdict.checks
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            "**Entailment (NLI)**  \n"
            + (
                f"{checks.nli_label} ({checks.nli_score:.2f})"
                if checks.nli_label is not None and checks.nli_score is not None
                else "not run"
            )
        )
        c2.markdown(
            "**Numbers match**  \n"
            + {True: "yes", False: "no", None: "not run"}[checks.numeric_ok]
            + (f" - {checks.numeric_detail}" if checks.numeric_detail else "")
        )
        c3.markdown("**Citation exists**  \n" + ("yes" if checks.citation_exists else "no"))

        chunks = evidence_for(verdict, store)
        if not chunks:
            st.caption("No policy text could be resolved for this claim's citations.")
        for chunk in chunks:
            st.markdown(f"**{chunk.section_path or chunk.citation_id}**")
            st.caption(f"{chunk.citation_id}  ·  [{chunk.university_name} source]({chunk.source_url})")
            st.text(chunk.text)


def draw_answer(answer: Answer, passages: list, store: VectorStore) -> None:
    st.subheader(answer.question)

    if answer.abstained:
        st.warning(f"**No answer given.** {answer.reason}", icon="🛑")
    elif answer.claims_kept:
        st.success(
            f"**{len(answer.claims_kept)} verified claim(s).** Every statement below "
            "was checked against the policy text it cites. Confidence runs from "
            "0.00 to 1.00.",
            icon="✅",
        )
        for i, verdict in enumerate(answer.claims_kept):
            draw_claim(verdict, store, key=f"kept-{i}")
        if answer.reason:
            st.info(answer.reason)

    if answer.claims_removed:
        st.markdown("---")
        st.markdown(f"#### Removed claims ({len(answer.claims_removed)})")
        st.caption(
            "The model generated these and the system would not stand behind them. "
            "They are shown rather than hidden, so you can see what was rejected and why."
        )
        for i, verdict in enumerate(answer.claims_removed):
            draw_claim(verdict, store, key=f"removed-{i}")

    st.markdown("---")
    t = answer.timings
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Kept", len(answer.claims_kept))
    m2.metric("Removed", len(answer.claims_removed))
    m3.metric("Retrieve", f"{t.retrieve_ms / 1000:.1f}s")
    m4.metric("Generate", f"{t.generate_ms / 1000:.1f}s")
    m5.metric("Verify", f"{t.verify_ms / 1000:.1f}s")

    with st.expander(f"Retrieved passages ({len(passages)})"):
        st.caption(
            "What the search returned for this question. The model may only use, "
            "and only cite, these."
        )
        for r in passages:
            st.markdown(f"`{r.score:.3f}` &nbsp; **{r.chunk.citation_id}**")
            st.caption(r.chunk.text[:400] + ("..." if len(r.chunk.text) > 400 else ""))

    with st.expander("Raw answer (JSON)"):
        st.json(json.loads(answer.model_dump_json()))


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def draw_evaluation(config) -> None:
    results_dir = config.paths.resolve("runs")
    logs = sorted(results_dir.glob("run-*.jsonl"), reverse=True)
    if not logs:
        st.info(
            "No evaluation runs yet. Produce one with:\n\n"
            "`python scripts/evaluate.py --split heldout`"
        )
        return

    log = st.selectbox("Run", logs, format_func=lambda p: p.stem)
    results = load_results(log)
    summary = summarise_results(results)

    meta_path = log.with_suffix(".summary.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("meta", {})
        scope = meta.get("split", "all")
        if meta.get("sample"):
            scope += f", sample of {meta['sample']}"
        if meta.get("category"):
            scope += f", {meta['category']} only"
        st.caption(
            f"Scope: {scope}  ·  model: {meta.get('config', {}).get('llm_model', '?')}  ·  "
            f"wall time: {meta.get('wall_seconds', '?')}s"
        )

    a, c = summary["abstention"], summary["claims"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Items scored", f"{summary['scored']} / {summary['items']}")
    k2.metric("Hallucination rate", pct(summary["hallucination_rate"]))
    k3.metric("Citation accuracy", pct(summary["citation_accuracy"]))
    k4.metric("Correct abstention", pct(a["correct_abstention_rate"]))
    k5.metric("False abstention", pct(a["false_abstention_rate"]))

    st.caption(
        "**Hallucination rate**: the system stood behind a figure the labels mark as "
        "wrong, or answered a question the corpus does not cover. "
        "**Citation accuracy**: answered questions resting on an expected section. "
        "**Correct abstention**: unanswerable questions declined. "
        "**False abstention**: answerable questions wrongly declined."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("##### By trap category")
        st.dataframe(
            [
                {
                    "category": name,
                    "passed": f"{row['passed']} / {row['items']}",
                    "accuracy": pct(row["accuracy"]),
                }
                for name, row in summary["by_category"].items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown("##### Claims")
        st.markdown(
            f"{c['generated']} generated → **{c['kept']} kept**, "
            f"**{c['removed']} removed** by verification ({pct(c['removal_rate'])})  \n"
            f"Citation precision {pct(c['citation_precision'])} · "
            f"citation recall {pct(c['citation_recall'])}"
        )
    with right:
        st.markdown("##### Latency")
        st.dataframe(
            [
                {
                    "stage": stage,
                    "mean (s)": round(row["mean_ms"] / 1000, 1),
                    "median (s)": round(row["median_ms"] / 1000, 1),
                    "p95 (s)": round(row["p95_ms"] / 1000, 1),
                }
                for stage, row in summary["latency"].items()
                if row
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown("##### Outcomes")
        st.dataframe(
            [{"outcome": k, "items": v} for k, v in sorted(summary["outcomes"].items())],
            hide_index=True,
            width="stretch",
        )

    st.markdown("##### Every item")
    st.dataframe(
        [
            {
                "": "✅" if r.passed else "❌",
                "id": r.id,
                "category": r.category,
                "outcome": r.outcome.value,
                "question": r.question,
                "kept": r.claims_kept,
                "removed": r.claims_removed,
                "detail": r.error
                or "; ".join(
                    part
                    for part in (
                        f"forbidden: {', '.join(r.forbidden_found)}" if r.forbidden_found else "",
                        f"missing: {', '.join(r.required_missing)}" if r.required_missing else "",
                    )
                    if part
                ),
                "seconds": round(r.total_ms / 1000, 1),
            }
            for r in results
        ],
        hide_index=True,
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def use_example(question: str) -> None:
    st.session_state["question_box"] = question
    st.session_state["pending"] = question


st.title("Policy Verification System")
st.caption(
    "Answers questions about SRM Institute of Science and Technology policies. "
    "Every answer is broken into individual claims, each claim is checked against "
    "the policy text it cites, and the system declines when the evidence is not there."
)

try:
    with st.spinner("Loading the search index and fact-checking model ..."):
        config, store, nli, llm = load_system()
except IndexMismatchError as exc:
    st.error(str(exc))
    st.stop()

with st.spinner(f"Loading {config.llm.model} into memory ..."):
    llm_problem = warm_up_llm()

if llm_problem:
    st.error(
        f"Language model not reachable:\n\n{llm_problem}\n\n"
        "Open the Ollama app from the Start menu, then refresh this page."
    )

ask_tab, eval_tab = st.tabs(["Ask a question", "Evaluation"])

with ask_tab:
    with st.form("ask"):
        question = st.text_input(
            "Your question",
            key="question_box",
            placeholder="e.g. What is the minimum attendance requirement?",
        )
        submitted = st.form_submit_button("Verify answer", type="primary")

    st.caption("Try one:")
    columns = st.columns(3)
    for i, (kind, example) in enumerate(EXAMPLES):
        columns[i % 3].button(
            example,
            key=f"example-{i}",
            help=kind,
            on_click=use_example,
            args=(example,),
            width="stretch",
        )

    to_run = question.strip() if submitted else st.session_state.pop("pending", None)

    if to_run:
        # Search every policy, with the configured number of passages.
        policy = None
        top_k = config.retrieval.top_k
        try:
            with st.status("Working on your question ...", expanded=True) as status:
                def show_stage(name: str) -> None:
                    status.write(STAGE_MESSAGES[name])

                answer = answer_question(
                    to_run,
                    k=top_k,
                    policy_type=policy,
                    config=config,
                    store=store,
                    llm=llm,
                    nli=nli,
                    on_stage=show_stage,
                )
                # Same call the pipeline made, repeated so the passages can be
                # shown. Retrieval is deterministic, so these are the same ones.
                passages = retrieve(
                    to_run, k=top_k, policy_type=policy, config=config, store=store
                )
                status.update(
                    label=f"Done in {answer.timings.total_ms / 1000:.0f} seconds",
                    state="complete",
                    expanded=False,
                )
            st.session_state["result"] = (answer, passages)
        except (LLMError, GenerationError) as exc:
            st.session_state.pop("result", None)
            st.error(
                f"Could not produce an answer: {exc}\n\n"
                "If this says Ollama could not be reached, open the Ollama app from "
                "the Start menu and ask again."
            )

    if "result" in st.session_state:
        draw_answer(*st.session_state["result"], store)

with eval_tab:
    draw_evaluation(config)
