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

Evaluation results are not shown here; see scripts/evaluate.py.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from policyverify.config import get_config  # noqa: E402
from policyverify.generate import GenerationError  # noqa: E402
from policyverify.indexing import IndexMismatchError, VectorStore  # noqa: E402
from policyverify.llm import LLMError, get_llm  # noqa: E402
from policyverify.pipeline import answer_question  # noqa: E402
from policyverify.retrieve import retrieve  # noqa: E402
from policyverify.schema import Answer, ClaimVerdict, VerdictStatus  # noqa: E402
from policyverify.verify import NLIChecker  # noqa: E402
from policyverify.verify.verifier import policy_says  # noqa: E402

st.set_page_config(page_title="Policy Verification System", page_icon="✅", layout="wide")

# Verdict -> (label, accent colour, translucent background). The backgrounds
# are translucent so the cards read correctly in both light and dark themes.
STYLES = {
    VerdictStatus.SUPPORTED: ("SUPPORTED", "#1a9c4b", "rgba(26, 156, 75, 0.10)"),
    VerdictStatus.REFUTED: ("REFUTED", "#d93025", "rgba(217, 48, 37, 0.10)"),
    VerdictStatus.NEUTRAL: ("UNVERIFIED", "#d98a00", "rgba(217, 138, 0, 0.10)"),
}

# A supported claim withheld because it does not answer the question asked.
OFF_TOPIC_STYLE = ("TRUE BUT OFF TOPIC", "#6b7280", "rgba(107, 114, 128, 0.10)")

# What the page says while each pipeline stage runs.
STAGE_MESSAGES = {
    "retrieve": "1/4  Searching the policy documents for relevant passages ...",
    "generate": "2/4  The language model is writing claims from those passages "
    "(the slow step: 1-2 minutes without a GPU) ...",
    "verify": "3/4  Checking every claim against the policy text it cites, "
    "and that it answers the question ...",
    "decide": "4/4  Keeping verified claims, removing the rest ...",
}

# Example questions. Each one was answered correctly, with every claim
# verified, in the evaluation run - and generation runs at temperature 0, so
# the same question gives the same answer. They cover four different policy
# documents.  (document, question)
EXAMPLES = [
    ("Academic regulations", "What is the minimum attendance percentage required in each course?"),
    ("Hostel rules", "How many days in advance must a parent inform the hostel office before staying?"),
    ("Hostel rules", "Are male visitors allowed in female hostels?"),
    ("Hostel rules", "How many floor representatives are there for each hostel floor?"),
    ("Academic regulations", "How many weeks does the summer semester run for?"),
    ("Plagiarism policy", "Do I need to cite a source if I completely rewrite it in my own words?"),
]

# What the system can answer from - shown when it declines, so it is clear
# that a declined question is outside these documents, not a crash.
DOCUMENTS = [
    "Academic Regulations 2021 (attendance, semesters, grading, examinations)",
    "Examination Policy",
    "Plagiarism Policy",
    "Scholarship Policy",
    "Hostel Rules and Regulations (Jan 2025)",
    "Code of Conduct for Students",
]

st.html(
    """
    <style>
      /* Streamlit reserves a large top margin on wide layouts. The title is
         the first thing on the page, so most of that space is wasted. */
      .stMainBlockContainer { padding-top: 2.5rem; }

      .claim-card { border-left: 5px solid; border-radius: 6px; padding: 0.7rem 1rem;
                    margin: 0.6rem 0 0.2rem 0; }
      .claim-head { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; }
      .claim-text { font-size: 1.05rem; margin: 0.25rem 0; }
      .claim-why  { font-size: 0.85rem; opacity: 0.8; }
      .cite { font-family: monospace; font-size: 0.8rem; opacity: 0.85; }
    </style>
    """
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
    # A claim the relevance gate withheld is TRUE of its passage but does not
    # answer the question, so "SUPPORTED" in green would be misleading on a
    # removed card. getattr keeps answers saved before the gate working.
    if getattr(verdict.checks, "relevant", None) is False:
        label, colour, background = OFF_TOPIC_STYLE
    citations = "".join(
        f'<div class="cite">&rarr; {html.escape(c)}</div>' for c in verdict.claim.citation_ids
    ) or '<div class="cite">&rarr; (no citation given)</div>'
    # The policy sentence behind a wrong claim, on the card itself. Empty for
    # supported claims and for old saved answers without the evidence fields;
    # also empty when the explanation above already quotes it.
    policy_lines = "".join(
        f'<div class="claim-why">{html.escape(line)}</div>' for line in policy_says(verdict)
    )

    st.markdown(
        f"""
        <div class="claim-card" style="border-color: {colour}; background: {background};">
          <div class="claim-head" style="color: {colour};">
            {label} &middot; support {verdict.score:.2f}
          </div>
          <div class="claim-text">{html.escape(verdict.claim.text)}</div>
          {citations}
          <div class="claim-why">{html.escape(verdict.explanation)}</div>
          {policy_lines}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Evidence and checks", key=key):
        checks = verdict.checks
        c1, c2, c3 = st.columns(3)
        # Two different numbers live here and they are easy to confuse. The
        # card above shows SUPPORT - how strongly the passage entails the
        # claim. This shows how certain the model was of whichever label it
        # picked. A neutral verdict can read "support 0.03" above and
        # "neutral, certainty 0.96" here: not a contradiction, two different
        # questions. Naming them differently is the whole point.
        c1.markdown(
            "**Entailment (NLI)**  \n"
            + (
                f"{checks.nli_label} &mdash; certainty {checks.nli_score:.2f}"
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

        # The one sentence that decided the verdict, shown before the whole
        # section so the reader does not have to hunt for it. getattr keeps
        # older saved answers (from before these fields existed) working.
        evidence_text = getattr(checks, "evidence_text", None)
        if evidence_text:
            heading = (
                "Contradicting sentence"
                if verdict.status is VerdictStatus.REFUTED
                else "Supporting sentence"
            )
            st.markdown(f"**{heading}** &middot; {html.escape(checks.evidence_chunk_id or '')}")
            st.text(evidence_text)
        contradicting_text = getattr(checks, "contradicting_text", None)
        if getattr(checks, "conflict", False) and contradicting_text:
            st.markdown(
                "**Contradicting sentence (another cited section)** &middot; "
                + html.escape(checks.contradicting_chunk_id or "")
            )
            st.text(contradicting_text)

        chunks = evidence_for(verdict, store)
        if not chunks:
            st.caption("No policy text could be resolved for this claim's citations.")
        for chunk in chunks:
            st.markdown(f"**{chunk.section_path or chunk.citation_id}**")
            st.caption(f"{chunk.citation_id}  ·  [{chunk.university_name} source]({chunk.source_url})")
            st.text(chunk.text)


def draw_declined(answer: Answer) -> None:
    """Explain a declined question in plain terms.

    A decline is the system working as designed, so the page says why in
    words a student would use, and lists what the system can answer from -
    otherwise "no answer" reads like a crash.
    """
    # abstain.decide() words this case as "do not appear to contain": the
    # relevance gate found the drafted claims TRUE of the policy but not an
    # answer to the question - so the fact-checker did not reject them.
    off_topic = "do not appear to contain" in (answer.reason or "")
    if off_topic:
        why = (
            "The search found related passages, but none of them states this. "
            "The statements drafted from them are true of the policy but do not "
            "answer the question; they are listed below under *Removed claims*."
        )
    elif answer.claims_removed:
        why = (
            "The language model drafted an answer, but the fact-checker could not "
            "confirm it from the policy text, so it is **not** shown as an answer. "
            "The rejected draft is listed below under *Removed claims*."
        )
    else:
        why = (
            "The search found the closest passages in the policy documents, but "
            "none of them states this information, so there was nothing to answer from."
        )

    st.warning(
        "**This is not stated in the SRM policy documents, so the system did not "
        "answer.**\n\n" + why,
        icon=":material/block:",
    )
    st.info(
        "The system answers **only** from these six documents. A question about "
        "anything else is declined instead of guessed:\n\n"
        + "\n".join(f"- {name}" for name in DOCUMENTS)
        + "\n\nThe passages it searched are under *Retrieved passages* at the bottom."
    )


def draw_answer(answer: Answer, passages: list, store: VectorStore) -> None:
    st.subheader(answer.question)

    if answer.abstained:
        draw_declined(answer)
    elif answer.claims_kept:
        st.success(
            f"**{len(answer.claims_kept)} verified claim(s).** Every statement below "
            "was checked against the policy text it cites. Support runs from "
            "0.00 to 1.00.",
            icon=":material/verified:",
        )
        for i, verdict in enumerate(answer.claims_kept):
            draw_claim(verdict, store, key=f"kept-{i}")
        if answer.reason:
            st.info(answer.reason, icon=":material/info:")

    if answer.claims_removed:
        st.space("small")
        st.markdown(f"##### Removed claims ({len(answer.claims_removed)})")
        st.caption(
            "The model generated these and the system would not stand behind them. "
            "They are shown rather than hidden, so you can see what was rejected and why."
        )
        for i, verdict in enumerate(answer.claims_removed):
            draw_claim(verdict, store, key=f"removed-{i}")

    st.space("small")
    t = answer.timings
    with st.container(border=True):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Kept", len(answer.claims_kept))
        m2.metric("Removed", len(answer.claims_removed))
        m3.metric("Retrieve", f"{t.retrieve_ms / 1000:.1f}s")
        m4.metric("Generate", f"{t.generate_ms / 1000:.1f}s")
        m5.metric("Verify", f"{t.verify_ms / 1000:.1f}s")

    with st.expander(
        f"Retrieved passages ({len(passages)})", icon=":material/search:"
    ):
        st.caption(
            "What the search returned for this question. The model may only use, "
            "and only cite, these."
        )
        for r in passages:
            boost = (
                f" +{r.boost:.3f} ({', '.join(r.matched_entities)})" if r.boost else ""
            )
            st.markdown(f"`{r.score:.3f}{boost}` &nbsp; **{r.chunk.citation_id}**")
            st.caption(r.chunk.text[:400] + ("..." if len(r.chunk.text) > 400 else ""))

    with st.expander("Raw answer (JSON)", icon=":material/code:"):
        st.json(json.loads(answer.model_dump_json()))


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

with st.form("ask"):
    question = st.text_input(
        "Your question",
        key="question_box",
        placeholder="e.g. What is the minimum attendance requirement?",
    )
    submitted = st.form_submit_button("Verify answer", type="primary")

# The examples are drawn into this slot further down, once we know whether
# there is an answer to show. Reserving the space here keeps them directly
# under the question box; filling it later lets the *first* answer collapse
# them too, which a plain top-to-bottom render cannot do.
examples_slot = st.container()


def draw_examples(collapsed: bool) -> None:
    def buttons() -> None:
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

    with examples_slot:
        if collapsed:
            # Six full-width buttons between the question and the answer
            # pushed every result below the fold.
            with st.expander("Try another example", icon=":material/lightbulb:"):
                buttons()
        else:
            st.caption("Try one (each of these is answered from the policy documents):")
            buttons()


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

has_result = "result" in st.session_state
draw_examples(collapsed=has_result)

if has_result:
    draw_answer(*st.session_state["result"], store)

