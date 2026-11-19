"""
nli.py - does this passage actually prove this claim?

Natural Language Inference. The model is given two texts and answers one
narrow question:

    premise:    "A student must maintain a minimum attendance record of at
                 least 75% in individual courses."
    hypothesis: "The attendance requirement is 75%."
    answer:     ENTAILMENT

Three possible answers, and all three are useful:
    ENTAILMENT     the passage proves the claim
    CONTRADICTION  the passage disproves it
    NEUTRAL        the passage neither proves nor disproves it

NEUTRAL is not a failure of the checker. It is the honest answer when a
passage simply does not address a claim, and it is exactly the situation in
which the system should consider staying quiet rather than answering.

WHY NOT JUST ASK QWEN TO CHECK ITS OWN WORK
    A model grading itself tends to repeat its own mistakes: the same
    misreading that produced the bad claim produces a confident bad grade.
    A separate NLI model answers one narrow question independently, runs on
    CPU, and is deterministic.

WHY THE PASSAGE IS CUT INTO SENTENCES, NOT 900-CHARACTER WINDOWS
    The first version scored the claim against 900-character windows of the
    cited section. Measured, that was wrong in a damaging way: the model
    rejected TRUE claims, including word-for-word quotes of the section they
    cite - several were marked REFUTED at 0.95 confidence or more. NLI models
    are trained on single-sentence premises (MNLI, FEVER, ANLI), and when
    handed a paragraph that mixes a rule, its exceptions and an unrelated
    clause, they get confused about what is being asked.

    The fix follows SummaC (Laban et al., TACL 2022): score the claim against
    each SENTENCE of the passage, and against short runs of two and three
    neighbouring sentences (because some claims join two sentences, e.g. a
    rule and the number it refers to). Support in any one unit is support
    from the passage. A read-only probe on our own data turned 10 of 11
    wrongly-removed live quotes into SUPPORTED this way.

    Text with no sentence punctuation (tables, bulleted lists flattened out
    of a PDF) would otherwise become one giant "sentence", so it falls back
    to overlapping ~300-character slices. Nothing is ever skipped: skipping
    text is the silent wrong answer this project exists to prevent.

WHY PDF NEWLINES ARE FLATTENED - BUT ONLY AFTER THE STRUCTURE IS READ
    PDF text has hard line breaks in the middle of sentences ("at least 75%\n
    in individual courses"). Left in, they hide sentence boundaries and break
    exact-quote matching, so whitespace inside a unit is collapsed to single
    spaces.

    The line breaks are read first, though, because some of them ARE the
    structure, and flattening everything at once lost it (measured after the
    ingestion repair, when answerable questions started being declined):
      - Table rows. Ingestion writes each row as one self-describing line
        ("SAT SCORE | Category: 1401 -1500 | ... | Benefit: 75% ..."). A row
        is a unit on its own and is never joined to its neighbour in a
        window, because two rows are two categories (see _layout).
      - List items. A line starting "a)", "(ii)" or "2." starts a new unit
        even with no full stop before it, a bare "1." is kept with its item,
        and a numbered subsection heading ("5.2 Temporary withdrawal ...")
        ends the stretch that windows may span (see _prose_sentences).
      - Lead-in lists. A timetable under "Food will be served at the
        following timings: -" is also offered item by item with its lead-in
        line, so "Dinner 07.30 pm 09.00 pm" is read as one fact
        (see _lead_in_units).
    These only ADD ways for the passage's own words to be read correctly or
    take away windows that glued unrelated rules together; the threshold,
    the model and the "support anywhere wins" rule are unchanged.

WHY THERE IS AN EXACT-QUOTE SHORTCUT
    If the claim appears word for word inside the passage, the passage
    proves it - no model needed, and no chance of the model talking itself
    out of it. The numeric guard in verifier.py still runs afterwards, as it
    does for every claim.

WHY THE MODEL IS CALLED IN BATCHES
    Sentences and short windows mean many more (shorter) pairs to score than
    three big windows did. Calling the model once per pair would pay the
    per-call overhead dozens of times per claim; handing it the whole list at
    once lets it pad them into batches and run them together. Measured on
    this laptop's CPU: one short pair alone 66 ms, eight in one batch 184 ms.

WHY IT IS FASTER, NOT SLOWER (TWO MORE THINGS)
    1. Float32. The checkpoint is stored in float16, and recent transformers
       versions load it that way. CPUs have no fast float16 arithmetic: one
       short pair took 416 ms in float16 and 66 ms in float32. The old
       verifier was paying this too; it is fixed in the pipeline below.
    2. Sentences first. If any single sentence already supports the claim,
       that settles it and no windows are scored. Most true claims stop here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from policyverify.config import Config, get_config

# Split on sentence ends. Deliberately simple: over-splitting is harmless
# here because windows overlap and are scored independently. Used only by
# split_windows(), which is kept for anything that still imports it.
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")

# Characters per window, for split_windows(). No longer used by check().
_WINDOW_CHARS = 900
_WINDOW_OVERLAP = 200

# A sentence ends at . ! or ? followed by a space and something that can
# start a sentence (a capital, a digit, an opening bracket or quote). The
# lookahead stops "e.g. the" and "approx. 5" splitting in most cases.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")

# Sentence ends are also recognised before a lettered or roman list marker
# ("program. b) If a student ..."): the general rule above wants a capital,
# so without this a whole a)/b) list became one long unit the model could
# not read (measured: a true claim about item a) scored 0.00 against it).
_LIST_ITEM_END_RE = re.compile(r"(?<=[.!?;])\s+(?=\(?(?:[a-z]|[ivx]{1,4})\)\s)")
# A numbered item that starts mid-line with no full stop before it
# ("Five days academic suspension 2. Any complaint ..."): PDF and HTML lists
# lose their line breaks, and the next item's number is the only boundary
# left. Only "<lowercase word> N. Capital", so "Rs.5000", "12.5 Menu" and
# "under Rule 2. The" never split. The match marks the gap with \x00, and
# the text is then split there.
_INLINE_NUMBER_RE = re.compile(r"(\b[a-z]+)\s+(?=\d{1,2}\.\s+[A-Z])")
# A piece that is nothing but a list marker ("1.", "12.1", "b)", "(ii)").
# The sentence rule splits "cards. 1. Intimation" after "1.", leaving the
# marker as a unit of its own that asserts nothing; it is joined back onto
# the item it introduces.
_MARKER_ONLY_RE = re.compile(r"^(?:\(?[a-zA-Z]\)|\(?[ivxIVX]{1,5}\)|\d{1,3}(?:\.\d{1,3})*\.?|[-•*–])$")
# A line opening a numbered subsection: "5.2 Temporary withdrawal ...",
# "12.10 If a day scholar ...", optionally after the stray "R" margin mark
# the PDFs carry. Windows never cross one (see _layout): measured, a window
# running from "(iii) M.Tech ... maximum of 14 semesters" over the 5.2
# heading into a 5.3 sentence naming B.Tech "supported" the false claim
# "a B.Tech degree may take at most 14 semesters" at 0.99.
_SUBSECTION_RE = re.compile(r"^(?:R\s+)?\d{1,2}(?:\.\d{1,2})+\s+[A-Z]")
# A piece that ENDS with the next item's number: "<lowercase word or colon>
# N." at the very end.
_TRAILING_MARKER_RE = re.compile(r"^(.*\b[a-z]+|.*:)\s+(\d{1,2}\.)$")
# Where a list item starts on its own line: a marker, as above, followed by
# a space. Used to find the items under a lead-in line (see _lead_in_units).
_ITEM_MARKER_RE = re.compile(r"^(?:\(?[a-zA-Z]\)|\(?[ivxIVX]{1,5}\)|\d{1,2}[.)]|[-•*–])\s")
# A short line starting with a capital ("Breakfast", "Dinner") also starts a
# list item under a lead-in; a long one is ordinary prose and ends the list.
_SHORT_ITEM_CHARS = 40
_MAX_LIST_ITEMS = 20

# One self-describing table row, as ingestion writes them: cells separated by
# " | ", with at least one "Label: value" cell, e.g.
# "SAT SCORE | Category: 1401 -1500 | Benefit: 75% Tuition Fee Waiver."
_TABLE_ROW_RE = re.compile(r"^[^|\n]+(?:\s\|\s[^|\n]+)+$")
_TABLE_CELL_LABEL_RE = re.compile(r"\|\s*[^|:]{1,60}:\s")

# A "sentence" longer than this almost certainly is not one - it is a table
# or list with no full stops. Such text is cut into overlapping slices
# instead, so the model still sees every part of it at a size it handles.
_MAX_SENTENCE_CHARS = 400
_SLICE_CHARS = 300
_SLICE_STEP = 220  # so neighbouring slices overlap by 80 characters

# Exact-quote shortcut: the claim must be at least this long (after
# normalising) AND at least this many words. Short strings match by
# accident and assert nothing - "at least 75%" or "the dean of students"
# occur in many passages that say something quite different about them. A
# real quoted claim from the generator is a full clause; the shortest
# verbatim claim in eval/claims.jsonl is well above these limits.
_QUOTE_MIN_CHARS = 40
_QUOTE_MIN_WORDS = 6
# If the quoted text is immediately preceded by one of these words in the
# passage, the passage may be saying the OPPOSITE ("it is not the case that
# students may ..."). Rare, but cheap to refuse - the model then decides.
_NEGATORS = {"not", "no", "never", "nor", "cannot", "without"}
_NEGATOR_LOOKBACK = 4

ENTAILMENT = "entailment"
NEUTRAL = "neutral"
CONTRADICTION = "contradiction"


@dataclass
class NLIResult:
    """One verdict from the NLI model.

    `entailment`, `neutral` and `contradiction` are the three probabilities
    for the ONE evidence unit (sentence or short window) that decided the
    verdict, and `window` is the full text of that unit - shown to the user
    as "this is the sentence that supports the claim".

    The optional fields describe the rest of the passage. verifier.py needs
    them to spot two cited sections that disagree: a section can support a
    claim in one sentence and the verdict for it is still ENTAILMENT, but
    another section's strongest contradiction must be visible too. Stub
    checkers in the tests leave them at their defaults, and the properties
    below fall back sensibly.
    """

    label: str
    entailment: float
    neutral: float
    contradiction: float
    # The full text of the evidence unit that produced this verdict.
    window: str = ""
    # Strongest contradiction seen in ANY unit of the passage, and its text.
    # None means "not recorded" (a stub, or the old interface).
    strongest_contradiction: float | None = None
    contradiction_window: str = ""
    # True when the verdict came from the exact-quote shortcut, not the model.
    exact_quote: bool = False
    # Strongest contradiction from a unit that is NOT a table row, and its
    # text. This is what verifier.py's "two sections disagree" rule uses; see
    # `conflict_strength`. None means "not recorded" (stubs, old interface).
    prose_contradiction: float | None = None
    prose_contradiction_window: str = ""

    @property
    def conflict_strength(self) -> float:
        """How strongly this passage contradicts the claim, for the purpose of
        deciding that two cited SECTIONS disagree.

        Table rows are left out. Each row states one value for one category
        ("SAT 1101-1200 -> 10% waiver"), and the model reads a different
        category's row as contradicting a claim about another category: a
        true "the 75% merit scholarship is a 75% waiver" was withheld as a
        conflict because the MAT table, which has no 75% row, "contradicted"
        it at 0.99 with its 100% row. That is two tables about different
        things, not two sources disagreeing. Prose rules still count in full,
        and a row can still REFUTE a claim nothing supports.
        """
        if self.prose_contradiction is None:
            return self.max_contradiction
        return self.prose_contradiction

    @property
    def conflict_text(self) -> str:
        """The unit that produced conflict_strength."""
        if self.prose_contradiction is None:
            return self.max_contradiction_text
        return self.prose_contradiction_window

    @property
    def score(self) -> float:
        """Confidence in whichever label was chosen."""
        return {
            ENTAILMENT: self.entailment,
            NEUTRAL: self.neutral,
            CONTRADICTION: self.contradiction,
        }.get(self.label, 0.0)

    @property
    def max_contradiction(self) -> float:
        """The strongest contradiction anywhere in the passage."""
        if self.strongest_contradiction is None:
            return self.contradiction
        return max(self.strongest_contradiction, self.contradiction)

    @property
    def max_contradiction_text(self) -> str:
        """The unit that produced max_contradiction."""
        if self.strongest_contradiction is not None and (
            self.strongest_contradiction > self.contradiction
        ):
            return self.contradiction_window
        return self.window


# ---------------------------------------------------------------------------
# Cutting a passage into evidence units (pure functions, no model)
# ---------------------------------------------------------------------------


def flatten(text: str) -> str:
    """Collapse every run of whitespace (including PDF hard newlines) to one space."""
    return re.sub(r"\s+", " ", text).strip()


def is_table_row(text: str) -> bool:
    """Is this one self-describing table row ("A | Label: value | ...")?

    Ingestion stores every HTML table row as one line of this shape, with the
    table's own lead-in and column names repeated in it, so a row can be read
    on its own. See `_layout` for why rows are kept apart from prose.
    """
    text = text.strip()
    return bool(_TABLE_ROW_RE.match(text)) and bool(_TABLE_CELL_LABEL_RE.search(text))


def _slice_long(piece: str) -> list[str]:
    """A piece too long to be a sentence, as overlapping slices.

    Slices start every _SLICE_STEP characters; the final slice is allowed to
    be short, and we stop once the end has been covered.
    """
    if len(piece) <= _MAX_SENTENCE_CHARS:
        return [piece]
    out: list[str] = []
    start = 0
    while True:
        out.append(piece[start : start + _SLICE_CHARS].strip())
        if start + _SLICE_CHARS >= len(piece):
            break
        start += _SLICE_STEP
    return out


def _prose_sentences(text: str) -> list[str]:
    """Sentences of a stretch of prose, after flattening whitespace.

    On top of the plain sentence rule, list structure is respected: a new
    sentence starts at a lettered item after a full stop ("... program. b)
    If ...") and at a numbered item that follows a word with no full stop
    ("suspension 2. Any complaint"), and a bare marker ("1.") is joined to
    the item it introduces instead of standing alone.

    A line that begins with a list marker also starts a new sentence, full
    stop or not. Lists are often written one item per line with no
    punctuation ("c) Withdrawal from Summer semester courses is not
    permitted" / "d) These courses will be conducted for 7 to 8 weeks"), and
    flattened they ran together into one unit in which the rule that
    mattered was buried.
    """
    segments: list[str] = []
    for line in text.splitlines() or [text]:
        if segments and _ITEM_MARKER_RE.match(line.strip()):
            segments.append(line)
        elif segments:
            segments[-1] += "\n" + line
        else:
            segments.append(line)
    pieces: list[str] = []
    for segment in segments:
        flat = flatten(segment)
        for part in _SENTENCE_END_RE.split(flat) if flat else []:
            for sub in _LIST_ITEM_END_RE.split(part):
                marked = _INLINE_NUMBER_RE.sub("\\1\x00", sub)
                pieces.extend(p.strip() for p in marked.split("\x00"))
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        if not piece:
            continue
        if _MARKER_ONLY_RE.match(piece):
            carry = f"{carry} {piece}".strip()
            continue
        piece = f"{carry} {piece}".strip() if carry else piece
        carry = ""
        # "Intimation to parents or / and 2." - the next item's number was
        # left on the end of this one. Move it forward. Only after a
        # lowercase word or a colon, so "... under Rule 2." keeps its number.
        trailing = _TRAILING_MARKER_RE.match(piece)
        if trailing:
            piece, carry = trailing.group(1), trailing.group(2)
        merged.append(piece)
    if carry:  # a trailing marker with nothing after it
        merged.append(carry)
    out: list[str] = []
    for piece in merged:
        out.extend(_slice_long(piece))
    return out


def _lead_in_units(lines: list[str]) -> list[str]:
    """A list item joined to the line that introduces it.

    Timetables and similar lists are written as a lead-in line ending in a
    colon followed by short items, one per line:

        Food will be served at the following timings: -
        Breakfast / 07.00 am / to 09.00 am / ... / Dinner / 07.30 pm / 09.00 pm

    Flattened, that is one long "sentence" in which the model cannot tell
    which time belongs to which meal (measured: "Dinner is served from 7:30
    pm to 9:00 pm" scored 0.01 against it). "<lead-in> Dinner 07.30 pm 09.00
    pm" scored 0.98. So each item is also offered with its lead-in.

    An item starts at a line beginning with a list marker or a short line
    starting with a capital; lines starting otherwise (a time, a lowercase
    word) continue the current item; a long capitalised line is ordinary
    prose and ends the list. Lines broken mid-sentence by a PDF therefore
    never become "items" - that would hand the model sentence fragments,
    which can drop a qualifier on the next line.
    """
    units: list[str] = []
    for i, line in enumerate(lines):
        if not re.search(r":\s*[-–]?\s*$", line):
            continue
        before = _prose_sentences("\n".join(lines[: i + 1]))
        lead = before[-1] if before else line.strip()
        items: list[list[str]] = []
        for nxt in lines[i + 1 :]:
            nxt = nxt.strip()
            if not nxt:
                continue
            starts_item = bool(_ITEM_MARKER_RE.match(nxt)) or (
                nxt[0].isupper() and len(nxt) <= _SHORT_ITEM_CHARS
            )
            # Two kinds of list. In a MARKED list (first item "a) ...") only a
            # marker starts an item, and any other line continues the current
            # item while that item has not finished its sentence ("... by
            # the respective" / "School/Department, ..."). In a list of SHORT
            # items ("Breakfast", "Lunch") a short capitalised line starts an
            # item. In both, a line that cannot open a sentence (a time, a
            # lowercase word) continues the item; anything else is prose and
            # the list is over.
            marked = bool(items) and bool(_ITEM_MARKER_RE.match(items[0][0]))
            if marked:
                starts_item = bool(_ITEM_MARKER_RE.match(nxt))
            open_item = bool(items) and not re.search(r"[.!?]$", items[-1][-1])
            if starts_item:
                if len(items) >= _MAX_LIST_ITEMS:
                    break
                items.append([nxt])
            elif items and (not nxt[0].isupper() or (marked and open_item)):
                items[-1].append(nxt)
            else:
                break  # not a list, or the list has ended in prose
        if len(items) < 2:
            continue
        for item in items:
            text = flatten(" ".join(item))
            if len(text) <= _MAX_SENTENCE_CHARS:
                units.append(f"{lead} {text}")
    return units


@dataclass
class _Layout:
    """A passage cut into evidence units, respecting its structure.

    `singles`  every single unit in reading order: prose sentences and table
               rows. Scored first.
    `runs`     stretches of consecutive prose sentences. Windows of 2-3
               neighbouring sentences are built inside a run only.
    `lead_ins` list items joined to their lead-in line (see _lead_in_units).
    """

    singles: list[str]
    runs: list[list[str]]
    lead_ins: list[str]


def _layout(text: str) -> _Layout:
    """Cut a passage into units, keeping table rows whole and on their own.

    Ingestion writes each table row as one self-describing line. Such a row
    is a unit by itself and is never joined to a neighbouring row in a
    window: two rows side by side describe two different categories, and
    the model read the pair as support for a claim that mixes them (measured:
    "a CLAT rank in the top 2500 receives a 75% waiver" scored 0.87 against
    the top-2500 row joined to the 2501-5000 row, and 0.00 against the
    top-2500 row alone). Everything that is not a row is handled as prose,
    with PDF line breaks flattened as before.
    """
    singles: list[str] = []
    runs: list[list[str]] = []
    prose: list[str] = []
    lead_ins: list[str] = []

    def close_prose() -> None:
        if not prose:
            return
        sentences = _prose_sentences("\n".join(prose))
        singles.extend(sentences)
        if sentences:
            runs.append(sentences)
        lead_ins.extend(_lead_in_units(prose))
        prose.clear()

    for line in text.splitlines() or [text]:
        if _SUBSECTION_RE.match(line.strip()):
            # A numbered subsection heading ("5.2 Temporary withdrawal from
            # the programme:") starts a new stretch, so no window joins the
            # end of one rule to the start of the next.
            close_prose()
        if is_table_row(line):
            close_prose()
            singles.extend(_slice_long(flatten(line)))
        else:
            prose.append(line)
    close_prose()
    return _Layout(singles=singles, runs=runs, lead_ins=lead_ins)


def split_sentences(text: str) -> list[str]:
    """The passage as a list of single evidence units: sentences, and each
    table row on its own.

    Anything too long to be a real sentence (a table, a list with no full
    stops) is cut into overlapping slices instead, so the model is never
    handed a wall of text and no part of it is dropped.
    """
    return _layout(text).singles


def sentence_windows(sentences: list[str], max_sentences: int = 3) -> list[str]:
    """Sliding runs of 2..max_sentences neighbouring sentences.

    Some true claims join two sentences ("Condonation is granted by the Dean.
    It may not exceed 10%." -> "The Dean may condone up to 10%"), so no
    single sentence entails them. Short runs catch those without going back
    to paragraph-sized premises that confuse the model.
    """
    out: list[str] = []
    for size in range(2, max_sentences + 1):
        for i in range(len(sentences) - size + 1):
            out.append(" ".join(sentences[i : i + size]))
    return out


def _dedupe(items: list[str]) -> list[str]:
    """Drop repeats and empties, keeping the first occurrence's position."""
    return list(dict.fromkeys(i for i in items if i))


def evidence_units(text: str, max_sentences: int = 3) -> list[str]:
    """Every unit the claim is scored against: sentences first, then windows.

    `max_sentences=1` means single sentences only. Deduplicated, because a
    short passage's 2-sentence window can equal the whole passage, and
    repeated text would only cost time.
    """
    layout = _layout(text)
    sentences = _dedupe(layout.singles)
    return sentences + _second_pass_units(layout, max_sentences, set(sentences))


def _second_pass_units(layout: _Layout, max_sentences: int, seen: set[str]) -> list[str]:
    """Windows of neighbouring prose sentences, then list items with their
    lead-in, minus anything already scored.

    Windows are built from the sentences in their real order (before removing
    repeats), so a window only ever joins true neighbours - and only inside
    one stretch of prose, never across a table row.
    """
    windows: list[str] = []
    for run in layout.runs:
        windows.extend(sentence_windows(run, max_sentences))
    return [u for u in _dedupe(windows + layout.lead_ins) if u not in seen]


def split_windows(
    text: str, window_chars: int = _WINDOW_CHARS, overlap: int = _WINDOW_OVERLAP
) -> list[str]:
    """Split a passage into overlapping windows on sentence boundaries.

    The original 900-character windowing. No longer used by check() - see
    the module docstring for why - but kept because it is part of the public
    interface and is useful for comparing old and new behaviour.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= window_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    windows: list[str] = []
    current = ""

    for sentence in sentences:
        # A single sentence longer than the window: hard-split it, since
        # there is no smaller natural unit left to break on.
        if len(sentence) > window_chars:
            if current:
                windows.append(current)
                current = ""
            for i in range(0, len(sentence), window_chars):
                windows.append(sentence[i : i + window_chars])
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= window_chars:
            current = candidate
        else:
            windows.append(current)
            tail = current[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = f"{tail} {sentence}".strip()

    if current:
        windows.append(current)
    return windows


# ---------------------------------------------------------------------------
# The exact-quote shortcut (pure function, no model)
# ---------------------------------------------------------------------------


def normalise_for_quote(text: str) -> str:
    """Lowercase, keep only letters, digits and %, collapse spaces.

    So that "75 %" vs "75%", curly vs straight quotes, a stray comma or a
    PDF line break do not stop a genuine quote from matching.
    """
    text = text.lower().replace(" %", "%")
    return re.sub(r"[^a-z0-9%]+", " ", text).strip()


def find_exact_quote(premise: str, hypothesis: str) -> str | None:
    """If `hypothesis` is quoted word for word in `premise`, return the
    smallest evidence unit (sentence or short window) that contains it.

    Returns None when there is no quote, when the claim is too short to
    count (see _QUOTE_MIN_CHARS), or when the quote is preceded by a negating
    word in the passage. If the quote spans more text than any one unit, the
    whole flattened passage is returned as the evidence.
    """
    claim = normalise_for_quote(hypothesis)
    if len(claim) < _QUOTE_MIN_CHARS or len(claim.split()) < _QUOTE_MIN_WORDS:
        return None

    # Pad with spaces so the match must start and end on whole words:
    # "attend" must not match inside "attendance".
    passage = f" {normalise_for_quote(premise)} "
    at = passage.find(f" {claim} ")
    if at < 0:
        return None
    # Look a few words back, to catch "not the case that <quote>" as well as
    # "not <quote>". Refusing here is harmless: the model then decides.
    before = passage[:at].split()[-_NEGATOR_LOOKBACK:]
    if _NEGATORS.intersection(before):
        return None

    units = sorted(evidence_units(premise), key=len)
    for unit in units:
        if f" {claim} " in f" {normalise_for_quote(unit)} ":
            return unit
    return flatten(premise)


def normalise_label(raw: str) -> str:
    """Different checkpoints spell the labels differently. Map to ours."""
    value = raw.lower().replace("-", "_")
    if "entail" in value or value in ("label_0", "0"):
        return ENTAILMENT
    if "contradict" in value or value in ("label_2", "2"):
        return CONTRADICTION
    if "neutral" in value or value in ("label_1", "1"):
        return NEUTRAL
    return value


def combine_unit_results(results: list[NLIResult], threshold: float) -> NLIResult:
    """Pick the deciding unit for one passage. Pure - no model.

    Support anywhere in the passage counts as support. A contradiction only
    wins if nothing entailed, so a passage that states a rule and then its
    exception is not written off. Whatever wins, the strongest contradiction
    seen anywhere is attached, so verifier.py can compare sections.
    """
    if not results:
        return NLIResult(label=NEUTRAL, entailment=0.0, neutral=1.0, contradiction=0.0)

    best_entail = max(results, key=lambda r: r.entailment)
    best_contra = max(results, key=lambda r: r.contradiction)

    if best_entail.entailment >= threshold:
        chosen = best_entail
    elif best_contra.contradiction >= threshold:
        chosen = best_contra
    else:
        # Nothing decisive either way - report the strongest signal we saw.
        chosen = max(results, key=lambda r: max(r.entailment, r.contradiction))

    # The strongest contradiction from prose alone, for the cross-section
    # conflict rule (see NLIResult.conflict_strength).
    prose = [r for r in results if not is_table_row(r.window)]
    best_prose = max(prose, key=lambda r: r.contradiction) if prose else None
    return replace(
        chosen,
        strongest_contradiction=best_contra.contradiction,
        contradiction_window=best_contra.window,
        prose_contradiction=best_prose.contradiction if best_prose else 0.0,
        prose_contradiction_window=best_prose.window if best_prose else "",
    )


class NLIChecker:
    """The fact-checking model, loaded once and reused."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            cfg = self.config.nli
            self._pipeline = pipeline(
                "text-classification",
                model=cfg.model_name,
                # -1 means CPU. Deliberate: this model is small enough to be
                # fast there, and it keeps the whole GPU free for Qwen.
                device=-1 if cfg.device == "cpu" else 0,
                # Full 32-bit precision, explicitly. The checkpoint is saved
                # in float16 and recent transformers versions load it in
                # whatever precision it was saved in. CPUs have no fast
                # float16 arithmetic, so that ran about 10x slower - measured
                # 416 ms for one short sentence pair (see the module notes).
                dtype="float32",
                top_k=None,
                truncation=True,
            )
        return self._pipeline

    @staticmethod
    def _to_result(scores, premise: str) -> NLIResult:
        """Turn the pipeline's raw label scores for one pair into an NLIResult."""
        by_label = {normalise_label(s["label"]): float(s["score"]) for s in scores}
        entail = by_label.get(ENTAILMENT, 0.0)
        neutral = by_label.get(NEUTRAL, 0.0)
        contra = by_label.get(CONTRADICTION, 0.0)
        label = max(
            ((ENTAILMENT, entail), (NEUTRAL, neutral), (CONTRADICTION, contra)),
            key=lambda pair: pair[1],
        )[0]
        return NLIResult(
            label=label,
            entailment=entail,
            neutral=neutral,
            contradiction=contra,
            window=premise,
        )

    def _score_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """Score one pair. Kept for probes and one-off debugging."""
        return self._score_many([premise], hypothesis)[0]

    def _score_many(self, premises: list[str], hypothesis: str) -> list[NLIResult]:
        """Score many premises against one hypothesis in batches.

        The pairs are sorted by length before batching so each batch holds
        similar-length texts and little time is wasted on padding, then put
        back in their original order.
        """
        if not premises:
            return []
        order = sorted(range(len(premises)), key=lambda i: len(premises[i]))
        inputs = [{"text": premises[i], "text_pair": hypothesis} for i in order]
        raw = list(self.pipeline(inputs, batch_size=self.config.nli.batch_size))
        results: list[NLIResult | None] = [None] * len(premises)
        # strict: one answer per pair, or something is badly wrong.
        for i, scores in zip(order, raw, strict=True):
            # One input can come back as a bare list of label dicts, or
            # wrapped once more, depending on the transformers version.
            if scores and isinstance(scores[0], list):
                scores = scores[0]
            results[i] = self._to_result(scores, premises[i])
        return results  # type: ignore[return-value]

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Does `premise` prove `hypothesis`?

        1. An exact quote of the passage is entailment, without the model.
        2. Otherwise score every sentence. If one entails, that settles it.
        3. Otherwise also score the 2- and 3-sentence windows.

        Step 2 stopping early never changes the verdict: once any unit
        entails, the passage supports the claim whatever the windows say.
        It only saves time on the common case of a claim drawn from one
        sentence.
        """
        # Line breaks are kept until the passage has been cut into units:
        # they are what marks a table row or a list item. Each unit is
        # flattened on its own.
        if not flatten(premise):
            return NLIResult(label=NEUTRAL, entailment=0.0, neutral=1.0, contradiction=0.0)

        quote = find_exact_quote(premise, hypothesis)
        if quote is not None:
            return NLIResult(
                label=ENTAILMENT,
                entailment=1.0,
                neutral=0.0,
                contradiction=0.0,
                window=quote,
                strongest_contradiction=0.0,
                exact_quote=True,
                prose_contradiction=0.0,
            )

        cfg = self.config.nli
        threshold = cfg.entailment_threshold
        layout = _layout(premise)
        sentences = _dedupe(layout.singles)
        results = self._score_many(sentences, hypothesis)
        if any(r.entailment >= threshold for r in results):
            return combine_unit_results(results, threshold)

        # Windows of neighbouring sentences, and list items with their
        # lead-in line.
        extra = _second_pass_units(layout, cfg.max_window_sentences, set(sentences))
        results += self._score_many(extra, hypothesis)
        return combine_unit_results(results, threshold)
