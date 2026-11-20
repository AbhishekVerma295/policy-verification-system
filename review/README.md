# Review — first academic review materials

All files for the first academic review live in this folder.

## Literature review (shared doc)

**Literature Review: Hallucination and Citation Verification for Policy QA**
https://claude.ai/code/artifact/855e15e0-5d41-45f8-b15e-69cb1f63f69a

Covers the final six papers in detail (problem, method, data, findings with
numbers, limitations, relevance to our project, how to use each in the survey),
a comparison matrix, the candidate research gap, the three dropped papers,
corrections to the teammate's AI-written study guide, supporting papers, and
references.

Local copies of the same content:

- `Literature_Review.docx` — Word version (11 pages, A4).
- `Literature_Review.md` — Markdown source of the Word file.

## Task 2 — Project audit

- `Task2_Project_Audit.docx` — Word version (7 pages, A4).
- `Task2_Project_Audit.md` — Markdown source.

What the code really does, verified numbers, implemented vs pending work,
findings ranked by severity, and documentation discrepancies.

## First review presentation

- `First_Review_Presentation.pptx` — 11 slides, 16:9, speaker notes on every slide.
- `PPT_Slide_Content.md` — the same speaker notes and likely professor questions, slide by slide.

Slide 1 shows the title, subtitle and an example panel (no team details).

## Final review presentation

- `Final_Review_Presentation.pptx` — 13 slides, same style, speaker notes on every slide:
  Title · Abstract · Introduction · Literature Survey ·
  Architecture Diagram (System Overview; Decision Flow for One Question) ·
  Low-Level Design (Preparation & Retrieval; Generation & Verification) ·
  Experimental Setup (Data; Environment and Configuration) ·
  Results and Discussion (measured spot checks; discussion and next steps) ·
  References (one slide: the six papers, what we took from each, and the SRM data source).

Results are development spot checks, labelled as such; the full evaluation has
not been run. `First_Review_Presentation.pptx` is kept unchanged.

## Task 3 — High-level flowchart

- `Architecture_Block_Diagram.svg` / `.png` — system overview at component level:
  offline index build and online question path (16:9, transparent; PNG is 3840×2160).
- `Task3_Architecture_Flowchart.svg` / `.png` — decision flow for one question
  (16:9, transparent background; PNG is 3840×2160).
- `Architecture_LLD_Mermaid.md` — Mermaid code for the block diagram, the flowchart
  and both LLD slides, for draw.io (Arrange → Insert → Advanced → Mermaid) or mermaid.live.

## Task 4 — Low-level design (two slides)

- `Task4_LLD_Slide1_Preparation_Retrieval.svg` / `.png` — fetch → extract →
  sections → chunk → embed → Chroma → retrieval.
- `Task4_LLD_Slide2_Generation_Verification.svg` / `.png` — claim generation →
  parsing → citation validation → NLI → numeric → verdict → final answer.

Each module card shows IN / DO / OUT / NEXT and its source file and function.

**Current versions (17 Nov 2026, match the code after the hybrid-retrieval,
sentence-level NLI, relevance-gate and ingestion-repair commits):**

- `LLD_Slide1_Preparation_Retrieval_v2.svg` / `.png`: Roman-numeral headings, table rows,
  212 sections (192 kept), 262 chunks, hybrid search (embedding + BM25, merged by RRF).
- `LLD_Slide2_Generation_Verification_v2.svg` / `.png`: passage screening, whole passages,
  claim salvage, citation repair, sentence-level NLI, clock-time check, cross-section
  conflict rule, the new relevance gate, and the final rules without the retired
  0.25 retrieval-score check.

- `LLD_Slide1_Simple_v2` / `LLD_Slide2_Simple_v2` (`.svg` / `.png`): presenter versions
  (IN / OUT only, large text) of the two diagrams above.
- `Architecture_Basic_v2.svg` / `.png`: the simple high-level architecture with the
  current numbers, hybrid retriever and relevance check.

The Task4 files above (and the older Simple versions and Mermaid code) show the older pipeline.

## Final review deck

- `RAI REVIEW 1 PPT (final).pptx`: 15 slides rebuilt in the team deck's template and style,
  content as approved in `Final_Deck_Content.md`, speaker notes with likely questions on
  every slide. Supersedes the "(updated)" file below.

## Updated review deck (superseded)

- `RAI REVIEW 1 PPT (updated).pptx`: the team's 12-slide deck brought up to date with the
  code and the final held-out results (17 Nov 2026). The diagrams on slides 5–7 are
  replaced, the literature and methodology tables on slides 3–4 are now native tables
  (no longer cut off), the setup slides use the current counts and configuration, the
  results slide shows the final Run A / B / C numbers from `RESEARCH_PAPER_DATA.md`, and
  references add SummaC and Reciprocal Rank Fusion.

Presenter versions (module name, file, IN and OUT only, large text):
`Task4_LLD_Slide1_Simple.svg` / `.png` and `Task4_LLD_Slide2_Simple.svg` / `.png`.
Use these on the main slides and keep the detailed ones as backup slides.
In PowerPoint: Insert → Pictures → the .svg, then right-click → Convert to
Shape to edit.

## Final six papers (PDFs in `papers/`, arXiv versions)

| # | File | Paper |
|---|---|---|
| 1 | `1_Lewis2020_RAG.pdf` | Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020) |
| 2 | `2_Min2023_FActScore.pdf` | Min et al., FActScore (EMNLP 2023) |
| 3 | `3_Gao2023_ALCE.pdf` | Gao et al., Enabling LLMs to Generate Text with Citations (EMNLP 2023) |
| 4 | `4_Tang2024_MiniCheck.pdf` | Tang, Laban & Durrett, MiniCheck (EMNLP 2024) |
| 5 | `5_Asai2024_SelfRAG.pdf` | Asai et al., Self-RAG (ICLR 2024) |
| 6 | `6_Magesh2025_LegalRAGHallucination.pdf` | Magesh et al., Hallucination-Free? Legal AI Research Tools (JELS 2025; arXiv version) |

These PDFs are for team reading only. They are not part of the codebase, so
don't commit them.

## Preparation phase — detailed low-level design

Three diagrams of the offline pipeline, describing the current code
(16:9, transparent background; PNG is 3840×2160):

- `Prep_LLD_A_Corpus_Preparation.svg` / `.png` — manifest → download → checksums → raw files → text extraction (HTML and PDF branches).
- `Prep_LLD_B_Document_Processing.svg` / `.png` — heading detection → sections → validated Document JSON, with the attendance rule 7.3 traced as a worked example.
- `Prep_LLD_C_Chunking_Indexing.svg` / `.png` — citation IDs → chunks → embeddings → Chroma index and index manifest → smoke test.
- `Prep_LLD_Mermaid.md` — the same three diagrams as Mermaid code for draw.io (Arrange → Insert → Advanced → Mermaid) or mermaid.live.

## Final review presentation — version 2 (21 Nov 2026)

- `RAI REVIEW 2 PPT (version 2).pptx` — 17 slides, same template as `RAI REVIEW 1 PPT (final).pptx`
  (kept unchanged), speaker notes and likely questions on every slide. Updated to version 2:
  Qwen3.5 9B, 7 documents (225 sections, 307 chunks), the table-row check, and the final test
  (99 questions, run once, `run-20261120-165959`). New slides: Errors and limitations, The system in
  use (screenshots), Conclusion and future work. The Abstract panel quotes three real outputs from
  the final run (`fb-33`, `fb-22`, `fa-25`).
- `Architecture_Basic_v3`, `LLD_Slide1_Preparation_Retrieval_v3`, `LLD_Slide2_Generation_Verification_v3`
  (`.svg` / `.png`): the v2 diagrams with the version 2 numbers, model and table-row check.
