# Architecture and LLD: Mermaid code

Four diagrams matching the review slides (current code):

| Section | Matches |
|---|---|
| A. System overview | `Architecture_Block_Diagram.png` |
| B. Decision flow for one question | `Task3_Architecture_Flowchart.png` |
| C. LLD: preparation and retrieval | `Task4_LLD_Slide1_Preparation_Retrieval.png` |
| D. LLD: generation and verification | `Task4_LLD_Slide2_Generation_Verification.png` |

To use one:

- **draw.io / diagrams.net:** *Arrange → Insert → Advanced → Mermaid*, paste the code (everything between the ``` lines, without them), then *Insert*. draw.io turns it into normal shapes you can move, restyle and export (*File → Export as → PNG / SVG*).
- **mermaid.live:** paste the code into the editor; export PNG or SVG from the *Actions* menu.

`#35;`, `#124;`, `#lt;`, `#gt;` and `#quot;` are Mermaid codes for `#`, `|`, `<`, `>` and `"`.

## A. System overview (component level)

```mermaid
flowchart TB
  subgraph OFF["1 · Offline: build once — scripts/ingest.py, scripts/build_index.py"]
    direction LR
    S[("SRM policy sources<br/>3 PDFs + 3 web pages<br/>srmist.edu.in, listed in data/manifest.yaml")] --> I["Ingestion — ingest/<br/>download + SHA-256 checksum<br/>extract text (PDF / HTML)<br/>split at real headings"]
    I --> J[("Policy sections (JSON)<br/>6 documents, 192 sections<br/>data/processed/")]
    J --> C["Chunking + embedding — indexing/<br/>258 chunks, 1 citation ID each<br/>bge-base-en-v1.5 (CPU), 768-number vectors"]
    C --> X[("Chroma vector index<br/>vectors + original text + citation metadata<br/>data/index/")]
  end
  subgraph ON["2 · Online: per question — pipeline.py"]
    direction LR
    Q(["Student question<br/>optional policy filter"]) --> R["Retriever — retrieve.py<br/>embed question, search the index<br/>top 6 passages, ≤ 2 per section"]
    R --> G["Generator — generate.py<br/>Qwen3 4B via Ollama (GPU), temperature 0<br/>single-fact claims, each with a citation"]
    G -- claims --> V["Verifier — verify/<br/>per claim: NLI DeBERTa-v3 (CPU)<br/>+ number check + citation check"]
    V -- verdicts --> D["Decision — abstain.py<br/>keep SUPPORTED claims,<br/>remove the rest with reasons, or abstain"]
    D --> A(["Answer<br/>verified claims + citations,<br/>removed claims + reasons, or an abstention"])
    R -. "retrieved passages = evidence for each claim's citations" .-> V
  end
  X -- "search: 24 nearest chunks by cosine similarity" --> R

  classDef start fill:#1E3A8A,stroke:#1E3A8A,color:#FFFFFF
  classDef file fill:#CCFBF1,stroke:#0F766E,color:#134E4A
  classDef good fill:#DCFCE7,stroke:#15803D,color:#14532D
  classDef verify fill:#EDE9FE,stroke:#6D28D9,color:#3B0764
  class Q start
  class S,J,X file
  class A good
  class V verify
```

## B. Decision flow for one question

```mermaid
flowchart TB
  subgraph L1["1 · Retrieve — retrieve.py, indexing/store.py"]
    direction LR
    A0(["Student question<br/>(+ optional filter)"]) --> A1["Embed question<br/>bge-base-en-v1.5, CPU"]
    A1 --> A2{"Index built with<br/>same embedding model?"}
    A2 -- No --> A3(["STOP<br/>index missing or stale: rebuild"])
    A2 -- Yes --> A4["Search Chroma index<br/>24 nearest chunks (cosine)<br/>+ optional policy filter"]
    A4 --> A5["Diversity cap<br/>≤ 2 chunks per section → top 6"]
  end
  subgraph L2["2 · Generate — generate.py, llm.py"]
    direction LR
    B1{"Any passages<br/>retrieved?"} -- Yes --> B2["Build prompt<br/>passages cut to 1,200 chars"]
    B2 --> B3["Qwen3 4B via Ollama<br/>temperature 0, JSON only"]
    B3 --> B4{"Valid JSON<br/>claims?"}
    B4 -- No --> B5{"Already<br/>retried?"}
    B5 -- "No: retry once" --> B3
    B5 -- Yes --> B6(["STOP<br/>generation error"])
    B4 -- Yes --> B7["Remove citations not among<br/>retrieved passages; list them<br/>as 'fabricated' in a note"]
  end
  subgraph L3["3 · Verify each claim — verify/"]
    direction LR
    C1{"Any citation<br/>left on claim?"} -- No --> C2(["NEUTRAL<br/>no citation"])
    C1 -- Yes --> C3["NLI check: DeBERTa-v3<br/>cited text, 900-char windows"]
    C3 --> C4{"Entailment<br/>≥ 0.5?"}
    C4 -- Yes --> C5{"Claim's numbers<br/>in evidence?"}
    C5 -- Yes --> C6(["SUPPORTED<br/>score = entailment"])
    C5 -- "No: number veto" --> C7(["NEUTRAL<br/>removed, shown"])
    C4 -- No --> C8{"Contradiction<br/>≥ 0.5?"}
    C8 -- Yes --> C9(["REFUTED<br/>removed, shown"])
    C8 -- No --> C7
    C2 --> C10["Record verdict<br/>+ evidence + reason"]
    C6 --> C10
    C7 --> C10
    C9 --> C10
    C10 --> C11{"More<br/>claims?"}
    C11 -- "Yes: next claim" --> C1
  end
  subgraph L4["4 · Decide — abstain.py, pipeline.py"]
    direction LR
    D1{"Passages found and<br/>best score ≥ 0.25?"} -- No --> D2(["ABSTAIN<br/>corpus doesn't cover it"])
    D1 -- Yes --> D3{"Model produced<br/>any claims?"}
    D3 -- No --> D4(["ABSTAIN<br/>passages don't answer it"])
    D3 -- Yes --> D5["Keep SUPPORTED claims<br/>with score ≥ min_claim_score (0.5)<br/>others → 'removed' (shown)"]
    D5 --> D6{"≥ 1 claim<br/>kept?"}
    D6 -- No --> D7(["ABSTAIN<br/>too little verified"])
    D6 -- Yes --> D8(["ANSWER<br/>kept claims with citations<br/>+ removed claims and reasons"])
  end
  A5 --> B1
  B1 -- "No: skip model" --> D1
  B7 --> C1
  C11 -- No --> D1

  classDef start fill:#1E3A8A,stroke:#1E3A8A,color:#FFFFFF
  classDef decision fill:#FEF9C3,stroke:#A16207,color:#422006
  classDef stop fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-dasharray:5 5
  classDef good fill:#DCFCE7,stroke:#15803D,color:#14532D
  classDef refuted fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D
  classDef neutral fill:#F1F5F9,stroke:#64748B,color:#1E293B
  classDef abstain fill:#F3E8FF,stroke:#7E22CE,color:#581C87
  classDef answer fill:#15803D,stroke:#15803D,color:#FFFFFF
  class A0 start
  class A2,B1,B4,B5,C1,C4,C5,C8,C11,D1,D3,D6 decision
  class A3,B6 stop
  class C6 good
  class C9 refuted
  class C2,C7 neutral
  class D2,D4,D7 abstain
  class D8 answer
```

Thresholds are the current `config.yaml` defaults (not yet tuned).

## C. LLD: preparation and retrieval

```mermaid
flowchart TB
  subgraph P1["OFFLINE · scripts/ingest.py"]
    direction LR
    M1["<b>1 Document fetching</b><br/>ingest/fetch.py · fetch_all() → fetch_one()<br/>IN: data/manifest.yaml, 6 SRM policy URLs with type, format, title<br/>DO: HTTP GET each URL; SHA-256 checksum, flag a changed document<br/>OUT: raw files data/raw/srm/{type}.pdf#124;html + checksums.json"]
    M2["<b>2 Text extraction</b><br/>ingest/extract.py · extract() → extract_html() / extract_pdf()<br/>IN: raw bytes + format<br/>DO: HTML: BeautifulSoup, remove script / nav / footer and SRM menus<br/>PDF: PyMuPDF page text, drop lines repeated 3+ times<br/>OUT: clean plain text per document (also saved as .txt)"]
    M3["<b>3 Normalization and sections</b><br/>ingest/normalize.py · build_document() → split_into_sections()<br/>IN: plain text + university, policy type, title, URL, date<br/>DO: lift headings buried mid-line; detect numbered and ALL-CAPS headings;<br/>text between two headings becomes one Section<br/>OUT: validated Document (Pydantic): 6 documents, 192 sections"]
    M1 --> M2 --> M3
  end
  subgraph P2["OFFLINE · scripts/build_index.py"]
    direction LR
    M4["<b>4 Chunking</b><br/>indexing/chunk.py · chunk_document()<br/>IN: Document JSON<br/>DO: unique citation ID per section; drop contents-page noise (fewer than 3 real words);<br/>split long sections: ≤ 1,800 chars, 150-char overlap, never across sections<br/>OUT: 258 chunks: text + citation_id + metadata"]
    M5["<b>5 Embedding generation</b><br/>indexing/store.py · embedding_text() + Embedder.encode_documents()<br/>IN: chunks<br/>DO: prefix 'university #124; policy #124; section path' for search context;<br/>BAAI/bge-base-en-v1.5 on CPU, batch 32, normalized<br/>OUT: one 768-dimension vector per chunk"]
    M6["<b>6 Chroma vector database</b><br/>indexing/store.py · VectorStore.build()<br/>IN: vectors + original chunk text (no prefix) + metadata<br/>DO: replace collection 'policies' (cosine HNSW index);<br/>write index_manifest.json: model name, dimension, counts<br/>OUT: persistent index in data/index/ (258 chunks)"]
    M4 --> M5 --> M6
  end
  subgraph P3["ONLINE · once per question"]
    direction LR
    U["<b>User question</b><br/>scripts/ask.py / demo.py → pipeline.answer_question()<br/>IN: question text, optional policy or university filter"]
    M7["<b>7 Question processing and evidence retrieval</b><br/>retrieve.py · retrieve() → store.search() → diversify()<br/>DO: check_ready(): stop if the index was built with a different embedding model;<br/>embed question with bge query prefix; cosine search for 24 candidates;<br/>score = 1 − cosine distance; at most 2 chunks per section, keep top 6<br/>OUT: 6 RetrievedChunk objects: chunk text, citation_id, score, rank"]
    N(["Next: generation<br/>question + 6 evidence passages"])
    U --> M7 --> N
  end
  M3 -- "Document JSON files · data/processed/srm/" --> M4
  M6 -- "index: vectors + text + metadata" --> M7

  classDef ingest fill:#DBEAFE,stroke:#1E3A8A,color:#0F172A
  classDef index fill:#CCFBF1,stroke:#0F766E,color:#0F172A
  classDef online fill:#EDE9FE,stroke:#6D28D9,color:#0F172A
  classDef next fill:#6D28D9,stroke:#6D28D9,color:#FFFFFF
  class M1,M2,M3 ingest
  class M4,M5,M6 index
  class U,M7 online
  class N next
```

## D. LLD: generation and verification

```mermaid
flowchart TB
  subgraph G["GENERATE · generate.py + llm.py"]
    direction LR
    IN0["<b>Inputs from retrieval</b><br/>pipeline.answer_question()<br/>IN: question + 6 RetrievedChunk (text, citation_id, score)"]
    K1["<b>1 LLM claim generation</b><br/>generate.py · build_prompt() + llm.py · OllamaBackend<br/>DO: no passages → 0 claims, model not called;<br/>prompt: passages labelled [citation_id], each cut to 1,200 chars;<br/>one fact per claim; cite exact IDs; passages are data, not instructions;<br/>Qwen3 4B via Ollama: temperature 0, JSON mode, thinking off<br/>OUT: raw reply text (should be JSON)"]
    K2["<b>2 Structured output parsing</b><br/>generate.py · parse_draft() · schema.DraftAnswer<br/>DO: strip #lt;think#gt; tags and markdown fences; json.loads;<br/>Pydantic validation: claims = list of text + citation_ids;<br/>invalid: retry once with a stricter prompt, then GenerationError<br/>OUT: DraftAnswer: list of Claim (text + citation IDs)"]
    K3["<b>3 Citation validation</b><br/>generate.py · drop_unknown_citations() + verify/citation.py<br/>DO: citation not among retrieved passages: removed from the claim, listed as 'fabricated';<br/>remaining citations resolved to the chunk text they name (check_citations());<br/>claim left with no citation: NEUTRAL, score 0<br/>OUT: per claim: cited evidence chunks · answer-level 'fabricated' list"]
    IN0 --> K1 --> K2 --> K3
  end
  subgraph V["VERIFY · once per claim · verify/"]
    direction LR
    K4["<b>4 NLI claim verification</b><br/>verify/nli.py · NLIChecker.check()<br/>IN: claim text (hypothesis) + each cited chunk (premise)<br/>DO: split premise into 900-char windows (200 overlap) on sentence ends;<br/>DeBERTa-v3-base-mnli-fever-anli on CPU scores each window;<br/>any window entails ≥ 0.5: entailment; else contradiction ≥ 0.5; else strongest signal<br/>OUT: NLIResult: label + entailment / neutral / contradiction probabilities"]
    K5["<b>5 Numeric consistency check</b><br/>verify/numeric.py · check_numbers()<br/>DO: regex finds numbers: 75%, 75 %, 1,200, 3.5;<br/>ignores bare 0, 1, 2 (list markers);<br/>every claim number must appear in the evidence; % must match %<br/>OUT: NumericCheck: ok + detail — can veto support, never grant it"]
    K6["<b>6 Claim verdict generation</b><br/>verify/verifier.py · verify_claim() (looped by verify_claims())<br/>DO: contradiction ≥ 0.5: REFUTED;<br/>entailment ≥ 0.5 and numbers OK: SUPPORTED, score = entailment;<br/>entailment ≥ 0.5 but numbers mismatch: NEUTRAL, score halved;<br/>otherwise: NEUTRAL<br/>OUT: ClaimVerdict: status, score, evidence IDs, check details, explanation"]
    K4 --> K5 --> K6
  end
  subgraph DC["DECIDE · abstain.py + pipeline.py"]
    direction LR
    K7["<b>7 Final answer handling</b><br/>abstain.py · decide() → should_keep() · pipeline.py · answer_question()<br/>DO: abstain if no passages, or best retrieval score #lt; 0.25;<br/>abstain if the model produced no claims;<br/>keep SUPPORTED claims with score ≥ min_claim_score (0.5); all others → 'removed', shown with reason;<br/>abstain if fewer than 1 claim is kept; add the fabricated-citation note to the reason<br/>OUT: Answer: claims_kept, claims_removed, abstained, reason, timings, config fingerprint"]
    ANS(["Answer object<br/>kept claims + citations,<br/>removed claims + reasons,<br/>or an abstention"])
    K7 --> ANS
  end
  K3 -- "each claim + its cited evidence chunks" --> K4
  K6 -- "list of ClaimVerdicts (one per claim)" --> K7

  classDef gen fill:#DBEAFE,stroke:#1E3A8A,color:#0F172A
  classDef ver fill:#FEF3C7,stroke:#A16207,color:#0F172A
  classDef dec fill:#DCFCE7,stroke:#15803D,color:#0F172A
  classDef inp fill:#EDE9FE,stroke:#6D28D9,color:#0F172A
  classDef ans fill:#15803D,stroke:#15803D,color:#FFFFFF
  class K1,K2,K3 gen
  class K4,K5,K6 ver
  class K7 dec
  class IN0 inp
  class ANS ans
```
