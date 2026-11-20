# Preparation-phase LLD: Mermaid code

Three diagrams matching `Prep_LLD_A/B/C` (current code). To use one:

- **draw.io / diagrams.net:** *Arrange → Insert → Advanced → Mermaid*, paste the code (everything between the ``` lines, without them), then *Insert*.
- **mermaid.live:** paste the code into the editor; export PNG/SVG from the *Actions* menu.

`#35;`, `#124;`, `#lt;` and `#gt;` are Mermaid codes for `#`, `|`, `<` and `>`.

## A. Corpus preparation (fetch.py, extract.py, scripts/ingest.py)

```mermaid
flowchart TB
  subgraph L1["1 · Download — fetch.py, scripts/ingest.py"]
    A0(["python scripts/ingest.py"]) --> A1[("data/manifest.yaml<br/>6 policies: URL, type, format, title, date")]
    A1 --> A2["load_manifest()<br/>YAML file to a Python dict"]
    A2 --> A3["fetch_all()<br/>reads checksums.json, then calls<br/>fetch_one() for each of the 6 policies"]
    A3 --> A4["fetch_one(): HTTP GET<br/>requests, 30 s timeout,<br/>browser-like User-Agent"]
  end
  subgraph L2["2 · Check and save — fetch.py"]
    B1{"HTTP status<br/>OK (2xx)?"} -- No --> B2(["STOP: error raised, run ends"])
    B1 -- Yes --> B3["SHA-256 of the bytes<br/>fingerprint of the exact file"]
    B3 --> B4{"Differs from a<br/>recorded checksum?"}
    B4 -- Yes --> B5(["Report flags it:<br/>CHANGED since last fetch"])
    B4 -- No --> B6[("data/raw/srm/{type}.pdf or .html<br/>raw bytes, as downloaded")]
    B5 --> B6
    B6 --> B7[("data/raw/checksums.json<br/>updated per policy, written after all 6")]
    B7 -.-> B8(["FetchResult per policy:<br/>size + changed flag, printed"])
  end
  subgraph L3["3 · Extract text — extract.py (after all 6 downloads finish)"]
    C1["scripts/ingest.py reads<br/>raw bytes + format from manifest"] --> C2{"File format?"}
    C2 -- html --> C3["extract_html()<br/>1. Parse with BeautifulSoup (lxml)<br/>2. Remove script, style, nav, header, footer, form, noscript<br/>3. Remove SRM page-builder blocks: icon lists, tabs, breadcrumbs<br/>4. Keep #lt;main#gt;, else #lt;article#gt;, else #lt;body#gt;<br/>5. Visible text, one block per line"]
    C2 -- pdf --> C4["extract_pdf()<br/>1. PyMuPDF: text page by page, in order<br/>2. Remove lines of ≤ 100 chars that appear 3+ times<br/>(running page headers and footers)"]
    C2 -- other --> C5(["STOP: unknown format, ValueError"])
    C3 --> C6["_clean_text()<br/>collapse blank-line runs, trim line ends,<br/>drop empty lines"]
    C4 --> C6
    C6 --> C7(["Plain text per document → Diagram B"])
    C6 -.-> C8[("data/processed/srm/{type}.txt<br/>copy for human review")]
  end
  A4 --> B1
  B7 --> C1

  classDef start fill:#1E3A8A,stroke:#1E3A8A,color:#FFFFFF
  classDef file fill:#CCFBF1,stroke:#0F766E,color:#134E4A
  classDef stop fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-dasharray:5 5
  classDef note fill:#F1F5F9,stroke:#64748B,color:#1E293B
  classDef good fill:#DCFCE7,stroke:#15803D,color:#14532D
  classDef decision fill:#FEF9C3,stroke:#A16207,color:#422006
  class A0 start
  class A1,B6,B7,C8 file
  class B2,C5 stop
  class B5,B8 note
  class C7 good
  class B1,B4,C2 decision
```

## B. Document processing (normalize.py, schema.py)

```mermaid
flowchart TB
  subgraph L1["1 · Find headings — normalize.py"]
    D0(["Plain text from Diagram A"]) --> D1["promote_inline_headings()<br/>a numbered heading found mid-line<br/>is moved onto its own line"]
    D1 --> D2["Check each non-empty line"]
    D2 --> D3{"Numbered heading?<br/>starts 4 / 4.2 / A.1 + text;<br/>text ≤ 100 chars, no final full stop;<br/>≥ half its words capitalised"}
    D3 -- Yes --> D4(["Record heading with its number"])
    D3 -- No --> D5{"ALL-CAPS heading?<br/>starts with a capital, 4–92 chars;<br/>≥ 4 letters, all uppercase"}
    D5 -- Yes --> D6(["Record heading:<br/>no number, Title-cased, colon removed"])
    D5 -- No --> D7(["Keep as body text"])
  end
  subgraph L2["2 · Cut into sections — normalize.py"]
    E1{"At least 2<br/>headings found?"} -- No --> E2["Whole document = 1 Section<br/>heading = title, path [title], level 1"]
    E1 -- Yes --> E3["For each heading, take the<br/>text up to the next heading"]
    E3 --> E4{"Text under it empty?"}
    E4 -- Yes --> E5(["Skip this heading"])
    E4 -- No --> E6["Section object<br/>number, heading, text,<br/>path = [title, heading], level 2"]
  end
  subgraph L3["3 · Build Document — normalize.py, schema.py"]
    F1["build_document()<br/>adds manifest metadata: university, name,<br/>policy type, title, URL, format, retrieved_at"]
    F1 --> F2["SHA-256 of raw bytes<br/>doc_id = srm/{type}"]
    F2 --> F3{"Pydantic validation OK?<br/>university slug, non-empty section text,<br/>policy type + format from fixed lists, valid date"}
    F3 -- No --> F4(["STOP: invalid field, ValidationError"])
    F3 -- Yes --> F5[("data/processed/srm/{type}.json<br/>6 documents, 192 sections")]
    F5 --> F6(["Validated Documents → Diagram C"])
  end
  subgraph L4["4 · Worked example — attendance PDF"]
    X1["Extracted PDF line<br/>R 7.3 Minimum Attendance: A student must strive to …"]
    X1 --> X2["After promote_inline_headings()<br/>R / 7.3 Minimum Attendance: / A student must strive to …"]
    X2 --> X3["Line classified<br/>numbered heading: number 7.3,<br/>heading 'Minimum Attendance:'"]
    X3 --> X4["Section saved<br/>number 7.3, heading 'Minimum Attendance:',<br/>path [Academic Regulations 2021 …, Minimum Attendance:]"]
  end
  D4 --> E1
  D6 --> E1
  E2 --> F1
  E6 --> F1

  classDef start fill:#1E3A8A,stroke:#1E3A8A,color:#FFFFFF
  classDef file fill:#CCFBF1,stroke:#0F766E,color:#134E4A
  classDef stop fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-dasharray:5 5
  classDef note fill:#F1F5F9,stroke:#64748B,color:#1E293B
  classDef good fill:#DCFCE7,stroke:#15803D,color:#14532D
  classDef decision fill:#FEF9C3,stroke:#A16207,color:#422006
  classDef example fill:#EDE9FE,stroke:#6D28D9,color:#3B0764
  class D0 start
  class F5 file
  class F4 stop
  class D7,E5 note
  class D4,D6,F6 good
  class D3,D5,E1,E4,F3 decision
  class X1,X2,X3,X4 example
```

## C. Chunking and indexing (chunk.py, store.py, scripts/build_index.py)

```mermaid
flowchart TB
  subgraph L1["1 · Load and cite — scripts/build_index.py, schema.py"]
    G0(["python scripts/build_index.py"]) --> G1[("data/processed/srm/*.json<br/>6 Document files")]
    G1 --> G2["load_documents()<br/>re-validates each file against the schema"]
    G2 --> G3["Document.citations()<br/>key = '{number}-{heading slug}'<br/>slug: lowercase, other chars → _, max 40;<br/>no number → slug only"]
    G3 --> G4{"Key already used<br/>in this document?"}
    G4 -- Yes --> G5["Append -2, -3, …<br/>e.g. consequence-2"]
    G4 -- No --> G6(["Citation ID<br/>srm/attendance/7.3-minimum_attendance"])
    G5 --> G6
  end
  subgraph L2["2 · Chunk — chunk.py"]
    H1["For each section<br/>with its citation ID"] --> H2{"Fewer than 3 real words?<br/>(real word = 2+ letters)"}
    H2 -- Yes --> H3(["Drop: contents-page entry<br/>e.g. 'Payment Of Fine' → 12"])
    H2 -- No --> H4{"≤ 1,800 characters?"}
    H4 -- Yes --> H5["One chunk = the whole section"]
    H4 -- No --> H6["split_long_text()<br/>1. Split at blank lines, then sentence ends;<br/>hard-cut any piece still over 1,800 chars<br/>2. Pack pieces into windows of ≤ 1,800 chars<br/>3. Start each window with the last ~150 chars<br/>of the one before (from a word start)"]
    H5 --> H7["Chunk object<br/>chunk_id = '{citation ID}#35;0', '#35;1', …<br/>text, citation_id, university, name,<br/>policy type, section path, URL<br/>(258 chunks from 192 sections)"]
    H6 --> H7
  end
  subgraph L3["3 · Embed — store.py"]
    I1{"Any chunks at all?"} -- No --> I2(["STOP: refuse to build from zero"])
    I1 -- Yes --> I3["Delete the old 'policies' collection<br/>(full rebuild, no leftovers)"]
    I3 --> I4["embedding_text()<br/>'{university name} #124; {policy type} #124; {section path}'<br/>+ new line + chunk text (prefix used for search only)"]
    I4 --> I5["Embedder.encode_documents()<br/>BAAI/bge-base-en-v1.5 on CPU, batch 32,<br/>normalised → one 768-number vector per chunk"]
  end
  subgraph L4["4 · Store and check — store.py, scripts/build_index.py"]
    J1["collection.add()<br/>id = chunk_id; document = ORIGINAL chunk text;<br/>embedding = vector; cosine-similarity HNSW index"]
    J1 -.-> J2(["metadata: chunk_id, citation_id, university,<br/>university_name, policy_type, section_path, source_url"])
    J1 --> J3[("data/index/<br/>Chroma database, 258 chunks, 768 dims")]
    J3 --> J4[("index_manifest.json<br/>model name, dims, chunk + doc counts,<br/>built_at, chunking (checked before every search)")]
    J4 --> J5{"--no-test flag given?"}
    J5 -- No --> J6["Smoke test<br/>3 sample questions, top 3 each"]
    J5 -- Yes --> J7(["Index ready<br/>searched at question time"])
    J6 --> J7
  end
  G6 --> H1
  H7 --> I1
  I5 --> J1

  classDef start fill:#1E3A8A,stroke:#1E3A8A,color:#FFFFFF
  classDef file fill:#CCFBF1,stroke:#0F766E,color:#134E4A
  classDef stop fill:#FEE2E2,stroke:#B91C1C,color:#7F1D1D,stroke-dasharray:5 5
  classDef note fill:#F1F5F9,stroke:#64748B,color:#1E293B
  classDef good fill:#DCFCE7,stroke:#15803D,color:#14532D
  classDef decision fill:#FEF9C3,stroke:#A16207,color:#422006
  class G0 start
  class G1,J3,J4 file
  class I2 stop
  class H3,J2 note
  class G6,J7 good
  class G4,H2,H4,I1,J5 decision
```
