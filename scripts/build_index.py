"""
Build the searchable index from the documents produced by Phase 1.

Usage:
    python scripts/build_index.py            # build, then run a smoke search
    python scripts/build_index.py --no-test  # build only

Reads data/processed/*/*.json (written by scripts/ingest.py), chunks each
document on its section boundaries, embeds the chunks, and stores them in
Chroma under data/index/.

Rebuilding is destructive by design: the old collection is dropped rather
than updated, so chunks whose sections no longer exist in the source cannot
linger in the index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.config import get_config  # noqa: E402
from policyverify.indexing import VectorStore, chunk_document  # noqa: E402
from policyverify.schema import Document  # noqa: E402

# Questions used only for the post-build smoke test. Chosen to cover
# different policy types so an obviously-broken index is visible immediately.
SMOKE_QUESTIONS = [
    "What is the minimum attendance requirement to sit the final examination?",
    "What happens if a student is caught plagiarising?",
    "Can guests stay overnight in the hostel?",
]


def load_documents(processed_dir: Path) -> list[Document]:
    """Read every Document JSON written by the ingestion step."""
    paths = sorted(processed_dir.glob("*/*.json"))
    if not paths:
        raise SystemExit(
            f"No processed documents found in {processed_dir}.\n"
            f"Run the ingestion step first:\n"
            f"    python scripts/ingest.py"
        )
    return [Document.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the policy vector index.")
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="skip the smoke search after building",
    )
    args = parser.parse_args()

    config = get_config()
    processed_dir = config.paths.resolve("processed")

    print("=== Phase 2: Chunking and indexing ===\n")

    print("[1/3] loading processed documents ...")
    documents = load_documents(processed_dir)
    print(f"  {len(documents)} documents")

    print("\n[2/3] chunking ...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        dropped = len(doc.sections) - len({c.citation_id for c in chunks})
        note = f"  ({dropped} noise sections dropped)" if dropped > 0 else ""
        print(
            f"  {doc.doc_id:<28} {len(doc.sections):>3} sections -> "
            f"{len(chunks):>4} chunks{note}"
        )
    print(f"  total: {len(all_chunks)} chunks")

    sizes = [len(c.text) for c in all_chunks]
    print(
        f"  chunk size: min={min(sizes)} median={sorted(sizes)[len(sizes) // 2]} "
        f"max={max(sizes)} chars"
    )

    print(f"\n[3/3] embedding with {config.embedding.model_name} on "
          f"{config.embedding.device} ...")
    print("  (first run downloads the model - this can take a few minutes)")
    store = VectorStore(config)
    manifest = store.build(all_chunks, document_count=len(documents))
    print(f"  embedded {manifest.chunk_count} chunks, {manifest.embedding_dim} dimensions")
    print(f"  index written to {store.index_dir}")
    print(f"  manifest: {store.manifest_path.name}")

    if not args.no_test:
        print("\n=== Smoke test ===")
        for question in SMOKE_QUESTIONS:
            print(f"\nQ: {question}")
            results = store.search(question, k=3)
            for r in results:
                preview = " ".join(r.chunk.text.split())[:95]
                print(f"  [{r.score:.3f}] {r.chunk.citation_id}")
                print(f"          {preview}...")

    print("\nDone. Next: read the smoke test results above and check they look sensible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
