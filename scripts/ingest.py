"""
Run the full Phase 1 ingestion pipeline: manifest -> raw files -> Document
objects, saved to data/processed/ for Phase 2 (chunking) to pick up.

Usage:
    python scripts/ingest.py

Downloads every document listed in data/manifest.yaml, extracts clean text,
splits it into sections, validates the result against the Document schema,
and writes both a .json (the structured Document) and a .txt (the plain
extracted text, for a human to skim) per document under data/processed/.

After running this, read at least 3 of the .txt files by hand - that is the
actual Phase 1 "done" bar, not just "the script exited 0".
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.config import get_config  # noqa: E402
from policyverify.ingest import build_document, extract, fetch_all, load_manifest  # noqa: E402
from policyverify.schema import PolicyType, SourceFormat  # noqa: E402


def main() -> int:
    config = get_config()
    manifest = load_manifest()
    processed_dir = config.paths.resolve("processed")
    raw_dir = config.paths.resolve("raw")

    print("=== Phase 1: Ingestion ===\n")

    print("[1/2] fetching documents from manifest ...")
    fetch_results = fetch_all(manifest)
    for r in fetch_results:
        flag = "  <-- CHANGED since last fetch!" if r.changed_since_last_fetch else ""
        print(f"  {r.university}/{r.policy_type:<20} {r.byte_count:>8,} bytes{flag}")

    print("\n[2/2] extracting + normalizing ...")
    documents = []
    for uni in manifest.get("universities", []):
        for policy in uni.get("policies", []):
            ext = "pdf" if policy["format"] == "pdf" else "html"
            raw_path = raw_dir / uni["university"] / f"{policy['type']}.{ext}"
            raw_bytes = raw_path.read_bytes()
            text = extract(raw_bytes, policy["format"])

            doc = build_document(
                university=uni["university"],
                university_name=uni["name"],
                policy_type=PolicyType(policy["type"]),
                title=policy["title"],
                source_url=policy["url"],
                source_format=SourceFormat(policy["format"]),
                retrieved_at=date.fromisoformat(policy["retrieved_at"]),
                raw_bytes=raw_bytes,
                text=text,
            )
            documents.append(doc)

            out_dir = processed_dir / uni["university"]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{policy['type']}.json").write_text(
                doc.model_dump_json(indent=2), encoding="utf-8"
            )
            (out_dir / f"{policy['type']}.txt").write_text(text, encoding="utf-8")

            print(f"  {doc.doc_id:<30} {len(doc.sections):>3} sections  ({len(text):,} chars)")

    total_sections = sum(len(d.sections) for d in documents)
    print(f"\nDone - {len(documents)} documents, {total_sections} sections total.")
    print(f"Saved to {processed_dir}")
    print("\nNow read at least 3 of the .txt files by hand before moving on to Phase 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
