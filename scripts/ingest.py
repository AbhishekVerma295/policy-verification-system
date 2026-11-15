"""
Run the full Phase 1 ingestion pipeline: manifest -> raw files -> Document
objects, saved to data/processed/ for Phase 2 (chunking) to pick up.

Usage:
    python scripts/ingest.py            # download, then process
    python scripts/ingest.py --offline  # re-process the files already in data/raw/

--offline exists for changes to extraction or sectioning: it rebuilds
data/processed/ from the exact bytes that were downloaded before, so any
difference in the output is caused by the code change and not by the
university having edited a page in the meantime. Each raw file is checked
against data/raw/checksums.json first, and the run stops if one is missing
or does not match.

Downloads every document listed in data/manifest.yaml, extracts clean text,
splits it into sections, validates the result against the Document schema,
and writes both a .json (the structured Document) and a .txt (the plain
extracted text, for a human to skim) per document under data/processed/.

After running this, read at least 3 of the .txt files by hand - that is the
actual Phase 1 "done" bar, not just "the script exited 0".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.config import get_config  # noqa: E402
from policyverify.ingest import build_document, extract, fetch_all, load_manifest  # noqa: E402
from policyverify.schema import PolicyType, SourceFormat  # noqa: E402


def check_raw_files(manifest: dict, raw_dir: Path) -> list[str]:
    """Problems with the already-downloaded files, for --offline. [] means all good."""
    checksums_path = raw_dir / "checksums.json"
    if not checksums_path.exists():
        return [f"{checksums_path} not found - run without --offline once"]
    recorded = json.loads(checksums_path.read_text(encoding="utf-8"))
    problems = []
    for uni in manifest.get("universities", []):
        for policy in uni.get("policies", []):
            doc_id = f"{uni['university']}/{policy['type']}"
            ext = "pdf" if policy["format"] == "pdf" else "html"
            path = raw_dir / uni["university"] / f"{policy['type']}.{ext}"
            if not path.exists():
                problems.append(f"{doc_id}: {path} is missing")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = recorded.get(doc_id, {}).get("checksum")
            if actual != expected:
                problems.append(f"{doc_id}: checksum {actual[:12]} does not match recorded {str(expected)[:12]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the policy documents.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not download; re-process the files already in data/raw/",
    )
    args = parser.parse_args()

    config = get_config()
    manifest = load_manifest()
    processed_dir = config.paths.resolve("processed")
    raw_dir = config.paths.resolve("raw")

    print("=== Phase 1: Ingestion ===\n")

    if args.offline:
        print("[1/2] --offline: checking the files already in data/raw/ ...")
        problems = check_raw_files(manifest, raw_dir)
        if problems:
            for p in problems:
                print(f"  {p}")
            print("\nStopping: --offline only re-processes files matching checksums.json.")
            return 1
        print("  all raw files present and matching their recorded checksums")
    else:
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
