"""
fetch.py - download the documents listed in data/manifest.yaml.

Reads the manifest, downloads each policy document, and saves the raw bytes
under data/raw/{university}/{policy_type}.{ext}. Nothing here parses or
cleans the content - that is extract.py's job. This stage only answers one
question: "did we get the bytes we expected, and are they the same bytes as
last time?"

Why checksums live in a separate file instead of being written back into
manifest.yaml: manifest.yaml is a hand-written, heavily-commented file that
explains WHY each source was chosen. Round-tripping it through a YAML
library would silently strip every comment. checksums.json is a generated,
disposable record instead - gitignored, rebuilt on every fetch - and it is
also what lets a re-fetch notice "the university silently changed this
document," which a one-shot download could never tell you.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests
import yaml

from policyverify.config import get_config

TIMEOUT = 30
HEADERS = {
    # Some university sites block requests that don't look like a browser.
    "User-Agent": "Mozilla/5.0 (compatible; PolicyVerifyIngest/0.1; academic use)"
}


@dataclass
class FetchResult:
    """What happened when we downloaded one document."""

    university: str
    policy_type: str
    title: str
    url: str
    format: str
    path: Path
    checksum: str
    byte_count: int
    changed_since_last_fetch: bool


def load_manifest(path: Path | None = None) -> dict:
    """Read data/manifest.yaml as a plain dict."""
    config = get_config()
    manifest_path = path or config.paths.resolve("manifest")
    with open(manifest_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_checksums(checksums_path: Path) -> dict:
    if checksums_path.exists():
        return json.loads(checksums_path.read_text(encoding="utf-8"))
    return {}


def _save_checksums(checksums_path: Path, data: dict) -> None:
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def fetch_one(
    university: str, policy: dict, raw_dir: Path, checksums: dict
) -> FetchResult:
    """Download one document and save it under data/raw/{university}/{type}.{ext}."""
    url = policy["url"]
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    raw = resp.content

    checksum = hashlib.sha256(raw).hexdigest()
    doc_id = f"{university}/{policy['type']}"
    previous = checksums.get(doc_id, {}).get("checksum")
    changed = previous is not None and previous != checksum

    ext = "pdf" if policy["format"] == "pdf" else "html"
    out_dir = raw_dir / university
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{policy['type']}.{ext}"
    out_path.write_bytes(raw)

    checksums[doc_id] = {
        "checksum": checksum,
        "byte_count": len(raw),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": url,
    }

    return FetchResult(
        university=university,
        policy_type=policy["type"],
        title=policy["title"],
        url=url,
        format=policy["format"],
        path=out_path,
        checksum=checksum,
        byte_count=len(raw),
        changed_since_last_fetch=changed,
    )


def fetch_all(manifest: dict | None = None) -> list[FetchResult]:
    """Download every document in the manifest. One FetchResult per document."""
    config = get_config()
    manifest = manifest or load_manifest()
    raw_dir = config.paths.resolve("raw")
    checksums_path = raw_dir / "checksums.json"
    checksums = _load_checksums(checksums_path)

    results = []
    for uni in manifest.get("universities", []):
        for policy in uni.get("policies", []):
            results.append(fetch_one(uni["university"], policy, raw_dir, checksums))

    _save_checksums(checksums_path, checksums)
    return results
