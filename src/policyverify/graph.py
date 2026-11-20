"""
graph.py - a graph of policy sections, for finding related rules.

WHAT IT IS
    Nodes are citable sections. Two kinds of edge connect them:

      refers_to       a section names another one explicitly ("as indicated in
                      the table under section R8.1"). Found by pattern, and
                      only kept when the named number is a real section of
                      the same document.
      shares_entity   two sections are both about a distinctive group of
                      people (parents, visitors, wardens, day scholars ...),
                      using the entity tags and corpus weights from
                      entities.py. Weighted by the rarest shared entity, so
                      "student" - in most sections - links almost nothing,
                      and each section keeps only its strongest few links.

WHAT IT IS FOR
    "Related sections" beside an answer: someone asking whether parents can
    stay over is also shown the visitors rules and the admission rule that a
    parent must accompany a new resident. It is a reading aid. Answers still
    come only from retrieval and verification; nothing here is evidence.

WHAT THE DATA SAYS ABOUT IT (measured 2026-11-20)
    Only 2 explicit cross-references exist in the SRM body text
    (8.2 -> 8.1, 4.2 -> 2.5). Twenty-two more "references" turned out to be
    sub-section headings that extraction merged into the section before
    them (R 3.6 inside 3.5-summer_semester), so they are not edges. Most of
    the graph's links are therefore entity links. That is why it is a
    display feature and not a retrieval step: on this corpus it would add
    little to search, which already finds the expected section for 48 of 51
    answerable evaluation questions.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from policyverify.schema import Chunk

GRAPH_NAME = "section_graph.json"

# "R 8.1", "R8.1", "section 7.3", "section R 5.1"
_REF_RE = re.compile(r"\b(?:R\s?(\d+\.\d+)|[Ss]ection\s+R?\s?(\d+\.\d+))\b")
# A section citing more numbers than this is a contents page, not a rule.
_CONTENTS_PAGE_REFS = 15
# An entity link needs at least one shared entity this distinctive (its
# corpus weight, 0..1 - see entities.entity_weights).
MIN_ENTITY_WEIGHT = 0.5
# Strongest entity links kept per section, so a hub does not link everything.
MAX_ENTITY_LINKS = 4


@dataclass
class SectionGraph:
    titles: dict[str, str] = field(default_factory=dict)
    # citation -> {neighbour: (weight, kind, why)}
    edges: dict[str, dict[str, tuple[float, str, str]]] = field(default_factory=dict)

    def related(self, citation: str, k: int = 3, exclude: set[str] | None = None) -> list[dict]:
        """The `k` strongest neighbours of a section, best first."""
        exclude = exclude or set()
        out = [
            {"citation": n, "title": self.titles.get(n, n), "weight": w, "kind": kind, "why": why}
            for n, (w, kind, why) in self.edges.get(citation, {}).items()
            if n not in exclude
        ]
        # Explicit references first, then by weight, then by id for stability.
        out.sort(key=lambda e: (e["kind"] != "refers_to", -e["weight"], e["citation"]))
        return out[:k]

    def to_dict(self) -> dict:
        return {
            "titles": self.titles,
            "edges": {
                a: {b: list(v) for b, v in nbrs.items()} for a, nbrs in self.edges.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> SectionGraph:
        return cls(
            titles=data.get("titles", {}),
            edges={
                a: {b: tuple(v) for b, v in nbrs.items()}
                for a, nbrs in data.get("edges", {}).items()
            },
        )

    def save(self, folder: Path) -> Path:
        path = Path(folder) / GRAPH_NAME
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, folder: Path) -> SectionGraph | None:
        path = Path(folder) / GRAPH_NAME
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _section_number(citation: str) -> str | None:
    """'srm/attendance/8.1-course_wise_grading' -> '8.1'."""
    slug = citation.rsplit("/", 1)[-1]
    match = re.match(r"(\d+(?:\.\d+)*)-", slug)
    return match.group(1) if match else None


def build_section_graph(chunks: list[Chunk], weights: dict[str, float]) -> SectionGraph:
    """Build the graph from the indexed chunks and the entity weights."""
    graph = SectionGraph()
    text: dict[str, list[str]] = defaultdict(list)
    entities: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        graph.titles.setdefault(
            chunk.citation_id, chunk.section_path.split(" > ")[-1] or chunk.citation_id
        )
        text[chunk.citation_id].append(chunk.text)
        entities[chunk.citation_id].update(chunk.entities)

    def link(a: str, b: str, weight: float, kind: str, why: str) -> None:
        for x, y in ((a, b), (b, a)):
            current = graph.edges.setdefault(x, {}).get(y)
            if current is None or (current[1] != "refers_to" and weight > current[0]):
                graph.edges[x][y] = (round(weight, 4), kind, why)

    # --- explicit references --------------------------------------------
    by_number: dict[tuple[str, str], str] = {}
    for citation in graph.titles:
        number = _section_number(citation)
        if number:
            by_number[(citation.rsplit("/", 1)[0], number)] = citation
    for citation, pieces in text.items():
        refs = [m.group(1) or m.group(2) for m in _REF_RE.finditer("\n".join(pieces))]
        if len(refs) > _CONTENTS_PAGE_REFS:
            continue
        document = citation.rsplit("/", 1)[0]
        for ref in refs:
            target = by_number.get((document, ref))
            if target and target != citation and ref != _section_number(citation):
                link(citation, target, 1.0, "refers_to", f"mentions R{ref}")

    # --- shared distinctive entities ------------------------------------
    candidates: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    cites = sorted(entities)
    for i, a in enumerate(cites):
        for b in cites[i + 1:]:
            shared = entities[a] & entities[b]
            if not shared:
                continue
            best = max(shared, key=lambda e: weights.get(e, 0.0))
            w = weights.get(best, 0.0)
            if w < MIN_ENTITY_WEIGHT:
                continue
            candidates[a].append((w, b, best))
            candidates[b].append((w, a, best))
    for a, options in candidates.items():
        options.sort(key=lambda o: (-o[0], o[1]))
        for w, b, entity in options[:MAX_ENTITY_LINKS]:
            link(a, b, w, "shares_entity", f"both concern {entity.replace('_', ' ')}")
    return graph


__all__ = ["GRAPH_NAME", "SectionGraph", "build_section_graph"]
