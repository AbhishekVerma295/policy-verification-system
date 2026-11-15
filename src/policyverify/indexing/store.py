"""
store.py - the vector index: embedding chunks and searching them.

Wraps Chroma so the rest of the system never imports it directly. Two things
in here are worth understanding, because both are load-bearing.

1. THE INDEX MANIFEST
   Every build writes data/index/index_manifest.json recording which
   embedding model produced the vectors. Every search checks it against the
   current config and refuses to run if they disagree.

   This guard exists because the failure it prevents is silent. Embeddings
   from two different models are not comparable, but nothing about comparing
   them raises an error: the maths still works, the search still returns its
   top k, the app still looks fine. It just returns near-random passages.
   Without this check, "I switched the embedding model in config.yaml and
   forgot to rebuild" is a bug you would debug for a day. With it, you get a
   sentence telling you to rebuild.

2. WHAT GETS EMBEDDED IS NOT WHAT GETS VERIFIED
   The text we embed has the university, policy type and section heading
   prepended. The text we store as the chunk's content does not.

   Those serve different jobs. For SEARCH, a passage reading "Students must
   attend at least 75% of classes" is ambiguous on its own - the heading is
   what tells the embedding model this is the attendance policy rather than
   an exam rule, so including it makes retrieval markedly better. For
   VERIFICATION, the claim has to be checked against what the university
   actually wrote. Checking it against a string we assembled ourselves would
   mean partly verifying a claim against our own text, which is not
   verification at all.

3. THE KEYWORD INDEX RIDES ALONG
   Retrieval also runs a keyword (BM25) search - see indexing/bm25.py for why.
   That index is not stored on disk: it is built in memory from the chunks
   already in Chroma, the first time it is asked for, and kept on this object.
   A few hundred chunks take well under a second. Because it is derived from
   Chroma rather than saved separately, the two can never disagree about what
   is in the corpus - and `build()` throws the cached copy away so a rebuild
   in the same process cannot leave a stale keyword index behind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from policyverify.config import Config, get_config
from policyverify.indexing.bm25 import BM25Index
from policyverify.schema import Chunk, PolicyType, RetrievedChunk

COLLECTION_NAME = "policies"
MANIFEST_NAME = "index_manifest.json"


class IndexMismatchError(RuntimeError):
    """The index on disk was built by a different embedding model than the
    one currently configured. Comparing the two would silently return
    nonsense, so we refuse rather than guess."""


@dataclass
class IndexManifest:
    """A record of exactly what built the index sitting on disk."""

    embedding_model: str
    embedding_dim: int
    chunk_count: int
    document_count: int
    built_at: str
    chunking: dict

    def to_dict(self) -> dict:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "chunk_count": self.chunk_count,
            "document_count": self.document_count,
            "built_at": self.built_at,
            "chunking": self.chunking,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IndexManifest:
        return cls(
            embedding_model=data["embedding_model"],
            embedding_dim=data["embedding_dim"],
            chunk_count=data["chunk_count"],
            document_count=data["document_count"],
            built_at=data["built_at"],
            chunking=data.get("chunking", {}),
        )


def embedding_text(chunk: Chunk) -> str:
    """The string actually handed to the embedding model for a chunk.

    Prepends the source context so the passage is interpretable on its own.
    See the note at the top of this module for why this differs from
    `chunk.text`.
    """
    return (
        f"{chunk.university_name} | {chunk.policy_type.value} | "
        f"{chunk.section_path}\n{chunk.text}"
    )


def _chunk_from_row(chunk_id: str, text: str, meta: dict) -> Chunk:
    """Rebuild a Chunk from one row as Chroma hands it back (the inverse of
    Chunk.to_metadata). One place, so every reader rebuilds chunks the same way."""
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        citation_id=meta["citation_id"],
        university=meta["university"],
        university_name=meta["university_name"],
        policy_type=PolicyType(meta["policy_type"]),
        section_path=meta["section_path"],
        source_url=meta["source_url"],
    )


def _similarity_from_distance(distance: float) -> float:
    """Chroma reports cosine *distance*; convert to a similarity in [0, 1] so
    that "higher is better", which is what the abstention thresholds in
    config.yaml assume."""
    return max(0.0, min(1.0, 1.0 - float(distance)))


class Embedder:
    """The embedding model, loaded once and reused.

    Loading is deferred until first use because importing
    sentence-transformers costs a few seconds, and things like `--help` or a
    config check should not pay that.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            cfg = self.config.embedding
            self._model = SentenceTransformer(cfg.model_name, device=cfg.device)
        return self._model

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for storage."""
        vecs = self.model.encode(
            texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vecs]

    def encode_query(self, text: str) -> list[float]:
        """Embed a question for searching.

        bge models are trained with a short instruction prefix on the query
        side only - not on documents - so the prefix is applied here and not
        in encode_documents.
        """
        prefixed = f"{self.config.embedding.query_prefix}{text}"
        vec = self.model.encode(
            [prefixed], show_progress_bar=False, normalize_embeddings=True
        )[0]
        return vec.tolist()

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


class VectorStore:
    """Chroma-backed store of policy chunks."""

    def __init__(self, config: Config | None = None, embedder: Embedder | None = None):
        self.config = config or get_config()
        self.embedder = embedder or Embedder(self.config)
        self.index_dir = self.config.paths.resolve("index")
        self._client = None
        self._collection = None
        # The keyword index, built on first use. See point 3 at the top.
        self._bm25: BM25Index | None = None

    # -- plumbing ---------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.index_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                # Cosine similarity, because the embeddings are normalized and
                # cosine is what bge models are trained to be compared with.
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / MANIFEST_NAME

    # -- manifest ---------------------------------------------------------

    def read_manifest(self) -> IndexManifest | None:
        if not self.manifest_path.exists():
            return None
        return IndexManifest.from_dict(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )

    def check_ready(self) -> IndexManifest:
        """Confirm an index exists and matches the current config.

        Raises IndexMismatchError with an actionable message otherwise. This
        is called before every search - see the module docstring for why.
        """
        manifest = self.read_manifest()
        if manifest is None:
            raise IndexMismatchError(
                f"No index found at {self.index_dir}. Build one first:\n"
                f"    python scripts/build_index.py"
            )
        configured = self.config.embedding.model_name
        if manifest.embedding_model != configured:
            raise IndexMismatchError(
                f"Index/model mismatch - refusing to search.\n"
                f"  index was built with : {manifest.embedding_model}\n"
                f"  config.yaml now says : {configured}\n"
                f"Embeddings from different models are not comparable, so this "
                f"would return nonsense rather than fail. Rebuild the index:\n"
                f"    python scripts/build_index.py"
            )
        return manifest

    # -- writing ----------------------------------------------------------

    def build(self, chunks: list[Chunk], document_count: int = 0) -> IndexManifest:
        """Embed and store chunks, replacing any existing index."""
        if not chunks:
            raise ValueError("refusing to build an index from zero chunks")

        # Drop the old collection outright. Rebuilding in place would leave
        # chunks behind whose sections no longer exist in the source.
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # nothing to delete on a first build
        self._collection = None
        # The cached keyword index describes the corpus we are about to throw
        # away. Drop it too; it will be rebuilt from the new chunks on the
        # next search that needs it.
        self._bm25 = None

        texts = [embedding_text(c) for c in chunks]
        vectors = self.embedder.encode_documents(texts)

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[c.to_metadata() for c in chunks],
        )

        manifest = IndexManifest(
            embedding_model=self.config.embedding.model_name,
            embedding_dim=len(vectors[0]),
            chunk_count=len(chunks),
            document_count=document_count,
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            chunking=self.config.chunking.model_dump(),
        )
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        return manifest

    # -- reading ----------------------------------------------------------

    def search(
        self,
        question: str,
        k: int | None = None,
        university: str | None = None,
        policy_type: PolicyType | str | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Find the chunks most relevant to `question`.

        The optional filters restrict the search rather than re-ranking it,
        so a filtered search cannot return a passage from the wrong source.

        `query_embedding`, if given, must be `embed_query(question)`. Hybrid
        retrieval embeds the question once and reuses the vector for both
        this search and `similarities()`, rather than paying for the model
        twice per question.
        """
        self.check_ready()
        top_k = k or self.config.retrieval.top_k

        clauses = []
        if university:
            clauses.append({"university": university})
        if policy_type:
            value = (
                policy_type.value if isinstance(policy_type, PolicyType) else policy_type
            )
            clauses.append({"policy_type": value})

        where = None
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        if query_embedding is None:
            query_embedding = self.embed_query(question)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        out: list[RetrievedChunk] = []
        for rank, (cid, text, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True)
        ):
            chunk = _chunk_from_row(cid, text, meta)
            score = _similarity_from_distance(dist)
            out.append(RetrievedChunk(chunk=chunk, score=score, rank=rank))
        return out

    def embed_query(self, question: str) -> list[float]:
        """The question's embedding, exactly as `search` would compute it."""
        return self.embedder.encode_query(question)

    def similarities(
        self, query_embedding: list[float], chunk_ids: list[str]
    ) -> dict[str, float]:
        """Cosine similarity between a query vector and specific stored chunks.

        WHY THIS EXISTS: hybrid retrieval can return a chunk that only the
        keyword search found, so the embedding search never gave it a score.
        But `RetrievedChunk.score` has one meaning everywhere - the cosine
        similarity of the passage to the question, in [0, 1] - because the
        abstention gate compares it with `min_retrieval_score` and the app
        shows it to the user. A made-up number from the keyword side would
        quietly break both. So we look up the chunk's stored embedding and
        compute the same similarity the embedding search would have.

        The stored vectors and the query vector are both normalised to length
        1, so the cosine similarity is just their dot product - the same
        quantity Chroma turns into a distance, converted back the same way.
        """
        if not chunk_ids:
            return {}
        result = self.collection.get(ids=list(chunk_ids), include=["embeddings"])
        out: dict[str, float] = {}
        for cid, vector in zip(result["ids"], result["embeddings"], strict=True):
            dot = sum(float(q) * float(v) for q, v in zip(query_embedding, vector, strict=True))
            out[cid] = _similarity_from_distance(1.0 - dot)
        return out

    def all_chunks(self) -> list[Chunk]:
        """Every chunk in the index, sorted by chunk_id.

        Sorted because Chroma promises no row order (see get_by_citation), and
        the keyword index built from this list breaks score ties by position -
        an order that changed between runs would change results between runs.
        """
        result = self.collection.get(include=["documents", "metadatas"])
        chunks = [
            _chunk_from_row(cid, text, meta)
            for cid, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]
        return sorted(chunks, key=lambda c: c.chunk_id)

    def bm25_index(self) -> BM25Index:
        """The keyword index over every stored chunk, built on first use.

        Cached on this object, and dropped by `build()`. See point 3 at the
        top of this module.
        """
        if self._bm25 is None:
            cfg = self.config.retrieval
            self._bm25 = BM25Index(self.all_chunks(), k1=cfg.bm25_k1, b=cfg.bm25_b)
        return self._bm25

    def count(self) -> int:
        return int(self.collection.count())

    def get_by_citation(self, citation_id: str) -> list[Chunk]:
        """Every chunk belonging to one citation, in document order.

        Phase 4 needs this to answer "does this cited section actually exist?"
        - the check that separates a fabricated citation from a misused one.

        The ordering is part of the contract, not incidental. Chroma's get()
        promises nothing about the order it returns rows in, and two callers
        depend on that order being stable: the verifier scores the chunks as
        premises in sequence, and the eval-set drift check hashes them joined
        together. An order that varied between runs would make a verdict's
        explanation non-deterministic and would make the drift check report
        phantom changes. Oversized sections are split into `{citation}#0`,
        `#1`, `#2`..., so sorting on that suffix restores document order.
        """
        result = self.collection.get(where={"citation_id": citation_id})
        chunks = [
            _chunk_from_row(cid, text, meta)
            for cid, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]

        def piece_number(chunk: Chunk) -> tuple[int, str]:
            """Sort key: the integer after the final '#', then the id itself.

            Falling back to the id keeps the sort total (and stable) if a
            chunk_id ever lacks the suffix, rather than raising.
            """
            _, _, suffix = chunk.chunk_id.rpartition("#")
            return (int(suffix) if suffix.isdigit() else -1, chunk.chunk_id)

        return sorted(chunks, key=piece_number)
