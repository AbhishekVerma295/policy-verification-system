"""
Indexing - splitting documents into chunks and making them searchable.

chunk.py splits a Document on its section boundaries so that each chunk is a
whole citable thing, which is what makes a citation like "4.2" point at
something exact. store.py wraps Chroma, the vector database that holds them.

One rule worth remembering, and enforced in store.py rather than trusted to
memory: the index records which embedding model built it. If you rebuild with
a different model and forget, search silently returns nonsense with no error
message at all. That is the most common bug in systems like this one, so the
store refuses to search a mismatched index instead of guessing.

Built in Phase 2.
"""

from policyverify.indexing.chunk import (
    chunk_document,
    chunk_documents,
    is_noise_section,
    split_long_text,
)
from policyverify.indexing.store import (
    COLLECTION_NAME,
    Embedder,
    IndexManifest,
    IndexMismatchError,
    VectorStore,
    embedding_text,
)

__all__ = [
    "COLLECTION_NAME",
    "Embedder",
    "IndexManifest",
    "IndexMismatchError",
    "VectorStore",
    "chunk_document",
    "chunk_documents",
    "embedding_text",
    "is_noise_section",
    "split_long_text",
]
