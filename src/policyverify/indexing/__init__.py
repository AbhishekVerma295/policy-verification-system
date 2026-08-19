"""
Indexing - splitting documents into chunks and making them searchable.

chunk.py splits a Document on its headings so that each chunk is a whole
section, which is what makes a citation like "4.2" point at something exact.
store.py wraps Chroma, the vector database that holds the chunks.

One rule worth remembering: the index records which embedding model built it.
If you rebuild with a different model and forget, search silently returns
nonsense with no error message at all. That is the most common bug in systems
like this one.

Built in Phase 2.
"""
