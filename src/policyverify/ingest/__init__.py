"""
Ingestion - turning messy public policy documents into clean Document objects.

The pipeline is: fetch.py (download) -> extract.py (PDF/HTML to text) ->
normalize.py (text to Document).

All of the ugliness lives in here. Every university publishes policies in a
different shape, and this package exists so that nothing downstream ever has to
know that. After ingestion, everything else only sees one clean Document shape.

Built in Phase 1.
"""
