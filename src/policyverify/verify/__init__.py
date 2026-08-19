"""
Verification - the heart of the project.

Three independent checks run on every claim the language model produces:

  nli.py       Does the cited passage actually prove this claim?
  numeric.py   Do the numbers agree? ("75%" vs "80%" is the error that hurts.)
  citation.py  Does the cited section exist, and does it support the claim?

Why not just ask the language model to check its own work: a model grading
itself tends to repeat its own mistakes, because the same misunderstanding that
produced the wrong claim also produces the wrong grade. A separate NLI model
answers one narrow question - "does text A prove text B?" - and it answers it
independently.

The citation checks are deliberately two separate things. A section that does
not exist is a fabricated citation. A section that exists but does not support
the claim is a misused citation. Those are different failures with different
causes, and telling them apart is most of the value here.

Built in Phase 4.
"""
