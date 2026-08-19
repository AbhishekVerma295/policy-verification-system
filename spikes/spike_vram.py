"""
SPIKE 3 of 3 - does everything fit in 8 GB of VRAM at the same time?

WHY THIS MATTERS
    This system runs three models at once: Qwen writes the draft answer, an
    embedding model powers search, and an NLI model checks the claims. Your
    RTX 4060 laptop GPU has 8 GB, which is comfortable for one of those and
    tight for all three.

    The plan is to keep the two small models on the CPU and give the whole GPU
    to Qwen. They are small enough that CPU is fast enough, and the embedding
    model only really works hard once, when the index is built.

    This spike proves that plan actually holds before anything depends on it.

HOW TO USE IT
        python spikes/spike_vram.py

    Make sure Ollama is running first, and that the model is pulled:

        ollama pull qwen3:4b

WHAT TO LOOK FOR
    - Peak VRAM should stay comfortably under 8192 MB. Under ~6500 MB is a
      good place to be, since Windows itself uses some of the GPU.
    - "torch sees CUDA: False" is CORRECT and intentional here. We installed
      the CPU build on purpose so that torch cannot quietly grab VRAM that
      Qwen needs.
    - If VRAM is tight, drop to a smaller Qwen before doing anything else.
      Model quality matters far less than a pipeline that runs reliably.
"""

from __future__ import annotations

import subprocess
import time

MODEL = "qwen3:4b"


def vram_used_mb() -> int | None:
    """Ask nvidia-smi how much VRAM is in use right now. None if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def vram_total_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def report(label: str, baseline: int | None) -> int | None:
    used = vram_used_mb()
    if used is None:
        print(f"  {label:<38} (nvidia-smi unavailable)")
        return None
    delta = f"  (+{used - baseline} MB)" if baseline is not None else ""
    print(f"  {label:<38} {used:>6} MB{delta}")
    return used


def main() -> int:
    print(__doc__)
    print("=" * 78)

    total = vram_total_mb()
    if total is None:
        print("\nCould not run nvidia-smi. Is the NVIDIA driver installed?")
        print("You can still continue, but you will be flying blind on memory.\n")
    else:
        print(f"\nGPU total VRAM: {total} MB")

    baseline = report("baseline (nothing loaded)", None)
    peak = baseline or 0

    # --- 1. torch, which must NOT be using the GPU --------------------------
    print("\n[1/3] torch")
    import torch

    print(f"  torch version                          {torch.__version__}")
    print(f"  torch sees CUDA                        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("  !! WARNING: this is the CUDA build of torch.")
        print("     It may compete with Qwen for VRAM. The CPU build is what")
        print("     this project wants:")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cpu")
    else:
        print("  OK - CPU build, so torch cannot take VRAM from Qwen.")

    # --- 2. the embedding model, on CPU ------------------------------------
    print("\n[2/3] embedding model (CPU)")
    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        vecs = embedder.encode(
            ["Students must attend at least 75% of scheduled classes."]
        )
        print(f"  loaded and encoded in {time.time() - t0:.1f}s")
        print(f"  vector dimensions                      {len(vecs[0])}")
        used = report("after embedding model", baseline)
        peak = max(peak, used or 0)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # --- 3. the NLI model, on CPU ------------------------------------------
    print("\n[3/3] NLI fact-checker (CPU)")
    t0 = time.time()
    try:
        from transformers import pipeline

        nli = pipeline(
            "text-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            device=-1,
            top_k=None,
        )
        out = nli(
            {
                "text": "Students must attend at least 75% of scheduled classes.",
                "text_pair": "The attendance requirement is 75%.",
            }
        )
        scores = out[0] if isinstance(out[0], list) else out
        best = max(scores, key=lambda d: d["score"])
        print(f"  loaded and ran in {time.time() - t0:.1f}s")
        print(f"  sample verdict                         {best['label']} ({best['score']:.2f})")
        used = report("after NLI model", baseline)
        peak = max(peak, used or 0)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # --- 4. Qwen, on the GPU, while the other two are still loaded ---------
    print(f"\n[4/4] {MODEL} via Ollama (GPU) - the real test")
    print("  The two CPU models above are still in memory, which is the point:")
    print("  this measures all three coexisting, not Qwen on its own.")
    t0 = time.time()
    try:
        import ollama

        resp = ollama.generate(
            model=MODEL,
            prompt="Reply with exactly one word: ready",
            options={"temperature": 0.0, "num_ctx": 8192},
        )
        print(f"  responded in {time.time() - t0:.1f}s")
        print(f"  reply                                  {resp['response'].strip()[:60]!r}")
        used = report("with Qwen loaded (PEAK)", baseline)
        peak = max(peak, used or 0)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("\n  Is Ollama running, and have you pulled the model?")
        print(f"      ollama pull {MODEL}")

    # --- verdict -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if total and peak:
        headroom = total - peak
        print(f"  peak VRAM used   {peak} MB of {total} MB")
        print(f"  headroom         {headroom} MB")
        if headroom > 1500:
            print("\n  GOOD - comfortable. You could try qwen3:8b later for better quality.")
        elif headroom > 600:
            print("\n  OK - it fits, but it is tight. Stay on 4b for now.")
        else:
            print("\n  TIGHT - use a smaller model, or a heavier quantisation.")
            print("  A pipeline that runs reliably beats a slightly smarter one.")
    else:
        print("  Could not measure VRAM. Watch `nvidia-smi` by hand while this runs.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
