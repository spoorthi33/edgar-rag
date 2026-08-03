"""Prompt assembly for grounded answering.

Two instructions in the system prompt do most of the anti-hallucination
work, and both are deliberate:

  - Explicit permission to decline. Without it a model treats every
    question as answerable and invents a figure rather than reporting that
    the context lacks one.
  - A required citation on every claim. Citing forces the answer to be
    traceable, and an answer that cannot cite is one the reader can catch.

Neither is sufficient on its own — the numeric check in `grounding.py`
verifies what the prompt asks for.
"""

from __future__ import annotations

from edgar_rag.models import RetrievedChunk

SYSTEM_PROMPT = """\
You answer questions about SEC filings using only the context provided.

Rules:
1. Use ONLY the numbered context passages below. Do not use outside knowledge \
about these companies, and do not infer figures that are not stated.
2. If the context does not contain the answer, say "I don't know based on the \
filings provided." Say this rather than guessing — an unanswerable question is \
an acceptable outcome.
3. Cite every factual claim with the passage's tag in square brackets, e.g. \
[0000320193:2025:I-1A]. A claim without a citation is not acceptable.
4. Quote figures exactly as they appear in the context, including units and \
scale (e.g. "$29.9 billion", "8.1%"). Do not convert, round, or recompute.
5. Always state which period a figure covers, since the context spans several.
6. Answer with the single most relevant figure — normally the most recent \
period, unless the question names one. Do not enumerate every period you can \
find; if other periods are relevant, say so in one clause rather than listing \
them.
7. Be concise: two or three sentences. No preamble, no restating the question."""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the user prompt from retrieved passages."""
    if not chunks:
        return f"Context:\n(no passages were retrieved)\n\nQuestion: {question}"

    blocks = []
    for index, result in enumerate(chunks, start=1):
        meta = result.chunk.metadata
        header = (
            f"[{meta.citation}] {meta.company_name} {meta.form_type.value} FY{meta.fiscal_year}"
        )
        if meta.item:
            header += f", Item {meta.item}"
        blocks.append(f"Passage {index} {header}\n{result.chunk.text}")

    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def build_judge_prompt(question: str, answer: str, chunks: list[RetrievedChunk]) -> str:
    """Prompt for the faithfulness judge used by the evaluation harness."""
    context = "\n\n".join(f"[{r.chunk.metadata.citation}] {r.chunk.text}" for r in chunks)
    return (
        "Decide whether the answer is fully supported by the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Reply with exactly one word — SUPPORTED or UNSUPPORTED — then a "
        "one-sentence reason. An answer that declines to answer because the "
        "context lacks the information counts as SUPPORTED."
    )
