"""Faithfulness scoring.

Two independent checks, because they catch different failures:

  - The numeric check is deterministic and free. It catches invented
    figures, which is the failure that matters most on financial questions.
  - The LLM judge catches unsupported *claims* — a sentence that asserts
    something the passages do not, without necessarily inventing a number.

The judge runs on a cheaper model than generation: deciding whether text
is supported by other text is a verification task, not a reasoning-heavy
one, and it runs on every question of every re-run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from edgar_rag.config import Settings, get_settings
from edgar_rag.generation.base import LLMClient
from edgar_rag.generation.grounding import has_declined
from edgar_rag.generation.prompts import build_judge_prompt
from edgar_rag.models import Answer

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You verify whether an answer is supported by the provided context. "
    "You are strict: a claim not present in the context is unsupported, "
    "even if it is true in the wider world."
)


@dataclass
class FaithfulnessVerdict:
    """Whether an answer is supported by the passages it was given."""

    numerically_grounded: bool
    ungrounded_figures: list[str]
    judge_supported: bool | None
    judge_reason: str | None
    declined: bool

    @property
    def is_faithful(self) -> bool:
        """Faithful when nothing was invented and the judge agrees.

        Declining is faithful: reporting that the filings do not contain
        the answer is the correct behaviour, not a failure.
        """
        if self.declined:
            return True
        if not self.numerically_grounded:
            return False
        return self.judge_supported is not False


def judge_answer(
    answer: Answer,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    use_llm: bool = True,
) -> FaithfulnessVerdict:
    """Score one answer for faithfulness."""
    settings = settings or get_settings()
    declined = has_declined(answer.answer)

    verdict = FaithfulnessVerdict(
        numerically_grounded=not answer.ungrounded_figures,
        ungrounded_figures=list(answer.ungrounded_figures),
        judge_supported=None,
        judge_reason=None,
        declined=declined,
    )

    # A declined answer asserts nothing, so there is nothing to judge and
    # no reason to pay for the call.
    if not use_llm or llm is None or declined:
        return verdict

    response = llm.complete(
        prompt=build_judge_prompt(answer.question, answer.answer, answer.retrieved),
        system=JUDGE_SYSTEM,
        max_tokens=150,
        temperature=0.0,
    )

    verdict.judge_supported, verdict.judge_reason = _parse_verdict(response.text)
    return verdict


def _parse_verdict(text: str) -> tuple[bool | None, str]:
    """Read SUPPORTED / UNSUPPORTED from the judge's reply.

    Checked in that order because "UNSUPPORTED" contains "SUPPORTED" as a
    substring — testing for the shorter one first would score every
    unsupported answer as supported.
    """
    upper = text.upper()
    reason = text.strip()

    if "UNSUPPORTED" in upper:
        return False, reason
    if "SUPPORTED" in upper:
        return True, reason

    logger.warning("could not parse judge verdict from: %r", text[:120])
    return None, reason
