"""Spend guard and cost accounting for LLM calls.

Generation and the evaluation judge are the only parts of this system that
cost money — embeddings and indexing run locally. The failure mode worth
guarding against is not a single expensive call but a loop: a retry bug can
issue thousands of calls in minutes. Every client is wrapped by a budget
that aborts once a run exceeds its ceiling.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# USD per million tokens. Sonnet 5 carries an introductory rate through
# 2026-08-31; the standard rate is $3 / $15.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class BudgetExceeded(RuntimeError):
    """Raised when a run hits its call or cost ceiling."""


@dataclass
class Budget:
    """Tracks spend for one run and stops it going over.

    Two independent ceilings: `max_calls` catches runaway loops quickly,
    `max_cost_usd` catches a smaller number of unexpectedly large calls.
    """

    max_calls: int = 200
    max_cost_usd: float = 5.00
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        """Raise if another call would exceed a ceiling."""
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(
                    f"call ceiling reached: {self.calls} calls "
                    f"(limit {self.max_calls}). Raise MAX_API_CALLS_PER_RUN "
                    "if this run genuinely needs more."
                )
            if self.cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(
                    f"cost ceiling reached: ${self.cost_usd:.2f} "
                    f"(limit ${self.max_cost_usd:.2f}). Raise MAX_COST_USD_PER_RUN "
                    "if this run genuinely needs more."
                )

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
    ) -> float:
        """Record one call. Returns its cost in USD."""
        cost = estimate_cost(model, input_tokens, output_tokens) if not cached else 0.0
        with self._lock:
            if cached:
                self.cached_calls += 1
            else:
                self.calls += 1
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens
                self.cost_usd += cost
        return cost

    def summary(self) -> str:
        return (
            f"{self.calls} calls ({self.cached_calls} cached), "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out, "
            f"${self.cost_usd:.4f}"
        )

    def reset(self) -> None:
        """Clear the counters, keeping the ceilings.

        A long-lived service needs the loop guard per request, not per
        process: sharing one budget across every request turns a ceiling
        meant to stop a runaway loop into a permanent outage once enough
        legitimate requests have been served.
        """
        with self._lock:
            self.calls = 0
            self.cached_calls = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.cost_usd = 0.0


@dataclass
class CumulativeSpend:
    """Running totals across a process, for reporting rather than limiting."""

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def absorb(self, budget: Budget) -> None:
        with self._lock:
            self.calls += budget.calls
            self.cached_calls += budget.cached_calls
            self.input_tokens += budget.input_tokens
            self.output_tokens += budget.output_tokens
            self.cost_usd += budget.cost_usd

    def summary(self) -> str:
        return (
            f"{self.calls} calls ({self.cached_calls} cached), "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out, "
            f"${self.cost_usd:.4f}"
        )


def _rates_for(model: str) -> tuple[float, float] | None:
    """Look up pricing, tolerating the dated ids the API returns.

    A request for `claude-haiku-4-5` comes back as
    `claude-haiku-4-5-20251001`. Matching only on the exact string priced
    every one of those calls at zero, which does not merely misreport cost
    — it exempts the model from the spend ceiling entirely.
    """
    if model in PRICING:
        return PRICING[model]

    # Longest first, so `claude-opus-4-8` is not matched by a shorter key.
    for known in sorted(PRICING, key=len, reverse=True):
        if model.startswith(known):
            return PRICING[known]
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for one call. Unknown models are priced at zero.

    An unknown model is logged rather than guessed at: a wrong price is
    worse than a missing one, because it makes the ceiling meaningless.
    """
    rates = _rates_for(model)
    if rates is None:
        logger.warning("no pricing for %s; cost not tracked for this call", model)
        return 0.0
    input_rate, output_rate = rates
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
