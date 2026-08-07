"""Phase 6 tests. No API calls: a stub client stands in for the provider."""

from __future__ import annotations

from datetime import date

import pytest

from edgar_rag.config import Settings
from edgar_rag.generation.anthropic_client import AnthropicClient
from edgar_rag.generation.base import LLMClient, LLMResponse
from edgar_rag.generation.budget import Budget, BudgetExceeded, estimate_cost
from edgar_rag.generation.cache import ResponseCache
from edgar_rag.generation.grounding import (
    extract_citations,
    find_ungrounded_figures,
    has_declined,
)
from edgar_rag.generation.pipeline import AnswerPipeline
from edgar_rag.generation.prompts import SYSTEM_PROMPT, build_prompt
from edgar_rag.models import Chunk, ChunkMetadata, FormType, RetrievedChunk, SearchFilter


def _retrieved(text: str, *, item: str = "1A", year: int = 2025) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"chunk-{item}-{year}",
            filing_id="0000320193/acc",
            text=text,
            order=0,
            metadata=ChunkMetadata(
                cik="0000320193",
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type=FormType.TEN_K,
                fiscal_year=year,
                item=item,
                part="I",
                filing_date=date(2025, 10, 31),
                accession_number="acc",
            ),
        ),
        score=0.9,
        dense_rank=1,
    )


class StubLLM(LLMClient):
    """Returns a canned answer and records what it was asked."""

    def __init__(self, text: str = "Revenue was $383.3 billion [0000320193:2025:I-1A].") -> None:
        self.text = text
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    @property
    def model(self) -> str:
        return "stub-model"

    def complete(self, prompt, system=None, max_tokens=None, temperature=None) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        return LLMResponse(text=self.text, model=self.model, input_tokens=100, output_tokens=20)


class StubRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple] = []

    def retrieve(self, query, top_k, filters=None, mode="hybrid"):
        self.calls.append((query, top_k, filters, mode))
        return self.results


# --- Budget --------------------------------------------------------------


def test_call_ceiling_stops_a_runaway_loop() -> None:
    """The failure that actually costs money is a retry loop, not one call."""
    budget = Budget(max_calls=3)
    for _ in range(3):
        budget.check()
        budget.record("claude-sonnet-5", 1000, 100)

    with pytest.raises(BudgetExceeded, match="call ceiling"):
        budget.check()


def test_cost_ceiling_stops_a_few_large_calls() -> None:
    budget = Budget(max_calls=1000, max_cost_usd=0.05)
    budget.record("claude-opus-5", 1_000_000, 100_000)  # $5 + $2.50

    with pytest.raises(BudgetExceeded, match="cost ceiling"):
        budget.check()


def test_cost_uses_published_rates() -> None:
    # Sonnet 5 introductory: $2 in / $10 out per million.
    assert estimate_cost("claude-sonnet-5", 1_000_000, 0) == pytest.approx(2.00)
    assert estimate_cost("claude-sonnet-5", 0, 1_000_000) == pytest.approx(10.00)


def test_unknown_model_costs_zero_rather_than_guessing() -> None:
    """A wrong price is worse than none: it makes the ceiling meaningless."""
    assert estimate_cost("some-future-model", 1_000_000, 1_000_000) == 0.0


def test_dated_model_ids_are_priced() -> None:
    """The API returns `claude-haiku-4-5-20251001` for a `claude-haiku-4-5`
    request. Exact-match lookup priced every such call at zero, which
    exempted the model from the spend ceiling rather than merely
    misreporting it."""
    assert estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.00)
    assert estimate_cost("claude-sonnet-5-20260101", 1_000_000, 0) == pytest.approx(2.00)


def test_prefix_match_prefers_the_longest_key() -> None:
    """A shorter key must not shadow a more specific one."""
    assert estimate_cost("claude-opus-5", 1_000_000, 0) == pytest.approx(5.00)


def test_cached_calls_do_not_count_against_the_ceiling() -> None:
    budget = Budget(max_calls=2)
    for _ in range(50):
        budget.record("claude-sonnet-5", 0, 0, cached=True)

    budget.check()  # still under the ceiling
    assert budget.cached_calls == 50
    assert budget.cost_usd == 0.0


# --- Cache ---------------------------------------------------------------


def test_repeated_request_is_served_from_cache(tmp_path) -> None:
    """The property that protects the eval budget across re-runs."""
    cache = ResponseCache(tmp_path)
    key = ResponseCache.key("m", "prompt", "system", 512, 0.0)
    cache.put(key, LLMResponse(text="answer", model="m", input_tokens=10, output_tokens=5))

    restored = cache.get(key)
    assert restored is not None
    assert restored.text == "answer"
    assert restored.cached is True


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "other"},
        {"prompt": "different question"},
        {"system": "different system"},
        {"max_tokens": 1024},
        {"temperature": 0.7},
    ],
)
def test_any_field_that_changes_the_response_changes_the_key(changed: dict) -> None:
    base = {
        "model": "m",
        "prompt": "p",
        "system": "s",
        "max_tokens": 512,
        "temperature": 0.0,
    }
    assert ResponseCache.key(**base) != ResponseCache.key(**{**base, **changed})


def test_key_is_stable_across_runs() -> None:
    """An unstable key never hits and silently re-bills every run."""
    args = ("m", "p", "s", 512, 0.0)
    assert ResponseCache.key(*args) == ResponseCache.key(*args)


def test_missing_entry_is_a_miss(tmp_path) -> None:
    assert ResponseCache(tmp_path).get("nonexistent") is None


def test_corrupt_entry_is_treated_as_a_miss(tmp_path) -> None:
    """A truncated write from an interrupted run must not fail the run."""
    cache = ResponseCache(tmp_path)
    (tmp_path / "abc.json").write_text("{ truncated")
    assert cache.get("abc") is None


def test_truncation_survives_the_cache_round_trip(tmp_path) -> None:
    """On evaluation re-runs every answer is served from cache — an
    incomplete answer must not come back looking complete."""
    cache = ResponseCache(tmp_path)
    key = ResponseCache.key("m", "p", "s", 512, 0.0)
    cache.put(key, LLMResponse(text="cut off mid-", model="m", truncated=True))

    restored = cache.get(key)
    assert restored is not None
    assert restored.truncated is True


def test_cache_can_be_disabled(tmp_path) -> None:
    cache = ResponseCache(tmp_path, enabled=False)
    cache.put("k", LLMResponse(text="a", model="m"))
    assert cache.get("k") is None


def test_client_serves_a_repeat_call_without_hitting_the_api(tmp_path) -> None:
    class CountingAPI:
        def __init__(self) -> None:
            self.calls = 0
            self.messages = self

        def create(self, **kwargs):
            self.calls += 1
            usage = type("U", (), {"input_tokens": 100, "output_tokens": 20})()
            block = type("B", (), {"type": "text", "text": "answer"})()
            return type(
                "M",
                (),
                {
                    "content": [block],
                    "model": "claude-sonnet-5",
                    "usage": usage,
                    "stop_reason": "end_turn",
                },
            )()

    api = CountingAPI()
    settings = Settings(_env_file=None, llm_cache_path=tmp_path, anthropic_api_key="test")
    client = AnthropicClient(settings=settings, client=api)

    first = client.complete("same prompt")
    second = client.complete("same prompt")

    assert api.calls == 1  # the second was served from disk
    assert first.text == second.text
    assert second.cached is True
    assert client.budget.calls == 1


def test_client_stops_at_the_call_ceiling(tmp_path) -> None:
    class CountingAPI:
        def __init__(self) -> None:
            self.calls = 0
            self.messages = self

        def create(self, **kwargs):
            self.calls += 1
            usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()
            block = type("B", (), {"type": "text", "text": f"answer {self.calls}"})()
            return type(
                "M",
                (),
                {
                    "content": [block],
                    "model": "claude-sonnet-5",
                    "usage": usage,
                    "stop_reason": "end_turn",
                },
            )()

    api = CountingAPI()
    settings = Settings(
        _env_file=None,
        llm_cache_path=tmp_path,
        anthropic_api_key="test",
        max_api_calls_per_run=3,
    )
    client = AnthropicClient(settings=settings, client=api)

    with pytest.raises(BudgetExceeded):
        for i in range(100):  # a runaway loop
            client.complete(f"prompt {i}")

    assert api.calls == 3


def test_hitting_the_output_cap_is_reported(tmp_path) -> None:
    """A truncated answer otherwise reaches the caller looking complete."""

    class TruncatingAPI:
        def __init__(self) -> None:
            self.messages = self

        def create(self, **kwargs):
            usage = type("U", (), {"input_tokens": 100, "output_tokens": 512})()
            block = type("B", (), {"type": "text", "text": "The figure was $1"})()
            return type(
                "M",
                (),
                {
                    "content": [block],
                    "model": "claude-sonnet-5",
                    "usage": usage,
                    "stop_reason": "max_tokens",
                },
            )()

    settings = Settings(_env_file=None, llm_cache_path=tmp_path, anthropic_api_key="test")
    client = AnthropicClient(settings=settings, client=TruncatingAPI())

    assert client.complete("prompt").truncated is True


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("length", True), ("stop", False)],
)
def test_openai_client_reports_truncation_too(tmp_path, finish_reason, expected) -> None:
    """Otherwise switching provider silently drops the check."""
    from edgar_rag.generation.openai_client import OpenAIClient

    class StubOpenAI:
        def __init__(self) -> None:
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            message = type("Msg", (), {"content": "The figure was $1"})()
            choice = type("C", (), {"message": message, "finish_reason": finish_reason})()
            usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 512})()
            return type("R", (), {"choices": [choice], "model": "gpt-4o-mini", "usage": usage})()

    settings = Settings(_env_file=None, llm_cache_path=tmp_path, openai_api_key="test")
    client = OpenAIClient(settings=settings, client=StubOpenAI())

    assert client.complete("prompt").truncated is expected


# --- Prompting -----------------------------------------------------------


def test_system_prompt_permits_declining() -> None:
    """Without explicit permission, models invent rather than decline."""
    assert "I don't know" in SYSTEM_PROMPT


def test_system_prompt_requires_citations() -> None:
    assert "Cite every factual claim" in SYSTEM_PROMPT


def test_prompt_includes_passages_and_tags() -> None:
    prompt = build_prompt("What are the risks?", [_retrieved("Supply chain risk text.")])
    assert "Supply chain risk text." in prompt
    assert "[0000320193:2025:I-1A]" in prompt
    assert "What are the risks?" in prompt


def test_prompt_handles_no_passages() -> None:
    prompt = build_prompt("Anything?", [])
    assert "no passages" in prompt


# --- Grounding -----------------------------------------------------------


def test_figure_present_in_context_is_grounded() -> None:
    chunks = [_retrieved("Research and development expense was $29.9 billion.")]
    assert find_ungrounded_figures("R&D was $29.9 billion.", chunks) == []


def test_invented_figure_is_flagged() -> None:
    """The claim this system makes about financial figures, made testable."""
    chunks = [_retrieved("Research and development expense was $29.9 billion.")]
    assert find_ungrounded_figures("R&D was $31.4 billion.", chunks) == ["31.4"]


def test_comma_formatting_does_not_cause_a_false_flag() -> None:
    chunks = [_retrieved("Research and development 8,866 for the quarter.")]
    assert find_ungrounded_figures("R&D was $8,866 million.", chunks) == []


def test_trailing_zeros_do_not_cause_a_false_flag() -> None:
    """Comparison is numeric, not textual."""
    chunks = [_retrieved("Net sales were 383.30 billion.")]
    assert find_ungrounded_figures("Net sales were $383.3 billion.", chunks) == []


def test_digits_inside_citations_are_not_treated_as_figures() -> None:
    chunks = [_retrieved("Some text with no figures at all.")]
    assert find_ungrounded_figures("The risk is described [0000320193:2025:I-1A].", chunks) == []


def test_fiscal_years_are_not_treated_as_figures() -> None:
    """ "Q1 FY2026" is a period the model states, not a quoted figure. Checking
    them flagged answers that were otherwise entirely grounded."""
    chunks = [_retrieved("Revenue for the quarter ended December 27, 2025 was $124 billion.")]
    assert find_ungrounded_figures("For Q1 FY2026, revenue was $124 billion.", chunks) == []


def test_scale_restatement_is_not_a_second_figure() -> None:
    """Filings tabulate in millions; answers often restate in billions."""
    chunks = [_retrieved("Repurchases totalled 62,184 for the year.")]
    answer = "Alphabet repurchased $62,184 million ($62.184 billion) of stock."
    assert find_ungrounded_figures(answer, chunks) == []


def test_a_genuinely_invented_figure_still_fails_after_the_year_exemption() -> None:
    chunks = [_retrieved("Revenue was $124 billion in the period.")]
    assert find_ungrounded_figures("Revenue was $999 billion.", chunks) == ["999"]


def test_small_integers_are_not_flagged() -> None:
    """Prose counts would otherwise bury the real findings."""
    chunks = [_retrieved("Text without those numbers.")]
    assert find_ungrounded_figures("There are 3 main risks and 2 segments.", chunks) == []


def test_percentages_are_checked() -> None:
    chunks = [_retrieved("Margin was 46.2% for the year.")]
    assert find_ungrounded_figures("Margin was 46.2%.", chunks) == []
    assert find_ungrounded_figures("Margin was 51.7%.", chunks) == ["51.7"]


def test_each_ungrounded_figure_is_reported_once() -> None:
    chunks = [_retrieved("No figures here.")]
    assert find_ungrounded_figures("It was 99.9, again 99.9.", chunks) == ["99.9"]


def test_citations_resolve_to_retrieved_passages() -> None:
    chunks = [_retrieved("Risk factor text.")]
    citations = extract_citations("The risk [0000320193:2025:I-1A] is material.", chunks)

    assert len(citations) == 1
    assert citations[0].citation == "0000320193:2025:I-1A"
    assert "Risk factor text." in citations[0].excerpt


def test_citation_to_an_unretrieved_passage_is_dropped() -> None:
    """It cannot be verified, so presenting it as a source would mislead."""
    chunks = [_retrieved("Risk factor text.")]
    assert extract_citations("Claim [0000789019:2024:II-7].", chunks) == []


def test_repeated_citation_appears_once() -> None:
    chunks = [_retrieved("Risk factor text.")]
    answer = "One [0000320193:2025:I-1A] and two [0000320193:2025:I-1A]."
    assert len(extract_citations(answer, chunks)) == 1


def test_declining_is_detected() -> None:
    assert has_declined("I don't know based on the filings provided.")
    assert not has_declined("Revenue was $383.3 billion.")


def test_unclosed_citation_from_truncation_is_not_a_figure() -> None:
    """A cut-off "[0001500412:2025:I" has no closing bracket, so the
    citation pattern misses it and the CIK reaches the figure check."""
    chunks = [_retrieved("Some risk factor text with no figures.")]
    answer = "Suppliers may stop selling to us [0001500412:2025:I"

    assert find_ungrounded_figures(answer, chunks) == []


def test_dates_are_not_treated_as_figures() -> None:
    chunks = [_retrieved("A plan was adopted during the period.")]

    assert find_ungrounded_figures("The plan was adopted 12/18/2025.", chunks) == []
    assert find_ungrounded_figures("Adopted on March 15, 2026.", chunks) == []
    assert find_ungrounded_figures("Effective 2025-10-31.", chunks) == []


def test_digits_inside_a_name_are_not_figures() -> None:
    """A 1,800-filer corpus contains names like "Data443 Risk Mitigation"."""
    chunks = [_retrieved("A company operating in risk mitigation.")]

    assert find_ungrounded_figures("Data443 Risk Mitigation provides tools.", chunks) == []


def test_a_figure_mashed_against_letters_in_a_filing_still_counts_as_evidence() -> None:
    """Table extraction runs cells together, so this is how real figures
    appear in filings. Requiring clean boundaries on the evidence side made
    four grounded answers look invented."""
    chunks = [_retrieved("Three Months Ended June 28, 2025AmericasEurope Net sales$41,198 total")]

    assert find_ungrounded_figures("Net sales were $41,198 million.", chunks) == []


def test_a_real_invented_figure_still_survives_the_new_filters() -> None:
    """The filters must not be so broad that they hide actual failures."""
    chunks = [_retrieved("Research and development expense was $29.9 billion.")]

    assert find_ungrounded_figures("R&D was $31.4 billion.", chunks) == ["31.4"]


# --- Pipeline ------------------------------------------------------------


def _pipeline(llm: StubLLM, results: list[RetrievedChunk]) -> AnswerPipeline:
    return AnswerPipeline(
        retriever=StubRetriever(results),
        llm=llm,
        settings=Settings(_env_file=None),
    )


def test_answer_carries_citations_and_passages() -> None:
    chunks = [_retrieved("Total net sales were $383.3 billion.")]
    answer = _pipeline(StubLLM(), chunks).answer("What were net sales?")

    assert "383.3" in answer.answer
    assert len(answer.citations) == 1
    assert answer.retrieved == chunks
    assert answer.is_grounded


def test_truncated_answer_skips_the_figure_check() -> None:
    """Its trailing text is cut mid-claim, so it is not a claim to verify."""

    class TruncatingLLM(StubLLM):
        def complete(self, prompt, system=None, max_tokens=None, temperature=None):
            return LLMResponse(
                text="Net sales were $999.9 billion [0000320193:2025:I",
                model=self.model,
                truncated=True,
            )

    answer = _pipeline(TruncatingLLM(), [_retrieved("Net sales were $383.3 billion.")]).answer("q")

    assert answer.truncated is True
    assert answer.ungrounded_figures == []


def test_answer_flags_an_invented_figure() -> None:
    chunks = [_retrieved("Total net sales were $383.3 billion.")]
    llm = StubLLM("Net sales were $999.9 billion [0000320193:2025:I-1A].")

    answer = _pipeline(llm, chunks).answer("What were net sales?")

    assert answer.ungrounded_figures == ["999.9"]
    assert not answer.is_grounded


def test_empty_retrieval_declines_instead_of_answering() -> None:
    """Answering with no passages means answering from the model's memory."""
    llm = StubLLM()
    answer = _pipeline(llm, []).answer("Anything?")

    assert "I don't know" in answer.answer
    assert llm.prompts == []  # no API call was made at all


def test_filters_reach_the_retriever() -> None:
    retriever = StubRetriever([_retrieved("text")])
    pipeline = AnswerPipeline(retriever=retriever, llm=StubLLM(), settings=Settings(_env_file=None))

    filters = SearchFilter(tickers=["AAPL"])
    pipeline.answer("q", top_k=3, filters=filters, mode="dense")

    query, top_k, passed, mode = retriever.calls[0]
    assert (top_k, passed, mode) == (3, filters, "dense")


def test_system_prompt_is_sent() -> None:
    llm = StubLLM()
    _pipeline(llm, [_retrieved("text")]).answer("q")
    assert llm.systems[0] == SYSTEM_PROMPT


def test_latency_is_recorded() -> None:
    answer = _pipeline(StubLLM(), [_retrieved("text")]).answer("q")
    assert answer.latency_ms is not None and answer.latency_ms >= 0
