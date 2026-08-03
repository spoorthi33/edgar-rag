# edgar-rag

Retrieval-augmented question answering over SEC EDGAR 10-K and 10-Q filings.

Ask a question in natural language, get an answer grounded in — and cited back to — the exact
passages it came from, with retrieval quality and answer faithfulness measured rather than assumed.

> **Status: Phase 8 (measured).** Retrieval and answer quality are measured against a 52-question
> labelled set rather than asserted. Hybrid retrieval reaches **Hit@5 = 1.00**; every unanswerable
> question is correctly declined. See [Results](#results).

## Why

A 10-K runs 100–200 pages. Asking an LLM directly fails three ways: the filing may postdate its
training data, the corpus is far too large for any context window, and when it doesn't know a
figure it tends to produce a plausible-looking one anyway. On financial numbers, a confident wrong
answer is worse than no answer.

This system retrieves the handful of relevant passages first, then asks the model to answer from
those alone — turning a recall problem into a reading-comprehension problem, and making every claim
traceable to a source.

## Architecture

```
SEC EDGAR ──> ingest ──> object store (raw HTML + parsed text)
                │
                ├─> section-aware + semantic chunking
                ├─> sentence-transformer embeddings (local)
                ├─> FAISS dense index + BM25 sparse index
                │            └──> RRF fusion ──> hybrid retrieval
                │                                      │
                └─> PostgreSQL                         └─> LLM (Anthropic / OpenAI)
                    (filings, chunk provenance,               │
                     extractions, eval runs)     FastAPI <────┘
                                                     │
                                        eval harness (precision / recall / faithfulness)
```

### Design decisions

| Area | Decision | Rationale |
|---|---|---|
| Data source | EDGAR JSON APIs directly | Full control over the 10 req/s cap and the mandatory `User-Agent` |
| Storage | `ObjectStore` interface — local FS or S3 by env flag | Same code path either way; zero cost until the flag flips |
| Chunking | Section-aware split (Items 1A/7/7A/8), then semantic within | Section metadata is what prevents right-topic/wrong-filing answers |
| Embeddings | `BAAI/bge-small-en-v1.5`, run locally | Free, fast; dimension is config, so the model is swappable |
| Index | FAISS `flat` → `ivf` at scale | Exact while debugging retrieval; approximate only when scale demands |
| Search | Dense + BM25, fused by Reciprocal Rank Fusion | Sparse catches tickers and exact figures; RRF needs no score normalization |
| Grounding | Provenance on every chunk, forced `[cik:year:item]` citations, post-hoc numeric check | Makes the anti-hallucination claim measurable, not aspirational |
| LLM | Anthropic default, OpenAI swappable behind `LLMClient` | Provider becomes an eval axis rather than a rewrite |

**Metadata filtering is applied before ranking, not after.** Apple's 2023 revenue paragraph and
Microsoft's 2021 revenue paragraph are near-identical in embedding space, so similarity alone will
happily return the wrong one. Filtering to the right company and year first makes that class of
error structurally impossible.

## Project layout

```
src/edgar_rag/
  models.py          Core data types (Filing, Section, Chunk, RetrievedChunk, Answer)
  config.py          Settings — every swappable choice lives here
  ingest/            EDGAR client: rate limiting, download, manifest
  parsing/           HTML → text, Item boundary detection
  chunking/          Section-aware + semantic chunking
  storage/           ObjectStore contract (local FS / S3)
  embeddings/        Embedder contract (sentence-transformers)
  index/             VectorIndex contract (FAISS)
  retrieval/         Hybrid search + RRF fusion
  generation/        LLMClient contract, prompting, grounding checks
  api/               FastAPI service
  db/                SQLAlchemy models + migrations
  eval/              Metrics and evaluation harness
```

Four interfaces (`ObjectStore`, `Embedder`, `VectorIndex`, `LLMClient`) are defined before their
implementations, so local-vs-S3 and Anthropic-vs-OpenAI stay one-line config changes.

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Set EDGAR_USER_AGENT — the SEC returns 403 without a real name and email.
# Set ANTHROPIC_API_KEY for generation (needed from Phase 6).

docker compose up -d db     # PostgreSQL on :5432
pytest
```

### Ingesting filings

```bash
python scripts/ingest.py --tickers AAPL MSFT NVDA AMZN GOOGL --limit 4
```

Downloads the most recent filings per ticker into `data/raw/` and records them in
`data/manifest.json`. Re-running is safe and cheap: filings are immutable once accepted, so
anything already present is skipped rather than re-fetched.

### Parsing filings into sections

```bash
python scripts/parse.py                          # section summary for every filing
python scripts/parse.py --ticker AAPL --item 1A --show
```

### Chunking

```bash
python scripts/chunk.py --stats                  # corpus-wide chunk statistics
python scripts/chunk.py --ticker AAPL --item 1A --show 3
python scripts/chunk.py --stats --fixed          # skip embeddings, deterministic
```

The 20-filing corpus yields about 2,100 chunks with a median of 461 tokens, none exceeding the
embedding model's 512-token limit.

### Indexing and search

```bash
python scripts/index.py build
python scripts/index.py search "what are the risks from supply chain disruption?"
python scripts/index.py search "revenue growth" --ticker AAPL --year 2025 --item 7
python scripts/index.py search "mine safety disclosures" --compare   # dense vs sparse vs hybrid
```

Building the 2,105-chunk index takes about 140s on CPU; queries return in well under a second.
`--compare` runs all three retrieval modes side by side and shows which retriever found each
result and at what rank, e.g. `(d2+s1)`.

### Asking questions

```bash
python scripts/ask.py "How much did Apple spend on research and development?" --ticker AAPL
python scripts/ask.py "What supply chain risks does Apple disclose?" --ticker AAPL --item 1A
python scripts/ask.py "..." --show-context --no-cache
```

Needs `ANTHROPIC_API_KEY`. Roughly $0.006-0.011 per answer on Claude Sonnet 5; repeated questions
are served from disk at no cost.

### Evaluating

```bash
python scripts/evaluate.py --validate                    # check labels, free
python scripts/evaluate.py --compare --retrieval-only    # all 3 modes, free, ~11s
python scripts/evaluate.py --mode hybrid                 # full judged run, ~$0.58
```

Relevance is expressed as a predicate over chunk text and metadata, resolved against the corpus at
run time, so labels survive a change to chunking — a hand-listed chunk id would go stale exactly
when the metrics matter most. `--validate` reports any rule that matches nothing, since a broken
label would otherwise depress recall and misdirect the next fix.

### Running the service

```bash
docker compose up -d db
uvicorn edgar_rag.api.main:app --reload
```

| Endpoint | Purpose |
|---|---|
| `POST /query` | Ask a question; returns answer, citations, and grounding verdict |
| `GET /health` | Index size, database reachability, active model |
| `GET /queries` | Recent queries with their grounding verdicts |
| `GET /stats` | How often answers were fully traceable to retrieved passages |

```bash
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"How much did Apple spend on R&D?","tickers":["AAPL"]}'
```

The index and embedding model load once at startup (~11s), so query latency is about 1s rather
than paying the load cost per request. Interactive docs at `/docs`.

## Results

Measured on 52 labelled questions (44 answerable, 8 deliberately unanswerable) over the 20-filing
corpus, `k=5`:

| Retrieval | Hit@5 | Recall@5* | Precision@5 | MRR | Faithfulness |
|---|---|---|---|---|---|
| Dense only | 0.841 | 0.602 | 0.600 | 0.746 | — |
| Sparse only (BM25) | 0.909 | 0.782 | 0.755 | 0.868 | — |
| **Hybrid (RRF)** | **1.000** | 0.768 | 0.745 | 0.864 | **0.962** |

\* `achievable_recall@5`. Raw Recall@k is bounded by `k/|relevant|`, so a question with 100 relevant
chunks scores 0.05 even under perfect retrieval; dividing by `min(k, |relevant|)` asks the
answerable question instead — of the slots available, how many held evidence.

**Hybrid retrieves relevant evidence for every answerable question**, where dense alone misses 16%
and sparse misses 9%. Sparse edges hybrid on precision and MRR, so the gain is specifically in
*coverage*: the questions dense and sparse each miss are different ones, and fusing them leaves no
gap. This is the claim Phase 5 could only assert on a 12-query probe.

Answer quality on hybrid retrieval: **96.2% faithful** (LLM-judged), **98.1% of answers contain no
figure absent from the retrieved passages**, and **100% of unanswerable questions were correctly
declined** — the model does not fall back on training knowledge when the filings are silent. Two of
52 questions fail.

**The measurement needed fixing before the numbers meant anything.** Hand-checking the judge's
verdicts found errors in both directions: the numeric check flagged fiscal years the model derived
("Q1 FY2026") and scale restatements ("$62,184 million ($62.184 billion)") as invented figures,
while the judge marked correct answers unsupported for "attributing disclosures to Apple" because
the judge prompt showed only `[cik:year:item]` tags and never named the company. Fixing both moved
faithfulness from 0.885 to 0.962 — the system did not change, the measurement did. A judge trusted
without validation is not evidence.

The `rrf_k` fusion constant was also re-tuned against this set: the Phase 5 value of 5 came from a
12-query probe, and a sweep over all 44 answerable questions showed `k=10` matches it on coverage
while scoring better on recall and MRR (`k=60`, RRF's published default, drops Hit@5 to 0.932).

A full judged run costs about **$0.58**; re-runs after a retrieval change cost only what changed,
and `--retrieval-only` scores retrieval with no API calls at all (11s for all three modes).

## Roadmap

| Phase | Scope | Done when |
|---|---|---|
| 0 (done) | Scaffolding — layout, config, interfaces, Docker | Tests pass; `docker compose up -d db` starts Postgres |
| 1 (done) | Ingestion — rate-limited EDGAR client, download, manifest | 20 filings stored, re-runnable without re-downloading |
| 2 (done) | Parsing — HTML→text, Item boundary detection | Can print Item 1A of a real 10-K, correctly bounded |
| 3 (done) | Chunking — semantic split within sections, provenance tags | Correct company/year/item tags, no mid-sentence cuts |
| 4 (done) | FAISS index over the embeddings | "supply chain risk" returns sensible paragraphs |
| 5 (done) | Hybrid retrieval — BM25, RRF, metadata pre-filter | Hybrid measurably beats dense alone |
| 6 (done) | Generation — prompting, citations, numeric grounding check | Answers with verifiable citations; fabricated figures flagged |
| 7 (done) | FastAPI + PostgreSQL schema | `curl` a question, get structured JSON |
| 8 (done) | Evaluation harness — Recall@k, Precision@k, MRR, faithfulness | A metrics table worth quoting |
| 9 | Scale to 10K+ documents, switch to IVF | Corpus target met; p95 latency measured |
| 10 | Packaging and deployment | Runs from a clean machine |

Phase 8 lands before Phase 9 deliberately: a well-evaluated 20-filing system is worth more than an
unevaluated 10,000-filing one, and debugging chunking at scale is miserable.

## Notes on EDGAR

- **10 requests/second per IP.** Exceeding it returns 429 and can get the IP temporarily blocked.
- **`User-Agent` is mandatory** and must identify you with a real email; its absence is the most
  common cause of 403s.
- Raw filings are stored on first download. Chunking strategy will change; re-downloading 10,000
  filings from a rate-limited government server should not have to.
- **BM25 needs question words and `company` in its stopword list.** Users ask "how much did the
  company spend on research and development"; without stopping the grammar, the query's content
  terms are outweighed. `company` looks like a content word but is a filer fingerprint — Apple
  writes "the Company" where others write "we" — so it pulls in one filer's chunks regardless of
  topic. Adding both moved the passages containing "research and development expense" from absent
  from the top 20 into the top 3.
- **RRF's published `k=60` was wrong here.** It weights top ranks gently, favouring chunks both
  retrievers found. "Mine safety disclosures" is BM25's rank-1 hit and absent from dense results
  entirely, and at `k=60` it fell out of the fused top 5. At `k=5` a rank-1 exclusive hit outweighs
  deep agreement. Measured on a 12-query probe: dense 11/12, sparse 12/12, hybrid 12/12 at `k=5`
  versus 11/12 at `k=60`. Phase 8's labelled set should confirm this on more than 12 queries.
- **Token counts must come from the model's tokenizer, not a character estimate.** Filings are
  dense with dollar figures, percentages and tickers that fragment into far more tokens than prose.
  A characters÷4 estimate measured between 0.64x and 1.81x the true count, which put a fifth of the
  corpus over the embedding model's 512-token limit — where the overflow is discarded silently and
  becomes unretrievable.
- **Item headings are surrounded by decoys.** Every filing repeats its item labels in a table of
  contents, in running page headers (Microsoft's 10-K carries a bare `Item 8` about forty times),
  and in body cross-references. Filers also disagree on format: Amazon and Alphabet write
  `Item 1A.Risk Factors` with no space, Microsoft uses all-caps split across nested inline tags.
  Detection therefore requires a title, ends the contents block where item numbering restarts, and
  breaks ties between repeats by section length.
- **Fiscal years are not calendar years.** Apple's ends in September, NVIDIA's in January,
  Microsoft's in June, and a fiscal year is named for the calendar year it *ends* in — so a quarter
  ending December 2025 is Apple's FY2026 Q1. Quarters are derived by distance from fiscal year end
  rather than by counting months, because 52/53-week calendars drift period ends across month
  boundaries (Apple's fiscal Q3 can end on July 1). Mislabelled years defeat the metadata filtering
  that keeps answers on the right filing.

## License

MIT
