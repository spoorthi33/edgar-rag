"""CLI for Phase 8 evaluation.

python scripts/evaluate.py --validate            # check labels only, free
python scripts/evaluate.py --no-judge            # retrieval metrics only, free
python scripts/evaluate.py                       # full run
python scripts/evaluate.py --compare             # hybrid vs dense vs sparse
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from edgar_rag.config import get_settings
from edgar_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from edgar_rag.eval.dataset import EvalDataset, validate_against_corpus
from edgar_rag.eval.harness import EvalHarness, EvalReport
from edgar_rag.generation.anthropic_client import AnthropicClient
from edgar_rag.generation.budget import BudgetExceeded
from edgar_rag.generation.pipeline import AnswerPipeline, get_llm_client
from edgar_rag.index.builder import load_retrievers
from edgar_rag.retrieval.hybrid import HybridRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation harness")
    parser.add_argument("--dataset", type=Path, default=get_settings().eval_dataset_path)
    parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--compare", action="store_true", help="run all three retrieval modes")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="score retrieval without generating answers (no API calls)",
    )
    parser.add_argument("--validate", action="store_true", help="check labels and exit")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    settings = get_settings()

    dataset = EvalDataset.load(args.dataset)
    if args.limit:
        dataset.questions = dataset.questions[: args.limit]

    embedder = SentenceTransformerEmbedder(settings=settings)
    index, sparse = load_retrievers(settings=settings, embedder=embedder)

    problems = validate_against_corpus(dataset, index.chunks)
    if problems:
        print("label problems (a rule matching nothing is a broken label, not a miss):")
        for kind, ids in problems.items():
            print(f"  {kind}: {', '.join(ids)}")
        if args.validate:
            return 1
    elif args.validate:
        print(f"{len(dataset)} questions, all labels resolve against the corpus")
        return 0

    # The judge runs on every question of every re-run, so it uses the
    # cheaper model: verifying that text supports text is not the same
    # task as answering.
    judge = (
        None
        if args.no_judge or args.retrieval_only
        else AnthropicClient(model=settings.eval_judge_model, settings=settings)
    )

    harness = EvalHarness(
        pipeline=AnswerPipeline(
            retriever=HybridRetriever(
                index=index, embedder=embedder, sparse=sparse, settings=settings
            ),
            llm=get_llm_client(settings),
            settings=settings,
        ),
        chunks=index.chunks,
        judge_llm=judge,
        settings=settings,
    )

    modes = ["dense", "sparse", "hybrid"] if args.compare else [args.mode]
    reports: list[EvalReport] = []

    try:
        for mode in modes:
            print(f"\nrunning {len(dataset)} questions in {mode} mode...")
            reports.append(
                harness.run(
                    dataset,
                    mode=mode,
                    top_k=args.top_k,
                    use_judge=not args.no_judge,
                    retrieval_only=args.retrieval_only,
                )
            )
    except BudgetExceeded as exc:
        print(f"\nstopped: {exc}")
        return 1

    _report(reports, args.out)
    return 0


def _report(reports: list[EvalReport], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([r.summary() for r in reports])

    print("\n" + "=" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=" * 78)
    print(
        "\nrecall@k is bounded by k/|relevant| when a question has many relevant\n"
        "chunks; achievable_recall@k divides by min(k, |relevant|) instead."
    )

    for report in reports:
        frame = report.frame()
        path = out / f"results_{report.mode}.csv"
        frame.to_csv(path, index=False)

        print(f"\n--- {report.mode} by category ---")
        by_category = (
            frame[frame["answerable"]]
            .groupby("category")
            .agg(
                {"hit@k": "mean", "achievable_recall@k": "mean", "mrr": "mean", "faithful": "mean"}
            )
        )
        print(by_category.to_string(float_format=lambda v: f"{v:.3f}"))

        failures = frame[~frame["faithful"] | (frame["answerable"] & (frame["hit@k"] == 0))]
        if not failures.empty:
            print(f"\n  {len(failures)} question(s) needing attention:")
            for _, row in failures.iterrows():
                reason = "no relevant chunk retrieved" if row["hit@k"] == 0 else "unfaithful answer"
                print(f"    {row['id']:12} {reason}")

    summary.to_csv(out / "summary.csv", index=False)
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    sys.exit(main())
