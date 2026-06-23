#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon v4 submission pipeline.

Single entry point that chains all three phases:
  1. parse   — candidates.jsonl → flat DataFrames
  2. features — feature engineering + candidate document building
  3. embed   — load precomputed similarities (or compute if missing)
  4. rerank  — composite scoring → top-100
  5. reason  — generate per-candidate reasoning
  6. output  — write CSV + validate

Usage (fast path, using precomputed similarity scores):
  python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \\
                 --out ./team_soloking.csv \\
                 --precomputed-dir ./

Usage (full recompute, ~15+ min):
  python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \\
                 --out ./team_soloking.csv \\
                 --recompute-embeddings
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon v4 — candidate ranking pipeline"
    )
    parser.add_argument(
        "--candidates",
        default="./candidates/candidates.jsonl",
        help="Path to candidates.jsonl (or .jsonl.gz)",
    )
    parser.add_argument(
        "--out",
        default="./team_soloking.csv",
        help="Output CSV path (e.g. team_soloking.csv)",
    )
    parser.add_argument(
        "--precomputed-dir",
        default="./",
        help=(
            "Directory containing precomputed artifacts: "
            "similarities_2s.npy, candidate_documents_v4.csv, "
            "candidate_master_df_1.csv, and processed_csvs/. "
            "Defaults to current directory."
        ),
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Ignore precomputed similarities and recompute from scratch.",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Skip auto-generating the reasoning column.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    t0 = time.time()
    precomputed_dir = Path(args.precomputed_dir)

    print("=" * 60)
    print("Redrob Hackathon v4 — Ranking Pipeline")
    print("=" * 60)
    print(f"Candidates  : {args.candidates}")
    print(f"Output      : {args.out}")
    print(f"Precomputed : {precomputed_dir}")
    print()

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 1: Parse
    # ──────────────────────────────────────────────────────────────────────
    from pipeline.parse import parse_candidates

    parsed_csv_dir = precomputed_dir / "processed_csvs"
    dfs = parse_candidates(
        candidates_path=args.candidates,
        precomputed_dir=str(parsed_csv_dir) if parsed_csv_dir.exists() else None,
    )
    print(f"  [OK] Parse done  ({time.time() - t0:.1f}s)\n")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 2: Feature engineering
    # ──────────────────────────────────────────────────────────────────────
    from pipeline.features import build_features

    master_path = precomputed_dir / "processed_csvs/candidate_master_df_1.csv"
    docs_path = precomputed_dir / "processed_csvs/candidate_documents_v4.csv"

    candidate_master_df, candidate_docs = build_features(
        dfs=dfs,
        precomputed_master=str(master_path) if master_path.exists() else None,
        precomputed_docs=str(docs_path) if docs_path.exists() else None,
    )
    print(f"  [OK] Features done  ({time.time() - t0:.1f}s)\n")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 3: Semantic similarity
    # ──────────────────────────────────────────────────────────────────────
    from pipeline.embed import get_top_candidates, load_or_compute_similarities

    sim_path = precomputed_dir / "processed_csvs/similarities_2s.npy"
    emb_path = precomputed_dir / "processed_csvs/candidate_embedding_2s.npy"

    if args.recompute_embeddings:
        sim_path_arg = None
        emb_path_arg = None
    else:
        sim_path_arg = str(sim_path) if sim_path.exists() else None
        emb_path_arg = str(emb_path) if emb_path.exists() else None

    # The similarities array aligns with candidate_docs row order
    similarities = load_or_compute_similarities(
        candidate_docs=candidate_docs,
        precomputed_sim_path=sim_path_arg,
        precomputed_embeddings_path=emb_path_arg,
    )

    top_candidates = get_top_candidates(candidate_docs, similarities, top_n=3000)
    print(f"  [OK] Semantic retrieval done — top {len(top_candidates):,} shortlisted  ({time.time() - t0:.1f}s)\n")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 4: Rerank
    # ──────────────────────────────────────────────────────────────────────
    from pipeline.rerank import run_rerank

    assessment_df = dfs["skill_assessments"]

    final_top100 = run_rerank(
        top_candidates=top_candidates,
        candidate_master_df=candidate_master_df,
        assessment_df=assessment_df,
    )
    print(f"  [OK] Reranking done — top 100 selected  ({time.time() - t0:.1f}s)\n")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 5: Build submission DataFrame
    # ──────────────────────────────────────────────────────────────────────
    submission_df = final_top100[["candidate_id", "final_score"]].copy()
    submission_df["score"] = submission_df["final_score"]
    submission_df.drop(columns=["final_score"], inplace=True)
    submission_df["rank"] = (
        submission_df["score"].rank(method="first", ascending=False).astype(int)
    )
    submission_df = submission_df.sort_values("rank").reset_index(drop=True)

    # ── Reasoning ──────────────────────────────────────────────────────────
    if not args.no_reasoning:
        from pipeline.reason import add_reasoning_column

        # Pass the full final_top100 (with all feature columns) for richer reasoning
        ranked_full = final_top100.copy()
        ranked_full["rank"] = (
            ranked_full["final_score"].rank(method="first", ascending=False).astype(int)
        )
        ranked_full = ranked_full.sort_values("rank").reset_index(drop=True)

        ranked_full = add_reasoning_column(ranked_full)
        reason_map = ranked_full.set_index("candidate_id")["reasoning"].to_dict()
        submission_df["reasoning"] = submission_df["candidate_id"].map(reason_map).fillna("")
        submission_df = submission_df[["candidate_id", "rank", "score", "reasoning"]]
        print(f"  [OK] Reasoning generated  ({time.time() - t0:.1f}s)\n")
    else:
        submission_df = submission_df[["candidate_id", "rank", "score"]]

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 6: Write CSV + Validate
    # ──────────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    submission_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  [OK] Written to {out_path}  ({time.time() - t0:.1f}s)\n")

    from pipeline.validate import validate_submission

    errors = validate_submission(str(out_path))
    if errors:
        print(f"⚠  Validation FAILED ({len(errors)} issue(s)):\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("  [OK] Validation PASSED — submission is spec-compliant.\n")

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"Output: {out_path.resolve()}")
    print("=" * 60)

    # Print top-10 preview
    print("\nTop-10 preview:")
    print(submission_df.head(10)[["rank", "candidate_id", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
