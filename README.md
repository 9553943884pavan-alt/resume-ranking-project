# Redrob Hackathon v4 — Ranking Pipeline

**Team:** soloking  
**Stack:** Python, pandas, numpy, scikit-learn, sentence-transformers (CPU-only)

---

## Setup

```bash
pip install -r requirements.txt
```

**Python version:** 3.9+  
**Hardware:** CPU only, ≥16 GB RAM

---

## Quick Start (< 5 minutes)

Uses precomputed embeddings already in this repo:

```bash
python rank.py \
  --candidates ./candidates/candidates.jsonl \
  --out ./team_soloking.csv
```

The script auto-detects and loads:
- `similarities_2s.npy` — precomputed cosine similarities  
- `candidate_master_df_1.csv` — precomputed feature matrix  
- `candidate_documents_v4.csv` — precomputed candidate documents  
- `processed_csvs/` — precomputed flat CSVs from candidates.jsonl  

---

## Full Recompute (one-time, ~15+ min)

If you want to regenerate everything from scratch:

```bash
python rank.py \
  --candidates ./candidates/candidates.jsonl \
  --out ./team_soloking.csv \
  --recompute-embeddings
```

> Pre-computation (embedding generation) is allowed to exceed the 5-minute window per the submission spec. Only the ranking step (loading precomputed scores → producing CSV) must complete within 5 minutes.

---

## Architecture

```
candidates.jsonl
    │
    ▼
pipeline/parse.py       # Flatten JSONL → 8 DataFrames
    │
    ▼
pipeline/features.py    # Hard-filter + feature engineering + doc building
    │
    ▼
pipeline/embed.py       # Load similarities_2s.npy (or compute via sentence-transformers)
    │
    ▼
pipeline/rerank.py      # Composite score: semantic + search_evidence + assessment
    │                   #   + behavior + logistics + experience
    ▼
pipeline/reason.py      # Auto-generate per-candidate reasoning
    │
    ▼
team_soloking.csv       # 100 rows: candidate_id, rank, score, reasoning
```

### Scoring Formula (verbatim from notebooks)

```
final_raw_score = 0.30 × semantic_score_norm
                + 0.32 × search_evidence_score
                + 0.10 × assessment_score
                + 0.10 × behavior_score
                + 0.13 × logistics_score
                + 0.05 × experience_score

final_score = final_raw_score × (1.01 if open_to_work else 0.99)
```

---

## Flags

| Flag | Default | Description |
|---|---|---|
| `--candidates` | `./candidates/candidates.jsonl` | Input JSONL path |
| `--out` | `./team_soloking.csv` | Output CSV path |
| `--precomputed-dir` | `./` | Directory with precomputed artifacts |
| `--recompute-embeddings` | off | Force full re-embedding |
| `--no-reasoning` | off | Skip reasoning column generation |

---

## Compute Environment

- Windows 11, Intel i5 11th Gen, 8 GB RAM  
- Python 3.12.10  
- No GPU required or used

---

## Validation

The pipeline runs `validate_submission()` automatically after writing the CSV.
To validate manually:

```bash
python -c "
from pipeline.validate import validate_submission
errors = validate_submission('./team_soloking.csv')
print('PASS' if not errors else errors)
"
```
