"""
embed.py — Text preprocessing and semantic similarity.

Verbatim logic from 02_Semantic_Retrieval.ipynb.
Only change: loads precomputed similarities_2s.npy if available (default),
otherwise computes via sentence-transformers.
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# JD text (verbatim from 02_Semantic_Retrieval.ipynb)
# ---------------------------------------------------------------------------

JD_TEXT = """
TITLE:

Senior Machine Learning Engineer
CAREER TYPE:

Machine Learning
HEADLINE:

Senior ML Engineer | Search, Ranking & Retrieval Systems
SUMMARY:

Machine learning engineer with 7 years of experience, the last 4–5 focused on applied ML at product companies — primarily search, ranking, and recommendation systems, not pure research and not consulting. I've taken production embeddings-based retrieval (sentence-transformers/E5, hybrid search via OpenSearch) from prototype to real users, including handling embedding drift and index refresh, and I've built the offline/online evaluation infrastructure (NDCG, MRR, MAP, A/B testing) to know whether a ranking change actually helped. My retrieval and ranking grounding predates the current LLM wave — I built learning-to-rank systems before embeddings and vector databases were the default — and I've layered in LLM-based re-ranking and light fine-tuning (LoRA/QLoRA) more recently rather than starting there. I write up design decisions publicly (a conference talk, one open-source contribution) rather than only shipping behind closed doors. Distributed systems and large-scale inference optimization are areas I've touched but wouldn't call a core strength.
YEARS OF EXPERIENCE:

7
CURRENT INDUSTRY:

B2B SaaS — Search, Ranking & Recommendation Systems
CAREER HISTORY:

Search and recommendations engineering at an early-stage marketplace startup, pre-LLM era. Built one of the company's first ranking systems end-to-end — BM25-based retrieval plus an XGBoost learning-to-rank layer over it — and the evaluation harness (NDCG, MRR) to validate it before online rollout. This is where I built my retrieval and ranking fundamentals, well before vector databases were standard tooling. Machine learning engineering at a growth-stage B2B SaaS company. Owned production embeddings-based retrieval (sentence-transformers, later E5) and hybrid search over OpenSearch, serving real users at meaningful scale. Handled the unglamorous operational parts — embedding drift, index refresh cadence, retrieval-quality regressions — and set up the offline-to-online evaluation pipeline so ranking changes could be trusted before shipping. Also gave a conference talk on hybrid dense+sparse retrieval and contributed a small fix upstream to an open-source embeddings library. Senior ML engineering at the same company as it scaled. Added an LLM-based re-ranking stage on top of the existing dense retrieval pipeline and did light fine-tuning (LoRA/QLoRA/PEFT) for domain adaptation. Picked up some distributed-systems and inference-optimization work to keep ranking latency acceptable at higher traffic, though infra optimization stays a secondary skill rather than a core strength. Comfortable owning a ranking system from architecture decision through production monitoring.
SEARCH AND RANKING EVIDENCE:

Built and operated production embeddings-based retrieval (E5/sentence-transformers) + hybrid search (OpenSearch) at real user scale, owning embedding drift and index refresh. Designed offline-to-online evaluation framework for ranking (NDCG, MRR, MAP) and used it to validate A/B test results before rollout. Pre-LLM ranking experience (BM25 + XGBoost learning-to-rank) predating embeddings/vector DBs as standard tooling. Recent LLM-based re-ranking layer added on top of existing dense retrieval pipeline. External validation of thinking: conference talk + open-source contribution in the retrieval/embeddings space.
JD RELEVANT SKILLS:

Python, Embeddings (sentence-transformers/E5), Vector Databases (OpenSearch/FAISS), Hybrid Retrieval, Ranking Evaluation (NDCG/MRR/MAP), LoRA/QLoRA/PEFT, Learning-to-Rank (XGBoost), A/B Testing, Distributed Systems
EDUCATION:

Bachelor's/Master's
FIELDS OF STUDY:

Computer Science / Information Retrieval
"""

# ---------------------------------------------------------------------------
# Text preprocessing (verbatim from 02_Semantic_Retrieval.ipynb)
# ---------------------------------------------------------------------------

def preprocess_text(text: str, lower: bool = False) -> str:
    """
    Clean candidate document or JD text before embedding.

    Steps:
    - convert to string
    - normalize unicode
    - remove line breaks / tabs
    - collapse repeated spaces
    - remove weird bullet characters
    - optionally lowercase
    """
    if text is None:
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace newlines / tabs with spaces
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # Remove common bullet symbols
    text = re.sub(r"[•·●▪▶►■]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Optional lowercase
    if lower:
        text = text.lower()

    return text


# ---------------------------------------------------------------------------
# Similarity loading / computation
# ---------------------------------------------------------------------------

def load_or_compute_similarities(
    candidate_docs: pd.DataFrame,
    precomputed_sim_path: Optional[str] = None,
    precomputed_embeddings_path: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Return cosine similarity array (shape: [n_candidates]) between each
    candidate document and the JD.

    Priority:
    1. Load from precomputed_sim_path (.npy) — fastest, use for ranking step
    2. Load candidate embeddings from precomputed_embeddings_path + embed JD
    3. Embed everything from scratch (slowest, ~10–15 min)
    """
    # Option 1: pre-computed similarities
    if precomputed_sim_path and Path(precomputed_sim_path).exists():
        print(f"[embed] Loading precomputed similarities from {precomputed_sim_path}")
        similarities = np.load(precomputed_sim_path)
        if len(similarities) != len(candidate_docs):
            raise ValueError(
                f"Similarity array length {len(similarities)} does not match "
                f"candidate_docs length {len(candidate_docs)}. "
                f"Re-run with --recompute-embeddings to regenerate."
            )
        return similarities

    # Option 2: precomputed candidate embeddings
    if precomputed_embeddings_path and Path(precomputed_embeddings_path).exists():
        print(f"[embed] Loading precomputed candidate embeddings from {precomputed_embeddings_path}")
        candidate_embeddings = np.load(precomputed_embeddings_path)

        print("[embed] Embedding JD ...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        jd_clean = preprocess_text(JD_TEXT, lower=True)
        jd_embedding = model.encode([jd_clean], normalize_embeddings=True, show_progress_bar=False)

        # Normalize candidate embeddings if needed
        norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        candidate_embeddings = candidate_embeddings / norms

        similarities = (candidate_embeddings @ jd_embedding.T).flatten()
        return similarities

    # Option 3: compute from scratch
    print("[embed] Computing embeddings from scratch (this may take 10–15 minutes) ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    print("[embed] Preprocessing candidate documents ...")
    docs_clean = candidate_docs["candidate_document"].apply(
        lambda x: preprocess_text(x, lower=True)
    ).tolist()

    jd_clean = preprocess_text(JD_TEXT, lower=True)

    print(f"[embed] Encoding {len(docs_clean):,} candidate documents ...")
    candidate_embeddings = model.encode(
        docs_clean,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    print("[embed] Encoding JD ...")
    jd_embedding = model.encode(
        [jd_clean],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarities = (candidate_embeddings @ jd_embedding.T).flatten()
    return similarities


def get_top_candidates(
    candidate_docs: pd.DataFrame,
    similarities: np.ndarray,
    top_n: int = 3000,
) -> pd.DataFrame:
    """
    Attach similarity scores to candidate_docs and return top-N by semantic score.
    (verbatim from 02_Semantic_Retrieval.ipynb)
    """
    top_candidates = (
        candidate_docs
        .assign(semantic_score=similarities)
        .sort_values("semantic_score", ascending=False)
        .head(top_n)
    )
    return top_candidates
