"""
rerank.py — Composite scoring and final top-100 selection.

All scoring functions are verbatim from 02_Semantic_Retrieval.ipynb.
Zero changes to weights, thresholds, or logic.
"""

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Search evidence scoring (verbatim from 02_Semantic_Retrieval.ipynb)
# ---------------------------------------------------------------------------

SEARCH_TERMS = {
    # Tier 1: direct production retrieval / ranking evidence
    "search": 8,
    "ranking": 8,
    "retrieval": 8,
    "recommendation": 7,
    "recommender": 7,
    "relevance": 7,
    "semantic search": 8,
    "information retrieval": 9,
    "information retrieval systems": 9,
    "learning to rank": 10,
    "ltr": 8,
    "candidate matching": 7,
    "hybrid search": 9,
    "vector search": 9,
    "search ranking": 9,
    "retrieval system": 9,
    "ranking system": 9,

    # Tier 2: evaluation / experimentation
    "ndcg": 10,
    "mrr": 10,
    "map": 8,
    "offline-online": 10,
    "offline to online": 10,
    "a/b": 8,
    "a/b testing": 8,
    "ab test": 8,
    "experiment": 5,
    "evaluation framework": 9,
    "relevance metrics": 8,
    "retrieval quality": 9,

    # Tier 3: infra / production search stack
    "embedding": 7,
    "embeddings": 7,
    "vector database": 8,
    "vector db": 8,
    "index refresh": 8,
    "embedding drift": 9,
    "faiss": 8,
    "pinecone": 8,
    "weaviate": 8,
    "qdrant": 8,
    "milvus": 8,
    "opensearch": 8,
    "elasticsearch": 8,
    "bm25": 9,
    "open search": 7,

    # Tier 4: modern ML/LLM terms, lower weight here
    "llm": 3,
    "llms": 3,
    "rag": 4,
    "langchain": 1,
    "llamaindex": 1,
    "prompt engineering": 1,
    "fine-tuning": 3,
    "fine tuning": 3,
    "lora": 2,
    "qlora": 2,
    "peft": 2,
}

NEGATIVE_TERMS = {
    "tutorial": -4,
    "demo": -3,
    "toy": -3,
    "proof of concept": -2,
    "openai api call": -5,
    "langchain tutorial": -6,
    "chatgpt wrapper": -6,
}


def search_evidence_score(text) -> int:
    """
    Weighted evidence score for production search / ranking relevance.
    Returns raw integer score (scaled separately via MinMaxScaler).
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    score = 0

    # weighted phrase hits
    for term, weight in SEARCH_TERMS.items():
        hits = text.count(term)
        if hits:
            # diminishing returns per repeated mention
            score += weight * min(hits, 2)

    # direct strong signals from production / shipped work
    strong_prod_terms = [
        "deployed to real users",
        "in production",
        "shipped",
        "production system",
        "real users",
        "handled embedding drift",
        "index refresh",
        "retrieval quality regression",
        "offline-online correlation",
        "a/b testing",
        "learning-to-rank",
    ]
    for term in strong_prod_terms:
        if term in text:
            score += 8

    # penalize demo-only / framework-only language
    for term, penalty in NEGATIVE_TERMS.items():
        if term in text:
            score += penalty

    return int(score)


# ---------------------------------------------------------------------------
# Experience score for reranking (verbatim from 02_Semantic_Retrieval.ipynb)
# ---------------------------------------------------------------------------

def experience_score(exp) -> float:
    """
    Reranking experience score.
    Range: 0-100
    Purpose: Peak around 6-8 years (ideal JD fit)
    """
    if pd.isna(exp):
        return 0

    exp = float(exp)

    if exp < 3:
        return 0
    elif 3 <= exp < 4:
        return 20
    elif 4 <= exp < 5:
        return 60
    elif 5 <= exp < 6:
        return 90
    elif 6 <= exp <= 8:
        return 100
    elif 8 < exp <= 9:
        return 90
    elif 9 < exp <= 10:
        return 70
    elif 10 < exp <= 12:
        return 40
    elif 12 < exp <= 15:
        return 10
    else:  # >15 years
        return 0


# ---------------------------------------------------------------------------
# Behavior score (verbatim from 02_Semantic_Retrieval.ipynb)
# ---------------------------------------------------------------------------

def behavior_score(row) -> float:
    """
    Single composite Redrob signal score.
    Range: 0-100

    Combines: availability, recruiter responsiveness, market validation,
    trust / identity, technical credibility, logistics, compensation realism.
    """

    def clamp(x, lo=0.0, hi=100.0):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return lo
        return max(lo, min(hi, float(x)))

    def bool_score(v, points_true):
        return points_true if bool(v) else 0.0

    def log_bonus(v, cap, points):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0.0
        v = float(v)
        if v <= 0:
            return 0.0
        return min(np.log1p(v) / np.log1p(cap) * points, points)

    def recency_score(date_val):
        if pd.isna(date_val) or date_val is None or str(date_val).strip() == "":
            return 0.0
        try:
            d = pd.to_datetime(date_val, errors="coerce")
            if pd.isna(d):
                return 0.0
            days = (pd.Timestamp.utcnow().normalize() - d.tz_localize(None).normalize()).days
        except Exception:
            return 0.0

        if days <= 7:
            return 8.0
        if days <= 30:
            return 7.0
        if days <= 60:
            return 5.0
        if days <= 90:
            return 2.5
        if days <= 180:
            return 0.5
        return 0.0

    def response_time_score(hours):
        if pd.isna(hours) or hours is None:
            return 0.0
        try:
            h = float(hours)
        except Exception:
            return 0.0

        if h <= 6:
            return 8.0
        if h <= 24:
            return 6.0
        if h <= 48:
            return 4.0
        if h <= 72:
            return 2.0
        return 0.0

    def notice_score(days):
        if pd.isna(days) or days is None:
            return 0.0
        try:
            d = float(days)
        except Exception:
            return 0.0

        if d <= 30:
            return 8.0
        if d <= 60:
            return 6.0
        if d <= 90:
            return 3.0
        if d <= 120:
            return 1.0
        return 0.0

    def work_mode_score(mode):
        mode = str(mode).strip().lower() if mode is not None and not pd.isna(mode) else ""
        if mode in {"hybrid", "flexible"}:
            return 2.0
        if mode == "onsite":
            return 1.5
        if mode == "remote":
            return 1.0
        return 0.0

    def salary_reasonableness(min_sal, max_sal):
        if pd.isna(min_sal) or pd.isna(max_sal) or min_sal is None or max_sal is None:
            return 0.0
        try:
            mn = float(min_sal)
            mx = float(max_sal)
        except Exception:
            return 0.0

        if mn < 0 or mx < 0:
            return -3.0
        if mn > mx:
            return -5.0

        spread = mx - mn
        if spread <= 10:
            return 2.0
        if spread <= 20:
            return 1.5
        if spread <= 35:
            return 1.0
        return 0.5

    score = 0.0

    # 1) Availability / responsiveness (0-45)
    score += bool_score(row.get("open_to_work_flag"), 10.0)
    score += clamp(row.get("recruiter_response_rate"), 0.0, 1.0) * 16.0
    score += response_time_score(row.get("avg_response_time_hours"))
    score += recency_score(row.get("last_active_date"))
    score += clamp(row.get("interview_completion_rate"), 0.0, 1.0) * 8.0

    offer_rate = row.get("offer_acceptance_rate")
    if not pd.isna(offer_rate) and offer_rate != -1:
        score += clamp(offer_rate, 0.0, 1.0) * 3.0

    # 2) Market validation / recruiter interest (0-20)
    score += log_bonus(row.get("saved_by_recruiters_30d"), cap=30, points=8.0)
    score += log_bonus(row.get("search_appearance_30d"), cap=120, points=6.0)
    score += log_bonus(row.get("profile_views_received_30d"), cap=120, points=4.0)
    score += log_bonus(row.get("applications_submitted_30d"), cap=20, points=2.0)

    # 3) Trust / profile hygiene (0-10)
    score += clamp(row.get("profile_completeness_score"), 0.0, 100.0) / 100.0 * 3.0
    score += bool_score(row.get("verified_email"), 1.5)
    score += bool_score(row.get("verified_phone"), 2.0)
    score += bool_score(row.get("linkedin_connected"), 1.5)

    signup_date = row.get("signup_date")
    if not pd.isna(signup_date) and signup_date is not None and str(signup_date).strip() != "":
        try:
            sd = pd.to_datetime(signup_date, errors="coerce")
            if not pd.isna(sd):
                account_age_days = (pd.Timestamp.utcnow().normalize() - sd.tz_localize(None).normalize()).days
                if account_age_days >= 365:
                    score += 2.0
                elif account_age_days >= 180:
                    score += 1.5
                elif account_age_days >= 90:
                    score += 1.0
                else:
                    score += 0.5
        except Exception:
            pass

    # 4) Technical credibility / social proof (0-10)
    github = row.get("github_activity_score")
    if not pd.isna(github) and github is not None and float(github) >= 0:
        score += clamp(github, 0.0, 100.0) / 100.0 * 6.0

    score += log_bonus(row.get("endorsements_received"), cap=50, points=2.0)
    score += log_bonus(row.get("connection_count"), cap=200, points=2.0)

    # 5) Logistics / compensation realism (0-15)
    score += notice_score(row.get("notice_period_days"))
    score += bool_score(row.get("willing_to_relocate"), 4.0)
    score += work_mode_score(row.get("preferred_work_mode"))
    score += salary_reasonableness(
        row.get("expected_salary_min_inr_lpa"),
        row.get("expected_salary_max_inr_lpa"),
    )

    return round(score, 2)


# ---------------------------------------------------------------------------
# Logistics score (verbatim from 02_Semantic_Retrieval.ipynb)
# ---------------------------------------------------------------------------

PREFERRED_CITIES = {"pune", "noida"}

WELCOME_CITIES = {"delhi", "delhi ncr", "gurgaon", "gurugram", "mumbai", "hyderabad"}


def logistics_score(row) -> float:
    country = str(row["country"]).lower().strip()
    location = str(row["location"]).lower().strip()
    notice = row["notice_period_days"]
    willing = bool(row["willing_to_relocate"])

    # Location fit
    if country == "india":
        if any(city in location for city in PREFERRED_CITIES):
            location_fit = 100
        elif any(city in location for city in WELCOME_CITIES):
            location_fit = 80
        else:
            location_fit = 60
    else:
        if willing:
            location_fit = 35
        else:
            location_fit = 0

    # Notice fit
    if pd.isna(notice):
        notice_fit = 50
    elif notice <= 30:
        notice_fit = 100
    elif notice <= 60:
        notice_fit = 75
    elif notice <= 90:
        notice_fit = 50
    else:
        notice_fit = 0

    final_score = 0.55 * location_fit + 0.45 * notice_fit
    return round(final_score, 2)


# ---------------------------------------------------------------------------
# Assessment score (verbatim from 02_Semantic_Retrieval.ipynb)
# ---------------------------------------------------------------------------

ASSESSMENT_WEIGHTS = {
    # ===== CORE JD SKILLS =====
    "Information Retrieval": 10,
    "Learning to Rank": 10,
    "Recommendation Systems": 10,
    "Semantic Search": 10,
    "Embeddings": 10,
    "Vector Search": 10,
    "BM25": 10,

    # ===== VECTOR DATABASES =====
    "FAISS": 8,
    "Milvus": 8,
    "Qdrant": 8,
    "Pinecone": 8,
    "Weaviate": 8,
    "OpenSearch": 8,
    "Elasticsearch": 8,

    # ===== NLP / RETRIEVAL =====
    "Sentence Transformers": 7,
    "NLP": 7,
    "LLMs": 7,
    "RAG": 7,
    "Hugging Face Transformers": 7,
    "LlamaIndex": 7,
    "Haystack": 7,
    "LangChain": 5,

    # ===== STRONG ENGINEERING =====
    "Python": 6,
    "PyTorch": 5,
    "TensorFlow": 5,
    "Machine Learning": 5,
    "Deep Learning": 4,
    "scikit-learn": 4,

    # ===== NICE TO HAVE =====
    "Fine-tuning LLMs": 3,
    "LoRA": 3,
    "QLoRA": 3,
    "PEFT": 3,

    # ===== PRODUCTION ML =====
    "MLOps": 2,
    "MLflow": 2,
    "BentoML": 2,
    "Kubeflow": 2,
    "Weights & Biases": 2,
    "Feature Engineering": 2,
}


def calculate_assessment_score(candidate_group) -> float:
    weighted_sum = 0
    total_weight = 0
    matched_skills = 0

    for _, row in candidate_group.iterrows():
        skill = row["skill_name"]
        if skill not in ASSESSMENT_WEIGHTS:
            continue

        weight = ASSESSMENT_WEIGHTS[skill]
        weighted_sum += row["assessment_score"] * weight
        total_weight += weight
        matched_skills += 1

    if total_weight == 0:
        return 0

    avg_score = weighted_sum / total_weight

    # reward breadth
    coverage_bonus = min(matched_skills * 2, 20)

    return avg_score + coverage_bonus


# ---------------------------------------------------------------------------
# Main reranking function
# ---------------------------------------------------------------------------

def run_rerank(
    top_candidates: pd.DataFrame,
    candidate_master_df: pd.DataFrame,
    assessment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply all composite scoring and return final top-100 DataFrame.
    Verbatim scoring weights from 02_Semantic_Retrieval.ipynb.
    """
    rerank_df = top_candidates.copy()

    # Merge with master features
    master_cols_to_drop = ["hard_reject", "hard_filter_score"]
    master_slim = candidate_master_df.drop(
        columns=[c for c in master_cols_to_drop if c in candidate_master_df.columns],
        errors="ignore",
    )
    rerank_df = rerank_df.merge(master_slim, on="candidate_id", how="left")

    # ── Search evidence score ───────────────────────────────────────────────
    rerank_df["search_evidence_score"] = rerank_df["career_text"].apply(search_evidence_score)

    scaler = MinMaxScaler()
    rerank_df["search_evidence_score"] = (
        scaler.fit_transform(rerank_df[["search_evidence_score"]])
    ) * 100

    # ── Experience score ────────────────────────────────────────────────────
    rerank_df["experience_score"] = rerank_df["years_of_experience"].apply(experience_score)

    # ── Behavior score ──────────────────────────────────────────────────────
    rerank_df["behavior_score"] = rerank_df.apply(behavior_score, axis=1)

    # ── Logistics score ─────────────────────────────────────────────────────
    rerank_df["logistics_score"] = rerank_df.apply(logistics_score, axis=1)

    # ── Semantic score normalisation ────────────────────────────────────────
    semantic_scaler = MinMaxScaler()
    rerank_df["semantic_score_norm"] = (
        semantic_scaler.fit_transform(rerank_df[["semantic_score"]]) * 100
    )

    hard_filter_scaler = MinMaxScaler()
    rerank_df["hard_filter_score_norm"] = (
        hard_filter_scaler.fit_transform(rerank_df[["hard_filter_score"]]) * 100
    )

    # ── Assessment score ────────────────────────────────────────────────────
    rerank_df = rerank_df.drop(
        columns=["assessment_score", "assessment_score_old"], errors="ignore"
    )

    assessment_scores = (
        assessment_df
        .groupby("candidate_id")
        .apply(calculate_assessment_score)
        .reset_index(name="assessment_score")
    )

    rerank_df = rerank_df.merge(assessment_scores, on="candidate_id", how="left")
    rerank_df["assessment_score"] = rerank_df["assessment_score"].fillna(0)

    # ── Final composite (verbatim weights) ──────────────────────────────────
    rerank_df["final_raw_score"] = (
        0.30 * rerank_df["semantic_score_norm"] +
        0.32 * rerank_df["search_evidence_score"] +
        0.10 * rerank_df["assessment_score"] +
        0.10 * rerank_df["behavior_score"] +
        0.13 * rerank_df["logistics_score"] +
        0.05 * rerank_df["experience_score"]
    )

    rerank_df["final_score"] = np.where(
        rerank_df["open_to_work_flag"],
        rerank_df["final_raw_score"] * 1.01,
        rerank_df["final_raw_score"] * 0.99,
    )

    # ── Top-100 selection ───────────────────────────────────────────────────
    final_top100 = (
        rerank_df
        .sort_values("final_score", ascending=False)
        .reset_index(drop=True)
        .head(100)
    )

    return final_top100
