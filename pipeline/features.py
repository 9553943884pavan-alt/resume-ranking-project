"""
features.py — Feature engineering, hard-filtering, and candidate document building.

Verbatim logic from Resume_Project.ipynb.
Only change: functions accept DataFrames as arguments instead of reading from CSV.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Skill / Title taxonomy sets (verbatim from Resume_Project.ipynb)
# ---------------------------------------------------------------------------

TITLE_KEEP = {
    "ml engineer",
    "junior ml engineer",
    "ai research engineer",
    "senior software engineer (ml)",
    "data scientist",
    "ai specialist",
    "machine learning engineer",
    "search engineer",
    "recommendation systems engineer",
    "ai engineer",
    "applied ml engineer",
    "nlp engineer",
    "senior data scientist",
    "senior machine learning engineer",
    "staff machine learning engineer",
    "senior nlp engineer",
    "senior applied scientist",
    "lead ai engineer",
    "senior ai engineer",
    "senior ml engineer — search & ranking",
    "senior ml engineer - search & ranking",
}

TITLE_CONDITIONAL = {
    "software engineer",
    "full stack developer",
    "java developer",
    "cloud engineer",
    "mobile developer",
    ".net developer",
    "qa engineer",
    "devops engineer",
    "frontend engineer",
    "data engineer",
    "analytics engineer",
    "data analyst",
    "senior data engineer",
    "backend engineer",
    "senior software engineer",
}

TITLE_REJECT = {
    "business analyst",
    "graphic designer",
    "project manager",
    "mechanical engineer",
    "accountant",
    "civil engineer",
    "hr manager",
    "customer support",
    "operations manager",
    "marketing manager",
    "content writer",
    "sales executive",
}

RETRIEVAL_SKILLS = {
    "embeddings",
    "sentence transformers",
    "semantic search",
    "information retrieval",
    "information retrieval systems",
    "search backend",
    "search infrastructure",
    "search & discovery",
    "ranking systems",
    "learning to rank",
    "recommendation systems",
    "bm25",
    "haystack",
    "content matching",
    "indexing algorithms",
    "vector search",
}

VECTOR_SKILLS = {
    "faiss",
    "milvus",
    "qdrant",
    "pinecone",
    "weaviate",
    "pgvector",
    "opensearch",
    "elasticsearch",
}

NLP_LLM_SKILLS = {
    "nlp",
    "natural language processing",
    "llms",
    "hugging face transformers",
    "sentence transformers",
    "text encoders",
    "rag",
    "fine-tuning llms",
    "lora",
    "qlora",
    "peft",
    "model adaptation",
    "prompt engineering",
}

PROD_ML_SKILLS = {
    "python",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "mlflow",
    "mlops",
    "docker",
    "kubernetes",
    "airflow",
    "kafka",
    "ci/cd",
    "fastapi",
    "flask",
    "django",
    "workflow orchestration",
    "feature engineering",
    "data pipelines",
    "etl",
}

MANDATORY_CAREER_EVIDENCE = {
    "retrieval",
    "ranking",
    "recommendation",
    "search",
    "embedding",
    "embeddings",
    "vector",
    "semantic",
    "faiss",
    "qdrant",
    "milvus",
    "pinecone",
    "elasticsearch",
    "opensearch",
    "bm25",
    "learning to rank",
    "ndcg",
    "mrr",
    "map",
    "a/b test",
    "production",
    "deployed",
    "shipped",
    "real users",
    "at scale",
    "nlp",
    "transformers",
    "llm",
    "llms",
}

JD_RELEVANT_SKILLS = {
    "Python",
    "Embeddings",
    "Sentence Transformers",
    "Semantic Search",
    "Information Retrieval",
    "Information Retrieval Systems",
    "Vector Search",
    "FAISS",
    "Milvus",
    "Qdrant",
    "Weaviate",
    "OpenSearch",
    "Elasticsearch",
    "BM25",
    "Ranking Systems",
    "Learning to Rank",
    "Recommendation Systems",
    "Search Backend",
    "Search Infrastructure",
    "Search & Discovery",
    "NLP",
    "Natural Language Processing",
    "LLMs",
    "RAG",
    "PyTorch",
    "TensorFlow",
    "Hugging Face Transformers",
    "LoRA",
    "QLoRA",
    "PEFT",
    "FastAPI",
    "Docker",
    "Kubernetes",
    "MLflow",
    "MLOps",
}

EVIDENCE_TERMS = [
    "search",
    "retrieval",
    "ranking",
    "recommendation",
    "recommender",
    "embedding",
    "embeddings",
    "vector",
    "faiss",
    "qdrant",
    "milvus",
    "pinecone",
    "elasticsearch",
    "opensearch",
    "bm25",
    "ndcg",
    "mrr",
    "information retrieval",
    "semantic search",
    "rag",
    "llm",
    "transformer",
    "a/b",
    "ab test",
    "a/b testing",
    "offline",
    "online",
    "relevance",
    "click-through",
    "ctr",
    "dwell time",
    "recommendation",
    "recommender",
]

# ---------------------------------------------------------------------------
# Scoring functions (verbatim from Resume_Project.ipynb)
# ---------------------------------------------------------------------------

def title_score(title: str) -> int:
    t = str(title).strip().lower()
    if t in TITLE_KEEP:
        return 18
    if t in TITLE_CONDITIONAL:
        return 8
    if t in TITLE_REJECT:
        return -12
    return 0


def skill_score(candidate_id: str, candidate_skill_meta: dict) -> int:
    rows = candidate_skill_meta.get(candidate_id, [])
    score = 0
    for s in rows:
        name = str(s.get("skill_name_norm", "")).strip().lower()
        prof = str(s.get("proficiency", "")).strip().lower()
        dur = s.get("duration_months") or 0
        end = s.get("endorsements") or 0

        trusted = (
            (prof in {"advanced", "expert"} and dur >= 12 and end >= 5) or
            (prof == "intermediate" and dur >= 6 and end >= 2)
        )
        if not trusted:
            continue

        if name in RETRIEVAL_SKILLS:
            score += 6
        elif name in VECTOR_SKILLS:
            score += 5
        elif name in NLP_LLM_SKILLS:
            score += 4
        elif name in PROD_ML_SKILLS:
            score += 2

    return min(score, 25)


def career_evidence_score(candidate_id: str, career_text_map: dict) -> int:
    text = career_text_map.get(candidate_id, "")
    hits = sum(1 for term in MANDATORY_CAREER_EVIDENCE if term in text)
    return min(hits * 3, 30)


def experience_score_hard(exp) -> int:
    """
    Hard-filter experience score. Range: 0-10
    (verbatim from Resume_Project.ipynb, used for filtering stage)
    """
    if pd.isna(exp):
        return 0
    exp = float(exp)
    if exp < 3:
        return 0
    elif 3 <= exp < 4:
        return 2
    elif 4 <= exp < 5:
        return 5
    elif 5 <= exp < 6:
        return 8
    elif 6 <= exp <= 8:
        return 10
    elif 8 < exp <= 9:
        return 8
    elif 9 < exp <= 10:
        return 5
    elif 10 < exp <= 12:
        return 2
    else:  # >12 years
        return 0


def has_relevant_skills(candidate_id: str, candidate_skill_sets: dict) -> bool:
    skills = candidate_skill_sets.get(candidate_id, set())
    relevant = RETRIEVAL_SKILLS | VECTOR_SKILLS | NLP_LLM_SKILLS | PROD_ML_SKILLS
    return len(skills & relevant) > 0


def has_career_evidence(candidate_id: str, career_text_map: dict) -> bool:
    text = career_text_map.get(candidate_id, "")
    return any(term in text for term in MANDATORY_CAREER_EVIDENCE)


def hard_reject(row, candidate_skill_sets: dict, career_text_map: dict) -> bool:
    cid = row["candidate_id"]
    title = str(row["current_title"]).strip().lower()
    exp = float(row["years_of_experience"]) if pd.notna(row["years_of_experience"]) else 0

    relevant_skills = has_relevant_skills(cid, candidate_skill_sets)
    career_evidence = has_career_evidence(cid, career_text_map)

    # Rule 1: Non-tech titles need proof
    if title in TITLE_REJECT:
        if not career_evidence:
            return True

    # Rule 2: Very junior candidates need proof
    if exp < 3:
        if not career_evidence:
            return True

    # Rule 3: No AI/Search skills + no evidence
    if not relevant_skills and not career_evidence and title not in TITLE_KEEP:
        return True

    return False


# ---------------------------------------------------------------------------
# Document building functions (verbatim from Resume_Project.ipynb)
# ---------------------------------------------------------------------------

def get_relevant_skills(skill_text: str) -> list:
    skills = [s.strip() for s in str(skill_text).split("|")]
    return [s for s in skills if s in JD_RELEVANT_SKILLS]


def extract_evidence(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    evidence = []
    sentences = text.replace("\n", " ").split(".")
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(term in sentence_lower for term in EVIDENCE_TERMS):
            evidence.append(sentence.strip())
    return " ".join(evidence[:15])


def build_career_type(row) -> str:
    titles = str(row.get("career_titles", "")).lower()
    career_types = []

    if "search" in titles:
        career_types.append("Search Engineering")
    if "recommendation" in titles:
        career_types.append("Recommendation Systems")
    if "machine learning" or "ml" in titles:
        career_types.append("Machine Learning")
    if "nlp" in titles:
        career_types.append("Natural Language Processing")
    if "ai" in titles:
        career_types.append("Artificial Intelligence")
    if "backend" in titles:
        career_types.append("Backend Engineering")
    if "data engineer" in titles:
        career_types.append("Data Engineering")
    if "analytics" in titles:
        career_types.append("Analytics Engineering")
    if "frontend" in titles:
        career_types.append("Frontend Engineering")
    if "qa" in titles:
        career_types.append("Quality Assurance")

    return ", ".join(career_types)


def build_candidate_document(row) -> str:
    relevant_skills = get_relevant_skills(row.get("all_skills", ""))
    evidence = extract_evidence(row.get("career_text", ""))
    career_type = build_career_type(row)

    doc = f"""
TITLE:
{row.get('current_title', '')}

CAREER TYPE:
{career_type}

HEADLINE:
{row.get('headline', '')}

SUMMARY:
{row.get('summary', '')}

YEARS OF EXPERIENCE:
{row.get('years_of_experience', '')}

CURRENT INDUSTRY:
{row.get('current_industry', '')}

CAREER HISTORY:
{str(row.get('career_text', ''))[:3500]}

SEARCH AND RANKING EVIDENCE:
{evidence}

JD RELEVANT SKILLS:
{', '.join(relevant_skills)}

EDUCATION:
{row.get('education_text', '')}

FIELDS OF STUDY:
{row.get('field_of_study_text', '')}
"""
    return doc.strip()


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------

def build_features(
    dfs: Dict[str, pd.DataFrame],
    precomputed_master: Optional[str] = None,
    precomputed_docs: Optional[str] = None,
) -> tuple:
    """
    Run feature engineering pipeline.

    Args:
        dfs: dict of DataFrames from parse.parse_candidates()
        precomputed_master: path to candidate_master_df_1.csv if already built
        precomputed_docs: path to candidate_documents_v4.csv if already built

    Returns:
        (candidate_master_df, candidate_docs_df)
    """
    # Fast path: both precomputed files exist
    if precomputed_master and precomputed_docs:
        import os
        if os.path.exists(precomputed_master) and os.path.exists(precomputed_docs):
            print(f"[features] Loading precomputed master from {precomputed_master}")
            print(f"[features] Loading precomputed docs from {precomputed_docs}")
            master_df = pd.read_csv(precomputed_master, low_memory=False)
            docs_df = pd.read_csv(precomputed_docs, low_memory=False)
            return master_df, docs_df

    print("[features] Building candidate master DataFrame ...")

    profile_df = dfs["profile"]
    skills_df = dfs["skills"]
    career_df = dfs["career"]
    education_df = dfs["education"]
    certifications_df = dfs["certifications"]
    signals_df = dfs["signals"]

    # ── Normalize text columns ──────────────────────────────────────────────
    profile_df = profile_df.copy()
    profile_df["current_title_norm"] = profile_df["current_title"].fillna("").str.strip().str.lower()

    skills_df = skills_df.copy()
    skills_df["skill_name_norm"] = skills_df["skill_name"].fillna("").str.strip().str.lower()

    career_df = career_df.copy()
    career_df["description_norm"] = career_df["description"].fillna("").str.strip().str.lower()
    career_df["title_norm"] = career_df["title"].fillna("").str.strip().str.lower()
    career_df["company_norm"] = career_df["company"].fillna("").str.strip().str.lower()
    career_df["industry_norm"] = career_df["industry"].fillna("").str.strip().str.lower()

    # ── Build lookup structures ─────────────────────────────────────────────
    candidate_skill_sets = (
        skills_df.groupby("candidate_id")["skill_name_norm"]
        .apply(set)
        .to_dict()
    )

    candidate_skill_meta = (
        skills_df.groupby("candidate_id")[["skill_name_norm", "proficiency", "endorsements", "duration_months"]]
        .apply(lambda g: g.to_dict("records"))
        .to_dict()
    )

    career_text_df = (
        career_df.groupby("candidate_id", as_index=False)
        .agg({
            "description_norm": lambda x: " ".join(x),
            "title_norm": lambda x: " ".join(x),
            "company_norm": lambda x: " ".join(x),
            "industry_norm": lambda x: " ".join(x),
            "duration_months": "sum",
        })
    )
    career_text_df["career_text"] = (
        career_text_df["title_norm"].fillna("") + " " +
        career_text_df["company_norm"].fillna("") + " " +
        career_text_df["industry_norm"].fillna("") + " " +
        career_text_df["description_norm"].fillna("")
    ).str.lower()

    career_text_map = career_text_df.set_index("candidate_id")["career_text"].to_dict()

    # ── Score and filter ────────────────────────────────────────────────────
    scoring_df = profile_df[["candidate_id", "current_title", "years_of_experience", "location", "country"]].copy()
    scoring_df["current_title"] = scoring_df["current_title"].fillna("")
    scoring_df["years_of_experience"] = scoring_df["years_of_experience"].fillna(0)

    scoring_df["title_score"] = scoring_df["current_title"].apply(title_score)
    scoring_df["skill_score"] = scoring_df["candidate_id"].apply(
        lambda cid: skill_score(cid, candidate_skill_meta)
    )
    scoring_df["career_score"] = scoring_df["candidate_id"].apply(
        lambda cid: career_evidence_score(cid, career_text_map)
    )
    scoring_df["experience_score"] = scoring_df["years_of_experience"].apply(experience_score_hard)

    scoring_df["hard_filter_score"] = (
        scoring_df["title_score"] +
        scoring_df["skill_score"] +
        scoring_df["career_score"] +
        scoring_df["experience_score"]
    )
    scoring_df["hard_reject"] = scoring_df.apply(
        lambda row: hard_reject(row, candidate_skill_sets, career_text_map), axis=1
    )

    THRESHOLD = 25
    filtered_df = scoring_df[
        (~scoring_df["hard_reject"]) &
        (scoring_df["hard_filter_score"] >= THRESHOLD)
    ].copy()

    print(f"[features] Survivors after hard filter: {len(filtered_df):,}")

    # ── Aggregate skills ────────────────────────────────────────────────────
    survivor_ids = set(filtered_df["candidate_id"])

    skills_survivors = skills_df[skills_df["candidate_id"].isin(survivor_ids)].copy()
    skills_agg = (
        skills_survivors
        .groupby("candidate_id")
        .agg(
            all_skills=("skill_name", lambda x: " | ".join(sorted(set(x.astype(str))))),
            skill_count=("skill_name", "count"),
            expert_skill_count=(
                "proficiency",
                lambda x: x.astype(str).str.lower().isin(["expert"]).sum()
            ),
            advanced_skill_count=(
                "proficiency",
                lambda x: x.astype(str).str.lower().isin(["advanced"]).sum()
            ),
            total_endorsements=("endorsements", "sum"),
            avg_skill_duration=("duration_months", "mean"),
        )
        .reset_index()
    )

    candidate_master_df = filtered_df.merge(skills_agg, on="candidate_id", how="left")

    # ── Aggregate career ────────────────────────────────────────────────────
    career_survivors = career_df[career_df["candidate_id"].isin(survivor_ids)].copy()
    career_agg = (
        career_survivors
        .groupby("candidate_id")
        .agg(
            career_text=("description", lambda x: " ".join(x.fillna("").astype(str))),
            career_titles=("title", lambda x: " | ".join(x.fillna("").astype(str))),
            companies_worked=("company", "nunique"),
            total_career_months=("duration_months", "sum"),
            current_role_count=("is_current", "sum"),
        )
        .reset_index()
    )
    candidate_master_df = candidate_master_df.merge(career_agg, on="candidate_id", how="left")

    # ── Aggregate education ─────────────────────────────────────────────────
    education_survivors = education_df[education_df["candidate_id"].isin(survivor_ids)].copy()
    education_agg = (
        education_survivors
        .groupby("candidate_id")
        .agg(
            education_text=("degree", lambda x: " | ".join(x.fillna("").astype(str))),
            field_of_study_text=("field_of_study", lambda x: " | ".join(x.fillna("").astype(str))),
            institution_count=("institution", "nunique"),
        )
        .reset_index()
    )
    candidate_master_df = candidate_master_df.merge(education_agg, on="candidate_id", how="left")

    # ── Aggregate certifications ────────────────────────────────────────────
    cert_agg = (
        certifications_df
        .groupby("candidate_id")
        .agg(
            certification_text=("name", lambda x: " | ".join(x.fillna("").astype(str))),
            certification_count=("name", "count"),
        )
        .reset_index()
    )
    candidate_master_df = candidate_master_df.merge(cert_agg, on="candidate_id", how="left")

    # ── Merge signals ───────────────────────────────────────────────────────
    important_signals = [
        "candidate_id",
        "profile_completeness_score",
        "open_to_work_flag",
        "recruiter_response_rate",
        "avg_response_time_hours",
        "connection_count",
        "endorsements_received",
        "notice_period_days",
        "github_activity_score",
        "saved_by_recruiters_30d",
        "interview_completion_rate",
        "offer_acceptance_rate",
        "search_appearance_30d",
        "profile_views_received_30d",
        "last_active_date",
        "expected_salary_min_inr_lpa",
        "expected_salary_max_inr_lpa",
        "verified_email",
        "verified_phone",
        "linkedin_connected",
        "willing_to_relocate",
    ]

    candidate_master_df = candidate_master_df.merge(
        signals_df[important_signals], on="candidate_id", how="left"
    )

    # ── Fill NaN ────────────────────────────────────────────────────────────
    candidate_master_df.fillna(
        {
            "all_skills": "",
            "career_text": "",
            "career_titles": "",
            "education_text": "",
            "field_of_study_text": "",
            "certification_text": "",
            "skill_count": 0,
            "companies_worked": 0,
            "institution_count": 0,
            "certification_count": 0,
        },
        inplace=True,
    )

    # ── Merge headline + summary ────────────────────────────────────────────
    text_df = profile_df[["candidate_id", "headline", "summary", "current_industry"]]
    candidate_master_df = candidate_master_df.merge(text_df, on="candidate_id", how="left")
    candidate_master_df.fillna({"headline": "", "summary": ""}, inplace=True)

    print(f"[features] candidate_master_df shape: {candidate_master_df.shape}")

    # ── Build candidate documents ───────────────────────────────────────────
    print("[features] Building candidate documents ...")
    candidate_master_df["candidate_document"] = candidate_master_df.apply(
        build_candidate_document, axis=1
    )

    candidate_docs = candidate_master_df[["candidate_id", "candidate_document", "hard_filter_score"]].copy()

    print(f"[features] candidate_docs shape: {candidate_docs.shape}")

    return candidate_master_df, candidate_docs
