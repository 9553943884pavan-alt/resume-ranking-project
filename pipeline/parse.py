"""
parse.py — Parse candidates.jsonl into flat DataFrames.

Verbatim logic from Ranking_Project.ipynb.
Only change: returns DataFrames in a dict instead of writing to CSV,
and accepts local file paths via arguments.
"""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (verbatim from Ranking_Project.ipynb)
# ---------------------------------------------------------------------------

PROFICIENCY_SCORE = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}

# ---------------------------------------------------------------------------
# Helper functions (verbatim)
# ---------------------------------------------------------------------------

def open_text_file(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open_text_file(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no}: {e}") from e
    return records


def safe_str(x):
    return "" if x is None else str(x)


def join_nonempty(parts, sep=" | "):
    parts = [safe_str(p).strip() for p in parts if safe_str(p).strip()]
    return sep.join(parts)


def flatten_salary(redrob: Dict[str, Any]) -> Dict[str, Any]:
    salary = redrob.get("expected_salary_range_inr_lpa") or {}
    return {
        "expected_salary_min_inr_lpa": salary.get("min"),
        "expected_salary_max_inr_lpa": salary.get("max"),
    }


def aggregate_features(rec: Dict[str, Any]) -> Dict[str, Any]:
    profile = rec.get("profile", {}) or {}
    career_history = rec.get("career_history", []) or []
    education = rec.get("education", []) or []
    skills = rec.get("skills", []) or []
    redrob = rec.get("redrob_signals", {}) or {}

    prof_scores = [PROFICIENCY_SCORE.get(s.get("proficiency"), 0) for s in skills]
    durations = [s.get("duration_months") for s in skills if isinstance(s.get("duration_months"), (int, float))]
    endorsements = [s.get("endorsements") for s in skills if isinstance(s.get("endorsements"), (int, float))]

    trusted_skills = 0
    partial_skills = 0
    for s in skills:
        prof = s.get("proficiency")
        dur = s.get("duration_months") or 0
        end = s.get("endorsements") or 0
        if prof in ("advanced", "expert") and dur >= 12 and end >= 5:
            trusted_skills += 1
        elif prof == "intermediate" and dur >= 6 and end >= 2:
            partial_skills += 1

    txt = ((profile.get("summary") or "") + " " + " ".join([j.get("description", "") for j in career_history])).lower()
    evidence_terms = [
        "embedding", "embeddings", "retrieval", "ranking", "search", "recommendation",
        "vector", "nlp", "rag", "faiss", "qdrant", "milvus", "opensearch", "elasticsearch",
        "bm25", "ndcg", "mrr", "map", "a/b test", "deployed", "shipped", "production"
    ]
    evidence_hits = sum(1 for term in evidence_terms if term in txt)

    return {
        "candidate_id": rec.get("candidate_id"),
        "years_of_experience": profile.get("years_of_experience"),
        "num_jobs": len(career_history),
        "num_education_entries": len(education),
        "num_skills": len(skills),
        "num_certifications": len(rec.get("certifications", []) or []),
        "num_languages": len(rec.get("languages", []) or []),
        "trusted_skills": trusted_skills,
        "partial_skills": partial_skills,
        "avg_skill_proficiency_score": (sum(prof_scores) / len(prof_scores)) if prof_scores else 0,
        "avg_skill_duration_months": (sum(durations) / len(durations)) if durations else 0,
        "avg_skill_endorsements": (sum(endorsements) / len(endorsements)) if endorsements else 0,
        "num_career_months": sum(int(j.get("duration_months") or 0) for j in career_history),
        "num_current_roles": sum(1 for j in career_history if j.get("is_current") is True),
        "text_evidence_hits": evidence_hits,
        "profile_completeness_score": redrob.get("profile_completeness_score"),
        "recruiter_response_rate": redrob.get("recruiter_response_rate"),
        "avg_response_time_hours": redrob.get("avg_response_time_hours"),
        "open_to_work_flag": redrob.get("open_to_work_flag"),
        "notice_period_days": redrob.get("notice_period_days"),
        "preferred_work_mode": redrob.get("preferred_work_mode"),
        "willing_to_relocate": redrob.get("willing_to_relocate"),
        "github_activity_score": redrob.get("github_activity_score"),
        "search_appearance_30d": redrob.get("search_appearance_30d"),
        "saved_by_recruiters_30d": redrob.get("saved_by_recruiters_30d"),
        "interview_completion_rate": redrob.get("interview_completion_rate"),
        "offer_acceptance_rate": redrob.get("offer_acceptance_rate"),
        "verified_email": redrob.get("verified_email"),
        "verified_phone": redrob.get("verified_phone"),
        "linkedin_connected": redrob.get("linkedin_connected"),
        **flatten_salary(redrob),
    }

# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_candidates(
    candidates_path: str,
    precomputed_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Parse candidates.jsonl into 8 flat DataFrames.

    If precomputed_dir is set and contains all 8 CSVs, loads from disk
    instead of re-parsing (faster for repeated runs).

    Returns a dict with keys:
        profile, career, education, skills, skill_assessments,
        certifications, signals, aggregated_features
    """
    csv_names = {
        "profile": "candidates_profile.csv",
        "career": "career_history.csv",
        "education": "education.csv",
        "skills": "skills.csv",
        "skill_assessments": "skill_assessments.csv",
        "certifications": "certifications.csv",
        "signals": "redrob_signals.csv",
        "aggregated_features": "aggregated_features.csv",
    }

    if precomputed_dir:
        precomputed_path = Path(precomputed_dir)
        all_exist = all((precomputed_path / v).exists() for v in csv_names.values())
        if all_exist:
            print(f"[parse] Loading pre-parsed CSVs from {precomputed_dir} ...")
            return {k: pd.read_csv(precomputed_path / v, low_memory=False) for k, v in csv_names.items()}
        else:
            print(f"[parse] Not all CSVs found in {precomputed_dir}, will parse from JSONL.")

    print(f"[parse] Reading candidates from {candidates_path} ...")
    records = load_jsonl(candidates_path)
    print(f"[parse] Loaded {len(records):,} records.")

    profile_rows = []
    career_rows = []
    education_rows = []
    skills_rows = []
    skill_assessment_rows = []
    cert_rows = []
    signal_rows = []
    agg_rows = []

    for rec in tqdm(records, desc="Parsing"):
        cid = rec.get("candidate_id")
        profile = rec.get("profile", {}) or {}
        career_history = rec.get("career_history", []) or []
        education = rec.get("education", []) or []
        skills = rec.get("skills", []) or []
        certs = rec.get("certifications", []) or []
        redrob = rec.get("redrob_signals", {}) or {}

        # Profile table
        profile_rows.append({
            "candidate_id": cid,
            "anonymized_name": profile.get("anonymized_name"),
            "headline": profile.get("headline"),
            "summary": profile.get("summary"),
            "location": profile.get("location"),
            "country": profile.get("country"),
            "years_of_experience": profile.get("years_of_experience"),
            "current_title": profile.get("current_title"),
            "current_company": profile.get("current_company"),
            "current_company_size": profile.get("current_company_size"),
            "current_industry": profile.get("current_industry"),
            **flatten_salary(redrob),
        })

        # Career history table
        for i, job in enumerate(career_history, start=1):
            career_rows.append({
                "candidate_id": cid,
                "career_order": i,
                "company": job.get("company"),
                "title": job.get("title"),
                "start_date": job.get("start_date"),
                "end_date": job.get("end_date"),
                "duration_months": job.get("duration_months"),
                "is_current": job.get("is_current"),
                "industry": job.get("industry"),
                "company_size": job.get("company_size"),
                "description": job.get("description"),
            })

        # Education table
        for i, edu in enumerate(education, start=1):
            education_rows.append({
                "candidate_id": cid,
                "education_order": i,
                "institution": edu.get("institution"),
                "degree": edu.get("degree"),
                "field_of_study": edu.get("field_of_study"),
                "start_year": edu.get("start_year"),
                "end_year": edu.get("end_year"),
                "grade": edu.get("grade"),
                "tier": edu.get("tier"),
            })

        # Skills table
        for i, s in enumerate(skills, start=1):
            skills_rows.append({
                "candidate_id": cid,
                "skill_order": i,
                "skill_name": s.get("name"),
                "proficiency": s.get("proficiency"),
                "endorsements": s.get("endorsements"),
                "duration_months": s.get("duration_months"),
            })

        # Skill assessments (inside redrob_signals)
        for skill_name, score in (redrob.get("skill_assessment_scores") or {}).items():
            skill_assessment_rows.append({
                "candidate_id": cid,
                "skill_name": skill_name,
                "assessment_score": score,
            })

        # Certifications
        for i, c in enumerate(certs, start=1):
            cert_rows.append({
                "candidate_id": cid,
                "cert_order": i,
                "name": c.get("name"),
                "issuer": c.get("issuer"),
                "year": c.get("year"),
            })

        # Signals table
        signal_rows.append({
            "candidate_id": cid,
            "profile_completeness_score": redrob.get("profile_completeness_score"),
            "signup_date": redrob.get("signup_date"),
            "last_active_date": redrob.get("last_active_date"),
            "open_to_work_flag": redrob.get("open_to_work_flag"),
            "profile_views_received_30d": redrob.get("profile_views_received_30d"),
            "applications_submitted_30d": redrob.get("applications_submitted_30d"),
            "recruiter_response_rate": redrob.get("recruiter_response_rate"),
            "avg_response_time_hours": redrob.get("avg_response_time_hours"),
            "connection_count": redrob.get("connection_count"),
            "endorsements_received": redrob.get("endorsements_received"),
            "notice_period_days": redrob.get("notice_period_days"),
            "expected_salary_min_inr_lpa": (redrob.get("expected_salary_range_inr_lpa") or {}).get("min"),
            "expected_salary_max_inr_lpa": (redrob.get("expected_salary_range_inr_lpa") or {}).get("max"),
            "preferred_work_mode": redrob.get("preferred_work_mode"),
            "willing_to_relocate": redrob.get("willing_to_relocate"),
            "github_activity_score": redrob.get("github_activity_score"),
            "search_appearance_30d": redrob.get("search_appearance_30d"),
            "saved_by_recruiters_30d": redrob.get("saved_by_recruiters_30d"),
            "interview_completion_rate": redrob.get("interview_completion_rate"),
            "offer_acceptance_rate": redrob.get("offer_acceptance_rate"),
            "verified_email": redrob.get("verified_email"),
            "verified_phone": redrob.get("verified_phone"),
            "linkedin_connected": redrob.get("linkedin_connected"),
        })

        # Aggregated features table
        agg_rows.append(aggregate_features(rec))

    dfs = {
        "profile": pd.DataFrame(profile_rows),
        "career": pd.DataFrame(career_rows),
        "education": pd.DataFrame(education_rows),
        "skills": pd.DataFrame(skills_rows),
        "skill_assessments": pd.DataFrame(skill_assessment_rows),
        "certifications": pd.DataFrame(cert_rows),
        "signals": pd.DataFrame(signal_rows),
        "aggregated_features": pd.DataFrame(agg_rows),
    }

    # Optionally cache parsed CSVs for future runs
    if precomputed_dir:
        out = Path(precomputed_dir)
        out.mkdir(parents=True, exist_ok=True)
        for key, fname in csv_names.items():
            dfs[key].to_csv(out / fname, index=False, encoding="utf-8")
        print(f"[parse] Cached parsed CSVs to {precomputed_dir}")

    return dfs
