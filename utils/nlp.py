import os
import re
import joblib
import math
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'models', 'artifacts')


def load_skills_list():
    skills_path = os.path.join(BASE_DIR, 'data', 'skills.txt')
    with open(skills_path, 'r', encoding='utf-8') as f:
        return [s.strip().lower() for s in f if s.strip()]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s,.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_skill_matches(text: str, skills_list=None):
    if skills_list is None:
        skills_list = load_skills_list()
    text_l = text.lower()
    found = set()
    for skill in skills_list:
        if skill in text_l:
            found.add(skill)
    return sorted(found)


def load_model_artifacts():
    vect_path = os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.joblib')
    clf_path = os.path.join(ARTIFACTS_DIR, 'clf.joblib')
    jobs_meta = os.path.join(ARTIFACTS_DIR, 'jobs_meta.joblib')
    if os.path.exists(vect_path) and os.path.exists(clf_path) and os.path.exists(jobs_meta):
        vect = joblib.load(vect_path)
        clf = joblib.load(clf_path)
        jobs_meta = joblib.load(jobs_meta)
        return vect, clf, jobs_meta
    return None, None, None


def predict_job_roles(text: str, top_n=3):
    vect, clf, jobs_meta = load_model_artifacts()
    if vect is None:
        return []
    X = vect.transform([text])
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X)[0]
        classes = clf.classes_
        top_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_n]
        return [{'job': classes[i], 'score': float(probs[i])} for i in top_idx]
    else:
        preds = clf.predict(X)
        return [{'job': preds[0], 'score': 1.0}]


def job_matching_by_skills(found_skills, top_n=5):
    _, _, jobs_meta = load_model_artifacts()
    if jobs_meta is None:
        return []
    # jobs_meta is a list of dicts with 'title' and 'skills'
    def score_job(job_skills):
        if not job_skills:
            return 0.0
        s_job = set([s.strip().lower() for s in job_skills.split(',') if s.strip()])
        if not s_job:
            return 0.0
        matched = len(s_job.intersection(found_skills))
        return matched / len(s_job)

    scored = []
    for jm in jobs_meta:
        sc = score_job(jm.get('skills', ''))
        scored.append({'job': jm['title'], 'score': sc, 'required_skills': jm.get('skills', '')})
    scored = sorted(scored, key=lambda x: x['score'], reverse=True)
    return scored[:top_n]


def analyze_resume(text: str):
    text_clean = clean_text(text)
    skills = extract_skill_matches(text_clean)
    pred = predict_job_roles(text_clean, top_n=3)
    matched_jobs = job_matching_by_skills(set(skills), top_n=5)

    # skill gap analysis: for top matched job, list missing skills
    missing = []
    if matched_jobs:
        top = matched_jobs[0]
        req = [s.strip().lower() for s in top['required_skills'].split(',') if s.strip()]
        missing = [s for s in req if s not in skills]

    # resume score: simple metric combining predicted job prob and skill match avg
    prob_score = max([p['score'] for p in pred]) if pred else 0
    skill_scores = [m['score'] for m in matched_jobs] if matched_jobs else [0]
    skill_score_avg = sum(skill_scores) / len(skill_scores) if skill_scores else 0
    resume_score = int((prob_score * 0.6 + skill_score_avg * 0.4) * 100)

    return {
        'skills_found': skills,
        'predicted_jobs': pred,
        'matched_jobs': matched_jobs,
        'missing_skills': missing,
        'resume_score': resume_score,
    }
