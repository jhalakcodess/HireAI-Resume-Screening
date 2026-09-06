import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "screening.db"

app = FastAPI(
    title="HireAI - Resume Screening API",
    version="1.0.0",
    description="AI-powered resume screening and candidate analysis system."
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Load model artifacts once at startup.
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
classifier = joblib.load(BASE_DIR / "classifier.pkl")

REVIEW_THRESHOLD = 0.80


class ScoreRequest(BaseModel):
    candidate_name: str = "Anonymous Candidate"
    resume_text: str
    job_description: str


class ReviewRequest(BaseModel):
    candidate_id: int
    human_decision: str
    notes: str = ""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT NOT NULL,
            resume_text TEXT NOT NULL,
            job_description TEXT NOT NULL,
            fit_score REAL NOT NULL,
            skill_overlap INTEGER NOT NULL,
            decision TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            human_decision TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def skill_overlap(resume_text: str, job_text: str) -> int:
    r_words = set(resume_text.lower().replace(",", "").split())
    j_words = set(job_text.lower().replace(",", "").split())
    return len(r_words & j_words)


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "online", "model": "TF-IDF + Logistic Regression"}


@app.post("/score")
def score_resume(req: ScoreRequest):
    if not req.resume_text.strip() or not req.job_description.strip():
        raise HTTPException(status_code=400, detail="Resume and job description are required.")

    R = vectorizer.transform([req.resume_text])
    J = vectorizer.transform([req.job_description])

    sim = float(cosine_similarity(R, J)[0, 0])
    overlap = skill_overlap(req.resume_text, req.job_description)

    X_raw = np.array([[sim, overlap]])
    X = scaler.transform(X_raw)
    pred = int(classifier.predict(X)[0])
    proba = float(classifier.predict_proba(X)[0][pred])

    decision = "SHORTLIST" if pred == 1 else "REJECT"

    status = "REVIEW" if sim < 0.70 else "AUTO"


    now = datetime.now().isoformat(timespec="seconds")
    conn = get_db()
    cur = conn.execute("""
        INSERT INTO screenings
        (candidate_name, resume_text, job_description, fit_score,
         skill_overlap, decision, confidence, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        req.candidate_name.strip() or "Anonymous Candidate",
        req.resume_text,
        req.job_description,
        round(sim, 3),
        overlap,
        decision,
        round(proba, 3),
        status,
        now,
    ))
    candidate_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {
        "candidate_id": candidate_id,
        "fit_score": round(sim, 3),
        "skill_overlap": overlap,
        "decision": decision,
        "confidence": round(proba, 3),
        "status": status,
        "review_required": status == "REVIEW",
        "threshold": REVIEW_THRESHOLD,
    }


@app.get("/api/candidates")
def candidates():
    conn = get_db()
    rows = conn.execute("""
        SELECT id, candidate_name, fit_score, skill_overlap, decision,
               confidence, status, human_decision, notes, created_at
        FROM screenings ORDER BY id DESC
    """).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/api/candidates/{candidate_id}")
def candidate(candidate_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM screenings WHERE id = ?", (candidate_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return dict(row)


@app.post("/api/review")
def review(req: ReviewRequest):
    decision = req.human_decision.upper()
    if decision not in {"SHORTLIST", "REJECT"}:
        raise HTTPException(status_code=400, detail="Human decision must be SHORTLIST or REJECT.")

    conn = get_db()
    cur = conn.execute("""
        UPDATE screenings
        SET human_decision = ?, notes = ?, status = 'REVIEWED'
        WHERE id = ?
    """, (decision, req.notes, req.candidate_id))
    conn.commit()
    conn.close()

    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"status": "review_saved", "candidate_id": req.candidate_id, "human_decision": decision}


@app.get("/api/stats")
def stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
    shortlisted = conn.execute("SELECT COUNT(*) FROM screenings WHERE decision='SHORTLIST'").fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM screenings WHERE decision='REJECT'").fetchone()[0]
    review = conn.execute("SELECT COUNT(*) FROM screenings WHERE status='REVIEW'").fetchone()[0]
    reviewed = conn.execute("SELECT COUNT(*) FROM screenings WHERE status='REVIEWED'").fetchone()[0]
    avg_fit = conn.execute("SELECT AVG(fit_score) FROM screenings").fetchone()[0] or 0
    avg_conf = conn.execute("SELECT AVG(confidence) FROM screenings").fetchone()[0] or 0
    conn.close()

    return {
        "total": total,
        "shortlisted": shortlisted,
        "rejected": rejected,
        "needs_review": review,
        "human_reviewed": reviewed,
        "average_fit_score": round(avg_fit, 3),
        "average_confidence": round(avg_conf, 3),
        "review_threshold": REVIEW_THRESHOLD,
    }
