"""
model.py
--------
Trains the resume-screening model.

APPROACH:
1. Convert resume text and job-description text into TF-IDF vectors.
2. Compute cosine similarity between each (resume, job) pair -> "fit_score".
3. Also compute a simple skill-overlap count as a second feature.
4. Feed [fit_score, skill_overlap] into a Logistic Regression classifier
   that predicts shortlist (1) or reject (0).

WHY TF-IDF INSTEAD OF EMBEDDINGS (e.g. sentence-transformers / BERT):
- No internet/model download needed -> runs anywhere instantly.
- Easy to explain in a viva: "each word gets a weight based on how
  rare/important it is, then we measure the angle between vectors."
- For your PPT: mention this as a deliberate lightweight-model choice,
  and note that swapping in sentence-transformers embeddings is a
  natural "future improvement" (better semantic matching, e.g.
  understanding "React" and "Frontend framework" are related).

Outputs:
- tfidf_vectorizer.pkl : fitted shared TF-IDF vectorizer
- classifier.pkl       : trained logistic regression model
- metrics printed to console
"""

import csv
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_data(path="resumes.csv"):
    resumes, jobs, labels = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resumes.append(row["resume_text"])
            jobs.append(row["job_description"])
            labels.append(int(row["label"]))
    return resumes, jobs, labels


def skill_overlap(resume_text, job_text):
    """Simple feature: how many words overlap between resume and JD."""
    r_words = set(resume_text.lower().replace(",", "").split())
    j_words = set(job_text.lower().replace(",", "").split())
    return len(r_words & j_words)


def build_features(resumes, jobs, vectorizer, fit=False):
    if fit:
        # Fit ONE shared vocabulary on resumes + jobs together, so both
        # sides map to the same vector space and cosine similarity is valid.
        vectorizer.fit(resumes + jobs)
    R = vectorizer.transform(resumes)
    J = vectorizer.transform(jobs)

    # cosine similarity between matching resume/job pairs (row-wise, not full matrix)
    sims = np.array([
        cosine_similarity(R[i], J[i])[0, 0] for i in range(R.shape[0])
    ])
    overlaps = np.array([
        skill_overlap(resumes[i], jobs[i]) for i in range(len(resumes))
    ])

    X = np.column_stack([sims, overlaps])
    return X


def main():
    resumes, jobs, labels = load_data()
    y = np.array(labels)

    (r_train, r_test,
     j_train, j_test,
     y_train, y_test) = train_test_split(
        resumes, jobs, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english")

    X_train_raw = build_features(r_train, j_train, vectorizer, fit=True)
    X_test_raw = build_features(r_test, j_test, vectorizer, fit=False)

    # IMPORTANT: fit_score (0-1) and skill_overlap (0-10ish) are on very
    # different scales. Without scaling, logistic regression ends up
    # weighting skill_overlap far more heavily just because its raw
    # numbers are bigger -- not because it's actually more predictive.
    # StandardScaler puts both features on equal footing.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("=== Evaluation on held-out test set ===")
    print("Accuracy :", round(accuracy_score(y_test, preds), 3))
    print("Precision:", round(precision_score(y_test, preds), 3))
    print("Recall   :", round(recall_score(y_test, preds), 3))
    print("F1 score :", round(f1_score(y_test, preds), 3))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(clf, "classifier.pkl")
    print("\nSaved: tfidf_vectorizer.pkl, scaler.pkl, classifier.pkl")


if __name__ == "__main__":
    main()
