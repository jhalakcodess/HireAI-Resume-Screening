"""
data_prep.py
------------
Generates a synthetic resume-screening dataset so the whole project
runs offline without needing to log into Kaggle.

Each row = (resume_text, job_description_text, label)
label = 1  -> resume is a good fit for the job (shortlist)
label = 0  -> resume is a poor fit (reject)

WHY SYNTHETIC DATA:
Kaggle requires an account + API token to download datasets, and this
sandbox can't reach kaggle.com. The generator below builds realistic-
looking resumes by mixing skills from a target job category with some
noise from other categories, which is enough to demonstrate the full
pipeline. For your final submission, you can swap this out for a real
dataset (e.g. search "Resume Dataset" on Kaggle) — model.py doesn't
care where resume.csv comes from, as long as it has the same columns.
"""

import random
import csv

random.seed(42)

JOBS = {
    "Data Scientist": [
        "python", "machine learning", "pandas", "numpy", "sql",
        "statistics", "scikit-learn", "data visualization", "deep learning"
    ],
    "Frontend Developer": [
        "javascript", "react", "html", "css", "typescript",
        "redux", "webpack", "responsive design", "ui/ux"
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "ci/cd", "terraform",
        "linux", "jenkins", "monitoring", "networking"
    ],
    "Digital Marketer": [
        "seo", "google ads", "content strategy", "analytics",
        "social media", "email marketing", "branding", "copywriting"
    ],
}

RESUME_TEMPLATE = (
    "Experienced professional with {years} years of experience. "
    "Skilled in {skills}. Worked on multiple projects involving "
    "{skills2}. Strong background in {domain}."
)

JD_TEMPLATE = (
    "We are looking for a {job_title} with strong skills in {skills}. "
    "The candidate should have experience with {skills2} and a "
    "background in {domain}."
)


def make_text(template, job_title, skill_pool, n_main=4, n_extra=2, noise_pool=None, noise_ratio=0.0):
    skills = random.sample(skill_pool, k=min(n_main, len(skill_pool)))
    extra_pool = skill_pool
    if noise_pool and random.random() < noise_ratio:
        extra_pool = noise_pool
    skills2 = random.sample(extra_pool, k=min(n_extra, len(extra_pool)))
    return template.format(
        years=random.randint(1, 10),
        job_title=job_title,
        skills=", ".join(skills),
        skills2=", ".join(skills2),
        domain=job_title.lower(),
    )


def generate_dataset(n_per_job=150):
    rows = []
    job_titles = list(JOBS.keys())

    for job_title, skills in JOBS.items():
        other_skills_pool = [
            s for t, sk in JOBS.items() if t != job_title for s in sk
        ]

        # Positive examples: resume built from the SAME job's skills
        for _ in range(n_per_job // 2):
            jd = make_text(JD_TEMPLATE, job_title, skills, noise_ratio=0.0)
            resume = make_text(
                RESUME_TEMPLATE, job_title, skills,
                noise_pool=other_skills_pool, noise_ratio=0.15  # a little realistic noise
            )
            rows.append((resume, jd, 1))

        # Negative examples: resume built mostly from a DIFFERENT job's skills
        for _ in range(n_per_job // 2):
            wrong_job = random.choice([t for t in job_titles if t != job_title])
            jd = make_text(JD_TEMPLATE, job_title, skills, noise_ratio=0.0)
            resume = make_text(
                RESUME_TEMPLATE, wrong_job, JOBS[wrong_job],
                noise_pool=skills, noise_ratio=0.1
            )
            rows.append((resume, jd, 0))

    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    data = generate_dataset(n_per_job=200)
    with open("resumes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resume_text", "job_description", "label"])
        writer.writerows(data)

    print(f"Generated {len(data)} rows -> resumes.csv")
    print("Label balance:", sum(r[2] for r in data), "positive /", len(data), "total")
