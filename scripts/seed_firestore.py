"""Seed Firestore with synthetic test data for local development.

Usage:
    python scripts/seed_firestore.py          # default: 50 patients
    python scripts/seed_firestore.py --count 100
    python scripts/seed_firestore.py --clear  # wipe all test data first
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from firebase_admin import firestore
from src.firebase_admin import get_firestore_client, _init_app
from src.risk_model import get_risk_model

# ── Demographic pools ──────────────────────────────────────────────────────

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
    "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark", "Margaret",
    "Donald", "Sandra", "Steven", "Ashley", "Paul", "Dorothy", "Andrew",
    "Kimberly", "Joshua", "Emily", "Kenneth", "Donna", "Kevin", "Michelle",
    "Brian", "Carol", "George", "Amanda", "Timothy", "Melissa", "Ronald",
    "Deborah",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts",
]

DIAGNOSIS_POOLS = {
    "depression": ["F32.0", "F32.1", "F32.2", "F32.9", "F33.0", "F33.1"],
    "anxiety": ["F41.0", "F41.1", "F41.9"],
    "ptsd": ["F43.10", "F43.11", "F43.12"],
    "bipolar": ["F31.0", "F31.1", "F31.2", "F31.81"],
    "substance": ["F10.10", "F10.20", "F11.10", "F11.20"],
    "none": [],
}

MEDICATION_POOLS = {
    "depression": ["sertraline", "fluoxetine", "escitalopram", "venlafaxine", "bupropion"],
    "anxiety": ["lorazepam", "alprazolam", "buspirone", "hydroxyzine", "clonazepam"],
    "bipolar": ["lithium", "valproate", "lamotrigine", "quetiapine", "aripiprazole"],
    "sleep": ["trazodone", "melatonin", "zolpidem", "doxepin"],
    "none": [],
}

CLINICAL_PROFILES = [
    # (profile_name, phq9_range, gad7_range, sleep_range, hr_range, diag_keys, med_keys, review_prob)
    ("healthy",           (0, 4),    (0, 3),   (7.0, 9.0), (60, 78),  ["none"],      ["none"],         0.02),
    ("mild_anxiety",      (3, 9),    (5, 10),  (6.0, 8.0), (65, 82), ["anxiety"],   ["anxiety"],      0.05),
    ("mild_depression",   (5, 12),   (3, 8),   (5.5, 8.0), (62, 85), ["depression"],["depression"],   0.08),
    ("moderate_mixed",    (10, 18),  (10, 16), (4.5, 7.0), (70, 90), ["depression","anxiety"], ["depression","anxiety"], 0.25),
    ("severe_depression", (18, 25),  (8, 15),  (3.0, 6.0), (72, 95), ["depression"], ["depression","sleep"], 0.55),
    ("severe_anxiety",    (8, 16),   (16, 21), (3.5, 5.5), (80, 110),["anxiety"],   ["anxiety"],      0.45),
    ("critical",          (22, 27),  (15, 21), (2.0, 4.5), (85, 120),["depression","anxiety"], ["depression","anxiety","sleep"], 0.80),
    ("bipolar",           (12, 22),  (8, 16),  (4.0, 10.0),(65, 100),["bipolar"],   ["bipolar"],      0.60),
    ("substance_use",     (10, 20),  (10, 18), (3.0, 6.0), (75, 105),["substance","depression"], ["depression"], 0.50),
    ("ptsd",              (14, 24),  (12, 20), (3.5, 5.5), (78, 105),["ptsd","anxiety"], ["anxiety"], 0.65),
]


# ── Generators ─────────────────────────────────────────────────────────────

def _pick(pool: list) -> str:
    return random.choice(pool)


def _rand_range(lo: float, hi: float, as_int: bool = False) -> float | int:
    v = random.uniform(lo, hi)
    return int(round(v)) if as_int else round(v, 1)


def generate_patient(index: int, profile: dict | None = None) -> dict:
    """Generate one synthetic patient with linked screening, explanation, and optionally a review."""
    if profile is None:
        profile = random.choice(CLINICAL_PROFILES)

    pname, (phq_lo, phq_hi), (gad_lo, gad_hi), (slp_lo, slp_hi), (hr_lo, hr_hi), diag_keys, med_keys, review_prob = profile

    anonymized_id = f"synth_{index:04d}_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)
    created_at = now - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))

    # ── Survey data ────────────────────────────────────────────────────────
    phq9 = _rand_range(phq_lo, phq_hi, as_int=True)
    gad7 = _rand_range(gad_lo, gad_hi, as_int=True)

    # ── Wearable data ──────────────────────────────────────────────────────
    sleep_hours = _rand_range(slp_lo, slp_hi)
    avg_hr = _rand_range(hr_lo, hr_hi, as_int=True)

    # ── EMR data ───────────────────────────────────────────────────────────
    diag_codes = []
    for dk in diag_keys:
        diag_codes.extend(random.sample(DIAGNOSIS_POOLS[dk], k=min(2, len(DIAGNOSIS_POOLS[dk]))))

    medications = []
    for mk in med_keys:
        medications.extend(random.sample(MEDICATION_POOLS[mk], k=min(2, len(MEDICATION_POOLS[mk]))))

    # ── Build input and run through risk model ─────────────────────────────
    combined = {
        "phq9_score": phq9,
        "gad7_score": gad7,
        "sleep_hours": sleep_hours,
        "avg_heart_rate": avg_hr,
    }
    if diag_codes:
        combined["diagnosis_codes"] = diag_codes
    if medications:
        combined["medications"] = medications

    model = get_risk_model()
    assessment = model.assess(combined)

    screening_id = str(uuid.uuid4())

    screening_doc = {
        "id": screening_id,
        "anonymized_id": anonymized_id,
        "patient_name": f"{_pick(FIRST_NAMES)} {_pick(LAST_NAMES)}",
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "input_data": combined,
        "clinical_profile": pname,
        "created_at": created_at,
    }

    explanation_doc = {
        "id": screening_id,
        "screening_id": screening_id,
        "explanation_text": assessment.clinical_interpretation,
        "factors": {
            "contributing_factors": assessment.contributing_factors,
            "confidence": assessment.confidence,
            "top_features": [{"name": n, "value": v} for n, v in assessment.top_features],
            "counterfactual": assessment.counterfactual,
        },
        "created_at": created_at,
    }

    review_doc = None
    if assessment.requires_human_review or random.random() < review_prob:
        statuses = ["pending", "approved", "escalated", "closed"]
        status_weights = [0.4, 0.25, 0.15, 0.2]
        review_status = random.choices(statuses, weights=status_weights, k=1)[0]

        reviewer_uids = [None, "admin@example.com", "reviewer1@example.com", "reviewer2@example.com"]
        reviewer = random.choice(reviewer_uids[1:]) if review_status != "pending" else None

        review_doc = {
            "id": screening_id,
            "screening_id": screening_id,
            "status": review_status,
            "reviewer_uid": reviewer,
            "notes": _generate_review_notes(review_status) if review_status != "pending" else None,
            "created_at": created_at + timedelta(minutes=random.randint(5, 120)),
            "updated_at": created_at + timedelta(hours=random.randint(1, 48)),
        }

    return {
        "screening": screening_doc,
        "explanation": explanation_doc,
        "review": review_doc,
        "anonymized_id": anonymized_id,
        "risk_level": assessment.risk_level,
        "profile": pname,
    }


def _generate_review_notes(status: str) -> str:
    notes = {
        "approved": random.choice([
            "Reviewed. Patient stable. Continue current treatment plan.",
            "Confirmed assessment. Recommend follow-up in 2 weeks.",
            "Risk level verified. Existing care plan is appropriate.",
            "Reviewed and approved. No changes to treatment recommended.",
        ]),
        "escalated": random.choice([
            "Elevated risk confirmed. Escalating to attending psychiatrist.",
            "Patient reports worsening symptoms. Urgent evaluation recommended.",
            "Risk factors compounding. Recommend intensive outpatient referral.",
            "Critical indicators present. Immediate clinical consult needed.",
        ]),
        "closed": random.choice([
            "Patient connected with resources. Case closed.",
            "Follow-up completed. Patient engaged with treatment.",
            "Reassessed — risk level decreased. Closing review.",
            "Case resolved. Patient stabilized with medication adjustment.",
        ]),
    }
    return notes.get(status, "")


def generate_users(count: int = 5) -> list[dict]:
    """Generate synthetic user documents (one admin, the rest plain users)."""
    users = []
    # ponytail: the bootstrap allowlist is the source of truth — seed only
    # mirrors that. In production no synthetic users are created.
    admin_email = "aakashrraj2@gmail.com"
    for i in range(count):
        uid = f"synth_user_{i:03d}_{uuid.uuid4().hex[:6]}"
        if i == 0:
            email = admin_email
            role = "admin"
            display_name = "Admin"
        else:
            first = _pick(FIRST_NAMES)
            last = _pick(LAST_NAMES)
            email = f"{first.lower()}.{last.lower()}@example.com"
            role = "user"
            display_name = f"{first} {last}"
        users.append({
            "uid": uid,
            "email": email,
            "display_name": display_name,
            "photo_url": None,
            "role": role,
            "provider": random.choice(["google", "email"]),
            "created_at": datetime.now(timezone.utc) - timedelta(days=random.randint(1, 90)),
        })
    return users


# ── Main ───────────────────────────────────────────────────────────────────

def seed(count: int = 50, clear: bool = False):
    _init_app()
    db = get_firestore_client()
    model = get_risk_model()

    if clear:
        print("Clearing existing test data...")
        for col in ["screenings", "explanations", "reviews", "users"]:
            docs = list(db.collection(col).where("id", ">=", "0").limit(500).get())
            # Also grab synth docs
            synth_docs = list(db.collection(col).where("id", ">=", "synth_").limit(500).get())
            all_docs = {d.id: d for d in docs + synth_docs}
            for doc in all_docs.values():
                doc.reference.delete()
            print(f"  Cleared {len(all_docs)} docs from {col}")

    print(f"\nGenerating {count} synthetic patients...")
    stats = {"screenings": 0, "explanations": 0, "reviews": 0, "users": 0}
    profile_counts = {}

    # Batch write for performance
    batch = db.batch()
    batch_count = 0
    FLUSH_AT = 400

    for i in range(1, count + 1):
        patient = generate_patient(i)
        profile_counts[patient["profile"]] = profile_counts.get(patient["profile"], 0) + 1

        ref_s = db.collection("screenings").document(patient["screening"]["id"])
        batch.set(ref_s, patient["screening"])
        stats["screenings"] += 1
        batch_count += 1

        ref_e = db.collection("explanations").document(patient["explanation"]["id"])
        batch.set(ref_e, patient["explanation"])
        stats["explanations"] += 1
        batch_count += 1

        if patient["review"]:
            ref_r = db.collection("reviews").document(patient["review"]["id"])
            batch.set(ref_r, patient["review"])
            stats["reviews"] += 1
            batch_count += 1

        if batch_count >= FLUSH_AT:
            batch.commit()
            batch = db.batch()
            batch_count = 0
            print(f"  ... {i}/{count} patients written")

    # Write users
    users = generate_users(8)
    for u in users:
        batch.set(db.collection("users").document(u["uid"]), u)
        stats["users"] += 1
        batch_count += 1

    if batch_count > 0:
        batch.commit()

    print(f"\nDone! Seeded Firestore with:")
    print(f"  {stats['screenings']} screenings")
    print(f"  {stats['explanations']} explanations")
    print(f"  {stats['reviews']} reviews")
    print(f"  {stats['users']} users")
    print(f"\nProfile distribution:")
    for profile, cnt in sorted(profile_counts.items(), key=lambda x: -x[1]):
        print(f"  {profile}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Firestore with synthetic test data")
    parser.add_argument("--count", type=int, default=50, help="Number of patients to generate (default: 50)")
    parser.add_argument("--clear", action="store_true", help="Clear existing test data before seeding")
    args = parser.parse_args()
    seed(count=args.count, clear=args.clear)
