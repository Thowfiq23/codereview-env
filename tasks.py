"""
codereview_env/tasks.py
Complete task dataset: 3 tasks with code snippets and ground-truth issues.
All graders use deterministic AST + fuzzy matching.
"""

TASKS = [
    # ─────────────────────────────────────────────────────────────────────────
    # TASK 1 — EASY
    # Single off-by-one bug in a data-processing function.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "task_1_bug",
        "difficulty": "easy",
        "filename": "data_processor.py",
        "language": "python",
        "context": (
            "This module is used in a sensor-data aggregation pipeline. "
            "compute_average() is called thousands of times per second."
        ),
        "task_description": (
            "Review this Python function for correctness bugs. "
            "Look for logic errors, off-by-one mistakes, or incorrect computations."
        ),
        "code_snippet": """\
def compute_average(numbers: list) -> float:
    \"\"\"Return the average of a list of numbers.\"\"\"
    if not numbers:
        return 0.0

    total = 0
    for i in range(1, len(numbers)):   # BUG: skips index 0
        total += numbers[i]

    return total / len(numbers)


def normalize(data: list) -> list:
    \"\"\"Normalize values to [0, 1] relative to max.\"\"\"
    avg = compute_average(data)
    max_val = max(data) if data else 1
    return [(x - avg) / max_val for x in data]
""",
        "ground_truth_issues": [
            {
                "line_number": 7,
                "correct_type": "bug",
                "correct_severity": "high",
                "match_keywords": "range index off-by-one skips first element starts 1 loop zero",
            }
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 2 — MEDIUM
    # Authentication module with 3 security issues.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "task_2_security",
        "difficulty": "medium",
        "filename": "user_auth.py",
        "language": "python",
        "context": (
            "This module handles user login for a customer-facing web application "
            "with ~50 000 registered accounts."
        ),
        "task_description": (
            "Review this authentication module for security vulnerabilities. "
            "Look for SQL injection, hardcoded secrets, and weak cryptographic choices."
        ),
        "code_snippet": """\
import sqlite3
import hashlib

SECRET_KEY = "mysecretkey123"       # ISSUE 1: hardcoded secret

def get_db():
    return sqlite3.connect("users.db")

def login(username: str, password: str) -> bool:
    \"\"\"Return True if credentials are valid.\"\"\"
    db = get_db()
    cursor = db.cursor()

    # ISSUE 2: SQL injection
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    db.close()
    return user is not None

def hash_password(password: str) -> str:
    \"\"\"Hash a password before storing.\"\"\"
    # ISSUE 3: MD5 is cryptographically broken
    return hashlib.md5(password.encode()).hexdigest()
""",
        "ground_truth_issues": [
            {
                "line_number": 4,
                "correct_type": "security",
                "correct_severity": "high",
                "match_keywords": "hardcoded SECRET_KEY environ environment secret plaintext env var config",
            },
            {
                "line_number": 15,
                "correct_type": "security",
                "correct_severity": "critical",
                "match_keywords": "sql injection f-string parameterised parameterized format string interpolat",
            },
            {
                "line_number": 24,
                "correct_type": "security",
                "correct_severity": "high",
                "match_keywords": "md5 broken weak bcrypt argon2 pbkdf2 cryptographic hashing deprecated",
            },
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 3 — HARD
    # Payment processor with 5 mixed issues.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "task_3_multi",
        "difficulty": "hard",
        "filename": "payment_processor.py",
        "language": "python",
        "context": (
            "This PR adds a new payment processing module to a fintech API. "
            "It handles charging users and logging transactions."
        ),
        "task_description": (
            "Review this payment-processing PR diff. It contains a mix of bugs, "
            "security issues, and logic errors. Find ALL issues."
        ),
        "code_snippet": """\
import os
import logging
import requests

API_KEY = "sk-live-abc123XYZ"        # ISSUE 1: hardcoded API key

logger = logging.getLogger(__name__)


def charge_user(user_id: int, amount: float, currency: str = "USD") -> dict:
    \"\"\"Charge a user via the external payments API.\"\"\"

    # ISSUE 2: missing validation for negative amounts
    payload = {
        "user_id": user_id,
        "amount": amount,
        "currency": currency,
    }

    response = requests.post(
        "https://payments.internal/charge",
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"},
        # ISSUE 3: no timeout mapped
    )

    if response.status_code != 200:
        logger.error(f"Charge failed for user {user_id}: {response.text}")
        return {"success": False}

    return response.json()


def log_transaction(user_id: int, amount: float, status: str) -> None:
    \"\"\"Write transaction record to the audit log.\"\"\"
    # ISSUE 4: sensitive data in plaintext log
    logger.info(f"Transaction: user={user_id} amount={amount} status={status}")


def calculate_fee(amount: float, rate: float = 0.029) -> float:
    \"\"\"Calculate the processing fee.\"\"\"
    # ISSUE 5: float math for currency
    return amount * rate
""",
        "ground_truth_issues": [
            {
                "line_number": 5,
                "correct_type": "security",
                "correct_severity": "critical",
                "match_keywords": "hardcoded live api key secret token source environment sk-live",
            },
            {
                "line_number": 14,
                "correct_type": "bug",
                "correct_severity": "high",
                "match_keywords": "validat check negative amount zero assert raise ValueError positive",
            },
            {
                "line_number": 20,
                "correct_type": "bug",
                "correct_severity": "high",
                "match_keywords": "timeout hang block indefinitely requests post thread stuck",
            },
            {
                "line_number": 37,
                "correct_type": "security",
                "correct_severity": "high",
                "match_keywords": "sensitive plaintext log mask redact financial money audit pii",
            },
            {
                "line_number": 43,
                "correct_type": "bug",
                "correct_severity": "medium",
                "match_keywords": "float floating-point decimal precise money rounding arithmetic",
            },
        ],
    },
]

def get_task(task_id: str) -> dict:
    """Return a task dict by ID, or raise KeyError."""
    for t in TASKS:
        if t["id"] == task_id:
            return t
    raise KeyError(f"Unknown task_id: {task_id!r}")

def list_task_ids() -> list:
    return [t["id"] for t in TASKS]

