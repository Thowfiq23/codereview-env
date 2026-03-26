"""
codereview_env/tasks.py
Complete task dataset: 3 tasks with code snippets and ground-truth issues.
All graders are fully deterministic — no LLM judge required.
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
                "issue_type": "bug",
                "severity": "high",
                "description": (
                    "Loop starts at index 1 instead of 0, causing the first element "
                    "to be excluded from the sum. The average will always be wrong."
                ),
                # Keywords the grader checks for in the agent's description
                "match_keywords": ["range(1", "index 0", "index 1", "off-by-one",
                                   "skips first", "first element", "starts at 1"],
                "correct_type": "bug",
                "correct_severity": "high",
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

    # ISSUE 2: SQL injection — user input directly in query string
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    db.close()
    return user is not None

def hash_password(password: str) -> str:
    \"\"\"Hash a password before storing.\"\"\"
    # ISSUE 3: MD5 is cryptographically broken for passwords
    return hashlib.md5(password.encode()).hexdigest()
""",
        "ground_truth_issues": [
            {
                "line_number": 4,
                "issue_type": "security",
                "severity": "high",
                "description": "Hardcoded secret key in source code. Should be loaded from environment variables or a secrets manager.",
                "match_keywords": ["hardcoded", "SECRET_KEY", "environment variable",
                                   "secret", "env var", "os.environ", "config"],
                "correct_type": "security",
                "correct_severity": "high",
            },
            {
                "line_number": 15,
                "issue_type": "security",
                "severity": "critical",
                "description": "SQL injection vulnerability — user input is directly interpolated into the query string. Use parameterised queries.",
                "match_keywords": ["sql injection", "f-string", "parameterised",
                                   "parameterized", "user input", "format string",
                                   "interpolat", "query string", "cursor.execute"],
                "correct_type": "security",
                "correct_severity": "critical",
            },
            {
                "line_number": 23,
                "issue_type": "security",
                "severity": "high",
                "description": "MD5 is cryptographically broken and must not be used for password hashing. Use bcrypt, argon2, or PBKDF2.",
                "match_keywords": ["md5", "broken", "weak", "bcrypt", "argon2",
                                   "pbkdf2", "cryptographic", "password hashing"],
                "correct_type": "security",
                "correct_severity": "high",
            },
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 3 — HARD
    # A realistic PR diff with 5 mixed issues across two files.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "task_3_multi",
        "difficulty": "hard",
        "filename": "payment_processor.py",
        "language": "python",
        "context": (
            "This PR adds a new payment processing module to a fintech API. "
            "It handles charging users and logging transactions. "
            "The code will be deployed to production next week."
        ),
        "task_description": (
            "Review this payment-processing PR diff. "
            "It contains a mix of bugs, security issues, and logic errors. "
            "Find ALL issues — partial credit is awarded per issue found."
        ),
        "code_snippet": """\
import os
import logging
import requests

API_KEY = "sk-live-abc123XYZ"        # ISSUE 1: hardcoded API key in source

logger = logging.getLogger(__name__)


def charge_user(user_id: int, amount: float, currency: str = "USD") -> dict:
    \"\"\"Charge a user via the external payments API.\"\"\"

    # ISSUE 2: amount is never validated — negative charges are possible
    payload = {
        "user_id": user_id,
        "amount": amount,
        "currency": currency,
    }

    response = requests.post(
        "https://payments.internal/charge",
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"},
        # ISSUE 3: no timeout — hangs indefinitely if the API is slow
    )

    if response.status_code != 200:
        logger.error(f"Charge failed for user {user_id}: {response.text}")
        return {"success": False}

    return response.json()


def log_transaction(user_id: int, amount: float, status: str) -> None:
    \"\"\"Write transaction record to the audit log.\"\"\"
    # ISSUE 4: sensitive financial data written to plaintext log
    logger.info(f"Transaction: user={user_id} amount={amount} status={status}")


def calculate_fee(amount: float, rate: float = 0.029) -> float:
    \"\"\"Calculate the processing fee.\"\"\"
    # ISSUE 5: floating-point arithmetic for money — precision errors accumulate
    return amount * rate
""",
        "ground_truth_issues": [
            {
                "line_number": 5,
                "issue_type": "security",
                "severity": "critical",
                "description": "Live API key hardcoded in source. Must be loaded from environment variables. Key must be rotated immediately.",
                "match_keywords": ["hardcoded", "api key", "API_KEY", "environment variable",
                                   "os.environ", "secret", "sk-live"],
                "correct_type": "security",
                "correct_severity": "critical",
            },
            {
                "line_number": 13,
                "issue_type": "bug",
                "severity": "high",
                "description": "The amount parameter is never validated. Negative or zero amounts would produce invalid charges.",
                "match_keywords": ["validat", "negative", "amount", "zero", "check",
                                   "assert", "raise", "ValueError"],
                "correct_type": "bug",
                "correct_severity": "high",
            },
            {
                "line_number": 21,
                "issue_type": "bug",
                "severity": "high",
                "description": "HTTP request has no timeout. If the payments API hangs, this will block indefinitely, causing thread exhaustion.",
                "match_keywords": ["timeout", "hang", "block", "indefinitely",
                                   "requests.post", "thread"],
                "correct_type": "bug",
                "correct_severity": "high",
            },
            {
                "line_number": 35,
                "issue_type": "security",
                "severity": "high",
                "description": "Sensitive financial data (user ID and amount) is logged in plaintext. Logs must be scrubbed or masked.",
                "match_keywords": ["sensitive", "plaintext", "log", "mask", "redact",
                                   "financial data", "amount", "audit"],
                "correct_type": "security",
                "correct_severity": "high",
            },
            {
                "line_number": 41,
                "issue_type": "bug",
                "severity": "medium",
                "description": "Floating-point arithmetic must not be used for monetary calculations. Use Python's decimal.Decimal instead.",
                "match_keywords": ["float", "decimal", "Decimal", "floating-point",
                                   "money", "precision", "rounding"],
                "correct_type": "bug",
                "correct_severity": "medium",
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
