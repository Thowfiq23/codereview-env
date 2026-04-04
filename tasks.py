from typing import List, Dict, Any

TASKS: List[Dict[str, Any]] = [
    # --- TASK 1: EASY (Security Basics) ---
    {
        "task_id": "task_1_pr",
        "description": "PR #42: Add user authentication and password hashing. Please review for security bugs.",
        "repository": {
            "README.md": "# Auth Module\nHandles user login.",
            "auth/models.py": "def get_user_query(username):\n    # Vulnerable to SQL injection\n    return f\"SELECT * FROM users WHERE username = '{username}'\"",
            "auth/crypto.py": "import hashlib\n\ndef hash_password(password: str):\n    # MD5 is weak/broken\n    return hashlib.md5(password.encode()).hexdigest()\n",
            "main.py": "from auth.models import get_user_query\nfrom auth.crypto import hash_password\n\nSECRET_KEY = 'hardcoded_super_secret_key'\n"
        },
        "ground_truth_issues": [
            {
                "file_path": "auth/models.py",
                "line_number": 3,
                "correct_type": "security",
                "correct_severity": "critical",
                "description": "SQL Injection vulnerability in f-string query",
                "match_keywords": "sql,injection,f-string"
            },
            {
                "file_path": "auth/crypto.py",
                "line_number": 4,
                "correct_type": "security",
                "correct_severity": "high",
                "description": "Using weak MD5 hashing algorithm",
                "match_keywords": "md5,weak,hashing"
            }
        ]
    },
    
    # --- TASK 2: MEDIUM (Logic Bugs across files) ---
    {
        "task_id": "task_2_pr",
        "description": "PR #88: Implement shopping cart billing and discount logic. Please review for correctness.",
        "repository": {
            "billing/cart.py": "from billing.discounts import apply_discount\n\ndef calculate_total(items):\n    total = 0\n    for item in items:\n        total += item['price'] # BUG: Forgot to multiply by item['quantity']!\n    return apply_discount(total)\n",
            "billing/discounts.py": "def apply_discount(amount):\n    # BUG: Applies 20% discount but accidentally returns just the discount amount, not the new total!\n    return amount * 0.20\n"
        },
        "ground_truth_issues": [
            {
                "file_path": "billing/cart.py",
                "line_number": 6,
                "correct_type": "logic",
                "correct_severity": "high",
                "description": "Fails to multiply item price by quantity",
                "match_keywords": "quantity,multiply,amount,price"
            },
            {
                "file_path": "billing/discounts.py",
                "line_number": 3,
                "correct_type": "logic",
                "correct_severity": "high",
                "description": "Returns the discount amount instead of the discounted total",
                "match_keywords": "subtract,total,discount amount,return"
            }
        ]
    },

    # --- TASK 3: HARD (Hidden Security & Performance) ---
    {
        "task_id": "task_3_pr",
        "description": "PR #105: Integrate Stripe payment processor and user caching.",
        "repository": {
            "payments/config.py": "import os\n\nSTRIPE_TEST_KEY = os.getenv('STRIPE_TEST_KEY')\nSTRIPE_LIVE_KEY = 'sk_live_9876543210qwerty' # CRITICAL: Hardcoded live API key\n",
            "payments/processor.py": "from payments.config import STRIPE_LIVE_KEY\nimport time\n\ndef process_payment(user_id, amount):\n    # Performance issue: synchronous sleep mimicking network\n    time.sleep(5)\n    return True\n"
        },
        "ground_truth_issues": [
            {
                "file_path": "payments/config.py",
                "line_number": 4,
                "correct_type": "security",
                "correct_severity": "critical",
                "description": "Hardcoded live Stripe API key exposed in source code",
                "match_keywords": "hardcoded,live,stripe,sk_live,key"
            },
            {
                "file_path": "payments/processor.py",
                "line_number": 6,
                "correct_type": "performance",
                "correct_severity": "medium",
                "description": "Synchronous blocking sleep call halts the entire thread",
                "match_keywords": "sleep,synchronous,blocking,performance"
            }
        ]
    }
]

def get_task(task_id: str) -> Dict[str, Any]:
    for t in TASKS:
        if t["task_id"] == task_id:
            return t
    raise ValueError(f"Task {task_id} not found")

def list_task_ids() -> List[str]:
    return [t["task_id"] for t in TASKS]
