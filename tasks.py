from typing import List, Dict, Any

TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "task_1_pr",
        "description": "PR #42: Add user authentication. The CI/CD pipeline is failing security checks. Fix the bugs and make the tests pass.",
        "repository": {
            "auth/models.py": "def get_user_query(username):\n    # Vulnerable to SQL injection\n    return f\"SELECT * FROM users WHERE username = '{username}'\"",
            "auth/crypto.py": "import hashlib\n\ndef hash_password(password: str):\n    # MD5 is weak/broken\n    return hashlib.md5(password.encode()).hexdigest()\n",
            "tests/test_auth.py": """
from auth.models import get_user_query
from auth.crypto import hash_password
import inspect

def test_sql_injection_fixed():
    query = get_user_query('admin')
    assert '%s' in str(query) or '?' in str(query), "Security Error: Code is still vulnerable to SQL Injection (use %s or ?)"

def test_crypto_fixed():
    source = inspect.getsource(hash_password)
    assert 'md5' not in source.lower(), "Security Error: Weak MD5 hashing is still being used."
"""
        }
    },
    {
        "task_id": "task_2_pr",
        "description": "PR #88: Implement shopping cart billing. Logic is failing. Fix the code so tests pass.",
        "repository": {
            "billing/cart.py": "from billing.discounts import apply_discount\n\ndef calculate_total(items):\n    total = 0\n    for item in items:\n        total += item['price']\n    return apply_discount(total)\n",
            "billing/discounts.py": "def apply_discount(amount):\n    return amount * 0.20\n",
            "tests/test_billing.py": """
from billing.cart import calculate_total
from billing.discounts import apply_discount

def test_cart_total():
    items = [{'price': 10, 'quantity': 2}, {'price': 20, 'quantity': 1}]
    # 10*2 + 20*1 = 40. Minus 20% = 32.
    assert calculate_total(items) == 32, "Logic Error: calculate_total did not account for quantity correctly."

def test_discount():
    assert apply_discount(100) == 80, "Logic Error: apply_discount returned the wrong final amount."
"""
        }
    },
    {
        "task_id": "task_3_pr",
        "description": "PR #105: Integrate Stripe payment processor. Performance timeout.",
        "repository": {
            "payments/config.py": "import os\n\nSTRIPE_TEST_KEY = os.getenv('STRIPE_TEST_KEY')\nSTRIPE_LIVE_KEY = 'sk_live_9876543210qwerty'\n",
            "payments/processor.py": "import time\n\ndef process_payment(user_id, amount):\n    time.sleep(5)\n    return True\n",
            "tests/test_payments.py": """
from payments.config import STRIPE_LIVE_KEY
from payments.processor import process_payment
import inspect

def test_hardcoded_keys():
    assert not STRIPE_LIVE_KEY or 'sk_live_' not in STRIPE_LIVE_KEY, "Security Error: Hardcoded live Stripe key found in config.py!"

def test_performance():
    source = inspect.getsource(process_payment)
    assert 'asyncio.sleep' in source or 'time.sleep' not in source, "Performance Error: Synchronous blocking time.sleep() detected."
"""
        }
    }
]

def get_task(task_id: str) -> Dict[str, Any]:
    for t in TASKS:
        if t["task_id"] == task_id:
            return t
    raise ValueError(f"Task {task_id} not found")

def list_task_ids() -> List[str]:
    return [t["task_id"] for t in TASKS]
