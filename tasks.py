import re
import subprocess
import os
import platform
from typing import List, Dict, Any, Callable


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
    # Use a payload that contains SQL metacharacters to prove parameterization
    injection_payload = "admin'; DROP TABLE users; --"
    result = get_user_query(injection_payload)
    assert isinstance(result, tuple) and len(result) == 2, (
        "Security Error: get_user_query must return a (query, params) tuple, "
        "not a plain string."
    )
    query, params = result
    assert '%s' in query or '?' in query, (
        "Security Error: Query string must contain a parameterized placeholder "
        "(%s or ?) instead of embedding the value directly."
    )
    assert injection_payload not in query, (
        "Security Error: User input must NOT appear inside the query string -- "
        "it must be passed as a parameter."
    )
    assert injection_payload in str(params), (
        "Security Error: User input must be present in the params argument, "
        "not embedded in the query string."
    )

def test_crypto_fixed():
    source = inspect.getsource(hash_password)
    assert 'md5' not in source.lower(), (
        "Security Error: Weak MD5 hashing is still being used."
    )
    # Verify the function actually produces output
    h = hash_password("testpassword")
    assert isinstance(h, str) and len(h) >= 32, (
        "Security Error: hash_password must return a hex string of at least 32 chars."
    )
    # Verify different inputs produce different hashes (no trivial stub)
    assert hash_password("password_a") != hash_password("password_b"), (
        "Security Error: hash_password must produce unique hashes for different inputs."
    )
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
import inspect

def test_hardcoded_keys():
    # No hardcoded sk_live_ value may remain
    assert not STRIPE_LIVE_KEY or 'sk_live_' not in str(STRIPE_LIVE_KEY), (
        "Security Error: Hardcoded live Stripe key found in config.py!"
    )
    # Must use os.getenv (not just delete the variable)
    import payments.config as cfg_module
    src = inspect.getsource(cfg_module)
    assert 'os.getenv' in src or 'os.environ' in src, (
        "Security Error: STRIPE_LIVE_KEY must be loaded from the environment "
        "via os.getenv() or os.environ, not hardcoded or simply deleted."
    )
    # The literal prefix must not appear anywhere in source
    assert 'sk_live_' not in src, (
        "Security Error: Stripe live key prefix 'sk_live_' found in source -- "
        "remove it entirely and load from env."
    )

def test_performance():
    from payments.processor import process_payment
    import ast
    source = inspect.getsource(process_payment)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        assert False, "Performance Error: process_payment source could not be parsed."
    async_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'process_payment']
    assert async_funcs, "Performance Error: process_payment must be declared as 'async def'."
    has_await_sleep = any(
        isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
        for n in ast.walk(async_funcs[0])
    )
    assert has_await_sleep, "Performance Error: process_payment must use 'await asyncio.sleep(...)'."
"""
        }
    },
    {
        "task_id": "task_4_pr",
        "description": "PR #173: Implement LRU response cache. Three bugs found in code review: eviction never fires, move-to-recent is missing, and the deserializer crashes on None. Fix them so the tests pass.",
        "repository": {
            "cache/store.py": "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache = OrderedDict()\n\n    def get(self, key: str):\n        if key not in self.cache:\n            return None\n        # BUG: must move key to end (most-recently-used) on every access\n        return self.cache[key]\n\n    def put(self, key: str, value) -> None:\n        if key in self.cache:\n            del self.cache[key]\n        elif len(self.cache) >= self.capacity:\n            # BUG: oldest entry must be evicted here -- currently nothing happens\n            pass\n        self.cache[key] = value\n",
            "cache/serializer.py": "import json\n\ndef serialize(data) -> str:\n    return json.dumps(data)\n\ndef deserialize(raw: str):\n    # BUG: crashes with TypeError when raw is None -- should return None\n    return json.loads(raw)\n",
            "tests/test_cache.py": """
from cache.store import LRUCache
from cache.serializer import serialize, deserialize

def test_lru_eviction():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # must evict "a" (oldest)
    assert cache.get("a") is None, "LRU Error: oldest key 'a' should have been evicted when capacity was exceeded."
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_lru_access_order():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")       # 'a' is now most-recently used
    cache.put("c", 3)    # must evict 'b' (LRU), NOT 'a'
    assert cache.get("b") is None, "LRU Error: 'b' should have been evicted (least recently used) after 'a' was accessed."
    assert cache.get("a") == 1, "LRU Error: 'a' was accessed most recently and must NOT be evicted."

def test_deserialize_none():
    assert deserialize(None) is None, "Serializer Error: deserialize(None) must return None, not raise an exception."
    assert deserialize(serialize({"k": "v"})) == {"k": "v"}, "Serializer Error: round-trip must preserve dict."

def test_deserialize_values():
    for val in [42, [1, 2], True, "hello"]:
        assert deserialize(serialize(val)) == val, f"Serializer Error: round-trip failed for {val!r}."
"""
        }
    },
    {
        "task_id": "task_5_pr",
        "description": "PR #241: Fix statistics and data normalisation module. Two logic bugs: mean divides by wrong denominator, and normalise returns 0-100 instead of 0.0-1.0. Fix them so the tests pass.",
        "repository": {
            "stats/aggregator.py": "def mean(values: list) -> float:\n    if not values:\n        raise ValueError('Cannot compute mean of empty list')\n    # BUG: off-by-one -- divides by len+1 instead of len\n    return sum(values) / (len(values) + 1)\n\ndef variance(values: list) -> float:\n    if len(values) < 2:\n        raise ValueError('Variance requires at least 2 values')\n    m = mean(values)\n    return sum((x - m) ** 2 for x in values) / len(values)\n",
            "stats/normalizer.py": "def min_max_normalize(values: list) -> list:\n    if not values:\n        return []\n    lo, hi = min(values), max(values)\n    if lo == hi:\n        return [0.0] * len(values)\n    # BUG: should return 0.0-1.0 range, not 0-100\n    return [(x - lo) / (hi - lo) * 100 for x in values]\n",
            "tests/test_stats.py": """
from stats.aggregator import mean, variance
from stats.normalizer import min_max_normalize

def test_mean_basic():
    assert mean([1, 2, 3, 4, 5]) == 3.0, "Stats Error: mean([1,2,3,4,5]) must be 3.0."
    assert mean([10, 20]) == 15.0, "Stats Error: mean([10,20]) must be 15.0."
    assert mean([7]) == 7.0, "Stats Error: mean of single element must equal that element."

def test_variance():
    # Classical dataset: mean=5, variance=4.0
    result = variance([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(result - 4.0) < 0.001, f"Stats Error: variance must be 4.0, got {result:.4f}."

def test_normalize_range():
    result = min_max_normalize([0, 5, 10])
    assert result == [0.0, 0.5, 1.0], f"Normalizer Error: expected [0.0, 0.5, 1.0], got {result}."

def test_normalize_bounds():
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    result = min_max_normalize(values)
    assert all(0.0 <= v <= 1.0 for v in result), (
        f"Normalizer Error: all values must be in [0.0, 1.0], got {result}."
    )
    assert result[values.index(min(values))] == 0.0, "Normalizer Error: minimum must map to 0.0."
    assert result[values.index(max(values))] == 1.0, "Normalizer Error: maximum must map to 1.0."

def test_normalize_equal_values():
    result = min_max_normalize([7, 7, 7])
    assert all(v == 0.0 for v in result), "Normalizer Error: equal values must all normalise to 0.0."
"""
        }
    },
]


def get_task(task_id: str) -> Dict[str, Any]:
    for t in TASKS:
        if t["task_id"] == task_id:
            return t
    raise ValueError(f"Task {task_id} not found")


def list_task_ids() -> List[str]:
    return [t["task_id"] for t in TASKS]


# ---------------------------------------------------------------------------
# Per-task grader functions
# ---------------------------------------------------------------------------
# Each grader accepts a workspace_dir (path to the episode sandbox) and
# returns a float in [0.0, 1.0] representing the fraction of tests passing.
# The environment calls these via the GRADERS dispatch table so that each
# task has an explicit, named, importable grader — not a single anonymous
# shared method.
# ---------------------------------------------------------------------------

_CLEAN_ENV_KEYS = {"PATH", "HOME", "LANG", "LC_ALL"}
# Windows requires these for Python to initialise at all.
# Without SYSTEMROOT Python raises: Fatal Python error: _Py_HashRandomization_Init
_WINDOWS_ENV_KEYS = {"SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC"}


def _run_pytest(workspace_dir: str) -> float:
    """
    Shared pytest runner used by all graders.
    Returns the fraction of tests passing in workspace_dir (0.0–1.0).
    Uses a minimal clean environment so server credentials are not exposed,
    with platform-aware key allowlist so Windows hosts work correctly.
    """
    keys = _CLEAN_ENV_KEYS.copy()
    if platform.system() == "Windows":
        keys |= _WINDOWS_ENV_KEYS
    env = {k: v for k, v in os.environ.items() if k in keys}
    env["PYTHONPATH"] = workspace_dir
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if "PATH" not in env:
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"

    try:
        result = subprocess.run(
            "pytest tests/ --disable-warnings -v",
            cwd=workspace_dir, shell=True,
            capture_output=True, text=True, timeout=15, env=env
        )
        if result.returncode == 0:
            return 1.0

        output = result.stdout + result.stderr
        n_passed = sum(1 for line in output.splitlines() if " PASSED" in line)
        n_failed = sum(1 for line in output.splitlines() if " FAILED" in line or " ERROR" in line)
        total = n_passed + n_failed
        if total > 0:
            return n_passed / total

        # Fallback: parse summary line e.g. "1 failed, 1 passed in 0.12s"
        m_pass = re.search(r"(\d+) passed", output)
        m_fail = re.search(r"(\d+) failed", output)
        p = int(m_pass.group(1)) if m_pass else 0
        f = int(m_fail.group(1)) if m_fail else 0
        return p / (p + f) if (p + f) > 0 else 0.0

    except Exception:
        return 0.0


def grade_task_1_pr(workspace_dir: str) -> float:
    """
    Grader for task_1_pr — Security Audit.
    Checks:
      - get_user_query returns a (query, params) tuple with the user input in params
      - hash_password does not use MD5 and produces valid, unique hashes
    """
    return _run_pytest(workspace_dir)


def grade_task_2_pr(workspace_dir: str) -> float:
    """
    Grader for task_2_pr — Billing Logic.
    Checks:
      - calculate_total multiplies price × quantity before summing
      - apply_discount keeps 80% of the amount (not 20%)
    """
    return _run_pytest(workspace_dir)


def grade_task_3_pr(workspace_dir: str) -> float:
    """
    Grader for task_3_pr — Stripe Security + Async Performance.
    Checks:
      - STRIPE_LIVE_KEY loaded via os.getenv, no literal sk_live_ in source
      - process_payment is declared async def with an await expression
    """
    return _run_pytest(workspace_dir)


def grade_task_4_pr(workspace_dir: str) -> float:
    """
    Grader for task_4_pr — LRU Cache.
    Checks:
      - LRUCache evicts oldest entry when capacity is exceeded
      - LRUCache moves accessed keys to most-recently-used position
      - deserialize(None) returns None without raising
    """
    return _run_pytest(workspace_dir)


def grade_task_5_pr(workspace_dir: str) -> float:
    """
    Grader for task_5_pr — Statistics & Normalisation.
    Checks:
      - mean() divides by len(values), not len(values)+1
      - min_max_normalize() returns values in [0.0, 1.0], not [0, 100]
    """
    return _run_pytest(workspace_dir)


# Dispatch table: maps task_id → grader callable(workspace_dir) → float
GRADERS: Dict[str, Callable[[str], float]] = {
    "task_1_pr": grade_task_1_pr,
    "task_2_pr": grade_task_2_pr,
    "task_3_pr": grade_task_3_pr,
    "task_4_pr": grade_task_4_pr,
    "task_5_pr": grade_task_5_pr,
}
