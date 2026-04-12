"""
Task definitions for CodeReview-Env.

Each task uses a _repo_factory(seed) callable so every episode reset generates
a fresh problem instance with different concrete values.  The LLM cannot
memorise answers — it must read the code, understand the bug, and fix the
algorithm correctly.

Difficulty progression:
  task_1_pr  Medium        — 3 bugs, user registration
  task_2_pr  Medium-Hard   — 3 bugs, order pricing (random prices/rates)
  task_3_pr  Hard          — 3 bugs, async payment processor
  task_4_pr  Hard          — 3 bugs, LRU cache
  task_5_pr  Very Hard     — 3 algorithmic bugs, analytics pipeline
                             (EMA formula, sample vs population std, window boundary)
                             All test values computed from the random seed —
                             the LLM must understand the maths to fix the code.
"""
import re
import subprocess
import os
import platform
import random
import math
from typing import List, Dict, Any, Callable


# =============================================================================
# Repository factories
# Each factory accepts an integer seed and returns {filepath: source_code}.
# Seeds are derived from the episode UUID by environment.py so each
# reset() gets a reproducible but unique problem instance.
# =============================================================================

def _make_task_1_repo(seed: int) -> dict:
    """
    User Registration — 3 bugs across 2 files.
      1. users/validator.py  email regex uses '*' (0-or-more) → accepts @example.com
      2. users/validator.py  password length check uses '> 8' → rejects exactly-8-char PW
      3. users/storage.py    MD5 hashing (32 hex chars) — need ≥64 (SHA-256+)

    Randomised per episode:
      • Which 3 of 6 valid email addresses appear in the test
      • Which of 4 password variant sets (all structurally equivalent) is used
    """
    rng = random.Random(seed)

    # ── random test parameters ──────────────────────────────────────────────
    email_pool = [
        "user@example.com",
        "first.last@company.io",
        "alice+filter@mail.example.net",
        "bob.smith@subdomain.org",
        "carol@example-corp.co.uk",
        "x123@university.edu",
    ]
    valid_emails = rng.sample(email_pool, 3)

    # Four structurally identical variants: (valid-8, invalid-7, no-upper, no-digit)
    pw_variants = [
        ("Passw0rd", "Passw0r", "password1", "PasswordX"),
        ("MyPas1sX", "MyPas1s", "mypas1sx", "MyPasssXX"),
        ("Hello1Wo", "Hello1W", "hello1wo", "HelloWorld"),
        ("Abc1DefG", "Abc1Def", "abc1defg", "AbcDefGhi"),
    ]
    pw_valid8, pw_7, pw_no_upper, pw_no_digit = pw_variants[rng.randint(0, 3)]

    # ── static buggy sources ─────────────────────────────────────────────────
    validator_src = (
        "import re\n"
        "\n"
        "def validate_email(email: str) -> bool:\n"
        "    # BUG 1: '*' allows empty local part -- '@example.com' is accepted\n"
        "    return bool(re.match(r'^[\\w.+-]*@([\\w-]+\\.)+\\w{2,}$', email))\n"
        "\n"
        "def validate_password(password: str) -> bool:\n"
        "    \"\"\"Password must be >=8 chars, contain uppercase and digit.\"\"\"\n"
        "    # BUG 2: '> 8' rejects exactly-8-char passwords; must be '>= 8'\n"
        "    has_length = len(password) > 8\n"
        "    has_upper  = any(c.isupper() for c in password)\n"
        "    has_digit  = any(c.isdigit() for c in password)\n"
        "    return has_length and has_upper and has_digit\n"
    )
    storage_src = (
        "import hashlib\n"
        "\n"
        "def hash_password(password: str) -> str:\n"
        "    \"\"\"Return a secure hex-digest of the password.\"\"\"\n"
        "    # BUG 3: MD5 produces only 32 hex chars and is cryptographically broken\n"
        "    return hashlib.md5(password.encode()).hexdigest()\n"
        "\n"
        "def store_user(email: str, password: str) -> dict:\n"
        "    return {'email': email, 'password_hash': hash_password(password)}\n"
    )

    # ── parametric test file ─────────────────────────────────────────────────
    # Uses .format(); {{ }} produces literal { } in the generated test source.
    test_template = """\
from users.validator import validate_email, validate_password
from users.storage import hash_password

# ── email ──────────────────────────────────────────────────────────────────

def test_email_rejects_empty_local_part():
    assert not validate_email("@example.com"), (
        "Validation Error: address with no local part before '@' must be rejected."
    )

def test_email_accepts_valid_addresses():
    for addr in {valid_emails}:
        assert validate_email(addr), (
            f"Validation Error: {{addr!r}} is a valid address but was rejected."
        )

def test_email_rejects_malformed():
    for addr in ["", "notanemail", "user@", "@@domain.com"]:
        assert not validate_email(addr), (
            f"Validation Error: malformed address {{addr!r}} was incorrectly accepted."
        )

# ── password ───────────────────────────────────────────────────────────────

def test_password_accepts_8_char_minimum():
    # {pw_valid8!r} is exactly 8 characters with uppercase + digit — must be valid
    assert validate_password({pw_valid8}), (
        "Validation Error: 8-character password satisfying all rules was rejected. "
        "Minimum length must be 8 (>= 8), not 9 (> 8)."
    )

def test_password_rejects_7_chars():
    assert not validate_password({pw_7}), (
        "Validation Error: 7-character password must be rejected."
    )

def test_password_rejects_no_uppercase():
    assert not validate_password({pw_no_upper}), (
        "Validation Error: password with no uppercase letter must be rejected."
    )

def test_password_rejects_no_digit():
    assert not validate_password({pw_no_digit}), (
        "Validation Error: password with no digit must be rejected."
    )

# ── hashing ────────────────────────────────────────────────────────────────

def test_hash_uses_strong_algorithm():
    h = hash_password("hunter2")
    assert len(h) >= 64, (
        f"Security Error: hash length {{len(h)}} chars is too short. "
        "MD5=32 chars is cryptographically broken; use SHA-256 (64 chars) or stronger."
    )

def test_hash_determinism_and_uniqueness():
    assert hash_password("alpha") != hash_password("beta"), (
        "Security Error: different passwords produced the same hash."
    )
    assert hash_password("same_pw") == hash_password("same_pw"), (
        "Security Error: same password produced different hashes on two calls."
    )
"""
    test_src = test_template.format(
        valid_emails=repr(valid_emails),
        pw_valid8=repr(pw_valid8),
        pw_7=repr(pw_7),
        pw_no_upper=repr(pw_no_upper),
        pw_no_digit=repr(pw_no_digit),
    )

    return {
        "users/validator.py": validator_src,
        "users/storage.py": storage_src,
        "tests/test_registration.py": test_src,
    }


def _make_task_2_repo(seed: int) -> dict:
    """
    Order Pricing Engine — 3 bugs across 2 files.
      1. orders/cart.py     ignores item['quantity'] — sums prices, not price×qty
      2. orders/pricing.py  apply_discount keeps the X% instead of removing it
      3. orders/pricing.py  apply_tax adds tax_rate as a raw offset, not a multiplier

    Randomised per episode:
      • 2–3 items with different prices and quantities
      • discount_pct: one of [10, 15, 20, 25]
      • tax_rate: one of [0.05, 0.08, 0.10, 0.15]
      • All expected values computed from the correct formula — the LLM must
        fix the algorithm to produce the right numbers, not hardcode them.
    """
    rng = random.Random(seed)

    # ── random pricing scenario ──────────────────────────────────────────────
    n_items = rng.randint(2, 3)
    prices    = [rng.choice([5, 8, 10, 12, 15, 20, 25]) for _ in range(n_items)]
    quantities = [rng.randint(1, 5) for _ in range(n_items)]
    items = [{"price": float(p), "quantity": q} for p, q in zip(prices, quantities)]

    discount_pct = rng.choice([10, 15, 20, 25])
    tax_rate     = rng.choice([0.05, 0.08, 0.10, 0.15])

    # Correct expected values (LLM must produce these after fixing the code)
    subtotal   = sum(item["price"] * item["quantity"] for item in items)
    discounted = subtotal * (1 - discount_pct / 100)
    total      = round(discounted * (1 + tax_rate), 4)

    # Per-primitive expected values for unit tests
    disc_unit_input  = rng.choice([50.0, 80.0, 100.0, 120.0, 200.0])
    disc_unit_expect = round(disc_unit_input * (1 - discount_pct / 100), 4)

    tax_unit_input   = round(disc_unit_input * (1 - discount_pct / 100), 4)
    tax_unit_expect  = round(tax_unit_input * (1 + tax_rate), 4)

    # ── static buggy sources ─────────────────────────────────────────────────
    cart_src = (
        "from orders.pricing import apply_discount, apply_tax\n"
        "\n"
        "def calculate_total(items: list, discount_pct: float, tax_rate: float) -> float:\n"
        "    \"\"\"Return the final order total after discount and tax.\"\"\"\n"
        "    subtotal = 0.0\n"
        "    for item in items:\n"
        "        # BUG 1: ignores quantity — must multiply price by quantity\n"
        "        subtotal += item['price']\n"
        "    discounted = apply_discount(subtotal, discount_pct)\n"
        "    return apply_tax(discounted, tax_rate)\n"
    )
    pricing_src = (
        "def apply_discount(amount: float, discount_pct: float) -> float:\n"
        "    \"\"\"Return amount after removing discount_pct percent.\"\"\"\n"
        "    # BUG 2: returns the discount portion, not the remaining amount\n"
        "    return amount * (discount_pct / 100)\n"
        "\n"
        "def apply_tax(amount: float, tax_rate: float) -> float:\n"
        "    \"\"\"Return amount with tax_rate (decimal, e.g. 0.10 = 10%%) applied.\"\"\"\n"
        "    # BUG 3: adds tax_rate as raw offset instead of multiplicative factor\n"
        "    # e.g. apply_tax(80, 0.10) returns 80.10 instead of 88.00\n"
        "    return amount + tax_rate\n"
    )

    # ── parametric test file ─────────────────────────────────────────────────
    test_template = """\
from orders.cart import calculate_total
from orders.pricing import apply_discount, apply_tax

# ── primitive unit tests ───────────────────────────────────────────────────

def test_discount_removes_percentage():
    # A {discount_pct}% discount on {disc_unit_input} should leave {disc_unit_expect}
    result = apply_discount({disc_unit_input}, {discount_pct})
    assert abs(result - {disc_unit_expect}) < 0.01, (
        f"Pricing Error: apply_discount({disc_unit_input}, {discount_pct}) "
        f"returned {{result:.4f}}, expected {disc_unit_expect:.4f}. "
        "A {discount_pct}% discount means {keep_pct}% of the amount remains."
    )

def test_tax_is_multiplicative():
    # {tax_pct}% tax on {tax_unit_input} should give {tax_unit_expect}
    result = apply_tax({tax_unit_input}, {tax_rate})
    assert abs(result - {tax_unit_expect}) < 0.01, (
        f"Pricing Error: apply_tax({tax_unit_input}, {tax_rate}) "
        f"returned {{result:.4f}}, expected {tax_unit_expect:.4f}. "
        "tax_rate is a decimal multiplier (e.g. 0.10 means ×1.10), not an additive offset."
    )

# ── integration test ───────────────────────────────────────────────────────

def test_cart_total_with_quantities():
    # Items: {items_repr}
    # Correct subtotal (price × qty each): {subtotal:.2f}
    # After {discount_pct}% discount:      {discounted:.2f}
    # After {tax_pct}% tax:                {total:.4f}
    items = {items}
    result = calculate_total(items, discount_pct={discount_pct}, tax_rate={tax_rate})
    assert abs(result - {total}) < 0.01, (
        f"Cart Error: calculate_total returned {{result:.4f}}, expected {total:.4f}. "
        "Check: (1) quantity is multiplied, (2) discount removes the percentage, "
        "(3) tax is a multiplicative factor."
    )

def test_cart_no_discount_no_tax():
    # Simple sanity check: single item, no discount, no tax
    items = [{{'price': 10.0, 'quantity': 3}}]
    result = calculate_total(items, discount_pct=0.0, tax_rate=0.0)
    assert abs(result - 30.0) < 0.01, (
        f"Cart Error: 3 × $10 with no discount and no tax should be $30.00, got {{result:.4f}}."
    )
"""
    test_src = test_template.format(
        discount_pct=discount_pct,
        keep_pct=100 - discount_pct,
        tax_rate=tax_rate,
        tax_pct=int(tax_rate * 100),
        disc_unit_input=disc_unit_input,
        disc_unit_expect=disc_unit_expect,
        tax_unit_input=tax_unit_input,
        tax_unit_expect=tax_unit_expect,
        items=repr(items),
        items_repr=items,
        subtotal=subtotal,
        discounted=discounted,
        total=total,
    )

    return {
        "orders/cart.py": cart_src,
        "orders/pricing.py": pricing_src,
        "tests/test_orders.py": test_src,
    }


def _make_task_3_repo(seed: int) -> dict:
    """
    Async Payment Processor — 3 bugs across 3 files.
      1. payments/config.py    hardcoded sk_live_ key
      2. payments/processor.py synchronous time.sleep (must be async def + await)
      3. payments/retry.py     range(max_retries + 1) → one extra attempt

    Randomised per episode:
      • max_retries value used in the retry test (2, 3, 4, or 5)
        The LLM cannot guess the expected call count without reading the code.
    """
    rng = random.Random(seed)
    max_retries = rng.choice([2, 3, 4, 5])

    config_src = (
        "import os\n"
        "\n"
        "STRIPE_TEST_KEY = os.getenv('STRIPE_TEST_KEY')\n"
        "# BUG 1: live key must never be hardcoded — load from environment\n"
        "STRIPE_LIVE_KEY = 'sk_live_9876543210qwerty'\n"
    )
    processor_src = (
        "import time\n"
        "\n"
        "# BUG 2: synchronous sleep blocks the entire event loop\n"
        "# Must be declared 'async def' and use 'await asyncio.sleep(...)'\n"
        "def process_payment(user_id: str, amount: float) -> bool:\n"
        "    time.sleep(0)\n"
        "    return True\n"
    )
    retry_src = (
        "import time\n"
        "\n"
        "def retry(fn, max_retries: int = 3, delay: float = 0.0):\n"
        "    \"\"\"\n"
        "    Call fn() up to max_retries times on failure.\n"
        "    Raises the last exception when all attempts are exhausted.\n"
        "    \"\"\"\n"
        "    last_exc = None\n"
        "    # BUG 3: range(max_retries + 1) gives max_retries+1 attempts\n"
        "    for attempt in range(max_retries + 1):\n"
        "        try:\n"
        "            return fn()\n"
        "        except Exception as e:\n"
        "            last_exc = e\n"
        "            if attempt < max_retries:\n"
        "                time.sleep(delay)\n"
        "    raise last_exc\n"
    )

    test_template = """\
import ast
import inspect
import payments.config as cfg_module

# ── security: no hardcoded live key ──────────────────────────────────────

def test_no_hardcoded_live_key():
    src = inspect.getsource(cfg_module)
    assert "sk_live_" not in src, (
        "Security Error: a hardcoded Stripe live key (sk_live_...) is present in "
        "config.py. Remove it and load the value from the environment."
    )

def test_live_key_loaded_from_env():
    src = inspect.getsource(cfg_module)
    assert "os.getenv" in src or "os.environ" in src, (
        "Security Error: STRIPE_LIVE_KEY must be loaded from the environment "
        "(os.getenv or os.environ), not hardcoded or simply deleted."
    )

# ── async: must not block event loop ─────────────────────────────────────

def test_process_payment_is_async():
    from payments.processor import process_payment
    tree = ast.parse(inspect.getsource(process_payment))
    async_defs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "process_payment"
    ]
    assert async_defs, (
        "Performance Error: process_payment must be 'async def' so it does not "
        "block the event loop."
    )

def test_process_payment_awaits():
    from payments.processor import process_payment
    tree = ast.parse(inspect.getsource(process_payment))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "process_payment"
    )
    assert any(isinstance(n, ast.Await) for n in ast.walk(fn)), (
        "Performance Error: process_payment must use 'await' (e.g. await asyncio.sleep(...))."
    )

# ── retry: exact attempt count ───────────────────────────────────────────

def test_retry_exact_attempt_count():
    from payments.retry import retry
    calls = []

    def always_fails():
        calls.append(1)
        raise RuntimeError("fail")

    try:
        retry(always_fails, max_retries={max_retries}, delay=0.0)
    except RuntimeError:
        pass

    assert len(calls) == {max_retries}, (
        f"Retry Error: retry(fn, max_retries={max_retries}) must call fn exactly "
        f"{max_retries} times, but called it {{len(calls)}} times."
    )

def test_retry_returns_on_success():
    from payments.retry import retry
    assert retry(lambda: 42, max_retries={max_retries}) == 42, (
        "Retry Error: retry must return the function result when it succeeds."
    )
"""
    test_src = test_template.format(max_retries=max_retries)

    return {
        "payments/config.py": config_src,
        "payments/processor.py": processor_src,
        "payments/retry.py": retry_src,
        "tests/test_payments.py": test_src,
    }


def _make_task_4_repo(seed: int) -> dict:
    """
    LRU Cache + Serialiser — 3 bugs across 2 files.
      1. cache/store.py      get() doesn't call move_to_end() → wrong eviction order
      2. cache/store.py      put() doesn't popitem(last=False) → never evicts
      3. cache/serializer.py deserialize(None) raises TypeError instead of returning None

    Randomised per episode:
      • Cache capacity (2, 3, or 4)
      • Key names (drawn from a pool so access patterns differ)
      The LLM must implement both LRU invariants correctly regardless of capacity.
    """
    rng = random.Random(seed)

    capacity = rng.choice([2, 3, 4])
    key_pool  = ["alpha", "beta", "gamma", "delta", "epsilon"]
    keys = rng.sample(key_pool, capacity + 1)   # capacity + 1 keys → triggers eviction
    k = keys  # shorthand

    store_src = (
        "from collections import OrderedDict\n"
        "\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity: int):\n"
        "        self.capacity = capacity\n"
        "        self.cache: OrderedDict = OrderedDict()\n"
        "\n"
        "    def get(self, key: str):\n"
        "        \"\"\"Return value for key, or None if absent.\"\"\"\n"
        "        if key not in self.cache:\n"
        "            return None\n"
        "        # BUG 1: must call self.cache.move_to_end(key) to mark as recently used\n"
        "        return self.cache[key]\n"
        "\n"
        "    def put(self, key: str, value) -> None:\n"
        "        \"\"\"Insert/update key; evict the LRU entry when over capacity.\"\"\"\n"
        "        if key in self.cache:\n"
        "            del self.cache[key]\n"
        "        elif len(self.cache) >= self.capacity:\n"
        "            # BUG 2: must call self.cache.popitem(last=False) to evict oldest\n"
        "            pass\n"
        "        self.cache[key] = value\n"
    )
    serializer_src = (
        "import json\n"
        "\n"
        "def serialize(data) -> str:\n"
        "    return json.dumps(data)\n"
        "\n"
        "def deserialize(raw: str):\n"
        "    \"\"\"Decode a JSON string. Returns None when raw is None.\"\"\"\n"
        "    # BUG 3: json.loads(None) raises TypeError — must guard for None input\n"
        "    return json.loads(raw)\n"
    )

    # Access pattern for the MRU promotion test:
    # Insert k[0]..k[cap-1], access k[0] (makes it MRU), insert k[cap] → should evict k[1]
    mru_evicted = k[1]
    mru_kept    = k[0]
    overflow_key = k[capacity]

    test_template = """\
from cache.store import LRUCache
from cache.serializer import serialize, deserialize

# ── eviction when at capacity ─────────────────────────────────────────────

def test_evicts_lru_on_overflow():
    cache = LRUCache(capacity={capacity})
    # Fill to capacity
    for i, key in enumerate({fill_keys}):
        cache.put(key, i)
    # One more insert must evict the oldest key
    cache.put({overflow_key!r}, 99)
    assert cache.get({first_key!r}) is None, (
        "LRU Error: {first_key!r} was inserted first and must be evicted "
        "when capacity is exceeded, but it is still present."
    )
    assert cache.get({overflow_key!r}) == 99

# ── get() must promote key to MRU ────────────────────────────────────────

def test_get_promotes_to_most_recent():
    cache = LRUCache(capacity={capacity})
    for i, key in enumerate({fill_keys}):
        cache.put(key, i)
    cache.get({mru_kept!r})          # access makes this MRU
    cache.put({overflow_key!r}, 99)   # should now evict {mru_evicted!r}, not {mru_kept!r}
    assert cache.get({mru_evicted!r}) is None, (
        "LRU Error: {mru_evicted!r} should be evicted (LRU after {mru_kept!r} was accessed), "
        "but {mru_kept!r} was evicted instead. get() must call move_to_end()."
    )
    assert cache.get({mru_kept!r}) is not None, (
        "LRU Error: {mru_kept!r} was accessed most recently and must NOT be evicted."
    )

def test_update_promotes_key():
    cache = LRUCache(capacity={capacity})
    for i, key in enumerate({fill_keys}):
        cache.put(key, i)
    cache.put({first_key!r}, 999)     # update → becomes MRU
    cache.put({overflow_key!r}, 88)   # should evict {second_key!r}
    assert cache.get({second_key!r}) is None, (
        "LRU Error: after updating {first_key!r} (making it MRU), "
        "{second_key!r} should be the LRU and get evicted."
    )
    assert cache.get({first_key!r}) == 999

# ── serialiser None guard ─────────────────────────────────────────────────

def test_deserialize_none_returns_none():
    assert deserialize(None) is None, (
        "Serialiser Error: deserialize(None) must return None, not raise TypeError."
    )

def test_serialize_roundtrip():
    for val in [42, "hello", [1, 2], {{"k": "v"}}, True, None]:
        assert deserialize(serialize(val)) == val, (
            f"Serialiser Error: round-trip failed for {{val!r}}."
        )
"""
    fill_keys = k[:capacity]
    test_src = test_template.format(
        capacity=capacity,
        fill_keys=repr(fill_keys),
        first_key=fill_keys[0],
        second_key=fill_keys[1],
        mru_kept=mru_kept,
        mru_evicted=mru_evicted,
        overflow_key=overflow_key,
    )

    return {
        "cache/store.py": store_src,
        "cache/serializer.py": serializer_src,
        "tests/test_cache.py": test_src,
    }


def _make_task_5_repo(seed: int) -> dict:
    """
    Analytics Pipeline — 3 algorithmic bugs across 3 files.

      1. analytics/aggregator.py  EMA weights are SWAPPED:
           buggy:   ema = alpha * prev + (1-alpha) * new_value
           correct: ema = alpha * new_value + (1-alpha) * prev

      2. analytics/detector.py  Uses POPULATION std (divides by n) instead of
           SAMPLE std (divides by n-1), causing false positives on small datasets.

      3. analytics/window.py  Off-by-one: range(n - size) drops the last window;
           must be range(n - size + 1).

    Randomised per episode:
      • EMA alpha (0.2, 0.3, or 0.4) and a 4-value sequence.
        Expected EMA computed from the CORRECT formula — the LLM must fix the
        formula to produce the right answer, cannot hardcode it.

      • Base value for the false-positive anomaly test — proves sample std is needed.
        Mathematically: [base]*4 + [base*2] always gives z ≈ 1.789 with sample std
        (< threshold 2.0, NOT anomaly) but z = 2.0 with population std (false positive).

      • Window size and data length — the LLM must fix the range boundary to get
        the right window count, which changes every episode.
    """
    rng = random.Random(seed)

    # ── EMA parameters ───────────────────────────────────────────────────────
    alpha = rng.choice([0.2, 0.3, 0.4])
    seq   = [float(rng.randint(5, 50)) for _ in range(4)]

    # Compute correct EMA for the full sequence
    ema_val = seq[0]
    for v in seq[1:]:
        ema_val = alpha * v + (1.0 - alpha) * ema_val
    ema_expected = round(ema_val, 9)

    # Compute BUGGY EMA to confirm the test would fail with the bug
    ema_buggy = seq[0]
    for v in seq[1:]:
        ema_buggy = alpha * ema_buggy + (1.0 - alpha) * v
    # (used only for documentation; not embedded in tests)

    # Two-value check: prev=seq[-2] result, new=seq[-1]
    # After all but last update, correct ema_prev was:
    ema_prev = seq[0]
    for v in seq[1:-1]:
        ema_prev = alpha * v + (1.0 - alpha) * ema_prev

    # ── Anomaly detection parameters ─────────────────────────────────────────
    # [base]*4 + [base*2]: z_outlier ≈ 1.789 with sample std (NOT anomaly at 2.0)
    #                      z_outlier = 2.0   with pop std    (false positive)
    fp_base    = rng.randint(3, 15)
    fp_outlier = fp_base * 2
    fp_values  = [fp_base] * 4 + [fp_outlier]

    # Clear outlier: [mean_val]*8 + [outlier]; z_outlier = 8/3 ≈ 2.667 with sample std
    clear_base    = rng.randint(5, 20)
    clear_outlier = clear_base + rng.randint(20, 60)
    clear_values  = [clear_base] * 8 + [clear_outlier]

    # ── Window parameters ────────────────────────────────────────────────────
    window_size    = rng.choice([2, 3, 4])
    extra          = rng.randint(2, 4)
    data_len       = window_size + extra
    window_data    = list(range(1, data_len + 1))
    expected_count = data_len - window_size + 1
    expected_last  = window_data[data_len - window_size:]

    # ── Static buggy sources ─────────────────────────────────────────────────
    aggregator_src = (
        "class ExponentialMovingAverage:\n"
        "    \"\"\"\n"
        "    Exponential Moving Average.\n"
        "    Correct formula: ema_t = alpha * x_t + (1 - alpha) * ema_{t-1}\n"
        "    alpha in (0, 1] — higher alpha gives more weight to recent values.\n"
        "    \"\"\"\n"
        "    def __init__(self, alpha: float):\n"
        "        if not (0 < alpha <= 1):\n"
        "            raise ValueError('alpha must be in (0, 1]')\n"
        "        self.alpha = alpha\n"
        "        self._value = None\n"
        "\n"
        "    def update(self, new_value: float) -> None:\n"
        "        if self._value is None:\n"
        "            self._value = float(new_value)\n"
        "        else:\n"
        "            # BUG 1: alpha and (1-alpha) are applied to the WRONG terms.\n"
        "            # Should be: alpha * new_value + (1 - alpha) * self._value\n"
        "            self._value = self.alpha * self._value + (1 - self.alpha) * new_value\n"
        "\n"
        "    @property\n"
        "    def value(self) -> float:\n"
        "        if self._value is None:\n"
        "            raise RuntimeError('No data: call update() first')\n"
        "        return self._value\n"
    )

    detector_src = (
        "import math\n"
        "\n"
        "def _mean(values: list) -> float:\n"
        "    return sum(values) / len(values)\n"
        "\n"
        "def _std(values: list) -> float:\n"
        "    \"\"\"Standard deviation for z-score normalisation.\"\"\"\n"
        "    if len(values) < 2:\n"
        "        return 0.0\n"
        "    m = _mean(values)\n"
        "    # BUG 2: divides by len(values) — population std (biased, ddof=0).\n"
        "    # Must divide by len(values) - 1 — sample std (unbiased, ddof=1).\n"
        "    variance = sum((x - m) ** 2 for x in values) / len(values)\n"
        "    return math.sqrt(variance)\n"
        "\n"
        "def detect_anomalies(values: list, threshold: float = 2.0) -> list:\n"
        "    \"\"\"Return bool list; True where |z-score| >= threshold.\"\"\"\n"
        "    if len(values) < 2:\n"
        "        return [False] * len(values)\n"
        "    m = _mean(values)\n"
        "    s = _std(values)\n"
        "    if s == 0:\n"
        "        return [False] * len(values)\n"
        "    return [abs((x - m) / s) >= threshold for x in values]\n"
    )

    window_src = (
        "from typing import List\n"
        "\n"
        "def sliding_window(data: list, size: int) -> List[list]:\n"
        "    \"\"\"\n"
        "    Return all contiguous sub-lists of length 'size'.\n"
        "    E.g. sliding_window([1,2,3,4,5], 3) -> [[1,2,3],[2,3,4],[3,4,5]]\n"
        "    \"\"\"\n"
        "    if size <= 0 or size > len(data):\n"
        "        return []\n"
        "    # BUG 3: range(n - size) is one step short — drops the last window.\n"
        "    # Must be range(len(data) - size + 1).\n"
        "    return [data[i:i + size] for i in range(len(data) - size)]\n"
    )

    # ── Pre-computed values for test template ────────────────────────────────
    one_minus_alpha_r9   = round(1 - alpha, 9)
    seq_last             = seq[-1]
    seq_second_last      = seq[-2]
    seq_prefix_repr      = repr(seq[:-1])
    ema_prev_r6          = round(ema_prev, 6)
    two_step_expected_r6 = round(alpha * seq_last + (1 - alpha) * ema_prev, 6)
    steps_comment        = "\n".join(
        f"#   ema_{i+1} = {alpha}*{v} + {one_minus_alpha_r9}*prev"
        for i, v in enumerate(seq[1:])
    )

    # ── Parametric test file ─────────────────────────────────────────────────
    test_template = """\
from analytics.aggregator import ExponentialMovingAverage
from analytics.detector import detect_anomalies
from analytics.window import sliding_window

# ── EMA: correct alpha weighting ─────────────────────────────────────────
# alpha={alpha}, sequence={seq}
# Correct step-by-step:
#   ema_0 = {seq0}
{steps_comment}
# Final expected value: {ema_expected}

def test_ema_first_update_is_identity():
    ema = ExponentialMovingAverage(alpha={alpha})
    ema.update({seq0})
    assert abs(ema.value - {seq0}) < 1e-9, (
        f"EMA Error: first update({seq0}) should set value to {seq0}, got {{ema.value}}."
    )

def test_ema_full_sequence():
    ema = ExponentialMovingAverage(alpha={alpha})
    for v in {seq}:
        ema.update(v)
    assert abs(ema.value - {ema_expected}) < 1e-6, (
        f"EMA Error: with alpha={alpha} and sequence {seq}, "
        f"expected {ema_expected:.6f} but got {{ema.value:.6f}}. "
        "Check the formula: ema = alpha * new_value + (1 - alpha) * previous_ema."
    )

def test_ema_two_step():
    # Isolated two-step check: prev={seq_second_last}, new={seq_last}, alpha={alpha}
    # Correct: {alpha}*{seq_last} + {one_minus_alpha_r9}*{ema_prev_r6} = {two_step_expected_r6}
    ema = ExponentialMovingAverage(alpha={alpha})
    for v in {seq_prefix_repr}:
        ema.update(v)
    prev = ema.value
    ema.update({seq_last})
    expected = {alpha} * {seq_last} + {one_minus_alpha_r9} * prev
    assert abs(ema.value - expected) < 1e-9, (
        f"EMA Error: update({seq_last}) with prev={{prev:.4f}} and alpha={alpha} "
        f"should give {{expected:.6f}}, got {{ema.value:.6f}}."
    )

# ── Anomaly detection: sample std (ddof=1) ───────────────────────────────
# Dataset: {fp_values}
# With population std (buggy): z({fp_outlier}) = 2.0  -> flagged (FALSE POSITIVE)
# With sample    std (correct): z({fp_outlier}) ~= 1.789 -> not flagged (correct)

def test_no_false_positive_with_sample_std():
    values = {fp_values}
    result = detect_anomalies(values, threshold=2.0)
    assert not result[-1], (
        "Detector Error: {fp_outlier} in {fp_values} is NOT an anomaly at threshold=2.0 "
        "when using sample std (ddof=1). Population std gives a false positive here. "
        "Fix _std() to divide by (n-1) instead of n."
    )

def test_detects_clear_outlier():
    # [{clear_base}]*8 + [{clear_outlier}]: z_outlier ~= 2.667 with sample std (> 2.0)
    values = {clear_values}
    result = detect_anomalies(values, threshold=2.0)
    assert result[-1], (
        "Detector Error: {clear_outlier} in a baseline of {clear_base}s is a "
        "clear outlier and must be detected."
    )
    assert not any(result[:-1]), (
        "Detector Error: only the last value should be flagged -- the baseline values "
        "must not be false-positives."
    )

# ── Sliding window: correct boundary ─────────────────────────────────────
# data={window_data}, size={window_size} -> expect {expected_count} windows

def test_sliding_window_count():
    windows = sliding_window({window_data}, size={window_size})
    assert len(windows) == {expected_count}, (
        f"Window Error: sliding_window over {data_len} elements with size={window_size} "
        f"should produce {expected_count} windows, got {{len(windows)}}. "
        "Fix the range: range(n - size + 1), not range(n - size)."
    )

def test_sliding_window_last_is_correct():
    windows = sliding_window({window_data}, size={window_size})
    assert windows[-1] == {expected_last}, (
        f"Window Error: last window should be {expected_last}, got {{windows[-1] if windows else 'none'}}."
    )

def test_sliding_window_full_span():
    data = list(range(1, {window_size} + 1))
    windows = sliding_window(data, size={window_size})
    assert windows == [data], (
        "Window Error: size==len(data) must produce exactly one window equal to the full list."
    )
"""
    test_src = test_template.format(
        alpha=alpha,
        seq=repr(seq),
        seq0=seq[0],
        seq_prefix_repr=seq_prefix_repr,
        seq_last=seq_last,
        seq_second_last=seq_second_last,
        one_minus_alpha_r9=one_minus_alpha_r9,
        ema_expected=ema_expected,
        ema_prev_r6=ema_prev_r6,
        two_step_expected_r6=two_step_expected_r6,
        steps_comment=steps_comment,
        fp_values=repr(fp_values),
        fp_base=fp_base,
        fp_outlier=fp_outlier,
        clear_base=clear_base,
        clear_outlier=clear_outlier,
        clear_values=repr(clear_values),
        window_data=repr(window_data),
        window_size=window_size,
        expected_count=expected_count,
        expected_last=repr(expected_last),
        data_len=data_len,
    )

    return {
        "analytics/aggregator.py": aggregator_src,
        "analytics/detector.py": detector_src,
        "analytics/window.py": window_src,
        "tests/test_analytics.py": test_src,
    }


def _make_task_6_repo(seed: int) -> dict:
    """
    Document Search + Ranking — 3 bugs with a coupled scoring/ranking trap.

    Files:
      search/indexer.py  — Bug 1: splits on whitespace only (punctuation sticks to tokens)
      search/scorer.py   — Bug 2: divides by len(doc_terms) instead of len(query_terms)
      search/ranker.py   — Bug 3: sorts ascending instead of descending

    THE TRAP — why this is hard:
      test_rank_best_match_first PASSES with BOTH Bug 2 and Bug 3 present because
      the verbose document (all query terms + many extras) gets a LOW buggy score
      (few matches / many total terms) while the short imprecise document gets a
      HIGH buggy score (partial matches / few total terms).  Ascending sort then
      accidentally puts the verbose doc first — the correct answer.

      Fix Bug 2 alone → scores are now correct (verbose=1.0 > short<1.0) but the
      ascending sort (Bug 3) promotes the short doc → ranking test FAILS.

      Fix Bug 3 alone → descending sort with still-wrong scores promotes the short
      doc (higher buggy score) → ranking test FAILS.

      Both must be fixed simultaneously.

    Parametric per episode:
      • Words drawn from a 16-word pool, shuffled by seed
      • Query size: 2 or 3 terms
      • Extra term count in verbose doc: 3–6
    """
    rng = random.Random(seed)

    WORD_POOL = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
        "eta",   "theta","iota",  "kappa", "lambda",  "mu",
        "nu",    "xi",   "omicron","pi",
    ]
    pool = WORD_POOL[:]
    rng.shuffle(pool)

    # ── Parameters ────────────────────────────────────────────────────────────
    q_size      = rng.choice([2, 3])
    extra_count = rng.randint(3, 6)

    query_terms    = pool[:q_size]
    extra_terms    = pool[q_size : q_size + extra_count]
    # verbose doc: all query terms + many extras  (correct winner; low buggy score)
    verbose_terms  = query_terms + extra_terms         # len = q_size + extra_count
    # short doc: only (q_size-1) query matches + 1 non-query term (wrong loser; high buggy score)
    short_terms    = query_terms[:-1] + [pool[q_size + extra_count]]  # len = q_size

    # Invariant check: buggy_verbose < buggy_short  (ascending accidentally correct)
    buggy_verbose  = q_size / len(verbose_terms)
    buggy_short    = (q_size - 1) / len(short_terms)
    assert buggy_verbose < buggy_short, "Coupling invariant failed for this seed"

    correct_short_num = q_size - 1
    correct_short_str = f"{correct_short_num}/{q_size}"

    # Tokenizer test words (independent of coupling)
    tok_word1 = pool[q_size + extra_count + 1]
    tok_word2 = pool[q_size + extra_count + 2]
    tok_punc  = rng.choice([",", "!", "?"])
    tok_input = f"{tok_word1}{tok_punc} {tok_word2}!"
    tok_expected = [tok_word1, tok_word2]

    # Scorer standalone test: doc with all query terms + 1 extra → must score 1.0
    scorer_extra_word = pool[q_size + extra_count + 3]
    scorer_doc_terms  = query_terms + [scorer_extra_word]

    # ── Pre-computed repr strings (no Python expressions in format fields) ────
    query_repr        = repr(query_terms)
    verbose_repr      = repr(verbose_terms)
    short_repr        = repr(short_terms)
    tok_input_repr    = repr(tok_input)
    tok_expected_repr = repr(tok_expected)
    scorer_doc_repr   = repr(scorer_doc_terms)
    q_size_minus_1    = q_size - 1
    buggy_verbose_r4  = round(buggy_verbose, 4)
    buggy_short_r4    = round(buggy_short, 4)

    # ── Static source files ───────────────────────────────────────────────────
    indexer_src = (
        "import re\n"
        "\n"
        "def tokenize(text: str) -> list:\n"
        "    \"\"\"\n"
        "    Split text into lowercase tokens, stripping all punctuation.\n"
        "    E.g. tokenize('Hello, world!') -> ['hello', 'world']\n"
        "    \"\"\"\n"
        "    # BUG 1: splits on whitespace only -- punctuation sticks to tokens.\n"
        "    # Fix: use re.split(r'\\W+', text.lower()) and filter empty strings.\n"
        "    return [w.lower() for w in text.split() if w]\n"
    )

    scorer_src = (
        "def relevance_score(query_terms: list, doc_terms: list) -> float:\n"
        "    \"\"\"\n"
        "    Fraction of query terms found in the document.  Range [0.0, 1.0].\n"
        "    1.0 means every query term appears at least once in the document.\n"
        "    \"\"\"\n"
        "    if not query_terms:\n"
        "        return 0.0\n"
        "    matches = len(set(query_terms) & set(doc_terms))\n"
        "    # BUG 2: divides by len(doc_terms) -- score unfairly penalises long docs.\n"
        "    # Fix: divide by len(query_terms) so a full match always returns 1.0.\n"
        "    return matches / len(doc_terms)\n"
    )

    ranker_src = (
        "from search.scorer import relevance_score\n"
        "\n"
        "def rank_documents(query_terms: list, documents: list) -> list:\n"
        "    \"\"\"\n"
        "    Return documents sorted by relevance to query_terms, highest first.\n"
        "    Each document is a dict with at least a 'terms' key (list[str]).\n"
        "    \"\"\"\n"
        "    scored = [\n"
        "        (doc, relevance_score(query_terms, doc['terms']))\n"
        "        for doc in documents\n"
        "    ]\n"
        "    # BUG 3: sorted ascending (lowest score first) -- must be descending.\n"
        "    # Fix: add reverse=True to sorted().\n"
        "    return [doc for doc, _ in sorted(scored, key=lambda x: x[1])]\n"
    )

    # ── Parametric test file ──────────────────────────────────────────────────
    test_template = """\
from search.indexer import tokenize
from search.scorer import relevance_score
from search.ranker import rank_documents

# ── Tokenizer ────────────────────────────────────────────────────────────────

def test_tokenize_strips_punctuation():
    result = tokenize({tok_input_repr})
    assert result == {tok_expected_repr}, (
        "Indexer Error: tokenize() must strip punctuation. "
        "Use re.split(r'\\\\W+', ...) not str.split(). Got: " + repr(result)
    )

def test_tokenize_lowercases():
    assert tokenize("Alpha BETA") == ["alpha", "beta"], (
        "Indexer Error: tokenize() must lowercase all tokens."
    )

# ── Scorer ───────────────────────────────────────────────────────────────────

def test_full_query_match_scores_one():
    # doc has all {q_size} query terms plus 1 extra; every query term is present
    score = relevance_score({query_repr}, {scorer_doc_repr})
    assert abs(score - 1.0) < 1e-9, (
        f"Scorer Error: all query terms present -> score must be 1.0, got {{score:.4f}}. "
        "Fix: divide by len(query_terms), not len(doc_terms)."
    )

def test_empty_query_scores_zero():
    assert relevance_score([], ["any", "word"]) == 0.0, (
        "Scorer Error: empty query must score 0.0."
    )

def test_no_match_scores_zero():
    assert relevance_score(["missing"], ["other", "words"]) == 0.0, (
        "Scorer Error: no matching terms must score 0.0."
    )

# ── Ranker (coupled trap) ────────────────────────────────────────────────────
# verbose_doc = {verbose_repr}
#   all {q_size} query terms + {extra_count} extras -> correct score 1.0
#   buggy score (divide by doc length): {buggy_verbose_r4}
#
# short_doc = {short_repr}
#   only {q_size_minus_1} of {q_size} query terms -> correct score {correct_short_str}
#   buggy score (divide by doc length): {buggy_short_r4}
#
# TRAP: buggy_verbose ({buggy_verbose_r4}) < buggy_short ({buggy_short_r4})
#   => ascending sort accidentally puts verbose_doc first (correct answer).
#   Fix scorer alone: verbose=1.0 > short, but ascending puts short first -> FAIL.
#   Fix ranker alone: descending, but buggy scores put short first -> FAIL.
#   Must fix BOTH scorer and ranker together.

def test_rank_best_match_first():
    query       = {query_repr}
    verbose_doc = {{"id": "verbose", "terms": {verbose_repr}}}
    short_doc   = {{"id": "short",   "terms": {short_repr}}}
    results = rank_documents(query, [verbose_doc, short_doc])
    assert results[0]["id"] == "verbose", (
        f"Ranker Error: verbose_doc matches all {{len(query)}} query terms (score 1.0) "
        f"and must rank first. Got order: {{[d['id'] for d in results]}}. "
        "Hint: fix BOTH scorer (divide by len(query_terms)) "
        "AND ranker (reverse=True) -- one fix without the other regresses this test."
    )

def test_rank_preserves_count():
    query       = {query_repr}
    verbose_doc = {{"id": "verbose", "terms": {verbose_repr}}}
    short_doc   = {{"id": "short",   "terms": {short_repr}}}
    results = rank_documents(query, [verbose_doc, short_doc])
    assert len(results) == 2, (
        f"Ranker Error: rank_documents must return all documents. Got {{len(results)}}."
    )
"""
    test_src = test_template.format(
        tok_input_repr=tok_input_repr,
        tok_expected_repr=tok_expected_repr,
        q_size=q_size,
        q_size_minus_1=q_size_minus_1,
        extra_count=extra_count,
        query_repr=query_repr,
        verbose_repr=verbose_repr,
        short_repr=short_repr,
        scorer_doc_repr=scorer_doc_repr,
        buggy_verbose_r4=buggy_verbose_r4,
        buggy_short_r4=buggy_short_r4,
        correct_short_str=correct_short_str,
    )

    return {
        "search/__init__.py": "",
        "search/indexer.py":  indexer_src,
        "search/scorer.py":   scorer_src,
        "search/ranker.py":   ranker_src,
        "tests/test_search.py": test_src,
    }


# =============================================================================
# Task manifest
# =============================================================================

TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "task_1_pr",
        "description": (
            "PR #47: User registration service is failing CI. "
            "Validation rejects some valid inputs and the password storage "
            "is flagged as insecure. Fix the bugs so the tests pass."
        ),
        "_repo_factory": _make_task_1_repo,
    },
    {
        "task_id": "task_2_pr",
        "description": (
            "PR #91: Order totals are wrong for quantity-based line items. "
            "Discount and tax logic also produce incorrect results. "
            "Fix the pricing engine so the tests pass."
        ),
        "_repo_factory": _make_task_2_repo,
    },
    {
        "task_id": "task_3_pr",
        "description": (
            "PR #120: Stripe integration has three issues: a hardcoded live key, "
            "a blocking sleep that stalls the event loop, and a retry loop that "
            "runs one iteration too many. Fix all three so the tests pass."
        ),
        "_repo_factory": _make_task_3_repo,
    },
    {
        "task_id": "task_4_pr",
        "description": (
            "PR #173: LRU cache eviction is broken and the serialiser crashes "
            "on null input. Fix all three bugs so the tests pass."
        ),
        "_repo_factory": _make_task_4_repo,
    },
    {
        "task_id": "task_5_pr",
        "description": (
            "PR #261: Real-time analytics pipeline is producing incorrect metrics. "
            "Three algorithmic bugs: wrong EMA weighting, wrong standard deviation "
            "formula in the anomaly detector, and a sliding-window off-by-one. "
            "Fix all three so the tests pass."
        ),
        "_repo_factory": _make_task_5_repo,
    },
    {
        "task_id": "task_6_pr",
        "description": (
            "PR #412: Document search ranking is broken — CI is red on three files: "
            "search/indexer.py, search/scorer.py, and search/ranker.py. "
            "Fix all three bugs so the full test suite passes."
        ),
        "_repo_factory": _make_task_6_repo,
    },
]


def get_task(task_id: str) -> Dict[str, Any]:
    for t in TASKS:
        if t["task_id"] == task_id:
            return t
    raise ValueError(f"Task {task_id} not found")


def list_task_ids() -> List[str]:
    return [t["task_id"] for t in TASKS]


# =============================================================================
# Per-task grader functions
# =============================================================================

_CLEAN_ENV_KEYS = {"PATH", "HOME", "LANG", "LC_ALL"}
_WINDOWS_ENV_KEYS = {"SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC"}


def _run_pytest(workspace_dir: str) -> float:
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

        m_pass = re.search(r"(\d+) passed", output)
        m_fail = re.search(r"(\d+) failed", output)
        p = int(m_pass.group(1)) if m_pass else 0
        f = int(m_fail.group(1)) if m_fail else 0
        return p / (p + f) if (p + f) > 0 else 0.0

    except Exception:
        return 0.0


def grade_task_1_pr(workspace_dir: str) -> float:
    """Grader: user registration validation + hashing."""
    return _run_pytest(workspace_dir)


def grade_task_2_pr(workspace_dir: str) -> float:
    """Grader: order pricing (quantity, discount, tax)."""
    return _run_pytest(workspace_dir)


def grade_task_3_pr(workspace_dir: str) -> float:
    """Grader: async payment processor + retry count."""
    return _run_pytest(workspace_dir)


def grade_task_4_pr(workspace_dir: str) -> float:
    """Grader: LRU cache eviction order + serialiser None guard."""
    return _run_pytest(workspace_dir)


def grade_task_5_pr(workspace_dir: str) -> float:
    """Grader: EMA formula, sample std, sliding-window boundary."""
    return _run_pytest(workspace_dir)


def grade_task_6_pr(workspace_dir: str) -> float:
    """Grader: tokenizer punctuation, relevance scorer, coupled ranking trap."""
    return _run_pytest(workspace_dir)


GRADERS: Dict[str, Callable[[str], float]] = {
    "task_1_pr": grade_task_1_pr,
    "task_2_pr": grade_task_2_pr,
    "task_3_pr": grade_task_3_pr,
    "task_4_pr": grade_task_4_pr,
    "task_5_pr": grade_task_5_pr,
    "task_6_pr": grade_task_6_pr,
}
