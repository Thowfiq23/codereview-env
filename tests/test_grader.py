import sys
import os
import math

# Force Python to look in the root directory for our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ReviewAction, ReviewComment
from grader import evaluate_review, parse_agent_action, get_ast_blast_radius

def test_parse_agent_action_unbreakable():
    bad_json = "I am an AI. Here is the json: ```json {bad} ```"
    action = parse_agent_action(bad_json)
    assert action is None

def test_ast_blast_radius():
    code = "def foo():\n    a = 1\n    b = 2\n\ndef bar():\n    pass\n"
    start, end = get_ast_blast_radius(code, 2)
    # The AST will bound either the line or the function. We just verify it bounds the target.
    assert start <= 2 and end >= 2

def test_evaluate_review_exact_match():
    code = "def add(a, b): return a - b"
    action = ReviewAction(
        comments=[ReviewComment(line_number=1, issue_type="bug", severity="high", description="wrong math logic here")],
        summary="bug found here"
    )
    # Added 'match_keywords' to satisfy your grader's fuzzy matching logic!
    issues = [{
        "line_number": 1, 
        "correct_type": "bug", 
        "correct_severity": "high", 
        "description": "uses minus instead of plus",
        "match_keywords": "math,minus,wrong"
    }]
    
    reward, info = evaluate_review(action, code, issues)
    assert reward > 0.0  # Proves the fuzzy matching and true positive detection works

def test_evaluate_review_false_positive_penalty():
    code = "def noop(): pass"
    action = ReviewAction(
        comments=[
            # Descriptions MUST be >= 10 characters to pass your Pydantic validation!
            ReviewComment(line_number=1, issue_type="style", severity="low", description="Bad style formatting"),
            ReviewComment(line_number=1, issue_type="security", severity="critical", description="Fake bug hallucinated")
        ],
        summary="Found stuff."
    )
    issues = [] # No real bugs in the code

    reward, info = evaluate_review(action, code, issues)
    
    # Base score is 1.0 (no bugs missed). 2 False Positives = penalty of math.exp(-0.25 * 2)
    expected_reward = math.exp(-0.5) 
    assert math.isclose(reward, expected_reward, rel_tol=1e-5)