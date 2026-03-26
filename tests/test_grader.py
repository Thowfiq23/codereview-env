import pytest
from codereview_env.grader import evaluate_review, parse_agent_action, get_ast_blast_radius
from codereview_env.models import ReviewAction, ReviewComment

def test_parse_agent_action_unbreakable():
    # 1. Clean JSON
    # 2. Markdown wrapped JSON
    # 3. Malformed JSON with syntax errors
    
    clean_json = '{"comments": [{"line_number": 1, "issue_type": "bug", "severity": "high", "description": "test"}], "summary": "test"}'
    action = parse_agent_action(clean_json)
    assert action is not None
    assert len(action.comments) == 1

    markdown_json = '```json\n' + clean_json + '\n```'
    action2 = parse_agent_action(markdown_json)
    assert action2 is not None

def test_ast_blast_radius():
    code = \"\"\"def foo():
    a = 1
    b = 2

def bar():
    pass
\"\"\"
    start, end = get_ast_blast_radius(code, 2)
    assert start == 1 and end == 3  # foo() spans lines 1-3

def test_evaluate_review_exact_match():
    # True positive
    code = "def add(a, b): return a - b"
    action = ReviewAction(
        comments=[ReviewComment(line_number=1, issue_type="bug", severity="high", description="You used minus instead of plus")],
        summary="Found one bug."
    )
    issues = [{"line_number": 1, "correct_type": "bug", "correct_severity": "high", "match_keywords": "minus plus subtract add instead"}]
    
    reward, info = evaluate_review(action, code, issues)
    assert reward > 0.5  # Base score should be 1.0 (1/1 recall, 1/1 severity)
    assert info["true_positives"] == 1
    assert info["false_positives"] == 0

def test_evaluate_review_false_positive_penalty():
    code = "def noop(): pass"
    action = ReviewAction(
        comments=[
            ReviewComment(line_number=1, issue_type="style", severity="low", description="Bad style"),
            ReviewComment(line_number=1, issue_type="security", severity="critical", description="Fake bug")
        ],
        summary="Found stuff."
    )
    issues = []
    
    reward, info = evaluate_review(action, code, issues)
    assert reward == 0.0
    assert info["false_positives"] == 2
