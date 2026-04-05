import sys
import os
# Tell Python to look in the parent directory for our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AgentAction, ReviewComment
from grader import evaluate_review

def test_grader_math():
    repo = {"auth.py": "def login():\n    pass\n"}
    gt = [{"file_path": "auth.py", "line_number": 2, "correct_severity": "high", "match_keywords": "login,bug"}]

    print("\n--- RUNNING GRADER STRESS TESTS ---")

    # TEST 1: The Perfect, Efficient Agent (1 step)
    action_perfect = AgentAction(action_type="submit_review", review_comments=[
        ReviewComment(file_path="auth.py", line_number=2, issue_type="security", severity="high", description="Login bug found")
    ])
    score_1, info_1 = evaluate_review(action_perfect, repo, gt, step_count=1)
    print(f"Test 1 (Perfect 1-Step): Score = {score_1:.2f}")
    assert score_1 > 0.8, "Failed: Efficient agent didn't get bonus!"

    # TEST 2: The Slow Agent (12 steps to find the exact same bug)
    score_2, info_2 = evaluate_review(action_perfect, repo, gt, step_count=12)
    print(f"Test 2 (Slow 12-Step): Score = {score_2:.2f}")
    assert score_1 > score_2, "Failed: Slow agent wasn't penalized for inefficiency!"

    # TEST 3: The Hallucinating Agent (Found the bug, but guessed 3 fake ones)
    action_hallucinate = AgentAction(action_type="submit_review", review_comments=[
        ReviewComment(file_path="auth.py", line_number=2, issue_type="security", severity="high", description="Login bug found"),
        ReviewComment(file_path="auth.py", line_number=99, issue_type="logic", severity="low", description="Fake bug 1"),
        ReviewComment(file_path="auth.py", line_number=100, issue_type="logic", severity="low", description="Fake bug 2"),
        ReviewComment(file_path="auth.py", line_number=101, issue_type="logic", severity="low", description="Fake bug 3")
    ])
    score_3, info_3 = evaluate_review(action_hallucinate, repo, gt, step_count=1)
    print(f"Test 3 (Hallucinations): Score = {score_3:.2f}")
    assert score_3 < score_1, "Failed: Hallucinations weren't penalized!"

    print("✅ ALL MATH TESTS PASSED! Grader is flawless.\n")

if __name__ == "__main__":
    test_grader_math()
