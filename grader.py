"""
codereview_env/grader.py

Fully deterministic grader — no LLM required.

Scoring formula (all components clamped to [0, 1]):

    recall_score        = issues_matched / total_real_issues          (weight 0.50)
    severity_accuracy   = correct_severities / issues_matched         (weight 0.25)
    fp_penalty          = min(false_positives / total_real_issues, 1) (weight 0.25, subtracted)

    total = 0.50 * recall + 0.25 * severity_acc - 0.25 * fp_penalty
    total = max(0.0, min(1.0, total))

An issue is "matched" if:
  1. The agent's line_number is within ±3 of the ground-truth line_number, AND
  2. The agent's issue_type matches the ground-truth type, AND
  3. At least one keyword from the ground-truth match_keywords list appears
     (case-insensitive) in the agent's description.

This makes the grader robust to minor line-number drift while still requiring
the agent to understand what the issue actually is.
"""
from __future__ import annotations
from typing import List, Dict, Any

from .models import ReviewAction, GraderResult


def _keyword_hit(description: str, keywords: List[str]) -> bool:
    """Return True if any keyword appears in description (case-insensitive)."""
    desc_lower = description.lower()
    return any(kw.lower() in desc_lower for kw in keywords)


def _match_comment_to_issue(comment_dict: dict, gt_issue: dict) -> bool:
    """
    Return True if an agent comment matches a ground-truth issue.
    Criteria:
      - line within ±3 lines
      - issue_type matches
      - at least one match_keyword found in description
    """
    line_ok = abs(comment_dict["line_number"] - gt_issue["line_number"]) <= 3
    type_ok = comment_dict["issue_type"] == gt_issue["correct_type"]
    kw_ok = _keyword_hit(comment_dict["description"], gt_issue["match_keywords"])
    return line_ok and type_ok and kw_ok


def grade(action: ReviewAction, task: dict) -> GraderResult:
    """
    Grade an agent's ReviewAction against a task's ground-truth issues.

    Returns a GraderResult with total_score in [0.0, 1.0].
    """
    gt_issues: List[Dict[str, Any]] = task["ground_truth_issues"]
    total_issues = len(gt_issues)
    agent_comments = [c.model_dump() for c in action.comments]

    # ── Match each ground-truth issue to at most one agent comment ───────────
    matched_gt: List[bool] = [False] * total_issues   # which GT issues were found
    matched_agent: List[bool] = [False] * len(agent_comments)  # which agent comments matched
    severity_correct: List[bool] = []
    per_issue_detail = []

    for gt_idx, gt in enumerate(gt_issues):
        for ag_idx, ag in enumerate(agent_comments):
            if matched_agent[ag_idx]:
                continue  # already used
            if _match_comment_to_issue(ag, gt):
                matched_gt[gt_idx] = True
                matched_agent[ag_idx] = True
                sev_ok = ag["severity"] == gt["correct_severity"]
                severity_correct.append(sev_ok)
                per_issue_detail.append({
                    "gt_line": gt["line_number"],
                    "gt_type": gt["correct_type"],
                    "gt_severity": gt["correct_severity"],
                    "agent_line": ag["line_number"],
                    "agent_type": ag["issue_type"],
                    "agent_severity": ag["severity"],
                    "matched": True,
                    "severity_correct": sev_ok,
                })
                break
        else:
            per_issue_detail.append({
                "gt_line": gt["line_number"],
                "gt_type": gt["correct_type"],
                "gt_severity": gt["correct_severity"],
                "matched": False,
                "severity_correct": False,
            })

    issues_found = sum(matched_gt)
    false_positives = sum(1 for used in matched_agent if not used)

    # ── Compute component scores ─────────────────────────────────────────────
    recall_score = issues_found / total_issues if total_issues > 0 else 1.0

    severity_accuracy = (
        sum(severity_correct) / issues_found if issues_found > 0 else 0.0
    )

    fp_penalty = min(false_positives / max(total_issues, 1), 1.0)

    # ── Weighted total ────────────────────────────────────────────────────────
    total = 0.50 * recall_score + 0.25 * severity_accuracy - 0.25 * fp_penalty
    total = round(max(0.0, min(1.0, total)), 4)

    # ── Build feedback string ─────────────────────────────────────────────────
    feedback_lines = [
        f"Issues found: {issues_found}/{total_issues}",
        f"Recall score: {recall_score:.2f}",
        f"Severity accuracy: {severity_accuracy:.2f}",
        f"False positive penalty: {fp_penalty:.2f}",
        f"Total score: {total:.4f}",
    ]
    if false_positives > 0:
        feedback_lines.append(
            f"You flagged {false_positives} issue(s) that don't match any real problem. "
            "Keep comments focused."
        )
    missed = [gt_issues[i]["issue_type"] for i, found in enumerate(matched_gt) if not found]
    if missed:
        feedback_lines.append(f"Missed issue type(s): {', '.join(missed)}")

    return GraderResult(
        task_id=task["id"],
        total_score=total,
        issues_found=issues_found,
        total_issues=total_issues,
        recall_score=round(recall_score, 4),
        severity_accuracy=round(severity_accuracy, 4),
        false_positive_penalty=round(fp_penalty, 4),
        per_issue_detail=per_issue_detail,
        feedback="\n".join(feedback_lines),
    )
