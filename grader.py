"""
codereview_env/grader.py

Deterministic scorer. Uses AST mapping for line radius and fuzzy keyword matching.
"""
from __future__ import annotations

import ast
import json
import math
import re
from typing import Dict, Any, List, Tuple

from thefuzz import fuzz
from models import ReviewAction, ReviewComment


def get_ast_blast_radius(code_snippet: str, target_line: int) -> Tuple[int, int]:
    """
    Parse code snippet to find the smallest logical block enclosing the target_line.
    Returns (start_line, end_line). Falls back to +/- 3 if parsing fails or not found.
    """
    fallback = (max(1, target_line - 3), target_line + 3)
    try:
        tree = ast.parse(code_snippet)
    except SyntaxError:
        return fallback

    best_match = None
    smallest_range = float("inf")

    for node in ast.walk(tree):
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            # Check if target line is inside this AST node's start and end
            if node.lineno <= target_line <= node.end_lineno:
                block_size = node.end_lineno - node.lineno
                if block_size < smallest_range:
                    smallest_range = block_size
                    best_match = (node.lineno, node.end_lineno)

    if best_match:
        return best_match
    return fallback


def parse_agent_action(raw_output: str) -> ReviewAction | None:
    """
    Unbreakable Parser. Strips markdown fences, extracts JSON, safely parses.
    Returns None on complete failure.
    """
    try:
        # Strip markdown code block boundaries (```json ... ```)
        clean_text = re.sub(r"^```(?:json)?\s*", "", raw_output.strip(), flags=re.MULTILINE)
        clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.MULTILINE)

        # Look for the outermost JSON curlies/brackets in case there is text before/after
        start_idx_dict = clean_text.find("{")
        start_idx_list = clean_text.find("[")
        
        # Decide if the payload is trying to be a JSON object or an array
        start_idx = -1
        if start_idx_dict != -1 and (start_idx_list == -1 or start_idx_dict < start_idx_list):
            start_idx = start_idx_dict
            end_idx = clean_text.rfind("}")
        elif start_idx_list != -1:
            start_idx = start_idx_list
            end_idx = clean_text.rfind("]")

        if start_idx == -1 or end_idx == -1:
            return None

        json_str = clean_text[start_idx : end_idx + 1]
        data = json.loads(json_str)

        # Massage the parsed payload into our ReviewAction model
        if isinstance(data, dict):
            if "comments" in data:
                return ReviewAction(**data)
            else:
                # Top level dict missing 'comments' wrapper, try interpreting as a single comment
                return ReviewAction(comments=[ReviewComment(**data)])
        elif isinstance(data, list):
            # Model emitted raw array of comments
            comments = [ReviewComment(**c) for c in data]
            return ReviewAction(comments=comments)

        return None

    except Exception:
        # The ultimate zero-crash response
        return None


def evaluate_review(
    agent_action: ReviewAction, 
    code_snippet: str, 
    ground_truth_issues: List[Dict[str, Any]]
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate the parsed action against ground truth using AST bounding & fuzzy matching.
    Reward = Clamp( ((0.6 * recall) + (0.4 * severity_acc)) * exp(-0.25 * FP) )
    """
    total_gt = len(ground_truth_issues)
    
    # Graceful handling for tasks that intentionally have 0 flaws.
    if total_gt == 0:
        false_positives = len(agent_action.comments)
        multiplier = math.exp(-0.25 * false_positives)
        return float(multiplier), {
            "true_positives": 0,
            "false_positives": false_positives,
            "recall": 1.0,
            "severity_accuracy": 1.0,
            "base_score": 1.0,
            "penalty_multiplier": multiplier,
        }

    matched_gt: List[bool] = [False] * total_gt
    matched_agent: List[bool] = [False] * len(agent_action.comments)
    severity_correct = 0

    for gt_idx, gt in enumerate(ground_truth_issues):
        gt_line = gt["line_number"]
        gt_type = gt["correct_type"]
        gt_severity = gt["correct_severity"]
        gt_keywords: str = gt["match_keywords"]

        blast_start, blast_end = get_ast_blast_radius(code_snippet, gt_line)

        for ag_idx, comment in enumerate(agent_action.comments):
            if matched_agent[ag_idx]:
                continue
            
            # Constraint 1: Must fall inside the AST block where the flaw lives
            if not (blast_start <= comment.line_number <= blast_end):
                continue
                
            # Constraint 2: Must be correctly categorised (e.g. security vs bug)
            if comment.issue_type != gt_type:
                continue
                
            # Constraint 3: Fuzzy Description Match (>= 60 threshold)
            # Compare the comment description with the ground-truth keyword string
            fuzz_score = fuzz.token_set_ratio(comment.description.lower(), gt_keywords.lower())
            if fuzz_score >= 60:
                # True Positive
                matched_gt[gt_idx] = True
                matched_agent[ag_idx] = True
                
                # Check secondary property
                if comment.severity == gt_severity:
                    severity_correct += 1
                break

    true_positives = sum(matched_gt)
    false_positives = sum(not used for used in matched_agent)

    # Calculate Reward
    recall = true_positives / total_gt
    severity_accuracy = (severity_correct / true_positives) if true_positives > 0 else 0.0

    base_score = (0.60 * recall) + (0.40 * severity_accuracy)
    multiplier = math.exp(-0.25 * false_positives)
    
    final_reward = base_score * multiplier
    
    # Clamp to strictly [0.0, 1.0] bounds
    final_reward = max(0.0, min(1.0, final_reward))

    info = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "recall": float(recall),
        "severity_accuracy": float(severity_accuracy),
        "base_score": float(base_score),
        "penalty_multiplier": float(multiplier),
    }

    return float(final_reward), info

