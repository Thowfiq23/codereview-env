"""
codereview_env/grader.py
"""
import ast, json, math, re
from typing import Dict, Any, List, Tuple
from thefuzz import fuzz
from .models import ReviewAction, ReviewComment

def get_ast_blast_radius(code_snippet: str, target_line: int) -> Tuple[int, int]:
    fallback = (max(1, target_line - 3), target_line + 3)
    try: tree = ast.parse(code_snippet)
    except: return fallback
    best_match, smallest_range = None, float("inf")
    for node in ast.walk(tree):
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            if node.lineno <= target_line <= node.end_lineno:
                sz = node.end_lineno - node.lineno
                if sz < smallest_range:
                    smallest_range = sz
                    best_match = (node.lineno, node.end_lineno)
    return best_match if best_match else fallback

def parse_agent_action(raw_output: str) -> ReviewAction | None:
    try:
        c = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.MULTILINE)
        s1, s2 = c.find("{"), c.find("[")
        s = s1 if s1 != -1 and (s2 == -1 or s1 < s2) else s2
        e = c.rfind("}") if s == s1 else c.rfind("]")
        if s == -1 or e == -1: return None
        d = json.loads(c[s:e+1])
        if isinstance(d, dict):
            return ReviewAction(**d) if "comments" in d else ReviewAction(comments=[ReviewComment(**d)])
        elif isinstance(d, list):
            return ReviewAction(comments=[ReviewComment(**x) for x in d])
        return None
    except: return None

def evaluate_review(agent_action: ReviewAction, code_snippet: str, gt_issues: list) -> tuple:
    total_gt = len(gt_issues)
    if total_gt == 0: return 1.0, {}
    tp, fp, sev_ok = 0, 0, 0
    matched_agent = [False] * len(agent_action.comments)
    for gt in gt_issues:
        bs, be = get_ast_blast_radius(code_snippet, gt["line_number"])
        for ag_idx, comm in enumerate(agent_action.comments):
            if matched_agent[ag_idx] or not (bs <= comm.line_number <= be): continue
            if comm.issue_type == gt["correct_type"] and fuzz.token_set_ratio(comm.description.lower(), gt["match_keywords"]) >= 60:
                matched_agent[ag_idx] = True
                tp += 1
                if comm.severity == gt["correct_severity"]: sev_ok += 1
                break
    fp = sum(not m for m in matched_agent)
    recall = tp / total_gt
    base = (0.6 * recall) + (0.4 * (sev_ok / tp if tp else 0))
    mult = math.exp(-0.25 * fp)
    return max(0.0, min(1.0, base * mult)), {"true_positives": tp, "false_positives": fp, "penalty_multiplier": mult}
