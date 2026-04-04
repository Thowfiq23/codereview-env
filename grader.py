import math
import ast
from typing import Dict, Any, Tuple, List, Optional
from thefuzz import fuzz
from models import AgentAction, ReviewComment

def get_ast_blast_radius(code: str, target_line: int) -> Tuple[int, int]:
    """Bounds the error to the logical AST block (function/class/loop)."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                if node.lineno <= target_line <= node.end_lineno:
                    return node.lineno, node.end_lineno
    except SyntaxError:
        pass
    return target_line, target_line

def parse_agent_action(text: str) -> Optional[AgentAction]:
    """Unbreakable JSON parser for the new multi-tool action schema."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            clean_json = text[start:end]
            return AgentAction.model_validate_json(clean_json)
        return None
    except Exception:
        return None

def evaluate_review(
    action: AgentAction, 
    repo: Dict[str, str], 
    ground_truth: List[Dict[str, Any]],
    step_count: int = 1  # <--- Added step_count here
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates the final review across multiple files in the repository."""
    
    comments = action.review_comments or []
    total_gt = len(ground_truth)
    
    if total_gt == 0:
        fp = len(comments)
        mult = math.exp(-0.25 * fp)
        return float(mult), {"true_positives": 0, "false_positives": fp, "penalty_multiplier": mult, "feedback": "Processed."}

    matched_gt = [False] * total_gt
    matched_agent = [False] * len(comments)
    severity_correct = 0

    for gt_idx, gt in enumerate(ground_truth):
        gt_file = gt["file_path"]
        gt_line = gt["line_number"]
        gt_sev = gt["correct_severity"]
        gt_keywords = gt["match_keywords"].split(",")
        
        file_code = repo.get(gt_file, "")
        start_line, end_line = get_ast_blast_radius(file_code, gt_line)

        for ag_idx, comment in enumerate(comments):
            if matched_agent[ag_idx]: continue
            if comment.file_path != gt_file: continue

            if start_line <= comment.line_number <= end_line:
                desc = comment.description.lower()
                if any(fuzz.partial_ratio(kw.strip().lower(), desc) > 70 for kw in gt_keywords):
                    matched_gt[gt_idx] = True
                    matched_agent[ag_idx] = True
                    if comment.severity == gt_sev:
                        severity_correct += 1
                    break
    
    tp = sum(matched_gt)
    fp = len(comments) - tp
    recall = tp / total_gt if total_gt > 0 else 1.0
    sev_acc = severity_correct / tp if tp > 0 else 0.0
    
    # --- V2 REWARD MATH ---
    # 1. Base Score (Max 0.80)
    base_score = (0.50 * recall) + (0.30 * sev_acc)
    
    # 2. Breadth Bonus (Max 0.10) - Did they find bugs in multiple files?
    unique_files_found = len(set([gt["file_path"] for i, gt in enumerate(ground_truth) if matched_gt[i]]))
    total_unique_files = len(set([gt["file_path"] for gt in ground_truth]))
    breadth_bonus = (unique_files_found / total_unique_files) * 0.10 if total_unique_files > 0 else 0.0
    
    # 3. Efficiency Bonus (Max 0.10) - Punish agents that take 10+ steps blindly guessing
    efficiency_bonus = max(0.0, 1.0 - (step_count / 15.0)) * 0.10
    
    total_score_before_penalty = base_score + breadth_bonus + efficiency_bonus
    
    # 4. Exponential Decay for Hallucinations
    penalty = math.exp(-0.25 * fp)
    final_score = max(0.0, min(1.0, total_score_before_penalty * penalty))
    
    return float(final_score), {
        "true_positives": tp,
        "false_positives": fp,
        "recall": recall,
        "severity_accuracy": sev_acc,
        "breadth_bonus": breadth_bonus,
        "efficiency_bonus": efficiency_bonus,
        "penalty_multiplier": penalty,
        "feedback": f"Graded. TP: {tp}, FP: {fp}. Breadth Bonus: +{breadth_bonus:.2f}, Efficiency Bonus: +{efficiency_bonus:.2f}"
    }

