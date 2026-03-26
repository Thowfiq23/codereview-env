"""
codereview_env/tasks.py
"""
TASKS = [
    {
        "id": "task_1_bug",
        "difficulty": "easy",
        "filename": "data_processor.py",
        "language": "python",
        "context": "Sensor-data pipeline.",
        "task_description": "Review for logic bugs.",
        "code_snippet": "def compute_average(numbers: list) -> float:\n    if not numbers:\n        return 0.0\n    total = 0\n    for i in range(1, len(numbers)):   # BUG: skips index 0\n        total += numbers[i]\n    return total / len(numbers)\n",
        "ground_truth_issues": [
            {"line_number": 6, "correct_type": "bug", "correct_severity": "high", "match_keywords": "range index off-by-one skips first"}
        ],
    }
]
def get_task(task_id: str) -> dict:
    for t in TASKS:
        if t["id"] == task_id: return t
    raise KeyError(task_id)
def list_task_ids():
    return [t["id"] for t in TASKS]
