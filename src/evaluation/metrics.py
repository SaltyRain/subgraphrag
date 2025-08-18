# DEPRECATED
# import json
# from pathlib import Path
# from typing import List, Dict
#
#
# def summarize_evaluation(evaluations: List[Dict], output_path: Path) -> Dict:
#     """
#     Compute accuracy and count statistics from LLM evaluation results.
#
#     Args:
#         evaluations (List[Dict]): List of evaluation records.
#         output_path (Path): Path to save the summary file as JSON.
#
#     Returns:
#         Dict: Summary dictionary with evaluation metrics.
#     """
#     correct = 0
#     incorrect = 0
#     errors = 0
#
#     for row in evaluations:
#         eval_text = row["evaluation"].strip().lower()
#
#         if eval_text == "error":
#             errors += 1
#         elif eval_text.startswith("yes"):
#             correct += 1
#         elif eval_text.startswith("no"):
#             incorrect += 1
#         else:
#             # Treat anything else as an error (e.g. malformed output)
#             errors += 1
#
#     total = correct + incorrect
#     accuracy = (correct / total) * 100 if total > 0 else 0.0
#
#     summary = {
#         "total": len(evaluations),
#         "evaluated": total,
#         "correct": correct,
#         "incorrect": incorrect,
#         "errors": errors,
#         "accuracy": round(accuracy, 2)
#     }
#
#     # Save summary to JSON file
#     with output_path.open("w", encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)
#
#     return summary