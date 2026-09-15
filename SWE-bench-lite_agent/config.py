"""
Evaluation-only subset of config.py, derived automatically by
generate_artefact_components.py from the main project's real values (not
hand-copied, so these can't drift out of sync) -- only the settings
evaluate_run()/list_evaluation_groups*()/merge-logs actually read. Every
other setting (models, providers, skill/agent tuning) belongs to the
agent-running half this repo doesn't have.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent
RESULTS_DIR = REPO_ROOT / "results"
DATASET_NAME = "SWE-bench/SWE-bench_Lite"
DATASET_SPLIT = "test"
RESULTS_PATH = str(RESULTS_DIR / "swebench_ablation.jsonl")
SUMMARY_CSV_PATH = str(RESULTS_DIR / "swebench_ablation_summary.csv")
PREDICTIONS_PATH = str(RESULTS_DIR / "predictions.json")
SWEBENCH_RUN_ID = "quixbugs_swebench_lite_agent"
EVALUATE_USE_MODAL_BY_DEFAULT = True
