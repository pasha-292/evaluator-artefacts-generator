"""
Evaluation-only subset of config.py, derived automatically by
generate_artefact_components.py from the main project's real values (not
hand-copied, so these can't drift out of sync) -- only the settings
evaluate_run()/list_evaluation_groups*()/merge-logs actually read, plus the
model registry those settings are derived from. Every other setting
(providers, skill/agent tuning) belongs to the agent-running half this repo
doesn't have. This one file's ACTIVE_MODEL-derived defaults serve both
models' workflows -- see the workflow YAML files, which always pass
explicit --out/--predictions/--summary-out/--run-id flags rather than
relying on whichever model ACTIVE_MODEL happens to default to here.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent
MODEL_REGISTRY = {
    "haiku-4-5": {
        "claude_code_model": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5",
        "run_id": "quixbugs_swebench_lite_agent",
    },
    "sonnet-5": {
        "claude_code_model": "claude-sonnet-5",
        "display_name": "Claude Sonnet 5",
        "run_id": "quixbugs_swebench_lite_agent_sonnet5",
    },
}
def model_paths(model_key: str) -> dict:
    """Every path/setting that varies per Claude Code model: the model tag,
    where its runs/ trajectories, results/ output, and evaluation logs live,
    and its swebench run_id. Each registered model gets its own top-level
    <model_key>/ directory -- <model_key>/runs/, <model_key>/results/,
    <model_key>/logs/run_evaluation/ -- rather than a shared results/runs/logs
    tree with one subdirectory per model, specifically so that downloading a
    GitHub Actions evaluation artifact for one model and pasting it over
    <model_key>/ is a single directory overwrite: everything that changed
    (results AND logs together) lives under that one path, and nothing
    belonging to the other model is anywhere underneath it. See
    MODEL_REGISTRY above for what's registered. Used both to set this
    module's own ACTIVE_MODEL-derived constants below, and directly by
    notebooks that want to run more than one model's pipeline side by side
    without editing this file (pass the explicit values this returns to
    run()/evaluate_run() rather than relying on ACTIVE_MODEL, which only
    reflects one model at a time)."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key {model_key!r} -- add it to config.MODEL_REGISTRY first.")
    entry = MODEL_REGISTRY[model_key]
    model_dir = REPO_ROOT / model_key
    results_dir = model_dir / "results"
    return {
        "model": entry["claude_code_model"],
        "model_dir": model_dir,
        "runs_dir": model_dir / "runs",
        "results_dir": results_dir,
        "logs_dir": model_dir / "logs" / "run_evaluation",
        "out": str(results_dir / "swebench_ablation.jsonl"),
        "summary_out": str(results_dir / "swebench_ablation_summary.csv"),
        "predictions_out": str(results_dir / "predictions.json"),
        "run_id": entry["run_id"],
    }
ACTIVE_MODEL = "haiku-4-5"
_active_model_paths = model_paths(ACTIVE_MODEL)
RESULTS_DIR = _active_model_paths["results_dir"]
LOGS_DIR = _active_model_paths["logs_dir"]
DATASET_NAME = "SWE-bench/SWE-bench_Lite"
DATASET_SPLIT = "test"
RESULTS_PATH = _active_model_paths["out"]
SUMMARY_CSV_PATH = _active_model_paths["summary_out"]
PREDICTIONS_PATH = _active_model_paths["predictions_out"]
SWEBENCH_RUN_ID = _active_model_paths["run_id"]
EVALUATE_USE_MODAL_BY_DEFAULT = True
