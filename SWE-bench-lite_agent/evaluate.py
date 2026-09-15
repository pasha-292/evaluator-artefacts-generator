"""
Turns a batch of agent-produced patches into real pass/fail, by handing
them to the official, unmodified swebench evaluation harness -- not by
scoring them ourselves. This is deliberately a separate step from
run_swebench_ablation.py's agent runs: producing a patch spends LLM API
budget, scoring one spends Docker (or Modal cloud) compute budget, and
keeping them separate means re-running evaluation (switching from local
Docker to Modal cloud, or re-scoring after a swebench package update)
never requires re-spending on the agent runs that produced the patches
in the first place.

Needs `pip install -r requirements.txt` (and `modal`, only for the cloud
path -- see README.md, "Running on the cloud"). The requirements pin
SWE-bench 3.0.13: newer 5.0.2 packages an image-backed TestSpec alongside a
Modal runner that expects the old script-backed API. `run_evaluation.main`'s
signature and the report.json layout under logs/run_evaluation/ are the two
interfaces this file depends on.

IMPORTANT, found the hard way (real trace data showed every `full` row
scoring NaN while `no_skill` scored normally, for every single instance):
the official harness's own `main()` loads predictions like this --

    predictions = {pred["instance_id"]: pred for pred in predictions}

-- a plain dict comprehension keyed by instance_id. This project
deliberately writes MORE THAN ONE prediction per instance_id (one per
condition/trial, each tagged with a different model_name_or_path, since
the same instance gets attempted under `full`, `no_skill`, etc.) -- and a
dict comprehension with repeated keys silently keeps only the LAST one.
Handing the harness one predictions file with every condition's patches
in it means only whichever condition happened to be last in the file
ever actually gets evaluated for a given instance; every other
condition's patch for that same instance is discarded before scoring
even starts, not after -- which is exactly why it shows up as unscored
(NaN) rather than as a scored failure.

The fix: evaluate one (condition, trial) group at a time, each against
its own predictions file containing only that group's entries -- within
one group every instance_id really is unique, so the collapsing above is
harmless. run_swebench_ablation.py's evaluate_run() does this by calling
load_predictions_by_group() and run_official_evaluation() once per group,
not once for everything. This does mean N separate evaluation calls (N
Modal app deployments, if running on Modal) instead of one -- more
overhead than the original single-call design, but the original design
was never actually evaluating most conditions at all, so this is a
correctness fix, not an optimization tradeoff.
"""

import json
import inspect
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import config

# swebench.harness.run_evaluation's own default -- see
# _redirected_run_evaluation_log_dir's docstring for why this project needs
# to know it (to redirect it per model) rather than just using it as-is.
DEFAULT_LOGS_DIR = Path("logs/run_evaluation")


def _run_swebench_main(**kwargs):
    """Call the official evaluator across its compatible 3.x API variants.

    SWE-bench 3.0.13 is pinned because its Modal evaluator and TestSpec API
    are internally compatible.  Its entrypoint has a few optional image-cache
    arguments that later releases removed, so pass those only when supported.
    """
    from swebench.harness.run_evaluation import main

    optional = {
        "force_rebuild": False,
        "cache_level": "env",
        "clean": False,
        "namespace": None,
    }
    supported = inspect.signature(main).parameters
    return main(**{**kwargs, **{k: v for k, v in optional.items() if k in supported}})


def load_predictions_by_group(predictions_path: Path) -> dict:
    """Loads a predictions file (written by write_predictions) and splits
    it by model_name_or_path -- one list of predictions per (condition,
    trial), each with a unique instance_id within it. See this module's
    docstring for why evaluate_run() needs to evaluate these one group at
    a time rather than handing the whole file to run_official_evaluation
    in a single call."""
    predictions = json.loads(Path(predictions_path).read_text())
    by_group = defaultdict(list)
    for pred in predictions:
        by_group[pred["model_name_or_path"]].append(pred)
    return dict(by_group)


def write_predictions_group(group_predictions: list, path: Path) -> Path:
    """Writes one (condition, trial) group's predictions (a list from
    load_predictions_by_group's values) to its own file, in the same
    format write_predictions itself uses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(group_predictions, indent=2))
    return path


def _running_in_event_loop() -> bool:
    """True inside a Jupyter/IPython kernel (which always runs one),
    False in a plain script or the CLI. See run_official_evaluation's
    docstring for why this matters specifically for the Modal path."""
    import asyncio
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


@contextmanager
def _redirected_run_evaluation_log_dir(logs_dir: Path):
    """
    swebench.harness.run_evaluation hardcodes RUN_EVALUATION_LOG_DIR =
    Path("logs/run_evaluation") as a module-level constant, resolved
    relative to the process's cwd -- main() itself takes no parameter to
    redirect where it writes per-instance logs/reports. This monkeypatches
    that one binding (the name main()'s own body actually reads, via
    swebench.harness.run_evaluation's namespace -- not
    swebench.harness.constants', a separate copy of the same original value)
    for the duration of a single main() call, restoring it afterwards even
    on error, so run_official_evaluation() can honor a model-scoped
    logs_dir. load_resolution() below reads back from the same logs_dir
    value directly instead of re-importing swebench's constant, so the
    write and read sides can't drift apart.

    A no-op (no import, no patch) when logs_dir already equals swebench's
    own default -- the common case for any caller that doesn't pass
    logs_dir at all.
    """
    logs_dir = Path(logs_dir)
    if logs_dir == DEFAULT_LOGS_DIR:
        yield
        return
    from swebench.harness import run_evaluation as _run_evaluation_module
    original = _run_evaluation_module.RUN_EVALUATION_LOG_DIR
    _run_evaluation_module.RUN_EVALUATION_LOG_DIR = logs_dir
    try:
        yield
    finally:
        _run_evaluation_module.RUN_EVALUATION_LOG_DIR = original


def _run_modal_evaluation_via_subprocess(instance_ids: list, predictions_path: Path,
                                          run_id: str, max_workers: int, timeout: int,
                                          report_dir: str, logs_dir: Path = DEFAULT_LOGS_DIR) -> None:
    """
    Runs swebench.harness.run_evaluation.main(..., modal=True) in a fresh
    subprocess instead of in-process. Needed because Modal's own SDK
    can't bridge sync/async from inside an already-running event loop --
    every Jupyter kernel has one, even though the notebook cell that
    calls evaluate_run() looks like ordinary synchronous code. The
    official harness's own Modal path (run_instances_modal) does a plain
    `for result in results:` over a `Function.starmap()` call, which
    only works outside of a running loop; from inside one, Modal raises
    "You can't iter(Function.starmap()) from an async function" -- a
    genuine library-level conflict between Modal's synchronicity layer
    and Jupyter's own event loop, not a bug in this project's code, and
    not something fixable by changing arguments or config. A fresh
    subprocess has no ambient loop, so the identical call succeeds there.

    Output is streamed line by line as it arrives (not captured and
    printed only at the end), so Modal's own build/progress output still
    shows up live in the notebook the same way it would in a terminal.

    logs_dir is passed through and applied inside the subprocess itself
    (via the same monkeypatch _redirected_run_evaluation_log_dir uses),
    since the subprocess's own cwd (config.REPO_ROOT, set below) is what
    swebench's RUN_EVALUATION_LOG_DIR would otherwise resolve against.
    """
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from pathlib import Path;"
        "from evaluate import _run_swebench_main, DEFAULT_LOGS_DIR;"
        "import json;"
        "logs_dir = Path(sys.argv[10]);"
        "if logs_dir != DEFAULT_LOGS_DIR:\n"
        "    from swebench.harness import run_evaluation as _rem\n"
        "    _rem.RUN_EVALUATION_LOG_DIR = logs_dir\n"
        "_run_swebench_main(dataset_name=sys.argv[2], split=sys.argv[3],"
        "     instance_ids=json.loads(sys.argv[4]), predictions_path=sys.argv[5],"
        "     max_workers=int(sys.argv[6]), open_file_limit=8192, run_id=sys.argv[7],"
        "     timeout=int(sys.argv[8]), rewrite_reports=False, modal=True,"
        "     report_dir=sys.argv[9])"
    )
    args = [
        sys.executable, "-c", script,
        str(config.REPO_ROOT), config.DATASET_NAME, config.DATASET_SPLIT,
        json.dumps(instance_ids), str(predictions_path), str(max_workers),
        run_id, str(timeout), report_dir, str(logs_dir),
    ]
    proc = subprocess.Popen(args, cwd=str(config.REPO_ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Evaluation subprocess exited with code {proc.returncode} -- "
            f"see the output above for the real error from inside it."
        )


def write_predictions(results: list, predictions_path: Path) -> Path:
    """
    results: a list of agent_loop.run_instance()'s return dicts -- ONLY the
    current call's newly-computed results, not anything a resumed run()
    skipped (those were never re-run, so there's no fresh final_patch for
    them in memory).

    Writes the swebench-standard predictions format: a JSON list of
    {"instance_id", "model_patch", "model_name_or_path"} dicts, one per
    (instance, condition, trial) triple. model_name_or_path tags each
    condition+trial combination as its own "model" in swebench's result
    grouping (logs/run_evaluation/<run_id>/<model_name_or_path>/<instance_id>/),
    so the official harness's own per-instance logs and reports keep every
    condition and trial distinct, the same way it would for two genuinely
    different models.

    Merges into whatever predictions_path already holds, keyed by
    (instance_id, model_name_or_path), rather than overwriting it wholesale --
    this is what makes run_swebench_ablation.py's batching (run() called once
    per instance/skill-mode/usage-limited-session, each with resume=True)
    actually work end to end: a later batch's results land in the file
    alongside an earlier batch's, instead of replacing them, since resume
    only re-runs (and thus only returns) whatever's new this call. Existing
    entries whose key isn't in this call's results are left untouched;
    entries that do match are overwritten with this call's (presumably
    newer) patch.
    """
    existing_by_key = {}
    if predictions_path.exists():
        for pred in json.loads(predictions_path.read_text()):
            existing_by_key[(pred["instance_id"], pred["model_name_or_path"])] = pred

    for r in results:
        key = (r["instance_id"], model_name_or_path(r["condition"], r["trial"]))
        existing_by_key[key] = {
            "instance_id": r["instance_id"],
            "model_patch": r["final_patch"] or "",
            "model_name_or_path": key[1],
        }

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(list(existing_by_key.values()), indent=2))
    return predictions_path


def model_name_or_path(condition: str, trial: int) -> str:
    """The tag a (condition, trial) pair is scored under. Kept as one
    function so write_predictions and load_resolution can never drift
    apart on how this string is built."""
    return f"{condition}__trial{trial}"


def run_official_evaluation(predictions_path: Path, instance_ids: list, run_id: str,
                             modal: bool = False, max_workers: int = 4,
                             timeout: int = 1800, report_dir: str = ".",
                             logs_dir: Path = DEFAULT_LOGS_DIR) -> Path:
    """
    Calls swebench.harness.run_evaluation.main(...) -- the real, official
    entrypoint, unmodified -- against the predictions file. Needs either a
    local Docker daemon (modal=False) or a configured Modal account
    (modal=True; run `modal setup` first, see README.md).

    When modal=True and this is called from inside an already-running
    event loop (a Jupyter notebook kernel always has one), the actual
    evaluation runs in a fresh subprocess instead of in-process -- see
    _run_modal_evaluation_via_subprocess's docstring for why Modal's own
    SDK requires this. The local-Docker path (modal=False) isn't affected
    and always runs in-process, since it doesn't touch Modal's async
    machinery at all.

    Writes swebench's own per-instance logs and reports under
    logs_dir/<run_id>/ -- defaults to logs/run_evaluation/<run_id>/
    (relative to wherever this is run from), swebench's own hardcoded
    location, but honors a model-scoped logs_dir instead when one is
    passed (see _redirected_run_evaluation_log_dir). Returns the path to
    the aggregate report.json run_evaluation.main itself writes and
    returns, when run in-process; returns None when run via the
    subprocess path (the caller doesn't use this return value -- see
    run_swebench_ablation.py's evaluate_run, which reads the per-instance
    report.json files directly instead).
    """
    if modal and _running_in_event_loop():
        _run_modal_evaluation_via_subprocess(
            instance_ids=instance_ids, predictions_path=predictions_path, run_id=run_id,
            max_workers=max_workers, timeout=timeout, report_dir=report_dir, logs_dir=logs_dir,
        )
        return None

    with _redirected_run_evaluation_log_dir(logs_dir):
        return _run_swebench_main(
            dataset_name=config.DATASET_NAME,
            split=config.DATASET_SPLIT,
            instance_ids=instance_ids,
            predictions_path=str(predictions_path),
            max_workers=max_workers,
            open_file_limit=8192,
            run_id=run_id,
            timeout=timeout,
            rewrite_reports=False,
            modal=modal,
            report_dir=report_dir,
        )


def load_resolution(run_id: str, condition: str, trial: int, instance_id: str,
                    report_dir: str = ".", logs_dir: Path = DEFAULT_LOGS_DIR) -> dict:
    """
    Reads back the report for this exact (condition, trial, instance) triple.
    The local-Docker path writes per-instance report.json files, while the
    Modal path writes one aggregate report in report_dir, so support both.

    logs_dir must be whatever was passed to the run_official_evaluation()
    call that produced this report (default: swebench's own
    logs/run_evaluation) -- taken directly as a parameter here, rather than
    re-imported from swebench.harness.constants (a separate copy of the same
    original value, unaffected by run_official_evaluation's own redirect),
    so the write and read sides can never drift apart onto different paths.
    """
    from swebench.harness.constants import LOG_REPORT

    report_path = (
        Path(logs_dir) / run_id / model_name_or_path(condition, trial)
        / instance_id / LOG_REPORT
    )
    if not report_path.exists():
        aggregate_path = Path(report_dir) / (
            f"{model_name_or_path(condition, trial)}.{run_id}.json"
        )
        if not aggregate_path.exists():
            return {"resolved": False, "error": f"no report at {report_path} or {aggregate_path}"}

        aggregate = json.loads(aggregate_path.read_text())
        if instance_id in aggregate.get("resolved_ids", []):
            return {"resolved": True}
        if instance_id in aggregate.get("unresolved_ids", []):
            return {"resolved": False, "scored": True}
        if instance_id in aggregate.get("empty_patch_ids", []):
            return {"resolved": False, "scored": True, "empty_patch": True}
        if instance_id in aggregate.get("error_ids", []):
            return {"resolved": False, "error": "official evaluation error"}
        return {"resolved": False, "error": "instance_id not present in evaluation report"}

    report = json.loads(report_path.read_text())
    # report.json is keyed by instance_id (see swebench.harness.grading.get_eval_report)
    return report.get(instance_id, {"resolved": False, "error": "instance_id not in report"})


def load_all_resolutions(run_id: str, results: list, report_dir: str = ".",
                          logs_dir: Path = DEFAULT_LOGS_DIR) -> dict:
    """Convenience wrapper: given the same results list write_predictions
    was called with, looks up every (instance_id, condition, trial)'s
    resolution and returns {(instance_id, condition, trial): report_dict}."""
    out = {}
    for r in results:
        key = (r["instance_id"], r["condition"], r["trial"])
        out[key] = load_resolution(
            run_id, r["condition"], r["trial"], r["instance_id"],
            report_dir=report_dir, logs_dir=logs_dir,
        )
    return out


def merge_evaluated_logs(paths: list) -> list:
    """
    Merges several copies of the same trajectory JSONL, each independently
    scored by run_swebench_ablation.py's evaluate_run(only_groups=...) for a
    different (usually disjoint) subset of condition/trial groups -- e.g.
    one file downloaded from each job of a GitHub Actions matrix that
    evaluates one group per job (see that workflow and evaluate_run's
    only_groups docstring for why this split is safe to do in the first
    place: a group each call didn't touch is left completely unchanged, not
    marked "not scored").

    "step" records are taken from the first path (identical byte-for-byte
    across every copy -- evaluate_run() never touches them). For "result"
    records, whichever copy actually scored that (instance_id, condition,
    trial) -- i.e. has "evaluation_status" set -- wins; a row no copy
    touched (evaluated by neither this batch of jobs nor a prior one) is
    passed through as-is from the first path.

    Returns the merged list of records, in the first path's original order
    -- write it back with one `"\\n".join(json.dumps(r) for r in records)`.
    """
    paths = [Path(p) for p in paths]
    base_records = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]

    scored_by_key = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["record_type"] != "result" or record.get("evaluation_status") is None:
                continue
            key = (record["instance_id"], record["condition"], record["trial"])
            scored_by_key[key] = record

    merged = []
    for record in base_records:
        if record["record_type"] == "result":
            key = (record["instance_id"], record["condition"], record["trial"])
            record = scored_by_key.get(key, record)
        merged.append(record)
    return merged
