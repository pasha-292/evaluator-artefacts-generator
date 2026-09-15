#!/usr/bin/env python3
"""
Evaluation-only subset of run_swebench_ablation.py, derived automatically
by generate_artefact_components.py from the main project's real source --
scores already-produced patches (`evaluate`), lists the condition/trial
groups available to filter on (`list-groups`), and merges several
--groups-scoped evaluation runs back into one (`merge-logs`). Deliberately
has no "run" subcommand and no agent/skill-generation code at all -- see
generate_artefact_components.py's own docstring for why.
"""

import argparse
import json
import os
import re
from pathlib import Path

import config
import metrics
import evaluate


def _display_path(p) -> str:
    """A path for printing in log output, relative to the project root
    when possible -- keeps notebook cell output (which is what this text
    ends up in) free of this machine's own directory structure, in case
    a notebook ever gets committed with outputs saved. Falls back to the
    path as given if it isn't under the project root at all (e.g. an
    explicit out= pointed somewhere else entirely)."""
    p = Path(p)
    try:
        return str(p.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(p)


def evaluate_run(out: str = config.RESULTS_PATH,
                  predictions_path: str = config.PREDICTIONS_PATH,
                  summary_out: str = config.SUMMARY_CSV_PATH,
                  run_id: str = config.SWEBENCH_RUN_ID,
                  modal: bool = config.EVALUATE_USE_MODAL_BY_DEFAULT,
                  max_workers: int = 4,
                  timeout: int = 1800,
                  report_dir: str = ".",
                  logs_dir: str = config.LOGS_DIR,
                  quiet: bool = False,
                  only_groups: list = None):
    """
    Callable version of `run_swebench_ablation.py evaluate`. Scores the
    patches `run()` produced (in predictions_path) through the official
    swebench harness -- locally via Docker (modal=False) or on Modal's
    cloud (modal=True, needs `modal setup` first) -- and rewrites out's
    result records with the real pass/fail. Returns the updated summary.

    Evaluates one (condition, trial) group at a time, not the whole
    predictions file in one call -- see evaluate.py's module docstring
    for why: the official harness's own predictions-loading collapses
    multiple predictions for the same instance_id down to just the last
    one, and this project deliberately writes one prediction per
    (instance, condition, trial), so a single combined call would
    silently evaluate only whichever condition happened to be last for
    each instance. This means N separate evaluation calls for N groups
    (N Modal app deployments, if running on Modal) rather than one --
    more overhead, but the single-call version was never actually
    evaluating most conditions in the first place.

    only_groups (a list of model_name_or_path tags, e.g. "full__trial1")
    restricts evaluation to those groups only, and -- critically --
    leaves every OTHER group's rows in out untouched (not overwritten
    with a false "not scored"), so this call's rewrite of out only ever
    touches the groups it was actually asked to evaluate. This is what
    makes it safe to run several evaluate_run() calls concurrently (e.g.
    one GitHub Actions matrix job per group, see
    .github/workflows/swebench-evaluate.yml) against copies of the same
    out/predictions_path checked out independently on separate runners --
    each job's output only ever disagrees with the others on its own
    group's rows, so merging the copies back together afterwards (see
    evaluate.merge_evaluated_logs) is a simple last-scored-wins merge,
    not a 3-way conflict. list_evaluation_groups() below lists the group
    tags available to filter on.
    """
    def log(msg):
        if not quiet:
            print(msg)

    out_path = Path(out)
    raw_lines = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    results = [r for r in raw_lines if r["record_type"] == "result"]
    if not results:
        raise ValueError(f"No result records in {out_path} -- run `run_swebench_ablation.py run` first.")

    groups = evaluate.load_predictions_by_group(Path(predictions_path))
    if only_groups is not None:
        unknown = [g for g in only_groups if g not in groups]
        if unknown:
            raise ValueError(f"Unknown group(s) {unknown}. Available: {sorted(groups)}")
        groups = {tag: preds for tag, preds in groups.items() if tag in only_groups}

    log(f"Evaluating {len(results)} run(s) across {len(groups)} condition/trial group(s) "
        f"via the official swebench harness ({'Modal cloud' if modal else 'local Docker'})...")

    predictions_dir = Path(predictions_path).parent
    resolutions = {}
    for group_tag, group_predictions in groups.items():
        group_instance_ids = sorted(set(p["instance_id"] for p in group_predictions))
        log(f"\n[{group_tag}] {len(group_instance_ids)} instance(s)...")

        group_predictions_path = predictions_dir / f"predictions_{group_tag}.json"
        evaluate.write_predictions_group(group_predictions, group_predictions_path)

        evaluate.run_official_evaluation(
            predictions_path=group_predictions_path, instance_ids=group_instance_ids, run_id=run_id,
            modal=modal, max_workers=max_workers, timeout=timeout, report_dir=report_dir,
            logs_dir=logs_dir,
        )

        group_results = [r for r in results if evaluate.model_name_or_path(r["condition"], r["trial"]) == group_tag]
        resolutions.update(evaluate.load_all_resolutions(
            run_id, group_results, report_dir=report_dir, logs_dir=logs_dir,
        ))

    # Rewrite only the "result" lines with real pass/fail; every "step"
    # line is written back exactly as it already was, untouched -- no
    # DataFrame round-trip in between that could distort a value. A row
    # whose group wasn't in `groups` this call (only_groups excluded it)
    # is written back completely unchanged -- this call never attempted
    # it, so it must not be marked "not scored" or otherwise touched.
    with open(out_path, "w") as f:
        for record in raw_lines:
            if record["record_type"] != "result":
                f.write(json.dumps(record) + "\n")
                continue
            group_tag = evaluate.model_name_or_path(record["condition"], record["trial"])
            if group_tag not in groups:
                f.write(json.dumps(record) + "\n")
                continue
            key = (record["instance_id"], record["condition"], record["trial"])
            report = resolutions.get(key, {"resolved": False, "error": "not scored"})
            if report.get("error"):
                record["passed"] = None
                record["evaluation_status"] = "error"
            elif report.get("empty_patch"):
                record["passed"] = False
                record["evaluation_status"] = "empty_patch"
            else:
                # Both the primary per-instance report.json path and the
                # aggregate-report fallback (see evaluate.py's
                # load_resolution) land here once a real resolved value
                # was found -- "scored" only ever appears on the fallback
                # path, so checking report["resolved"] directly (rather
                # than branching on "scored") labels a genuinely-scored
                # primary-path result correctly instead of as
                # "not_evaluated", which it very much was.
                record["passed"] = bool(report["resolved"])
                record["evaluation_status"] = "passed" if record["passed"] else "failed"
            record["eval_error"] = report.get("error")
            f.write(json.dumps(record) + "\n")

    updated_df = metrics.load_jsonl(out_path)
    summary = metrics.condition_summary(updated_df)
    log("\n--- Per-condition summary (real, official pass/fail) ---")
    log(summary.to_string(index=False))

    if "no_skill" in summary["condition"].values and len(summary["condition"].unique()) > 1:
        gains = metrics.normalised_gain(summary, baseline="no_skill")
        log("\n--- Normalised gain over no_skill baseline ---")
        log(gains.to_string(index=False))

    summary_path = Path(summary_out)
    summary.to_csv(summary_path, index=False)
    log(f"\nSummary CSV written to {_display_path(summary_path)}")
    return out_path, summary


def list_evaluation_groups(predictions_path: str = config.PREDICTIONS_PATH) -> list:
    """The condition/trial group tags (model_name_or_path values) present in
    predictions_path -- e.g. for a CI matrix to evaluate one group per job
    (each job passing its tag back to evaluate_run(only_groups=[tag])), or
    just to see what evaluate_run() would otherwise loop over in one call."""
    return sorted(evaluate.load_predictions_by_group(Path(predictions_path)))


def list_evaluation_groups_by_condition(predictions_path: str = config.PREDICTIONS_PATH) -> list:
    """Like list_evaluation_groups(), but bundled by condition: one entry
    per condition, each {"condition": ..., "tags": "full__trial1,full__trial2,..."}
    -- every one of that condition's trials joined into a single comma list,
    which evaluate_run's only_groups (and the --groups CLI flag) already
    accepts as-is. Feeds a CI matrix with one job per CONDITION instead of
    one per (condition, trial) pair -- fewer, larger jobs, each still only
    touching its own rows (see evaluate_run()'s only_groups docstring), but
    now also rebuilding the Docker base/env/instance image only once per
    condition instead of once per (condition, trial) -- fewer total minutes
    AND fewer concurrent jobs needed to run everything in one wave, useful
    when trials x conditions exceeds the CI account's concurrent-job
    ceiling (see .github/workflows/swebench-evaluate.yml)."""
    tags = list_evaluation_groups(predictions_path)
    by_condition = {}
    for tag in tags:
        m = re.match(r"^(.*)__trial\d+$", tag)
        condition = m.group(1) if m else tag
        by_condition.setdefault(condition, []).append(tag)
    return [{"condition": c, "tags": ",".join(t)} for c, t in by_condition.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)


    p_eval = sub.add_parser("evaluate", help="Score run's patches through the official swebench harness.")
    p_eval.add_argument("--out", default=config.RESULTS_PATH)
    p_eval.add_argument("--predictions", default=config.PREDICTIONS_PATH)
    p_eval.add_argument("--summary-out", default=config.SUMMARY_CSV_PATH)
    p_eval.add_argument("--run-id", default=config.SWEBENCH_RUN_ID)
    p_eval.add_argument("--logs-dir", default=config.LOGS_DIR,
                         help="Where swebench's own per-instance evaluation logs/reports are "
                              "written and read back from (default: config.LOGS_DIR, i.e. "
                              "<model_key>/logs/run_evaluation -- see config.model_paths).")
    p_eval.add_argument("--local", action="store_true",
                         help="Evaluate on local Docker instead of Modal's cloud "
                              "(config.EVALUATE_USE_MODAL_BY_DEFAULT's default).")
    p_eval.add_argument("--max-workers", type=int, default=4)
    p_eval.add_argument("--timeout", type=int, default=1800)
    p_eval.add_argument("--report-dir", default=".")
    p_eval.add_argument("--groups", default=None,
                         help="Comma-separated condition/trial group tags (e.g. "
                              "'full__trial1,no_skill__trial1') to evaluate, instead of "
                              "every group in predictions -- see list-groups. Rows outside "
                              "the given groups are left untouched, not marked unscored, "
                              "which is what makes it safe to split evaluation across "
                              "several parallel calls (e.g. a CI matrix, one job per group).")

    p_list_groups = sub.add_parser("list-groups", help="List the condition/trial group tags in predictions.json.")
    p_list_groups.add_argument("--predictions", default=config.PREDICTIONS_PATH)
    p_list_groups.add_argument("--json", action="store_true", help="Print as a JSON array instead of one per line.")
    p_list_groups.add_argument("--by-condition", action="store_true",
                                help="Bundle every condition's trials into one entry each "
                                     "(comma-joined tags), instead of one entry per (condition, trial) -- "
                                     "for a CI matrix with one job per condition instead of per (condition, trial).")

    p_merge = sub.add_parser("merge-logs", help="Merge several evaluate --groups-scoped copies of the "
                                                 "trajectory JSONL (e.g. one per CI matrix job) into one.")
    p_merge.add_argument("--inputs", required=True, help="Comma-separated paths to the copies to merge.")
    p_merge.add_argument("--out", default=config.RESULTS_PATH)
    p_merge.add_argument("--summary-out", default=config.SUMMARY_CSV_PATH)

    args = parser.parse_args()

    if args.command == "evaluate":
        only_groups = [g.strip() for g in args.groups.split(",") if g.strip()] if args.groups else None
        evaluate_run(out=args.out, predictions_path=args.predictions, summary_out=args.summary_out,
                     run_id=args.run_id, modal=not args.local, max_workers=args.max_workers,
                     timeout=args.timeout, report_dir=args.report_dir, logs_dir=args.logs_dir,
                     only_groups=only_groups)
    elif args.command == "list-groups":
        groups = (list_evaluation_groups_by_condition(args.predictions) if args.by_condition
                  else list_evaluation_groups(args.predictions))
        if args.json:
            print(json.dumps(groups))
        else:
            for g in groups:
                print(f"{g['condition']}: {g['tags']}" if isinstance(g, dict) else g)
    elif args.command == "merge-logs":
        input_paths = [p.strip() for p in args.inputs.split(",") if p.strip()]
        merged = evaluate.merge_evaluated_logs(input_paths)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(json.dumps(r) for r in merged) + "\n")
        print(f"Merged {len(input_paths)} file(s) into {out_path}")

        summary = metrics.condition_summary(metrics.load_jsonl(out_path))
        print("\n--- Per-condition summary (real, official pass/fail) ---")
        print(summary.to_string(index=False))
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary CSV written to {summary_path}")


if __name__ == "__main__":
    main()
