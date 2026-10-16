"""
Turns a run's JSONL trajectory log into the numbers and tables people
actually want to look at: a per-condition pass rate summary (same
two-step averaging as the QuixBugs skill-ablation project's metrics.py --
trials average into an instance score, instance scores average into a condition
score), a SkillsBench-style normalised gain over the no_skill baseline,
tool-call counts, and a readable step-by-step trace for one run. Used by
both run_repo_ablation.py (prints a summary after a run) and the notebook.

Each JSONL line is one of two record types, distinguished by
record_type: "step" (one tool call or one text-only reply, in order) or
"result" (the final pass/fail verdict for one (instance, condition, trial)
triple, written once the loop ends). Loading mixes both into one
DataFrame; the helpers below split them back out as needed.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Tool names that count as "the agent modified a file", across every
# provider this project supports. "write_file" is this project's own
# canonical tool; "Write" and "Edit" are Claude Code's own built-in tools
# when a run goes through harness/claude_code_runner.py instead of the
# turn-by-turn agent_loop.py path -- see that module's docstring.
_EDIT_TOOL_NAMES = {"write_file", "Write", "Edit"}


def load_jsonl(path) -> pd.DataFrame:
    path = Path(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    return pd.DataFrame(records)


def steps_only(df: pd.DataFrame) -> pd.DataFrame:
    """Just the "step" records. Guarantees the columns every step-record
    function downstream expects (kind, tool, args, result, text,
    latency_s) actually exist, even when this particular df happens to
    have zero step records at all -- e.g. every run in a batch errored
    before logging a single step, or (specific to this project) a
    claude-code --dry-run, which returns immediately with an empty
    trajectory. Without this, pandas never creates those columns in the
    first place when no record anywhere has them, and every downstream
    function that reads df["kind"] etc. raises a bare KeyError instead of
    a normal empty result."""
    steps = df[df["record_type"] == "step"].copy()
    for col in ("kind", "tool", "args", "result", "text", "latency_s", "total_tokens"):
        if col not in steps.columns:
            steps[col] = None
    return steps


def steps_preview(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The first n step rows, formatted the same readable way
    trajectory_table is (tool name, a short summary of what was passed in
    and what came back -- not the raw args/result dicts, which for
    write_file or a large read_file are the entire file contents and make
    it hard to tell at a glance which tool even ran). load_jsonl mixes
    "step" and "result" records into one DataFrame, which unions every
    column across both record types -- a plain df.head() on that combined
    frame is full of NaN for whichever record type a given row *isn't*,
    which looks like something's broken even though it isn't; this sticks
    to step rows only, so that doesn't come up."""
    steps = steps_only(df).sort_values(["instance_id", "condition", "trial", "step"]).head(n)

    rows = []
    for _, r in steps.iterrows():
        base = {
            "instance_id": r.get("instance_id"),
            "condition": r.get("condition"),
            "trial": r.get("trial"),
            "step": int(r["step"]),
        }
        if r["kind"] == "tool_call":
            rows.append({
                **base,
                "action": r["tool"],
                "detail": format_tool_args(r["tool"], r.get("args")),
                "result": format_tool_result(r["tool"], r.get("result")),
            })
        elif r["kind"] == "final_check":
            rows.append({
                **base,
                "action": "(harness) run_tests",
                "detail": "automatic check after the run ended",
                "result": format_tool_result("run_tests", r.get("result")),
            })
        else:
            rows.append({
                **base,
                "action": "(message, no tool call)",
                "detail": r.get("text") or "",
                "result": "",
            })
    return pd.DataFrame(rows, columns=["instance_id", "condition", "trial", "step", "action", "detail", "result"])


def results_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["record_type"] == "result"].copy()


def tool_call_counts(df: pd.DataFrame, instance_id: str = None, condition: str = None) -> pd.DataFrame:
    """How many times each tool was called, across every run in the log
    (or narrowed to one instance and/or one condition)."""
    steps = steps_only(df)
    steps = steps[steps["kind"] == "tool_call"]
    if instance_id is not None:
        steps = steps[steps["instance_id"] == instance_id]
    if condition is not None:
        steps = steps[steps["condition"] == condition]
    if steps.empty:
        return pd.DataFrame(columns=["tool", "count"])
    return (
        steps.groupby("tool").size()
             .reset_index(name="count")
             .sort_values("count", ascending=False)
             .reset_index(drop=True)
    )


def repeated_call_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (instance, condition, trial): how many tool calls repeated a
    tool + arguments combination already made earlier in that same run,
    and whether the run ever reached a write_file call at all.

    A high repeat count is a direct, quantifiable version of "the agent
    got stuck re-exploring instead of making progress." In practice this
    usually isn't the same call twice *in a row* -- it's cycling between a
    small set of paths it's already seen (list_files '.', then
    'src/pkg', then '.' again, then 'src', then '.' again, ...), so this
    checks against everything made earlier in the run, not just the
    immediately preceding step. This is what actually distinguished a weak
    condition's failures in practice: not wrong fixes, but never reaching
    a fix at all because most of the step budget went to revisiting
    already-seen state. Cross-tab this against condition_summary's
    pass_rate (or just eyeball it next to reached_write_file) rather than
    reading it alone -- a repeat or two is normal, a run maxing out its
    step budget on repeats with reached_write_file=False is the pattern
    worth a closer look via trajectory_table on that specific (instance,
    condition, trial).
    """
    steps = steps_only(df)
    steps = steps[steps["kind"] == "tool_call"].sort_values(["instance_id", "condition", "trial", "step"]).copy()
    if steps.empty:
        return pd.DataFrame(columns=["instance_id", "condition", "trial", "repeated_calls", "reached_write_file"])

    steps["call_key"] = steps["tool"] + "|" + steps["args"].apply(lambda a: json.dumps(a or {}, sort_keys=True))

    rows = []
    for (instance_id, condition, trial), group in steps.groupby(["instance_id", "condition", "trial"]):
        keys = group["call_key"].tolist()
        seen = set()
        repeats = 0
        for k in keys:
            if k in seen:
                repeats += 1
            seen.add(k)
        rows.append({
            "instance_id": instance_id,
            "condition": condition,
            "trial": trial,
            "repeated_calls": repeats,
            # Broadened beyond just "write_file": Claude Code's own tools
            # use different names for the same underlying action (see
            # harness/claude_code_runner.py's module docstring) -- this
            # still counts as "made an edit" for stuck_run_rate's purposes.
            "reached_write_file": bool(group["tool"].isin(_EDIT_TOOL_NAMES).any()),
        })
    return pd.DataFrame(rows)


def stuck_run_rate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per condition: what fraction of its runs had at least one
    repeated call and never reached write_file -- "got stuck exploring and
    never even attempted a fix," as opposed to "attempted a fix that
    didn't work." Read this next to condition_summary's pass_rate: a
    condition with a low pass rate and a high stuck rate here is failing
    for a different reason than a condition with a low pass rate and a
    low stuck rate (the former never tried; the latter tried and got it
    wrong)."""
    calls = repeated_call_counts(df)
    if calls.empty:
        return pd.DataFrame(columns=["condition", "n_runs", "stuck_runs", "stuck_rate"])
    calls["stuck"] = (calls["repeated_calls"] > 0) & (~calls["reached_write_file"])
    return calls.groupby("condition", as_index=False).agg(
        n_runs=("stuck", "size"),
        stuck_runs=("stuck", "sum"),
        stuck_rate=("stuck", "mean"),
    )


def _per_pair_costs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, condition, trial): steps used, tool calls made,
    tokens spent, latency spent -- summed across every attempt in that
    run, not just the last one, since every step costs something whether
    or not it was the one that mattered."""
    results = results_only(df)[
        ["instance_id", "condition", "trial", "passed", "steps_used", "finished_explicitly"]
    ].copy()
    steps = steps_only(df)

    tool_call_steps = steps[steps["kind"] == "tool_call"]
    calls_per_pair = (
        tool_call_steps.groupby(["instance_id", "condition", "trial"]).size()
                        .reset_index(name="tool_calls")
    )
    cost_per_pair = steps.groupby(["instance_id", "condition", "trial"], as_index=False).agg(
        total_tokens=("total_tokens", lambda s: s.fillna(0).sum()),
        total_latency_s=("latency_s", "sum"),
    )

    merged = (
        results.merge(calls_per_pair, on=["instance_id", "condition", "trial"], how="left")
               .merge(cost_per_pair, on=["instance_id", "condition", "trial"], how="left")
    )
    merged["tool_calls"] = merged["tool_calls"].fillna(0)
    return merged


def task_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, condition): trials averaged into a single score
    for that pair. This is the finer-grained table underneath
    condition_summary below -- useful for spotting whether a condition's
    overall number is driven by one task rather than a general effect."""
    per_pair = _per_pair_costs(df)
    return per_pair.groupby(["instance_id", "condition"], as_index=False).agg(
        n_trials=("trial", "count"),
        pass_rate=("passed", "mean"),
        mean_steps_used=("steps_used", "mean"),
        mean_tool_calls=("tool_calls", "mean"),
        mean_total_tokens=("total_tokens", "mean"),
        mean_latency_s=("total_latency_s", "mean"),
    )


def steps_by_task(df: pd.DataFrame) -> pd.DataFrame:
    """Steps used, one row per task and one column per condition -- trials
    already averaged into each cell. Pass rate can end up identical across
    conditions while the actual difference is how many steps it took to
    get there; this table is for spotting that directly, instance by instance,
    rather than digging it out of the longer per-(task, condition) table."""
    per_task = task_condition_summary(df)
    pivot = per_task.pivot(index="instance_id", columns="condition", values="mean_steps_used")
    return pivot.round(2)


def steps_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """One row per condition: steps used, averaged across tasks the same
    way condition_summary averages pass rate (per instance first, then across
    instances at one vote per instance) -- plus the spread across instances, so a
    condition's average being dragged around by one unusually hard or easy
    instance is visible as a wide min-max range rather than hidden inside a
    single mean."""
    per_task = task_condition_summary(df)
    return per_task.groupby("condition", as_index=False).agg(
        n_tasks=("instance_id", "count"),
        mean_steps=("mean_steps_used", "mean"),
        median_steps=("mean_steps_used", "median"),
        min_steps=("mean_steps_used", "min"),
        max_steps=("mean_steps_used", "max"),
    )


def pending_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """(instance_id, condition, trial) triples whose "passed" is still
    None -- either not evaluated yet, or evaluate.py tried and couldn't
    find a resolution (see the eval_error column below for which).
    run_swebench_ablation.py's agent runs write results with passed=None,
    since there's no way to know pass/fail until evaluate.py has scored
    the patch through the official harness. Check this before trusting
    condition_summary's pass_rate: pandas' .mean() silently skips
    None/NaN rows, so a pass rate computed before evaluation finishes (or
    where evaluation failed to find a result for some pairs) isn't wrong
    exactly, but it's an average over whichever pairs happened to be
    scored already, not all of them.

    eval_error is None for pairs simply not evaluated yet (run evaluate);
    a real string means evaluate.py DID run for that pair and couldn't
    find a resolution -- read the string itself for why (a report.json
    never got written for that instance/condition, an aggregate report
    didn't list it, etc.) before assuming it's the same issue as an
    unevaluated pair."""
    results = results_only(df)
    pending = results[results["passed"].isna()]
    cols = ["instance_id", "condition", "trial"]
    if "eval_error" in pending.columns:
        cols.append("eval_error")
    return pending[cols].reset_index(drop=True)


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per condition: pass rate and cost, averaged the SkillsBench
    way -- trials into a per-task score (task_condition_summary above),
    then instance scores into one condition-level score at a fixed denominator
    of one vote per instance, regardless of how many trials any individual
    instance ran under that condition.
    """
    per_task = task_condition_summary(df)
    return per_task.groupby("condition", as_index=False).agg(
        n_tasks=("instance_id", "count"),
        # ``mean`` below deliberately excludes missing values: a run whose
        # evaluator crashed has no pass/fail verdict.  Surface that fact next
        # to pass_rate so a partial evaluation cannot look like a complete
        # zero-pass run.
        n_evaluated=("pass_rate", "count"),
        n_pending=("pass_rate", lambda values: values.isna().sum()),
        pass_rate=("pass_rate", "mean"),
        mean_steps_used=("mean_steps_used", "mean"),
        mean_tool_calls=("mean_tool_calls", "mean"),
        mean_total_tokens=("mean_total_tokens", "mean"),
        mean_latency_s=("mean_latency_s", "mean"),
    )


def normalised_gain(summary: pd.DataFrame, baseline: str = "no_skill") -> pd.DataFrame:
    """
    SkillsBench-style (Hake, 1998) normalised gain for every condition
    except the baseline: g = (pass_skill - pass_baseline) / (1 - pass_baseline).
    Shows how much of the *possible* improvement each condition captures,
    not just the raw pass-rate difference.

    When the baseline is already at a 1.0 pass rate there is no positive
    headroom left (denominator = 0), so the ratio is mathematically
    undefined -- but a condition that scores below a perfect baseline is
    still a real, informative result, not a missing one. In that case this
    falls back to the raw pass-rate delta (bounded in [-1, 0], since
    neither rate can exceed 1.0) instead of returning NaN, and flags the
    row via baseline_saturated. pass_rate_delta is always included, even
    when the ordinary ratio applies, so the raw effect size is never
    hidden behind the normalisation.
    """
    base_row = summary[summary["condition"] == baseline]
    if base_row.empty:
        raise ValueError(
            f"Baseline condition '{baseline}' not found in summary. "
            f"Available: {list(summary['condition'])}"
        )
    p_base = base_row["pass_rate"].iloc[0]
    denom = 1 - p_base

    columns = ["condition", "pass_rate", "baseline_pass_rate", "pass_rate_delta",
               "normalised_gain", "baseline_saturated"]
    rows = []
    for _, row in summary[summary["condition"] != baseline].iterrows():
        delta = row["pass_rate"] - p_base
        saturated = denom <= 0
        g = delta if saturated else delta / denom
        rows.append({
            "condition": row["condition"],
            "pass_rate": row["pass_rate"],
            "baseline_pass_rate": p_base,
            "pass_rate_delta": delta,
            "normalised_gain": g,
            "baseline_saturated": saturated,
        })
    # Explicit columns= even for the empty case (baseline was the only
    # condition run) -- pd.DataFrame([]) with no rows has NO columns at
    # all, not even "condition", so gains["condition"] would raise a bare,
    # confusing KeyError instead of the caller getting back a normal,
    # checkable empty result (gains.empty is True, gains["condition"] is
    # an empty Series, not an error).
    return pd.DataFrame(rows, columns=columns)


def pairwise_normalised_gain(summary: pd.DataFrame) -> pd.DataFrame:
    """
    normalised_gain(summary, baseline=B) for every condition B present in
    summary, not just a fixed no_skill baseline -- e.g. how does `full`
    compare to `minus_examples` directly, not just how each compares to
    `no_skill` separately. Answers questions a single fixed baseline
    can't: whether removing recovery advice hurts `full` about as much as
    removing examples does (compare `full`'s gain over
    `minus_recovery_advice` to its gain over `minus_examples` directly),
    or whether a leave-one-out variant lands closer to `full` or closer
    to `no_skill` (compare its gain over each).

    Tidy format: one row per (condition, baseline) ordered pair, tagged
    with which baseline that row used -- pairwise_gain_matrix pivots this
    into an easier-to-eyeball condition x baseline table.
    """
    columns = ["condition", "baseline", "pass_rate", "baseline_pass_rate",
               "pass_rate_delta", "normalised_gain", "baseline_saturated"]
    conditions = list(summary["condition"].unique())
    frames = []
    for baseline in conditions:
        gains = normalised_gain(summary, baseline=baseline)
        if gains.empty:
            continue
        gains = gains.copy()
        gains.insert(1, "baseline", baseline)
        frames.append(gains)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def pairwise_gain_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """pairwise_normalised_gain(summary), pivoted into a condition x
    baseline matrix of normalised_gain values -- read row R, column C as
    "R's normalised gain over C". Easier to eyeball than the tidy table
    when comparing several pairs at once. A condition against itself
    never appears (normalised_gain excludes the baseline row from its own
    output), left as NaN on the diagonal."""
    pairwise = pairwise_normalised_gain(summary)
    if pairwise.empty:
        return pd.DataFrame()
    return pairwise.pivot(index="condition", columns="baseline", values="normalised_gain").round(3)


def normalised_steps(summary: pd.DataFrame, baseline: str = "no_skill") -> pd.DataFrame:
    """
    The step-based analogue of normalised_gain: how many more (or fewer)
    steps each condition took relative to the baseline, as both a raw
    delta and a proportion of the baseline's own step count. Matters most
    once pass_rate alone stops differentiating conditions -- a capable
    enough agent solves everything eventually regardless of condition,
    given enough steps, so efficiency (not just success) is where a
    skill's effect would still show up. See RQ4 in the project proposal.

    steps_delta = mean_steps_skill - mean_steps_baseline (positive means
    MORE steps than baseline -- slower, less efficient; negative means
    FEWER -- more efficient).
    normalised_steps_increase = steps_delta / mean_steps_baseline (as a
    proportion of the baseline's own step count -- 0.20 means 20% more
    steps than baseline took; -0.20 means 20% fewer).

    If the baseline's own mean_steps_used is 0 (every baseline run
    somehow took zero steps -- not realistic, but guarded the same way
    normalised_gain guards a saturated denominator), this falls back to
    the raw steps_delta and flags the row via baseline_zero_steps rather
    than dividing by zero.
    """
    base_row = summary[summary["condition"] == baseline]
    if base_row.empty:
        raise ValueError(
            f"Baseline condition '{baseline}' not found in summary. "
            f"Available: {list(summary['condition'])}"
        )
    base_steps = base_row["mean_steps_used"].iloc[0]

    columns = ["condition", "mean_steps_used", "baseline_steps_used", "steps_delta",
               "normalised_steps_increase", "baseline_zero_steps"]
    rows = []
    for _, row in summary[summary["condition"] != baseline].iterrows():
        delta = row["mean_steps_used"] - base_steps
        zero_baseline = base_steps == 0
        ratio = delta if zero_baseline else delta / base_steps
        rows.append({
            "condition": row["condition"],
            "mean_steps_used": row["mean_steps_used"],
            "baseline_steps_used": base_steps,
            "steps_delta": delta,
            "normalised_steps_increase": ratio,
            "baseline_zero_steps": zero_baseline,
        })
    return pd.DataFrame(rows, columns=columns)


def pairwise_normalised_steps(summary: pd.DataFrame) -> pd.DataFrame:
    """normalised_steps(summary, baseline=B) for every condition B present
    in summary -- the step-based analogue of pairwise_normalised_gain."""
    columns = ["condition", "baseline", "mean_steps_used", "baseline_steps_used",
               "steps_delta", "normalised_steps_increase", "baseline_zero_steps"]
    conditions = list(summary["condition"].unique())
    frames = []
    for baseline in conditions:
        steps = normalised_steps(summary, baseline=baseline)
        if steps.empty:
            continue
        steps = steps.copy()
        steps.insert(1, "baseline", baseline)
        frames.append(steps)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


# Display order for every per-condition table and chart: the two anchors
# first and last, the four leave-one-out variants in between. Conditions a
# run didn't cover are skipped; conditions not listed here go at the end.
CONDITION_ORDER = [
    "full", "minus_examples", "minus_procedure",
    "minus_recovery_advice", "minus_validation", "no_skill",
]


def order_conditions(frame: pd.DataFrame, column: str = "condition") -> pd.DataFrame:
    """frame's rows sorted into CONDITION_ORDER by its `column`."""
    rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    key = frame[column].map(lambda c: rank.get(c, len(rank)))
    return frame.assign(_order=key).sort_values(["_order", column]).drop(columns="_order").reset_index(drop=True)


def pass_rate_matrix(df: pd.DataFrame, value: str = "pass_rate") -> pd.DataFrame:
    """Instance x condition table of per-instance pass rates (trials already
    averaged), columns in CONDITION_ORDER. The unit every statistical helper
    below resamples and pairs over. `value` picks any other
    task_condition_summary column instead, e.g. "mean_steps_used"."""
    per_task = task_condition_summary(df)
    matrix = per_task.pivot(index="instance_id", columns="condition", values=value)
    cols = [c for c in CONDITION_ORDER if c in matrix.columns] + \
           [c for c in matrix.columns if c not in CONDITION_ORDER]
    return matrix[cols].astype(float)


def run_outcomes(df: pd.DataFrame, step_cap: int = None) -> pd.DataFrame:
    """
    One row per condition describing *how* its runs ended, not just whether
    they passed: how often a patch was produced at all (has_patch), how
    often the agent stopped on its own (finished_explicitly) versus being
    cut off at step_cap, how often it got stuck (see stuck_run_rate), and
    the cost of each resolved instance. Rates are over runs, not instances;
    pass_rate is the same instance-averaged figure as condition_summary.

    steps_per_success / tokens_per_success divide per-attempt cost by pass
    rate, treating every failed attempt as pure waste. step_cap defaults to
    the largest steps_used seen in the log.
    """
    results = results_only(df)
    per_pair = _per_pair_costs(df)
    if "has_patch" in results.columns:
        per_pair = per_pair.merge(
            results[["instance_id", "condition", "trial", "has_patch"]],
            on=["instance_id", "condition", "trial"], how="left",
        )
    else:
        per_pair["has_patch"] = None
    cap = step_cap if step_cap is not None else per_pair["steps_used"].max()
    per_pair["hit_step_cap"] = per_pair["steps_used"] >= cap

    runs = per_pair.groupby("condition", as_index=False).agg(
        n_runs=("trial", "size"),
        has_patch_rate=("has_patch", lambda s: s.astype(float).mean()),
        finished_explicitly_rate=("finished_explicitly", lambda s: s.astype(float).mean()),
        hit_step_cap_rate=("hit_step_cap", "mean"),
    )
    stuck = stuck_run_rate(df)[["condition", "stuck_rate"]]
    summary = condition_summary(df)[["condition", "pass_rate", "mean_steps_used", "mean_total_tokens"]]
    out = summary.merge(runs, on="condition").merge(stuck, on="condition", how="left")
    out["pass_rate"] = out["pass_rate"].astype(float)
    out["stuck_rate"] = out["stuck_rate"].astype(float).fillna(0.0)
    out["steps_per_success"] = out["mean_steps_used"] / out["pass_rate"]
    out["tokens_per_success"] = out["mean_total_tokens"] / out["pass_rate"]
    cols = ["condition", "n_runs", "pass_rate", "has_patch_rate", "stuck_rate",
            "finished_explicitly_rate", "hit_step_cap_rate", "mean_steps_used",
            "steps_per_success", "mean_total_tokens", "tokens_per_success"]
    return order_conditions(out[cols])


def _bootstrap_indices(n_items: int, n_boot: int, seed: int):
    return np.random.default_rng(seed).integers(0, n_items, size=(n_boot, n_items))


def pass_rate_ci(df: pd.DataFrame, n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """
    Per-condition pass rate with a percentile bootstrap confidence interval,
    resampling *instances* (not individual runs) with replacement. Trials of
    the same instance are strongly correlated -- an instance is usually
    either solvable or not -- so treating runs as independent would make the
    interval far too narrow. Resampling at the instance level is the honest
    unit given how few instances this benchmark covers.
    """
    matrix = pass_rate_matrix(df)
    values = matrix.to_numpy()
    idx = _bootstrap_indices(len(matrix), n_boot, seed)
    alpha = (1 - level) / 2
    rows = []
    for j, condition in enumerate(matrix.columns):
        col = values[:, j]
        boots = np.nanmean(col[idx], axis=1)
        rows.append({
            "condition": condition,
            "n_instances": int((~np.isnan(col)).sum()),
            "pass_rate": np.nanmean(col),
            "ci_low": np.quantile(boots, alpha),
            "ci_high": np.quantile(boots, 1 - alpha),
        })
    return order_conditions(pd.DataFrame(rows))


def _sign_flip_p_value(diffs, n_resamples: int = 100_000, seed: int = 0) -> float:
    """Two-sided paired permutation test on per-instance differences: under
    the null, each instance's difference is equally likely to have either
    sign. Exact (every sign pattern enumerated) up to 16 instances, Monte
    Carlo above that."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0 or np.allclose(diffs, 0):
        return 1.0
    observed = abs(diffs.mean())
    if len(diffs) <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=len(diffs))))
    else:
        signs = np.random.default_rng(seed).choice([1, -1], size=(n_resamples, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((null >= observed - 1e-12).mean())


def paired_condition_test(df: pd.DataFrame, reference: str = "full", value: str = "pass_rate",
                          n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """
    Every other condition compared against `reference`, paired by instance:
    the mean per-instance difference in `value` (condition minus reference;
    pass rate by default, or e.g. "mean_steps_used" for cost),
    a bootstrap confidence interval on that difference (instances
    resampled, as in pass_rate_ci), an exact sign-flip permutation p-value,
    and on how many instances the condition scored higher / the same /
    lower than the reference.

    Pairing matters: instance difficulty varies far more than the condition
    effect does, so comparing two conditions' pooled pass rates without
    pairing would bury a consistent per-instance effect under that
    between-instance variance. With a dozen instances, an exact test can
    never reach small p-values unless nearly every instance moves the same
    way -- read p alongside the higher/same/lower counts.
    """
    matrix = pass_rate_matrix(df, value=value)
    if reference not in matrix.columns:
        raise ValueError(f"Reference condition {reference!r} not found. Available: {list(matrix.columns)}")
    idx = _bootstrap_indices(len(matrix), n_boot, seed)
    alpha = (1 - level) / 2
    rows = []
    for condition in matrix.columns:
        if condition == reference:
            continue
        diffs = (matrix[condition] - matrix[reference]).to_numpy()
        boots = np.nanmean(diffs[idx], axis=1)
        rows.append({
            "condition": condition,
            "reference": reference,
            "delta": np.nanmean(diffs),
            "ci_low": np.quantile(boots, alpha),
            "ci_high": np.quantile(boots, 1 - alpha),
            "p_value": _sign_flip_p_value(diffs, seed=seed),
            "instances_higher": int((diffs > 0).sum()),
            "instances_same": int((diffs == 0).sum()),
            "instances_lower": int((diffs < 0).sum()),
        })
    return order_conditions(pd.DataFrame(rows))


def format_tool_args(tool: str, args: dict) -> str:
    """One-line summary of what a tool was called with. write_file's
    content argument is the whole new file, which is the wrong thing to
    put in a one-line trace -- show its size instead and let someone go to
    the raw JSONL if they need the actual text.

    Handles both this project's own canonical tool names and Claude
    Code's built-in ones (Read/Write/Edit/Bash/Glob -- see
    harness/claude_code_runner.py), since a run can come from either."""
    args = args or {}
    if tool == "write_file":
        content = args.get("content", "")
        return f"path={args.get('path')!r}, content={len(content)} chars"
    if tool == "read_file":
        return f"path={args.get('path')!r}"
    if tool == "list_files":
        return f"path={args.get('path', '.')!r}"
    if tool == "finish":
        return args.get("summary", "")
    if tool == "run_tests":
        return ""
    if tool == "Read":
        return f"file_path={args.get('file_path')!r}"
    if tool == "Write":
        content = args.get("content", "")
        return f"file_path={args.get('file_path')!r}, content={len(content)} chars"
    if tool == "Edit":
        return f"file_path={args.get('file_path')!r}, old={len(args.get('old_string',''))} chars, new={len(args.get('new_string',''))} chars"
    if tool == "Bash":
        cmd = args.get("command", "")
        return cmd if len(cmd) <= 100 else cmd[:100] + "…"
    if tool == "Glob":
        return f"pattern={args.get('pattern')!r}"
    return str(args)


def format_tool_result(tool: str, result: dict) -> str:
    """One-line summary of what a tool call came back with, in the same
    spirit as format_tool_args -- readable, not exhaustive. Same
    both-vocabularies handling as format_tool_args."""
    result = result or {}
    if "error" in result:
        return f"error: {result['error']}"
    if tool == "list_files":
        entries = result.get("entries", [])
        shown = ", ".join(entries[:6])
        more = f", … ({len(entries) - 6} more)" if len(entries) > 6 else ""
        return f"{len(entries)} entries: {shown}{more}"
    if tool == "read_file":
        content = result.get("content", "")
        return f"{len(content)} chars"
    if tool == "write_file":
        return f"wrote {result.get('path')}"
    if tool == "run_tests":
        status = "PASSED" if result.get("passed") else "FAILED"
        lines = [l for l in result.get("output", "").strip().splitlines() if "truncated" not in l]
        last_line = lines[-1] if lines else ""
        return f"{status} -- {last_line}"
    if tool == "finish":
        return "acknowledged"
    if tool in ("Read", "Write", "Edit", "Bash", "Glob"):
        # Claude Code's own tools report a plain "output" string (see
        # claude_code_runner._parse_stream_json) -- not the structured
        # dicts this project's own tools return, so there's no per-tool
        # field to special-case beyond the error check above.
        output = result.get("output", "")
        output = output if len(output) <= 200 else output[:200] + "…"
        prefix = "ERROR: " if result.get("is_error") else ""
        return f"{prefix}{output}"
    return str(result)


def trajectory_table(df: pd.DataFrame, instance_id: str, condition: str, trial: int = 1) -> pd.DataFrame:
    """The step-by-step trace for one (task, condition, trial) run, as a
    table: what the model called, with what, and what came back. This is
    the thing to render in the notebook to actually see what the agent
    did, and how that differed (or didn't) between conditions.

    Raises if nothing matches (instance_id, condition, trial) rather than
    silently returning an empty table -- an empty table with no
    explanation looks like broken plumbing rather than what it usually
    actually is: instance_id or condition set to something that wasn't part of
    this particular run (e.g. every instance_id in the selected subset, when
    this run only covered a subset of tasks)."""
    steps = steps_only(df)
    subset = steps[
        (steps["instance_id"] == instance_id) & (steps["condition"] == condition) & (steps["trial"] == trial)
    ].sort_values("step")

    if subset.empty:
        available_instances = sorted(steps["instance_id"].dropna().unique())
        available_conditions = sorted(steps["condition"].dropna().unique())
        raise ValueError(
            f"No steps found for instance_id={instance_id!r}, condition={condition!r}, trial={trial!r}. "
            f"Instances actually present in this run: {available_instances}. "
            f"Conditions actually present: {available_conditions}. "
            f"If this run covered a subset of instances, instance_ids (every instance in "
            f"the selected subset) isn't the same list as what this particular run covered -- "
            f"pick from df['instance_id'].unique() instead."
        )

    rows = []
    for _, r in subset.iterrows():
        if r["kind"] == "tool_call":
            rows.append({
                "step": int(r["step"]),
                "action": r["tool"],
                "detail": format_tool_args(r["tool"], r.get("args")),
                "result": format_tool_result(r["tool"], r.get("result")),
                "latency_s": r["latency_s"],
            })
        elif r["kind"] == "final_check":
            rows.append({
                "step": int(r["step"]),
                "action": "(harness) run_tests",
                "detail": "automatic check after the run ended, not something the agent chose to do",
                "result": format_tool_result("run_tests", r.get("result")),
                "latency_s": r["latency_s"],
            })
        else:
            rows.append({
                "step": int(r["step"]),
                "action": "(message, no tool call)",
                "detail": r.get("text") or "",
                "result": "",
                "latency_s": r["latency_s"],
            })
    return pd.DataFrame(rows, columns=["step", "action", "detail", "result", "latency_s"])


def print_trajectory(df: pd.DataFrame, instance_id: str, condition: str, trial: int = 1) -> None:
    """Plain-text version of trajectory_table, for a quick look outside a
    notebook or for pasting into a report."""
    try:
        table = trajectory_table(df, instance_id, condition, trial)
    except ValueError as e:
        print(e)
        return
    for _, r in table.iterrows():
        print(f"[{r['step']:>2}] {r['action']:<24} {r['detail']}")
        if r["result"]:
            print(f"     -> {r['result']}")


# ---------------------------------------------------------------------------
# Trajectory behaviour: localisation, checking, dead-end calls, failure types
# ---------------------------------------------------------------------------
# Added after the final 6-trial runs. Everything below only *reads* an
# existing trajectory log (plus the task metadata and skill files already in
# the repo) -- nothing here writes to results/, runs/ or logs/. This block is
# identical in SWE-bench-lite_agent/metrics.py and
# quixbugs_repo_level_tasks/metrics.py: the run key ("instance_id" or "task")
# is detected from the log, so either project's log can be passed to either
# copy (the cross-dataset cell in the Sonnet 5 SWE-bench notebook relies on
# that).

import re as _re

_CHANGE_TOOLS = {"Edit", "MultiEdit", "write_file"}   # modify an existing file
_CREATE_TOOLS = {"Write"}                             # create or overwrite a whole file
_EXEC_RE = _re.compile(r"(^|[\s;&|(])(python3?|pytest|py\.test)\b")
_ANSI_RE = _re.compile(r"\x1b\[[0-9;]*m")
_PATH = r"""['"]?([^\s'";|&()<>]+\.\w+)['"]?"""
_BASH_WRITE_RES = [
    _re.compile(r"(?<![0-9&<>=-])>>?\s*" + _PATH),                          # shell redirect into a file
    _re.compile(r"\btee\s+(?:-a\s+)?" + _PATH),
    _re.compile(r"""open\(\s*['"]([^'"]+)['"]\s*,\s*['"][^'"]*[wa+]"""),     # Python heredoc writing a file
    _re.compile(r"""Path\(\s*['"]([^'"]+)['"]\s*\)\.write_text"""),
]
_IN_PLACE_RE = _re.compile(r"\b(sed|perl)\s+(-\S*\s+)*-i")


def bash_write_paths(command: str) -> list:
    """Files a shell command writes to, by a regex over the command text:
    `sed -i`/`perl -i` (every *.py file it names), redirects, `tee`, and
    Python heredocs that open a file for writing. Paths under /tmp and
    /dev are ignored. Heuristic -- it misses writes through a variable
    path, so it can only under-count."""
    command = command or ""
    paths = []
    if _IN_PLACE_RE.search(command):
        paths += _re.findall(r"[^\s'\";|&]+\.py\b", command)
    for pattern in _BASH_WRITE_RES:
        paths += pattern.findall(command)
    return [p for p in dict.fromkeys(paths) if not p.startswith(("/tmp", "/dev"))]
FAILURE_TYPES = [
    "never edited",
    "missed the target file(s)",
    "edited target, no working check after last edit",
    "edited target, checked, still failed",
]


def run_key(df: pd.DataFrame) -> str:
    """The column that names a task in this log: "instance_id" (SWE-bench
    Lite) or "task" (QuixBugs repo-level)."""
    if "instance_id" in df.columns and df["instance_id"].notna().any():
        return "instance_id"
    return "task"


def target_files_from_patches(subset_path) -> dict:
    """SWE-bench Lite: instance_id -> set of files the gold patch changes.
    The gold patch is never shown to the agent; it is only used here to ask
    whether the agent edited the file the real fix lives in."""
    instances = json.loads(Path(subset_path).read_text())
    return {x["instance_id"]: set(_re.findall(r"^diff --git a/(\S+) b/", x["patch"], _re.M))
            for x in instances}


def target_files_from_tasks(tasks_dir) -> dict:
    """QuixBugs repo-level: task_id -> set of files task.json's entry_point
    names as the buggy file(s) (two files for formula_validator)."""
    out = {}
    for path in sorted(Path(tasks_dir).glob("*/task.json")):
        task = json.loads(path.read_text())
        out[task["task_id"]] = set(_re.findall(r"src/\S+?\.py", task["entry_point"]))
    return out


def classify_dead_end(tool: str, args: dict, result: dict):
    """Why a tool call could not have done anything useful, or None if it
    could. Three kinds, all decided from the call's own output:
      - "command not found": the shell had no such program (e.g. `python`
        when only `python3` exists, or no `pytest`);
      - "no tests collected": a test runner ran but found nothing to run;
      - "missing path": the file or directory the call named did not exist.
    A failing test or a traceback is *not* a dead end -- it is information."""
    result = result or {}
    out = _ANSI_RE.sub("", result.get("output") or "")
    is_error = bool(result.get("is_error"))
    if tool == "Bash":
        if _re.search(r"command not found|python3? not found|pytest not found|No module named pytest", out):
            return "command not found"
        if _re.search(r"no tests ran|no tests collected", out):
            return "no tests collected"
        if is_error and _re.search(r"no such file or directory", out, _re.I):
            return "missing path"
    elif is_error and _re.search(r"does not exist", out):
        return "missing path"
    return None


def run_behaviour(df: pd.DataFrame, targets: dict, step_cap: int = None) -> pd.DataFrame:
    """
    One row per run describing what the agent did, read from its tool calls:

    - localisation: did it edit the target file(s) at all (edited_target,
      all_targets_edited), and at which step (first_target_edit_step;
      first_target_edit_capped counts a run that never edits the target at
      step_cap, so it can be averaged); edits to other existing files
      (off_target_edits) and whole-file writes elsewhere, usually scratch
      scripts (scratch_writes);
    - checking: how many calls executed code (exec_calls: python/pytest),
      and whether a working one (not a dead end) came after the last edit
      (checked_after_last_edit);
    - waste: dead-end calls by kind (see classify_dead_end);
    - failure_type for failed runs (see FAILURE_TYPES), None for passes.

    `targets` maps each task/instance to its target files
    (target_files_from_patches / target_files_from_tasks). Edits are matched
    by path suffix, so absolute and "./"-prefixed paths both count. An edit
    is an Edit/MultiEdit/Write call *or* a Bash command that writes a file
    (bash_write_paths: Haiku 4.5 often patches with `sed -i` or a Python
    heredoc), so first_edit_step here can be earlier than the Edit/Write-only
    first-edit metric used in the six-trial diagnostics cells.
    """
    key = run_key(df)
    results = results_only(df).set_index([key, "condition", "trial"])
    steps = steps_only(df)
    calls = steps[steps["kind"] == "tool_call"].sort_values("step")
    cap = step_cap if step_cap is not None else int(results["steps_used"].max())
    groups = dict(tuple(calls.groupby([key, "condition", "trial"])))

    rows = []
    for (task, condition, trial), r in results.iterrows():
        g = groups.get((task, condition, trial), calls.iloc[0:0])
        task_targets = targets.get(task, set())
        edited_paths = pd.Series([
            [(a or {}).get("file_path") or ""] if tool in _CHANGE_TOOLS | _CREATE_TOOLS
            else bash_write_paths((a or {}).get("command")) if tool == "Bash" else []
            for tool, a in zip(g["tool"], g["args"])
        ], index=g.index, dtype=object)
        is_edit = edited_paths.map(bool)
        on_target = edited_paths.map(lambda ps: any(p.endswith(t) for p in ps for t in task_targets))
        is_bash = g["tool"] == "Bash"
        edits = g[is_edit]
        target_edits = g[is_edit & on_target]
        hit = {t for ps in edited_paths for p in ps for t in task_targets if p.endswith(t)}

        dead = [classify_dead_end(t, a, res) for t, a, res in zip(g["tool"], g["args"], g["result"])]
        dead = pd.Series(dead, index=g.index, dtype=object)
        is_exec = (g["tool"] == "Bash") & g["args"].map(
            lambda a: bool(_EXEC_RE.search((a or {}).get("command") or "")))
        working_exec = g[is_exec & dead.isna()]
        last_edit = edits["step"].max() if len(edits) else None
        checked = last_edit is not None and bool((working_exec["step"] > last_edit).any())

        passed = bool(r["passed"])
        if passed:
            failure = None
        elif not len(edits):
            failure = FAILURE_TYPES[0]
        elif len(hit) < len(task_targets):
            failure = FAILURE_TYPES[1]
        elif not checked:
            failure = FAILURE_TYPES[2]
        else:
            failure = FAILURE_TYPES[3]

        first_target = target_edits["step"].min() if len(target_edits) else np.nan
        rows.append({
            key: task, "condition": condition, "trial": trial, "passed": passed,
            "steps_used": r["steps_used"], "hit_step_cap": r["steps_used"] >= cap,
            "tool_calls": len(g), "edited": bool(len(edits)),
            "edited_target": bool(len(target_edits)), "all_targets_edited": len(hit) == len(task_targets),
            "first_edit_step": edits["step"].min() if len(edits) else np.nan,
            "first_target_edit_step": first_target,
            "first_target_edit_capped": cap if np.isnan(first_target) else first_target,
            "target_edits_via_bash": int((is_bash & on_target).sum()),
            "off_target_edits": int(((g["tool"].isin(_CHANGE_TOOLS) | is_bash) & is_edit & ~on_target).sum()),
            "scratch_writes": int((g["tool"].isin(_CREATE_TOOLS) & ~on_target).sum()),
            "exec_calls": int(is_exec.sum()), "checked_after_last_edit": checked,
            "dead_end_calls": int(dead.notna().sum()),
            "dead_command_not_found": int((dead == "command not found").sum()),
            "dead_no_tests_collected": int((dead == "no tests collected").sum()),
            "dead_missing_path": int((dead == "missing path").sum()),
            "failure_type": failure,
        })
    return pd.DataFrame(rows)


def behaviour_summary(behaviour: pd.DataFrame) -> pd.DataFrame:
    """Per-condition rates and means over run_behaviour's rows (rates over
    runs, not trial-averaged per task)."""
    g = behaviour.groupby("condition")
    out = pd.DataFrame({
        "pass_rate": g["passed"].mean(),
        "edited_target": g["edited_target"].mean(),
        "median_first_target_edit": g["first_target_edit_step"].median(),
        "runs_with_off_target_edit": g["off_target_edits"].apply(lambda s: (s > 0).mean()),
        "runs_with_scratch_write": g["scratch_writes"].apply(lambda s: (s > 0).mean()),
        "exec_calls": g["exec_calls"].mean(),
        "checked_after_last_edit": g["checked_after_last_edit"].mean(),
        "dead_end_calls": g["dead_end_calls"].mean(),
        "dead_end_share_of_calls": g["dead_end_calls"].sum() / g["tool_calls"].sum(),
    })
    return order_conditions(out.reset_index())


def failure_type_table(behaviour: pd.DataFrame) -> pd.DataFrame:
    """Failed runs per condition, split into FAILURE_TYPES (counts)."""
    fails = behaviour[~behaviour["passed"]]
    table = (fails.groupby(["condition", "failure_type"]).size().unstack(fill_value=0)
                  .reindex(columns=FAILURE_TYPES, fill_value=0))
    table.insert(0, "failed runs", table.sum(axis=1))
    table = table.reindex(behaviour["condition"].unique(), fill_value=0)
    return order_conditions(table.reset_index())


def paired_run_metric_test(behaviour: pd.DataFrame, column: str, reference: str = "full",
                           n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """paired_condition_test for any run_behaviour column: runs averaged per
    task first, then every condition against `reference`, paired by task
    (mean difference, task-bootstrap CI, exact sign-flip p,
    higher/same/lower counts)."""
    key = run_key(behaviour)
    matrix = behaviour.groupby([key, "condition"])[column].mean().astype(float).unstack()
    idx = _bootstrap_indices(len(matrix), n_boot, seed)
    alpha = (1 - level) / 2
    rows = []
    for condition in matrix.columns:
        if condition == reference:
            continue
        diffs = (matrix[condition] - matrix[reference]).to_numpy()
        boots = np.nanmean(diffs[idx], axis=1)
        rows.append({
            "metric": column, "condition": condition, "reference": reference,
            "delta": np.nanmean(diffs), "ci_low": np.quantile(boots, alpha), "ci_high": np.quantile(boots, 1 - alpha),
            "p_value": _sign_flip_p_value(diffs, seed=seed),
            "higher/same/lower": f"{(diffs > 0).sum()}/{(diffs == 0).sum()}/{(diffs < 0).sum()}",
        })
    return order_conditions(pd.DataFrame(rows))


def _pass_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Task x condition pass rate (trials averaged), like pass_rate_matrix
    but keyed by run_key, so it accepts either project's log."""
    key = run_key(df)
    m = results_only(df).groupby([key, "condition"])["passed"].mean().astype(float).unstack()
    return m[[c for c in CONDITION_ORDER if c in m.columns] + [c for c in m.columns if c not in CONDITION_ORDER]]


def headroom_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per task/instance: no_skill and full pass rate, the five skill
    conditions pooled, the headroom no_skill leaves (1 - no_skill) and the
    pooled lift. Sorted hardest first."""
    m = _pass_matrix(df)
    skill = [c for c in m.columns if c != "no_skill"]
    out = pd.DataFrame({"no_skill": m["no_skill"], "full": m["full"], "skill_pooled": m[skill].mean(axis=1)})
    out["headroom"] = 1 - out["no_skill"]
    out["lift_pooled"] = out["skill_pooled"] - out["no_skill"]
    return out.sort_values("no_skill")


def headroom_gain(df: pd.DataFrame, condition: str = None, tasks=None,
                  n_boot: int = 10_000, seed: int = 0) -> dict:
    """Normalised gain computed only over the tasks where no_skill leaves
    headroom (summed lift / summed headroom), for `condition` or the five
    skill conditions pooled, with a bootstrap over those tasks. `tasks`
    restricts it further (e.g. to the tasks where two models both have
    headroom). Answers "is a small whole-dataset gain just a ceiling?": if
    the gain over the tasks that *can* move is still small, the ceiling is
    not the explanation."""
    m = _pass_matrix(df)
    if tasks is not None:
        m = m.loc[[t for t in tasks if t in m.index]]
    m = m[m["no_skill"] < 1]
    if m.empty:
        return {"tasks_with_headroom": 0, "gain": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    skill = m[condition] if condition else m[[c for c in m.columns if c != "no_skill"]].mean(axis=1)
    lift, head = (skill - m["no_skill"]).to_numpy(), (1 - m["no_skill"]).to_numpy()
    idx = _bootstrap_indices(len(m), n_boot, seed)
    boots = lift[idx].sum(axis=1) / head[idx].sum(axis=1)
    return {"tasks_with_headroom": len(m), "gain": lift.sum() / head.sum(),
            "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975)}


_SECTIONS = {"minus_procedure": "procedure", "minus_examples": "examples",
             "minus_validation": "validation", "minus_recovery_advice": "recovery_advice"}


def skill_section_sizes(task_variants_dir) -> pd.DataFrame:
    """Per task: characters in the full skill and in each section (full
    minus that section's leave-one-out variant), from the generated variant
    files. A rough token count is chars / 4."""
    rows = []
    for task_dir in sorted(p for p in Path(task_variants_dir).iterdir() if (p / "full.md").exists()):
        full = len((task_dir / "full.md").read_text())
        row = {"task": task_dir.name, "full_chars": full}
        for variant, section in _SECTIONS.items():
            path = task_dir / f"{variant}.md"
            row[section] = full - len(path.read_text()) if path.exists() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("task")


def _diff_stats(patch: str) -> dict:
    files = _re.findall(r"^diff --git a/(\S+) b/", patch, _re.M)
    new_files = set(_re.findall(r"^diff --git a/\S+ b/(\S+)\n(?:new file mode)", patch, _re.M))
    lines = patch.splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return {"files": files, "new_files": new_files, "lines_changed": added + removed}


def patch_footprint(predictions_path, subset_path, df: pd.DataFrame) -> pd.DataFrame:
    """
    SWE-bench Lite only. One row per submitted (non-empty) patch, comparing it
    with the gold patch: files touched, lines changed (added + removed), the
    size relative to the gold patch, and files outside the gold patch --
    split into new files (usually a reproduction script left in the diff)
    and other existing files the gold fix does not touch. Joined to the
    run's resolved flag. Read-only over predictions.json and the subset file.
    """
    gold = {x["instance_id"]: _diff_stats(x["patch"]) for x in json.loads(Path(subset_path).read_text())}
    resolved = results_only(df).set_index(["instance_id", "condition", "trial"])["passed"]
    rows = []
    for p in json.loads(Path(predictions_path).read_text()):
        if not p.get("model_patch"):
            continue
        condition, trial = p["model_name_or_path"].rsplit("__trial", 1)
        trial = int(trial)
        s, g = _diff_stats(p["model_patch"]), gold[p["instance_id"]]
        extra = set(s["files"]) - set(g["files"])
        rows.append({
            "instance_id": p["instance_id"], "condition": condition, "trial": trial,
            "resolved": bool(resolved.get((p["instance_id"], condition, trial), False)),
            "files_touched": len(s["files"]), "lines_changed": s["lines_changed"],
            "size_vs_gold": s["lines_changed"] / max(g["lines_changed"], 1),
            "touches_gold_file": bool(set(s["files"]) & set(g["files"])),
            "extra_new_files": len(extra & s["new_files"]),
            "extra_existing_files": len(extra - s["new_files"]),
        })
    return pd.DataFrame(rows)
