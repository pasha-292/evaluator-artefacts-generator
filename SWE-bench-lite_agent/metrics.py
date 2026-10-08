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
