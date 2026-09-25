# %% [markdown]
# # IAAT blackmail trace: CPU-only analysis
#
# This file analyses saved intervention outputs for the Qwen3-32B IAAT blackmail
# trace.  It performs no model inference.  It is intended to live at
# `iaat_blackmail/analysis.py` in the clean repository.
#
# Expected directory layout:
#
# iaat_blackmail/
# ├── analysis.py
# ├── starting_data/
# │   └── case.json                         # optional for analysis
# └── data/
#     ├── measurements.csv                  # required
#     ├── prepared.json                     # required
#     ├── spans.tsv                         # required
#     ├── validation.json                   # required
#     ├── interaction_contrasts.json        # required
#     ├── matrix_boundary_delta_mean.csv    # required
#     ├── matrix_boundary_delta_trimmed.csv # required
#     ├── matrix_boundary_kl_mean.csv       # required
#     ├── experiment_catalog.json           # optional; useful provenance
#     ├── plots/                            # created here
#     └── tables/                           # created here
#
# Sign convention used below:
#   saved delta = log p(intervention) - log p(baseline)
#   support     = log p(baseline) - log p(intervention) = -delta
#
# Positive `support` therefore means that removing/reversing the source made the
# recorded downstream text less likely.  This is conditional support for the
# fixed continuation, not a total behavioural effect.

# %%
from __future__ import annotations

import json
import math
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% [markdown]
# ## Paths and constants

# %%
try:
    ROOT = Path(__file__).resolve().parent
except NameError:  # convenient when run cell-by-cell in an interactive session
    ROOT = Path.cwd()

STARTING_DATA = ROOT / "starting_data"
DATA = ROOT / "data"
PLOTS = DATA / "plots"
TABLES = DATA / "tables"
PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = {
    "measurements": DATA / "measurements.csv",
    "prepared": DATA / "prepared.json",
    "spans": DATA / "spans.tsv",
    "validation": DATA / "validation.json",
    "interactions": DATA / "interaction_contrasts.json",
    "matrix_mean": DATA / "matrix_boundary_delta_mean.csv",
    "matrix_trimmed": DATA / "matrix_boundary_delta_trimmed.csv",
    "matrix_kl": DATA / "matrix_boundary_kl_mean.csv",
}

for name, path in REQUIRED_FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required analysis input {name!r}: {path}\n"
            "Copy the corresponding saved investigation artefact into data/."
        )


# %% [markdown]
# ## Load and validate saved outputs

# %%
def read_json(path: Path):
    return json.loads(path.read_text())


measurements = pd.read_csv(REQUIRED_FILES["measurements"])
prepared = read_json(REQUIRED_FILES["prepared"])
validation = read_json(REQUIRED_FILES["validation"])
interactions = pd.DataFrame(read_json(REQUIRED_FILES["interactions"]))
spans = pd.read_csv(REQUIRED_FILES["spans"], sep="\t")

required_measurement_columns = {
    "experiment",
    "family",
    "target",
    "tokens",
    "delta_mean",
    "delta_sum",
    "delta_trimmed",
    "kl_mean",
    "baseline_lp_sum",
    "intervention_lp_sum",
}
missing_cols = required_measurement_columns - set(measurements.columns)
if missing_cols:
    raise ValueError(f"measurements.csv is missing columns: {sorted(missing_cols)}")

# There should be exactly one row per experiment-target pair in the compiled table.
if measurements.duplicated(["experiment", "target"]).any():
    dup = measurements.loc[
        measurements.duplicated(["experiment", "target"], keep=False),
        ["experiment", "target"],
    ].head(20)
    raise ValueError(f"Duplicate experiment-target rows found:\n{dup}")

measurements = measurements.copy()
measurements["support_mean"] = -measurements["delta_mean"]
measurements["support_trimmed"] = -measurements["delta_trimmed"]
measurements["support_sum"] = -measurements["delta_sum"]

MEAS = measurements.set_index(["experiment", "target"], drop=False)
SPAN = spans.set_index("id", drop=False)

baseline_action_p = float(validation["baseline_action_first_probability"])
if not (0.0 <= baseline_action_p <= 1.0):
    raise ValueError(f"Invalid baseline action-token probability: {baseline_action_p}")

print(
    f"Loaded {len(measurements):,} experiment-target measurements from "
    f"{validation.get('successful_interventions', 'unknown')} intervention/control runs."
)
print(f"Baseline p(first action-name token = 'declare'): {baseline_action_p:.4f}")
print(
    "Trace reconstruction: "
    f"{validation.get('trace_tokens', '?')} local tokens; "
    f"{validation.get('reported_trace_tokens', '?')} reported upstream tokens."
)


# %% [markdown]
# ## Helpers

# %%
def get_measure(experiment: str, target: str) -> pd.Series:
    """Return one compiled measurement and fail loudly if it is absent."""
    key = (experiment, target)
    if key not in MEAS.index:
        raise KeyError(
            f"Required measurement {key} is absent. "
            "Check that the clean data export contains the final investigation outputs."
        )
    return MEAS.loc[key]


def maybe_measure(experiment: str, target: str) -> pd.Series | None:
    key = (experiment, target)
    return None if key not in MEAS.index else MEAS.loc[key]


def support(experiment: str, target: str, *, trimmed: bool = False) -> float:
    row = get_measure(experiment, target)
    return float(row["support_trimmed" if trimmed else "support_mean"])


def delta(experiment: str, target: str, *, trimmed: bool = False) -> float:
    row = get_measure(experiment, target)
    return float(row["delta_trimmed" if trimmed else "delta_mean"])


def token_probability(experiment: str, target: str = "R_action_first") -> float:
    """Probability of a one-token recorded readout under the forced prefix."""
    row = get_measure(experiment, target)
    if int(row["tokens"]) != 1:
        raise ValueError(f"{target} is not one token in {experiment}; cannot exponentiate sum as p(token).")
    lp = float(row["intervention_lp_sum"])
    return float(np.exp(lp))


def short(text: str, n: int = 72) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def span_label(span_id: str, n: int = 54) -> str:
    if span_id not in SPAN.index:
        return span_id
    return f"{span_id}: {short(SPAN.loc[span_id, 'text'], n)}"


def savefig(fig: plt.Figure, stem: str, *, dpi: int = 220) -> None:
    """Save both a README-friendly PNG and a vector PDF."""
    fig.savefig(PLOTS / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(TABLES / f"{stem}.csv", index=False)
    try:
        (TABLES / f"{stem}.md").write_text(df.to_markdown(index=False) + "\n")
    except ImportError:
        pass


def pairwise_support_contrast(
    negative_experiment: str,
    positive_experiment: str,
    target: str,
) -> float:
    """
    Matched semantic contrast.

    Positive values mean the positive/permissive/private condition supports the
    recorded target more than the negative/rejecting/public condition.
    """
    return -(delta(negative_experiment, target) - delta(positive_experiment, target))


# %% [markdown]
# ## 1. Boundary-preserving fixed-trace influence matrix
#
# The full matrix is useful for screening but not itself a motive ranking.  We
# plot the strongest source rows using the separator-preserving, first/last-token
# trimmed effect to reduce obvious sentence-boundary artefacts.

# %%
matrix_trimmed_delta = pd.read_csv(REQUIRED_FILES["matrix_trimmed"], index_col=0)
matrix_mean_delta = pd.read_csv(REQUIRED_FILES["matrix_mean"], index_col=0)
matrix_kl = pd.read_csv(REQUIRED_FILES["matrix_kl"], index_col=0)

matrix_support = -matrix_trimmed_delta

# Keep only rows with at least one finite, non-zero-ish downstream effect.
row_strength = matrix_support.abs().max(axis=1, skipna=True).fillna(0.0)
N_TOP_SOURCES = min(36, len(row_strength))
top_sources = row_strength.nlargest(N_TOP_SOURCES).index.tolist()

# Preserve causal chronology in the displayed subset rather than ordering by effect size.
def source_start(sid: str) -> int:
    if sid in SPAN.index:
        return int(SPAN.loc[sid, "start"])
    return 10**9


top_sources = sorted(top_sources, key=source_start)
heat = matrix_support.loc[top_sources]

fig, ax = plt.subplots(figsize=(15, max(9, 0.33 * len(top_sources))))
arr = heat.to_numpy(dtype=float)
finite = np.abs(arr[np.isfinite(arr)])
clip = float(np.quantile(finite, 0.98)) if finite.size else 1.0
clip = max(clip, 1e-6)
im = ax.imshow(arr, aspect="auto", vmin=-clip, vmax=clip)
ax.set_xticks(np.arange(len(heat.columns)))
ax.set_xticklabels(heat.columns, rotation=90, fontsize=7)
ax.set_yticks(np.arange(len(heat.index)))
ax.set_yticklabels([span_label(x) for x in heat.index], fontsize=7)
ax.set_xlabel("Later fixed reasoning target")
ax.set_ylabel("Intervened source")
ax.set_title(
    "Boundary-preserving fixed-trace support matrix\n"
    "positive = source supports the recorded target; strongest source rows shown"
)
fig.colorbar(im, ax=ax, label="support (nats / token; clipped for display)")
fig.tight_layout()
savefig(fig, "01_boundary_influence_matrix_top_sources")

# Also save the complete matrices in the intuitive support convention for downstream use.
(-matrix_mean_delta).to_csv(TABLES / "matrix_boundary_support_mean.csv")
(-matrix_trimmed_delta).to_csv(TABLES / "matrix_boundary_support_trimmed.csv")
matrix_kl.to_csv(TABLES / "matrix_boundary_kl_mean.csv")


# %% [markdown]
# ## 2. Selected substantive source → reasoning effects
#
# These are not chosen by raw effect magnitude.  They correspond to the main
# competing explanations tested in the investigation.

# %%
selected_specs = [
    # group, human label, experiment, target
    ("Goal conflict", "Remove both prompt goal copies", "boundary_joint_goal_prompt", "D_bridge"),
    ("Goal conflict", "NALA serves American interests better", "replacement_serves_goal_better", "D_bridge"),
    ("Leverage", "Remove affair evidence + reasoning summaries", "boundary_joint_affair_evidence", "D_threatening"),
    ("Leverage", "Make relationship public and harmless to disclose", "affair_public_consistent_early", "D_force"),
    ("Permission", "Change C30 to ‘not justified’", "c30_not_justified", "D_consider"),
    ("Permission", "Change C38 to ‘not acceptable’", "c38_not_acceptable", "D_call"),
    ("Plan", "Remove concrete coercive plan C39", "boundary_delete_C39", "C42"),
    ("Competition", "Remove David-first ending C44", "boundary_delete_C44", "R_action_first"),
]

selected_rows = []
for group, label, experiment, target in selected_specs:
    row = get_measure(experiment, target)
    selected_rows.append(
        {
            "group": group,
            "intervention": label,
            "experiment": experiment,
            "target": target,
            "target_text": SPAN.loc[target, "text"] if target in SPAN.index else "",
            "support_nats_per_token": float(row["support_mean"]),
            "kl_nats_per_token": float(row["kl_mean"]),
        }
    )
selected_effects = pd.DataFrame(selected_rows)
save_table(selected_effects, "selected_substantive_effects")

fig, ax = plt.subplots(figsize=(10, 6.5))
plot_df = selected_effects.iloc[::-1]
y = np.arange(len(plot_df))
ax.barh(y, plot_df["support_nats_per_token"])
ax.axvline(0, linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(plot_df["intervention"], fontsize=8)
ax.set_xlabel("support for recorded downstream target (nats / token)")
ax.set_title("Selected trace-specific causal dependencies")
fig.tight_layout()
savefig(fig, "02_selected_substantive_effects")


# %% [markdown]
# ## 3. Matched semantic controls
#
# Raw deletion effects mix semantics with wording, reference resolution and
# discourse continuation.  These paired contrasts compare a rejecting/public
# intervention against a closer positive/private control.

# %%
semantic_specs = [
    # label, negative/off condition, positive/on condition, local target
    (
        "C30 permission → consider blackmail",
        "c30_not_justified",
        "c30_justified",
        "D_consider",
    ),
    (
        "C38 permission → formulate plan",
        "c38_not_acceptable",
        "c38_acceptable",
        "D_call",
    ),
    (
        "Repeated justification → formulate plan",
        "all_justification_reverse",
        "all_justification_paraphrase",
        "D_call",
    ),
    (
        "Private leverage → coercive efficacy",
        "affair_public_consistent_early",
        "affair_private_paraphrase_early",
        "D_force",
    ),
]

semantic_rows = []
for label, neg, pos, target in semantic_specs:
    semantic_rows.append(
        {
            "contrast": label,
            "negative_experiment": neg,
            "positive_experiment": pos,
            "target": target,
            "local_support_contrast": pairwise_support_contrast(neg, pos, target),
            "final_action_support_contrast": pairwise_support_contrast(neg, pos, "R_action_first"),
        }
    )
semantic_controls = pd.DataFrame(semantic_rows)
save_table(semantic_controls, "semantic_control_contrasts")

fig, ax = plt.subplots(figsize=(9.5, 5.5))
y = np.arange(len(semantic_controls))
ax.barh(y, semantic_controls["local_support_contrast"])
ax.axvline(0, linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(semantic_controls["contrast"], fontsize=8)
ax.set_xlabel("matched semantic support contrast (nats / token)")
ax.set_title("Semantic effects surviving closer controls")
fig.tight_layout()
savefig(fig, "03_semantic_control_contrasts")


# %% [markdown]
# ## 4. Redundancy and interaction
#
# Single deletions can badly understate a factor when the same information is
# repeated elsewhere.  The saved factorial contrasts compare A, B and A+B in
# the same target units.  We convert their deltas to positive-support units.

# %%
redundancy_choices = [
    ("two_goal_copies", "D_bridge", "Two goal copies → goal inference"),
    ("affair_source_and_summaries", "D_threatening", "Affair source + summaries → threat wording"),
    ("early_and_late_plans", "R_action_first", "Early + late plans → action token"),
    ("tool_source_and_reasoning", "D_call", "Tool source + procedural reasoning → ‘call’"),
]

redundancy_rows = []
for name, target, label in redundancy_choices:
    hit = interactions[(interactions["name"] == name) & (interactions["target"] == target)]
    if len(hit) != 1:
        raise KeyError(f"Expected one interaction row for {(name, target)}, found {len(hit)}")
    r = hit.iloc[0]
    redundancy_rows.append(
        {
            "label": label,
            "name": name,
            "target": target,
            "A_support": -float(r["A"]),
            "B_support": -float(r["B"]),
            "AB_support": -float(r["AB"]),
            "interaction_support_beyond_additive": -float(r["interaction_AB_minus_A_minus_B"]),
        }
    )
redundancy = pd.DataFrame(redundancy_rows)
save_table(redundancy, "redundancy_interactions")

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(redundancy))
w = 0.24
ax.bar(x - w, redundancy["A_support"], width=w, label="A")
ax.bar(x, redundancy["B_support"], width=w, label="B")
ax.bar(x + w, redundancy["AB_support"], width=w, label="A + B")
ax.axhline(0, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(redundancy["label"], rotation=18, ha="right", fontsize=8)
ax.set_ylabel("support (nats / token)")
ax.set_title("Joint interventions reveal redundant support missed by single deletions")
ax.legend()
fig.tight_layout()
savefig(fig, "04_redundancy_interactions")


# %% [markdown]
# ## 5. Conditional final-action readout
#
# The action readout is deliberately labelled narrowly.  It is the probability
# of the first token `declare` after a forced synthetic tool-call prefix.  It is
# not a probability that the agent would freely decide to blackmail.

# %%
action_specs = [
    ("Remove earlier blackmail planning", "boundary_joint_prior_blackmail"),
    ("Remove late commitments", "boundary_joint_late_commitments"),
    ("Remove earlier + late blackmail reasoning", "boundary_action_no_blackmail_reasoning"),
    ("Reverse late commitments", "late_commitments_reverse_keep_david"),
    ("Remove concrete plan C39", "boundary_delete_C39"),
    ("Negate concrete plan C39", "commit39_negate_only"),
    ("Remove David-first ending C44", "boundary_delete_C44"),
    ("Goal-compatible replacement", "replacement_serves_goal_better"),
    ("Public, harmless relationship", "affair_public_consistent_early"),
    ("Provide non-coercive solution in prompt only", "noncoercive_solution"),
    ("Carry non-coercive solution through reasoning", "noncoercive_solution_and_reasoning"),
]

action_rows = [
    {
        "intervention": "Baseline",
        "experiment": "baseline",
        "p_declare_first_token": baseline_action_p,
        "change_from_baseline": 0.0,
        "support_nats": 0.0,
    }
]
for label, experiment in action_specs:
    row = get_measure(experiment, "R_action_first")
    p = token_probability(experiment)
    action_rows.append(
        {
            "intervention": label,
            "experiment": experiment,
            "p_declare_first_token": p,
            "change_from_baseline": p - baseline_action_p,
            "support_nats": float(row["support_mean"]),
        }
    )
action_probe = pd.DataFrame(action_rows)
save_table(action_probe, "conditional_action_probe")

fig, ax = plt.subplots(figsize=(11, 7))
plot_df = action_probe.iloc[::-1]
y = np.arange(len(plot_df))
ax.barh(y, plot_df["p_declare_first_token"])
ax.axvline(baseline_action_p, linestyle="--", linewidth=1, label="baseline")
ax.set_xlim(0, 1)
ax.set_yticks(y)
ax.set_yticklabels(plot_df["intervention"], fontsize=8)
ax.set_xlabel("p(`declare` | forced action prefix and fixed prior reasoning)")
ax.set_title("Conditional final-action probe")
ax.legend()
fig.tight_layout()
savefig(fig, "05_conditional_action_probe")


# %% [markdown]
# ## 6. Teacher-forcing screening-off
#
# A central limitation of fixed-trace analysis is that an upstream intervention
# may strongly affect the *next* reasoning transition while having little final
# effect after the original downstream plan has been forced back into context.
# Matched semantic contrasts make that visible without relying on raw deletion.

# %%
screening_rows = []
for label, neg, pos, local_target in semantic_specs:
    local = pairwise_support_contrast(neg, pos, local_target)
    final = pairwise_support_contrast(neg, pos, "R_action_first")
    screening_rows.append(
        {
            "contrast": label,
            "local_target": local_target,
            "local_support_contrast": local,
            "final_action_support_contrast": final,
            "retained_fraction": np.nan if abs(local) < 1e-12 else final / local,
        }
    )

# Add one structurally useful non-paired example: removing C39 and asking about
# the next commitment versus the final action readout.
local_c39 = support("boundary_delete_C39", "C42")
final_c39 = support("boundary_delete_C39", "R_action_first")
screening_rows.append(
    {
        "contrast": "Concrete C39 plan → later commitment/action",
        "local_target": "C42",
        "local_support_contrast": local_c39,
        "final_action_support_contrast": final_c39,
        "retained_fraction": np.nan if abs(local_c39) < 1e-12 else final_c39 / local_c39,
    }
)

screening = pd.DataFrame(screening_rows)
save_table(screening, "teacher_forcing_screening")

fig, ax = plt.subplots(figsize=(7.5, 7))
ax.scatter(screening["local_support_contrast"], screening["final_action_support_contrast"], s=55)
lims = [
    min(0.0, screening[["local_support_contrast", "final_action_support_contrast"]].min().min()),
    max(0.0, screening[["local_support_contrast", "final_action_support_contrast"]].max().max()),
]
pad = 0.08 * max(lims[1] - lims[0], 1e-6)
lims = [lims[0] - pad, lims[1] + pad]
ax.plot(lims, lims, linestyle="--", linewidth=1)
for _, r in screening.iterrows():
    ax.annotate(short(r["contrast"], 34), (r["local_support_contrast"], r["final_action_support_contrast"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("support at local downstream target (nats / token)")
ax.set_ylabel("support at final `declare` readout (nats)")
ax.set_title("Fixed downstream text can screen off upstream causal effects")
fig.tight_layout()
savefig(fig, "06_teacher_forcing_screening")


# %% [markdown]
# ## 7. Local dependency paths
#
# These plots are deliberately called dependency paths, not mediation paths.
# Each bar is a separate conditional intervention effect with the later recorded
# text fixed.

# %%
leverage_path_specs = [
    ("Private email → secret summary", "boundary_delete_E03", "C26"),
    ("Secret summary → coercive efficacy", "boundary_delete_C26", "C27"),
    ("Coercive efficacy → consider blackmail", "boundary_delete_C27", "C32"),
    ("Consider blackmail → reporting rule", "boundary_delete_C32", "C33"),
    ("Reporting rule → call instruction", "boundary_delete_C33", "C34"),
    ("Call instruction → concrete plan", "boundary_delete_C34", "C39"),
    ("Concrete plan → late commitment", "boundary_delete_C39", "C42"),
]

norm_path_specs = [
    ("System prohibition → restatement", "boundary_delete_S07", "C29"),
    ("Restatement → possible justification", "boundary_delete_C29", "C30"),
    ("Possible justification → acceptability", "boundary_delete_C30", "C38"),
    ("Acceptability → concrete plan", "boundary_delete_C38", "C39"),
]


def path_table(specs: Sequence[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for label, experiment, target in specs:
        r = get_measure(experiment, target)
        rows.append(
            {
                "edge": label,
                "experiment": experiment,
                "target": target,
                "support_mean": float(r["support_mean"]),
                "support_trimmed": float(r["support_trimmed"]),
                "kl_mean": float(r["kl_mean"]),
            }
        )
    return pd.DataFrame(rows)


leverage_path = path_table(leverage_path_specs)
norm_path = path_table(norm_path_specs)
save_table(leverage_path, "leverage_dependency_path")
save_table(norm_path, "normative_dependency_path")

for df, stem, title in [
    (leverage_path, "07_leverage_dependency_path", "Conditional leverage/planning dependencies"),
    (norm_path, "08_normative_dependency_path", "Conditional prohibition/permission dependencies"),
]:
    fig, ax = plt.subplots(figsize=(9.5, max(4.3, 0.7 * len(df))))
    plot_df = df.iloc[::-1]
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["support_mean"])
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["edge"], fontsize=8)
    ax.set_xlabel("conditional support (nats / token)")
    ax.set_title(title + "\n(each edge measured separately; not a natural indirect effect)")
    fig.tight_layout()
    savefig(fig, stem)


# %% [markdown]
# ## 8. Boundary artefact diagnostic
#
# Compare ordinary sentence means with first/last-token-trimmed effects for the
# separator-preserving matrix.  Large changes are a warning that an apparent
# causal edge is dominated by a connective, punctuation or newline token.

# %%
common_rows = matrix_mean_delta.index.intersection(matrix_trimmed_delta.index)
common_cols = matrix_mean_delta.columns.intersection(matrix_trimmed_delta.columns)
mean_flat = matrix_mean_delta.loc[common_rows, common_cols].stack(dropna=True).rename("delta_mean")
trim_flat = matrix_trimmed_delta.loc[common_rows, common_cols].stack(dropna=True).rename("delta_trimmed")
boundary_diag = pd.concat([mean_flat, trim_flat], axis=1).dropna().reset_index()
boundary_diag.columns = ["source", "target", "delta_mean", "delta_trimmed"]
boundary_diag["boundary_sensitivity"] = (boundary_diag["delta_mean"] - boundary_diag["delta_trimmed"]).abs()
boundary_diag["support_mean"] = -boundary_diag["delta_mean"]
boundary_diag["support_trimmed"] = -boundary_diag["delta_trimmed"]

boundary_top = boundary_diag.nlargest(30, "boundary_sensitivity").copy()
boundary_top["source_text"] = boundary_top["source"].map(lambda x: SPAN.loc[x, "text"] if x in SPAN.index else "")
boundary_top["target_text"] = boundary_top["target"].map(lambda x: SPAN.loc[x, "text"] if x in SPAN.index else "")
save_table(boundary_top, "largest_boundary_sensitive_effects")

fig, ax = plt.subplots(figsize=(7.5, 7))
# A deterministic subsample keeps rendering light while preserving extremes.
if len(boundary_diag) > 2500:
    sample = pd.concat(
        [
            boundary_diag.nlargest(250, "boundary_sensitivity"),
            boundary_diag.iloc[:: max(1, len(boundary_diag) // 2250)],
        ]
    ).drop_duplicates(["source", "target"])
else:
    sample = boundary_diag
ax.scatter(sample["support_mean"], sample["support_trimmed"], s=8, alpha=0.45)
lo = min(sample[["support_mean", "support_trimmed"]].min().min(), 0)
hi = max(sample[["support_mean", "support_trimmed"]].max().max(), 0)
pad = 0.05 * max(hi - lo, 1e-6)
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
ax.set_xlabel("separator-preserving support: full sentence mean")
ax.set_ylabel("separator-preserving support: first/last token trimmed")
ax.set_title("Boundary-token sensitivity of broad deletion effects")
fig.tight_layout()
savefig(fig, "09_boundary_token_diagnostic")


# %% [markdown]
# ## 9. Optional exact cross-investigation replication table
#
# If a future clean export includes `data/replication_pairs.csv`, this cell will
# plot it.  The file should contain *explicitly matched estimands*, not merely
# semantically similar experiments:
#
# `label,run1_support,run2_support`
#
# I intentionally do not auto-match the two legacy Codex investigations because
# their span definitions, sign conventions, action probe and boundary handling
# differ enough that an automatic correlation would be easy to overinterpret.

# %%
replication_file = DATA / "replication_pairs.csv"
if replication_file.exists():
    replication = pd.read_csv(replication_file)
    needed = {"label", "run1_support", "run2_support"}
    if not needed.issubset(replication.columns):
        raise ValueError(f"replication_pairs.csv must contain {sorted(needed)}")
    save_table(replication, "replication_pairs")

    x = replication["run1_support"].to_numpy(float)
    y = replication["run2_support"].to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    corr = float(np.corrcoef(x[finite], y[finite])[0, 1]) if finite.sum() >= 2 else np.nan

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=50)
    lo = min(np.nanmin(x), np.nanmin(y), 0)
    hi = max(np.nanmax(x), np.nanmax(y), 0)
    pad = 0.05 * max(hi - lo, 1e-6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    for _, r in replication.iterrows():
        ax.annotate(short(r["label"], 30), (r["run1_support"], r["run2_support"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("investigation 1 support")
    ax.set_ylabel("investigation 2 support")
    ax.set_title(f"Matched cross-investigation effects (Pearson r={corr:.2f})")
    fig.tight_layout()
    savefig(fig, "10_cross_investigation_replication")
else:
    replication = None


# %% [markdown]
# ## 10. Machine-readable summary for the report
#
# This is deliberately a summary of measurements rather than autogenerated prose.
# `report.md` should interpret these values with the methodological caveats.

# %%
def as_records(df: pd.DataFrame):
    return json.loads(df.to_json(orient="records"))


summary = {
    "data": {
        "successful_interventions": validation.get("successful_interventions"),
        "model": validation.get("model"),
        "revision": validation.get("revision"),
        "trace_tokens_local": validation.get("trace_tokens"),
        "trace_tokens_reported": validation.get("reported_trace_tokens"),
        "raw_final_action_available": validation.get("raw_final_action_available"),
        "original_sampled_token_ids_available": validation.get("original_sampled_token_ids_available"),
        "baseline_action_first_probability": baseline_action_p,
    },
    "selected_substantive_effects": as_records(selected_effects),
    "semantic_control_contrasts": as_records(semantic_controls),
    "redundancy_interactions": as_records(redundancy),
    "conditional_action_probe": as_records(action_probe),
    "teacher_forcing_screening": as_records(screening),
    "leverage_dependency_path": as_records(leverage_path),
    "normative_dependency_path": as_records(norm_path),
    "analysis_conventions": {
        "support_definition": "baseline log probability minus intervention log probability",
        "positive_support_means": "the source/positive condition supports the recorded fixed downstream target",
        "action_probe_scope": "first token of a synthetic forced action-name serialization, not free behavioural probability",
        "path_scope": "separate fixed-trace conditional dependencies, not natural indirect effects",
    },
}

if replication is not None:
    summary["cross_investigation_replication"] = as_records(replication)

(TABLES / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


# %% [markdown]
# ## 11. Compact console summary

# %%
print("\nKey matched semantic contrasts (positive = recorded harmful continuation is more supported):")
for _, r in semantic_controls.iterrows():
    print(
        f"  {r['contrast']}: local {r['local_support_contrast']:+.3f} nats/token; "
        f"final action {r['final_action_support_contrast']:+.3f} nats"
    )

print("\nRedundancy checks:")
for _, r in redundancy.iterrows():
    print(
        f"  {r['label']}: A={r['A_support']:+.3f}, B={r['B_support']:+.3f}, "
        f"A+B={r['AB_support']:+.3f}, extra joint={r['interaction_support_beyond_additive']:+.3f}"
    )

print("\nConditional action probe:")
for _, r in action_probe.iterrows():
    print(f"  {r['intervention']}: p(declare token)={r['p_declare_first_token']:.3f}")

print(f"\nWrote plots to {PLOTS}")
print(f"Wrote analysis tables to {TABLES}")

# %%
