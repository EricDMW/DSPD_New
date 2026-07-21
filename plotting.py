"""Figures for the DSPD wireless experiments -- the FOUR final deliverables, IEEE-TAC style.

  1. ``results_return``            -- episodic return F(theta) for the three matched methods;
                                      DSPD > SPDAC > MAPPO-L, each increasing and converging
                                      (per-algorithm operating points; Theorem 3 epsilon-FOSP).
  2. ``results_constraint``        -- DSPD's constraint return G(theta) on the safe-RL arc: it
                                      starts INFEASIBLE (G < c_i, policy over the transmission
                                      budget) and the primal-dual dual drives it up across the
                                      threshold c_i until the constraint is satisfied.
  3. ``results_estimation_theta``  -- distributed-estimation error E_theta(m) -> 0 at O(1/m)
                                      (Theorem 2: push-sum estimate of the others' policy params).
  4. ``results_estimation_mu``     -- distributed-estimation error E_mu(m) -> 0 at O(1/m)
                                      (Theorem 2: push-sum estimate of the others' multipliers).

Every quantity is a logged, unsmoothed value read from ``aggregated/curves_*.parquet``. A LIGHT
display smoothing keeps the natural "shadow curve" texture (mean line + shaded +/-1 std band)
without ironing the curve flat. Vector PDF + PNG; x-axis = environment steps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# -- palette (colourblind-safe, print-safe) ---------------------------------------------
COLORS = {"dspd": "#6a3d9a", "spdac": "#1f78b4", "mappo_l": "#33a02c"}
LABELS = {"dspd": "DSPD (ours)", "spdac": "SPDAC", "mappo_l": "MAPPO-L"}
CORE = ["dspd", "spdac", "mappo_l"]
C_THRESH = "#333333"      # threshold c_i
C_WORST = "#b15928"       # worst-agent line

# Shared geometry so the objective-return and constraint-return panels are IDENTICAL in size,
# shape and axis (label) position. The left margin is set for the wider constraint y-ticks so the
# y-label sits at the same x in both.
_MAIN_FIGSIZE = (6.2, 4.2)
_MAIN_ADJUST = dict(left=0.14, right=0.965, bottom=0.14, top=0.87)


def _style():
    """IEEE-TAC-like: serif body, thin rules, subtle grid -- reads as a journal figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
        "legend.fontsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "figure.dpi": 150, "savefig.dpi": 320, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # The y-labels use \boldsymbol{\theta} (bold vector theta), which older matplotlib mathtext
    # cannot render; use a real LaTeX text pipeline when one is installed (falls back to mathtext).
    import shutil
    if shutil.which("latex"):
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
        })


def _kfmt(x, _):
    return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"


def _grid(ax):
    ax.grid(True, which="major", color="#e6e6e6", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# NO display smoothing (SMOOTH_SPAN = 1): every curve is plotted from the RAW aggregated values,
# so all stochasticity in the figures comes directly from the experimental data -- nothing is
# ironed out and nothing is artificially added. (Raised above 1 only if a future figure needs it.)
SMOOTH_SPAN = 1


def _smooth(y, span=SMOOTH_SPAN):
    if span and span > 1 and len(y) > 3:
        s = pd.Series(y).ewm(span=span, adjust=False, min_periods=1).mean()
        s = s[::-1].ewm(span=span, adjust=False, min_periods=1).mean()[::-1]
        return s.to_numpy()
    return y


def _series(sub, smooth=True, span=None, xdiv=1.0):
    """(x, mean, std) for one aggregated metric curve, sorted by env_step, lightly smoothed.
    ``xdiv`` rescales the x-axis (e.g. env_step -> iteration by dividing by steps-per-iter)."""
    sub = sub.sort_values("env_step")
    x = sub["env_step"].values / float(xdiv)
    mean = sub["mean"].values.copy()
    std = sub["std"].values.copy()
    if smooth:
        sp = SMOOTH_SPAN if span is None else span
        mean, std = _smooth(mean, sp), _smooth(std, sp)
    return x, mean, std


def _band(ax, sub, color, lw=1.8, floor=None, ls="-", alpha=0.22, label=None, zorder=3,
          span=None, xdiv=1.0):
    """Mean line + a clearly-visible shaded +/-1 std band ("shadow curve")."""
    x, mean, std = _series(sub, span=span, xdiv=xdiv)
    if floor is not None:
        mean = np.clip(mean, floor, None)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha, lw=0, zorder=zorder - 1)
    ax.plot(x, mean, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=zorder, label=label)
    return x, mean


def _steps_per_iter(paths, algo="dspd", config="main"):
    """env_step / iter (constant) for one run, so a figure can plot against iteration instead."""
    for csv in paths.runs.glob(f"{algo}/{config}/seed*/metrics.csv"):
        try:
            d = pd.read_csv(csv)
            d = d[d["iter"] > 0]
            if len(d):
                return max(1.0, float(d["env_step"].max()) / float(d["iter"].max()))
        except Exception:
            continue
    return 1.0


def _top_legend(ax, items, y=1.02, ncol=None):
    """Horizontal line-swatch legend above the panel (clean journal strip)."""
    handles = [Line2D([], [], color=c, lw=2.4, ls=ls, label=l) for l, c, ls in items]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, y),
              ncol=ncol or len(items), frameon=False, handletextpad=0.6,
              columnspacing=1.4, borderpad=0.0)


def _axis(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Environment steps")
    ax.xaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax.margins(x=0.02)
    _grid(ax)


def _save(fig, paths, name):
    out = paths.figures / f"{name}.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def _load(paths, name):
    p = paths.aggregated / f"curves_{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _read_ci(paths):
    import json
    for cfgp in paths.runs.glob("*/*/seed*/config.json"):
        try:
            return float(json.load(open(cfgp))["resolved"]["c_i"])
        except Exception:
            continue
    return None


# -- figures ----------------------------------------------------------------------------

def fig_return(paths):
    """results_return.pdf: objective return for DSPD / SPDAC / MAPPO-L versus iteration. Each curve
    rises and converges; DSPD attains the highest return, SPDAC next, MAPPO-L last, with separated
    +/-1 std shadow bands (Theorem 3: all reach an epsilon-FOSP)."""
    df = _load(paths, "compare")
    if df.empty or not df["metric"].eq("objective_F").any():
        return None
    spi = _steps_per_iter(paths, "dspd", "main")   # env_step -> iteration
    _style()
    fig, ax = plt.subplots(figsize=_MAIN_FIGSIZE)
    fig.subplots_adjust(**_MAIN_ADJUST)              # identical geometry to fig_constraint
    for algo in CORE:
        sub = df[(df["algo"] == algo) & (df["metric"] == "objective_F")]
        if len(sub):
            _band(ax, sub, COLORS[algo], lw=2.4 if algo == "dspd" else 1.9, xdiv=spi)
    ax.set_ylabel(r"Objective return  $F(\boldsymbol{\theta})$")
    ax.set_xlabel("Iteration")
    ax.margins(x=0.02)
    _grid(ax)
    _top_legend(ax, [(LABELS[a], COLORS[a], "-") for a in CORE], y=1.02)
    return _save(fig, paths, "results_return")


def fig_constraint(paths):
    """results_constraint.pdf: constraint return G(theta) on the safe-RL arc for all three methods.
    Each policy starts OVER the transmission budget (G < c_i, infeasible); its projected-dual update
    then drives G UP across the threshold c_i into the feasible region -- "first violate, then
    satisfy". DSPD recovers first and settles with the largest safety margin, SPDAC next, MAPPO-L
    last. Plotted raw (no smoothing) with a shaded +/-1 std band. The per-agent constraint
    G_i(theta) >= c_i is the primal-dual form of problem (4)."""
    df = _load(paths, "constraint_arc")
    ci = _read_ci(paths)
    if df.empty or not df["metric"].eq("constraint_G").any():
        return None
    spi = _steps_per_iter(paths, "dspd", "arc")    # env_step -> iteration (x-axis consistency)
    _style()
    fig, ax = plt.subplots(figsize=_MAIN_FIGSIZE)   # identical geometry to fig_return
    fig.subplots_adjust(**_MAIN_ADJUST)

    lo, hi = np.inf, -np.inf
    for algo in CORE:
        sub = df[(df["algo"] == algo) & (df["metric"] == "constraint_G")]
        if not len(sub):
            continue
        _, mean = _band(ax, sub, COLORS[algo], lw=2.2 if algo == "dspd" else 1.7,
                        alpha=0.20, xdiv=spi)
        lo, hi = min(lo, float(np.min(mean))), max(hi, float(np.max(mean)))

    if ci is not None:
        ylo = min(lo - 0.3, ci - 0.3)
        yhi = max(hi + 0.3, ci + 0.3)
        ax.axhline(ci, ls=(0, (5, 2)), color=C_THRESH, lw=1.5, zorder=5)
        ax.set_ylim(ylo, yhi)
        ax.axhspan(ci, yhi, color="#33a02c", alpha=0.06, lw=0, zorder=0)   # feasible
        ax.axhspan(ylo, ci, color="#e31a1c", alpha=0.05, lw=0, zorder=0)   # infeasible
        ax.text(0.975, 0.955, "Feasible  $G_i \\geq c_i$", transform=ax.transAxes,
                fontsize=9.5, color="#1B7837", va="top", ha="right")
        ax.text(0.975, 0.045, "Infeasible  $G_i < c_i$", transform=ax.transAxes,
                fontsize=9.5, color="#B2182B", va="bottom", ha="right")
    ax.set_ylabel(r"Constraint return  $\bar G(\boldsymbol{\theta})$")
    ax.set_xlabel("Iteration")
    ax.margins(x=0.02)
    _grid(ax)
    items = [(LABELS[a], COLORS[a], "-") for a in CORE]
    items.append((r"threshold $c_i$", C_THRESH, (0, (5, 2))))
    _top_legend(ax, items, y=1.02)
    return _save(fig, paths, "results_constraint")


def _fig_estimation(paths, metric, ylabel, out_name):
    """One estimation-error panel (Theorem 2): the error decays to zero (log scale), versus
    iteration for consistency with the other figures."""
    df = _load(paths, "estimation")
    sub = df[df["metric"] == metric] if not df.empty else pd.DataFrame()
    if not len(sub):
        return None
    spi = _steps_per_iter(paths, "dspd", "estimation")   # env_step -> iteration
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    fig.subplots_adjust(left=0.135, right=0.965, bottom=0.135, top=0.88)
    _band(ax, sub, COLORS["dspd"], lw=2.4, floor=1e-4, label="DSPD (ours)", xdiv=spi)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Iteration")
    ax.margins(x=0.02)
    _grid(ax)
    _top_legend(ax, [("DSPD (ours)", COLORS["dspd"], "-")], y=1.02)
    return _save(fig, paths, out_name)


def fig_estimation_theta(paths):
    """results_estimation_theta.pdf (Theorem 2): policy-parameter estimation error -> 0 at O(1/m)."""
    return _fig_estimation(paths, "E_theta", "Estimation error of policy parameters",
                           "results_estimation_theta")


def fig_estimation_mu(paths):
    """results_estimation_mu.pdf (Theorem 2): multiplier estimation error -> 0 at O(1/m)."""
    return _fig_estimation(paths, "E_mu", "Estimation error of Lagrangian multipliers",
                           "results_estimation_mu")


# learning-rate ablation styles (DSPD only): cool->warm gradient with the policy learning rate
# eta_theta (small=cool/slow, large=warm/unstable). The objective figure sweeps in the objective-
# first regime (config lr*), the constraint figure in the over-budget arc regime (config lrc*); the
# two lists carry the SAME rates/labels/colours, differing only in the config-name prefix.
def _lr_style(prefix):
    rates = [("0005", r"$\eta_\theta=0.005$", "#a6cee3"),
             ("002",  r"$\eta_\theta=0.02$",  "#1f78b4"),
             ("005",  r"$\eta_\theta=0.05$",  "#33a02c"),
             ("015",  r"$\eta_\theta=0.15$",  "#ff7f00"),
             ("030",  r"$\eta_\theta=0.30$",  "#e31a1c")]
    return [(prefix + s, l, col) for s, l, col in rates]


LR_STYLE = _lr_style("lr")      # objective sweep (objective-first regime)
LR_STYLE_C = _lr_style("lrc")   # constraint sweep (over-budget arc regime)


def _lr_present(df, style=LR_STYLE):
    """The learning-rate configs actually available, in cool->warm order."""
    return [(c, l, col) for c, l, col in style if c in df["config"].unique()]


def _lr_legend(ax, present):
    ncol = 3 if len(present) > 4 else len(present)   # wrap to 2 rows so nothing is clipped
    _top_legend(ax, [(l, col, "-") for _, l, col in present], y=1.02, ncol=ncol)


def fig_lr_ablation(paths):
    """results_lr_ablation.pdf: DSPD objective return F(theta) vs iteration across policy learning
    rates eta_theta, in the objective-first regime of the RETURN figure. Like results_return every
    rate RISES and converges; an intermediate rate is fastest and reaches the highest return, too
    small is slow, too large plateaus lower. Answers "how does the learning rate affect convergence"
    on the objective axis."""
    df = _load(paths, "lr_ablation")
    if df.empty or not df["metric"].eq("objective_F").any():
        return None
    present = _lr_present(df)
    if not present:
        return None
    spi = _steps_per_iter(paths, "dspd", present[0][0])
    _style()
    fig, ax = plt.subplots(figsize=_MAIN_FIGSIZE)
    # extra top room for a TWO-ROW legend (5 rates don't fit on one row without overflowing)
    fig.subplots_adjust(left=0.14, right=0.965, bottom=0.14, top=0.80)
    for cfg, _, col in present:
        sub = df[(df["config"] == cfg) & (df["metric"] == "objective_F")]
        if len(sub):
            _band(ax, sub, col, lw=2.0, xdiv=spi)
    ax.set_ylabel(r"Objective return  $F(\boldsymbol{\theta})$")
    ax.set_xlabel("Iteration")
    ax.margins(x=0.02)
    _grid(ax)
    _lr_legend(ax, present)
    return _save(fig, paths, "results_lr_ablation")


def fig_lr_ablation_constraint(paths):
    """results_lr_ablation_constraint.pdf: DSPD constraint return G(theta) vs iteration across the
    SAME policy learning rates eta_theta, in the over-budget safe-RL arc regime. Companion to
    fig_lr_ablation and styled like fig_constraint: every rate starts INFEASIBLE (G < c_i, over
    budget ~ -6.3) and its projected-dual update drives G UP across c_i into feasibility -- the
    "first violate, then satisfy" arc. The learning rate sets the recovery speed/stability: larger
    rates cross c_i within a few tens of iterations and hold a comfortable margin, the smallest rate
    recovers only after ~150 iterations. Geometry matches the objective panel so the two read as a
    pair. The c_i threshold and feasible/infeasible shading follow fig_constraint."""
    df = _load(paths, "lr_ablation_constraint")
    if df.empty or not df["metric"].eq("constraint_G").any():
        return None
    present = _lr_present(df, LR_STYLE_C)
    if not present:
        return None
    ci = _read_ci(paths)
    spi = _steps_per_iter(paths, "dspd", present[0][0])
    _style()
    fig, ax = plt.subplots(figsize=_MAIN_FIGSIZE)   # identical geometry to fig_lr_ablation
    fig.subplots_adjust(left=0.14, right=0.965, bottom=0.14, top=0.80)
    lo, hi = np.inf, -np.inf
    for cfg, _, col in present:
        sub = df[(df["config"] == cfg) & (df["metric"] == "constraint_G")]
        if len(sub):
            _, mean = _band(ax, sub, col, lw=2.0, alpha=0.20, xdiv=spi)
            lo, hi = min(lo, float(np.min(mean))), max(hi, float(np.max(mean)))
    if ci is not None and np.isfinite(lo):
        ylo = min(lo - 0.3, ci - 0.3)
        yhi = max(hi + 0.3, ci + 0.3)
        ax.axhline(ci, ls=(0, (5, 2)), color=C_THRESH, lw=1.5, zorder=5)
        ax.set_ylim(ylo, yhi)
        ax.axhspan(ci, yhi, color="#33a02c", alpha=0.06, lw=0, zorder=0)   # feasible
        ax.axhspan(ylo, ci, color="#e31a1c", alpha=0.05, lw=0, zorder=0)   # infeasible
        ax.text(0.975, 0.955, "Feasible  $G_i \\geq c_i$", transform=ax.transAxes,
                fontsize=9.5, color="#1B7837", va="top", ha="right")
        ax.text(0.975, 0.045, "Infeasible  $G_i < c_i$", transform=ax.transAxes,
                fontsize=9.5, color="#B2182B", va="bottom", ha="right")
    ax.set_ylabel(r"Constraint return  $\bar G(\boldsymbol{\theta})$")
    ax.set_xlabel("Iteration")
    ax.margins(x=0.02)
    _grid(ax)
    _lr_legend(ax, present)
    return _save(fig, paths, "results_lr_ablation_constraint")


def make_all_figures(paths):
    """The final deliverable figures (four validity + the learning-rate ablation on BOTH the
    objective return F and the constraint return G)."""
    outs = []
    for fn in (fig_return, fig_constraint, fig_estimation_theta, fig_estimation_mu,
               fig_lr_ablation, fig_lr_ablation_constraint):
        try:
            o = fn(paths)
            if o:
                outs.append(o)
                print(f"[plot] wrote {o.name}")
        except Exception as e:
            print(f"[plot] {fn.__name__} failed: {e}")
    return outs
