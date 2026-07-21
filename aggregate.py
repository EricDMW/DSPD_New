"""Cross-seed aggregation onto a common env-step grid (brief section 11.5).

Reads only ``results/runs/.../metrics.csv`` (never mutates raw runs, brief section 11.7),
resamples each run onto a shared env-step grid by linear interpolation (ragged lengths,
brief section 14.27), and writes:
  * ``data_csv/all_metrics_raw.csv``  -- every raw, UNSMOOTHED row concatenated (the record)
  * ``aggregated/summary.parquet`` + ``.csv`` -- mean/std/sem/ci95 across seeds
  * ``aggregated/final_table.parquet`` + ``.csv`` -- final-policy metrics per method
  * ``aggregated/curves_<figure>.parquet`` -- exact arrays behind each figure
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCALAR_METRICS = ["objective_F", "objective_F_undisc", "constraint_G", "constraint_G_min",
                  "constraint_G_max", "constraint_G_std_agents", "constraint_slack",
                  "feasible_frac", "violation_total", "violation_mean", "violation_max",
                  "dual_mu", "lagrangian_L", "transmit_freq", "E_theta", "E_mu",
                  "E_theta_sq", "pushsum_invariant_err", "pushsum_disagreement", "policy_loss"]


def load_all_raw(paths) -> pd.DataFrame:
    frames = []
    for csv in sorted(paths.runs.rglob("metrics.csv")):
        try:
            frames.append(pd.read_csv(csv))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _common_grid(df_scalar: pd.DataFrame, n: int = 120) -> np.ndarray:
    """Shared env-step grid up to the min final step across runs (brief section 11.5)."""
    finals = df_scalar.groupby(["algo", "config", "seed"])["env_step"].max()
    hi = int(finals.min()) if len(finals) else 0
    return np.linspace(0, hi, n)


def _config_grids(df_scalar: pd.DataFrame, n: int = 120) -> dict:
    """PER-CONFIG env-step grid, each spanning that config's own full training range (min final
    step across its seeds). Configs use different batch sizes -> different env-step totals; a single
    global grid would truncate the longer configs to the shortest one's range. Each figure reads a
    single config, so a per-config grid lets every figure show its full 0..M iterations."""
    finals = df_scalar.groupby(["algo", "config", "seed"])["env_step"].max()
    grids = {}
    for config in df_scalar["config"].unique():
        sub = finals[finals.index.get_level_values("config") == config]
        hi = int(sub.min()) if len(sub) else 0
        grids[config] = np.linspace(0, hi, n)
    return grids


def aggregate(paths, ci_z: float = 1.96):
    raw = load_all_raw(paths)
    if raw.empty:
        print("[aggregate] no metrics found")
        return None
    # record: all raw rows, unsmoothed
    (paths.data_csv).mkdir(parents=True, exist_ok=True)
    raw.to_csv(paths.data_csv / "all_metrics_raw.csv", index=False)

    scal = raw[(raw["scope"].isin(["mean", "agg"])) & (raw["metric"].isin(SCALAR_METRICS))].copy()
    grid = _common_grid(scal)             # global grid (kept for the summary tables)
    cfg_grids = _config_grids(scal)       # per-config grid so each figure spans its own full range

    # interpolate each (algo,config,seed,metric) series onto its config's grid
    rows = []
    keys = ["algo", "config", "seed", "metric"]
    for (algo, config, seed, metric), g in scal.groupby(keys):
        g = g.sort_values("env_step")
        if g["env_step"].nunique() < 2:
            continue
        grid = cfg_grids.get(config, grid)
        yi = np.interp(grid, g["env_step"].values, g["value"].values)
        for x, y in zip(grid, yi):
            rows.append(dict(algo=algo, config=config, seed=seed, metric=metric,
                             env_step=x, value=y))
    resampled = pd.DataFrame(rows)

    # aggregate across seeds
    agg = (resampled.groupby(["algo", "config", "metric", "env_step"])["value"]
           .agg(["count", "mean", "std"]).reset_index()
           .rename(columns={"count": "n_seeds"}))
    agg["std"] = agg["std"].fillna(0.0)
    agg["sem"] = agg["std"] / np.sqrt(agg["n_seeds"].clip(lower=1))
    agg["ci95_low"] = agg["mean"] - ci_z * agg["sem"]
    agg["ci95_high"] = agg["mean"] + ci_z * agg["sem"]
    agg["scope"] = "mean"
    agg.to_parquet(paths.aggregated / "summary.parquet", index=False)
    agg.to_csv(paths.aggregated / "summary.csv", index=False)

    _final_table(scal, paths)
    _curves(agg, paths)
    _heatmap_data(raw, paths)
    print(f"[aggregate] summary rows={len(agg)}  grid=[0,{grid[-1]:.0f}] x{len(grid)}")
    return agg


_FINAL_METRICS = ["objective_F", "objective_F_undisc", "violation_total", "constraint_G",
                  "feasible_frac", "transmit_freq"]


def _final_table(scal: pd.DataFrame, paths):
    """Final-policy metrics per (algo,config), mean +/- std over seeds (execution/deployment)."""
    finals = []
    for (algo, config, seed), g in scal.groupby(["algo", "config", "seed"]):
        last = g["env_step"].max()
        gl = g[g["env_step"] == last]
        row = {"algo": algo, "config": config, "seed": seed}
        for metric in _FINAL_METRICS:
            v = gl[gl["metric"] == metric]["value"]
            row[metric] = float(v.iloc[0]) if len(v) else np.nan
        finals.append(row)
    fdf = pd.DataFrame(finals)
    aggs = {"n_seeds": ("seed", "count")}
    for m in _FINAL_METRICS:
        aggs[f"{m}_mean"] = (m, "mean")
        aggs[f"{m}_std"] = (m, "std")
    tbl = fdf.groupby(["algo", "config"]).agg(**aggs).reset_index().fillna(0.0)
    # attach wall-clock and env-steps from run_meta.json (exact, per run)
    tbl["wall_clock_s"] = 0.0
    tbl["env_steps"] = 0
    for i, r in tbl.iterrows():
        walls, steps = [], []
        for md in paths.runs.glob(f"{r['algo']}/{r['config']}/seed*/run_meta.json"):
            import json
            m = json.load(open(md))
            walls.append(m.get("wall_clock_s", 0.0))
            steps.append(m.get("env_steps", 0))
        if walls:
            tbl.at[i, "wall_clock_s"] = float(np.mean(walls))
            tbl.at[i, "env_steps"] = int(np.mean(steps))
    tbl.to_parquet(paths.aggregated / "final_table.parquet", index=False)
    tbl.to_csv(paths.aggregated / "final_table.csv", index=False)
    (paths.paper).mkdir(parents=True, exist_ok=True)
    tbl.to_csv(paths.paper / "final_table.csv", index=False)
    return tbl


def _heatmap_data(raw: pd.DataFrame, paths):
    """Per-agent final transmission frequency (5x5 grid), main config, mean over seeds."""
    ag = raw[(raw["scope"] == "agent") & (raw["metric"] == "transmit_freq")
             & (raw["config"] == "main")].copy()
    if ag.empty:
        return
    rows = []
    for (algo, seed), g in ag.groupby(["algo", "seed"]):
        last = g["env_step"].max()
        gl = g[g["env_step"] == last]
        for _, r in gl.iterrows():
            rows.append(dict(algo=algo, agent=int(r["agent"]), value=float(r["value"])))
    hd = pd.DataFrame(rows).groupby(["algo", "agent"])["value"].mean().reset_index()
    hd.to_parquet(paths.aggregated / "curves_heatmap.parquet", index=False)


def _curves(agg: pd.DataFrame, paths):
    """Persist the exact mean/band arrays each figure plots (brief section 11.5/12)."""
    def dump(name, sub):
        if len(sub):
            sub.to_parquet(paths.aggregated / f"curves_{name}.parquet", index=False)

    OV = ["objective_F", "violation_total"]
    main = agg[agg["config"] == "main"]
    # main comparison (Ying et al. Fig. 1 style): objective F + constraint violation, plus the
    # constraint return G vs c_i for the binding-constraint / feasibility detail figure.
    dump("compare", main[main["metric"].isin(
        ["objective_F", "constraint_G", "constraint_G_min", "constraint_G_max",
         "constraint_G_std_agents", "violation_total", "feasible_frac"])])
    # "safe-RL arc" (all methods): constraint return G(theta) starts infeasible and rises across
    # c_i (config 'arc'; separate over-budget operating point from the objective-first main config).
    arc = agg[agg["config"] == "arc"]
    dump("constraint_arc", arc[arc["metric"].isin(
        ["constraint_G", "constraint_G_min", "constraint_G_max", "violation_total"])])
    # DSPD learning-rate ablation: objective return across the policy learning rate eta_theta.
    # Two operating points, one per figure (each mirrors its main-figure regime, both INCREASING):
    #   objective F sweep  -> objective-first/main regime (F rises like results_return),
    #   constraint G sweep -> over-budget arc regime      (G rises across c_i like results_constraint).
    lr_cfgs = ["lr0005", "lr002", "lr005", "lr015", "lr030"]
    lrab = agg[(agg["algo"] == "dspd") & (agg["config"].isin(lr_cfgs))]
    dump("lr_ablation", lrab[lrab["metric"].isin(["objective_F", "constraint_G", "violation_total"])])
    lrc_cfgs = ["lrc0005", "lrc002", "lrc005", "lrc015", "lrc030"]
    lrcab = agg[(agg["algo"] == "dspd") & (agg["config"].isin(lrc_cfgs))]
    dump("lr_ablation_constraint",
         lrcab[lrcab["metric"].isin(["objective_F", "constraint_G", "violation_total"])])
    # estimation uses the dedicated theoretical-regime config if present, else falls back to main
    est_cfg = "estimation" if (agg["config"] == "estimation").any() else "main"
    dump("estimation", agg[(agg["algo"] == "dspd") & (agg["config"] == est_cfg)
                           & (agg["metric"].isin(["E_theta", "E_mu"]))])
    # ablations (DSPD): each sweep -> objective + violation (both indices, brief section 12)
    dspd = agg[agg["algo"] == "dspd"]
    dump("ablation_kappa", dspd[dspd["config"].isin(["main", "ablation_k2", "ablation_k3"])
                                & dspd["metric"].isin(OV)])
    dump("ablation_beta", dspd[dspd["config"].isin(["beta0.3", "main", "beta0.7"])
                               & dspd["metric"].isin(OV)])
    dump("ablation_eta", dspd[dspd["config"].isin(["eta1", "main", "eta100"])
                              & dspd["metric"].isin(OV)])
    # training-dynamics diagnostics (main config): dual, feasibility, violation, Lagrangian
    dump("training", main[main["metric"].isin(
        ["dual_mu", "feasible_frac", "violation_total", "lagrangian_L"])])
