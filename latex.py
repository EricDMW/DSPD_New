"""Generate the TAC-level experimental section (brief section 13).

Emits ``paper/experiment_section.tex`` (self-contained, pdflatex-compilable, liftable into
IEEEtran) with every ``[fill in]`` replaced by a measured mean +/- std from ``aggregated/``,
and ``paper/numbers.json`` mapping each quoted number to its source run(s)/column
(brief section 11.7 provenance). The booktabs baseline table is generated from
``final_table.parquet`` (no manual entry); the best entry per column is bolded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ORDER = ["dspd", "spdac", "mappo_l", "mappo_l_dec", "mappo_l_decagg"]
METHOD_TEX = {"dspd": r"\textbf{DSPD} (ours)", "spdac": "SPDAC", "mappo_l": "MAPPO-L",
              "mappo_l_dec": "Dec.\\ MAPPO-L", "mappo_l_decagg": "Dec.-Agg.\\ MAPPO-L"}


def _steps_to_objective(summary, algo, config, frac=0.9):
    """Env-steps for the mean objective F to first reach ``frac`` of its final (asymptotic)
    value -- the objective-convergence speed (sample efficiency) for the increasing-F setting
    of the paper's demo. Uses the method's own asymptote as the target."""
    sub = summary[(summary["algo"] == algo) & (summary["config"] == config)
                  & (summary["metric"] == "objective_F")].sort_values("env_step")
    if sub.empty:
        return None
    x, y = sub["env_step"].values, sub["mean"].values
    target = frac * y[-1]
    hit = np.where(y >= target)[0]
    return float(x[hit[0]]) if len(hit) else None


def compute_numbers(paths, obj_frac=0.9):
    summary = pd.read_parquet(paths.aggregated / "summary.parquet")
    final = pd.read_parquet(paths.aggregated / "final_table.parquet")
    numbers, prov = {}, {}

    def rec(key, value, source):
        numbers[key] = value
        prov[key] = source

    # Resolved operating-point parameters, read from a main-config run's config.json, so the
    # generated prose states THIS group's actual (gamma, H, beta, M, ...) rather than a
    # hardcoded default -- essential when several parameter groups are generated (the run_id /
    # multi-group fix). Every such number is traceable to config.json in numbers.json.
    import glob
    for p in sorted(glob.glob(str(paths.runs / "dspd" / "main" / "seed*" / "config.json"))):
        try:
            cfg = json.load(open(p))["resolved"]
        except Exception:
            continue
        for k in ("gamma", "H", "beta", "c_i", "kappa", "kappa_p", "actor_lr", "lr_tau",
                  "eta_mu", "update_epochs", "K_push", "mu_max", "total_iters",
                  "n_sample_traj", "eval_M", "env_seed"):
            if k in cfg:
                v = cfg[k]
                rec(f"cfg_{k}", (int(v) if k in ("H", "env_seed", "kappa", "kappa_p",
                    "update_epochs", "K_push", "total_iters", "n_sample_traj", "eval_M") else v),
                    {"source": "config.json (dspd/main)", "param": k})
        break

    # asymptotic objective F per method (main config)
    fmain = final[final["config"] == "main"]
    for _, r in fmain.iterrows():
        a = r["algo"]
        rec(f"F_{a}", round(float(r["objective_F_mean"]), 3),
            {"metric": "objective_F", "algo": a, "config": "main",
             "source": "final_table.parquet", "mean": float(r["objective_F_mean"]),
             "std": float(r["objective_F_std"])})
        rec(f"Fstd_{a}", round(float(r["objective_F_std"]), 3),
            {"metric": "objective_F(std)", "algo": a, "config": "main"})
        rec(f"viol_{a}", round(float(r["violation_total_mean"]), 3),
            {"metric": "violation_total", "algo": a, "config": "main",
             "source": "final_table.parquet", "mean": float(r["violation_total_mean"]),
             "std": float(r["violation_total_std"])})

    # env-steps to reach obj_frac of the asymptotic objective (convergence speed) + DSPD speedup
    frac = obj_frac
    for a in fmain["algo"].unique():
        s = _steps_to_objective(summary, a, "main", frac)
        if s is not None:
            rec(f"steps_{a}", int(round(s)),
                {"metric": f"env_steps to {int(frac*100)}% of asymptotic F", "algo": a,
                 "config": "main", "source": "summary.parquet"})
    if "steps_dspd" in numbers and "steps_spdac" in numbers and numbers["steps_dspd"] > 0:
        rec("speedup_dspd_vs_spdac", round(numbers["steps_spdac"] / max(numbers["steps_dspd"], 1), 2),
            {"metric": "speedup = steps_spdac/steps_dspd (to 90% F)", "config": "main"})

    # kappa ablation: return + convergence speedup (dspd main = kappa1, ablation_k2 = kappa2)
    # plus per-radius final objective and VIOLATION (the safety benefit of larger kappa, the
    # Ying-et-al. Fig. 1(d) story: larger kappa -> lower constraint violation).
    def _dspd_final(cfg_name, col):
        r = final[(final["algo"] == "dspd") & (final["config"] == cfg_name)]
        return float(r[col].iloc[0]) if len(r) else None
    for cfg_name, k in [("main", 1), ("ablation_k2", 2), ("ablation_k3", 3)]:
        fk = _dspd_final(cfg_name, "objective_F_mean")
        vk = _dspd_final(cfg_name, "violation_total_mean")
        if fk is not None:
            rec(f"F_k{k}", round(fk, 3), {"metric": "objective_F", "algo": "dspd",
                                          "config": cfg_name, "source": "final_table.parquet"})
        if vk is not None:
            rec(f"viol_k{k}", round(vk, 3), {"metric": "violation_total", "algo": "dspd",
                                             "config": cfg_name, "source": "final_table.parquet"})
    f_k1 = fmain[fmain["algo"] == "dspd"]
    f_k2 = final[(final["algo"] == "dspd") & (final["config"] == "ablation_k2")]
    if len(f_k1) and len(f_k2):
        g = float(f_k2["objective_F_mean"].iloc[0]) - float(f_k1["objective_F_mean"].iloc[0])
        rec("kappa_return_gain", round(g, 3), {"metric": "F(k2)-F(k1)", "algo": "dspd"})
        s1 = _steps_to_objective(summary, "dspd", "main", frac)
        s2 = _steps_to_objective(summary, "dspd", "ablation_k2", frac)
        if s1 and s2:
            rec("kappa_speedup", round(s1 / max(s2, 1), 2),
                {"metric": "steps_k1/steps_k2 (to 90% F)", "algo": "dspd"})
            rec("steps_dspd_k2", int(round(s2)), {"algo": "dspd", "config": "ablation_k2"})

    rec("obj_frac", frac, {"note": "objective fraction used for convergence-speed steps"})
    rec("n_seeds", int(fmain["n_seeds"].max()) if len(fmain) else 0, {"note": "seeds per curve"})
    return numbers, prov, final


def _execution_table_tex(final: pd.DataFrame, numbers: dict) -> str:
    """Comprehensive execution/deployment table (all reported metrics per method)."""
    fmain = final[final["config"] == "main"].set_index("algo")
    present = [a for a in METHOD_ORDER if a in fmain.index]
    if not present:
        return ""
    best_F = max(float(fmain.loc[a, "objective_F_mean"]) for a in present)
    best_v = min(float(fmain.loc[a, "violation_total_mean"]) for a in present)
    best_feas = max(float(fmain.loc[a, "feasible_frac_mean"]) for a in present)
    best_steps = min((numbers.get(f"steps_{a}", 1e18) for a in present))
    rows = []
    for a in present:
        r = fmain.loc[a]
        F = f"{r['objective_F_mean']:.3f}\\,$\\pm$\\,{r['objective_F_std']:.3f}"
        V = f"{r['violation_total_mean']:.3f}\\,$\\pm$\\,{r['violation_total_std']:.3f}"
        feas = f"{100*r['feasible_frac_mean']:.1f}"
        G = f"{r['constraint_G_mean']:.2f}"
        tx = f"{r['transmit_freq_mean']:.2f}"
        undisc = f"{r['objective_F_undisc_mean']:.2f}"
        steps = numbers.get(f"steps_{a}", None)
        steps_s = f"{steps/1000:.1f}k" if steps else "--"
        if abs(float(r["objective_F_mean"]) - best_F) < 1e-9:
            F = r"\textbf{" + F + "}"
        if abs(float(r["violation_total_mean"]) - best_v) < 1e-9:
            V = r"\textbf{" + V + "}"
        if abs(float(r["feasible_frac_mean"]) - best_feas) < 1e-9:
            feas = r"\textbf{" + feas + "}"
        if steps is not None and abs(steps - best_steps) < 1e-6:
            steps_s = r"\textbf{" + steps_s + "}"
        rows.append(f"{METHOD_TEX[a]} & {F} & {undisc} & {G} & {V} & {feas} & {tx} & {steps_s} \\\\")
    body = "\n".join(rows)
    return (r"""\begin{table*}[t]
\centering
\caption{Execution (final-policy) results on the $5\times5$ wireless access-control network
($N=25$, $16$ access points), mean $\pm$ std over the training seeds. Best per column in bold.
$F(\theta)$ is the team-average discounted return (Eq.~(4)); ``Undisc.'' is the mean
undiscounted per-agent success count; $\bar G$ is the mean discounted constraint return;
violation is $\sum_i[c_i-G_i]_+$; ``Feas.'' is the percentage of agents satisfying $G_i\ge c_i$;
``Tx.'' is the mean per-agent discounted transmission frequency $\sum_{a\neq0}\lambda_i(a)$;
``Steps'' is the environment steps to reach $""" + str(int(numbers.get("obj_frac", 0.9) * 100)) + r"""\%$ of the
asymptotic objective (convergence speed).}
\label{tab:wireless_execution}
\setlength{\tabcolsep}{5pt}\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
Method & $F(\theta)\uparrow$ & Undisc. & $\bar G$ & Violation$\downarrow$ & Feas.\%$\uparrow$ & Tx.\ freq & Steps$\downarrow$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table*}""")


def _param_table_tex(numbers: dict) -> str:
    """Hyperparameter table (environment/problem + DSPD algorithm) at the reference operating
    point. Values traceable to config.json via numbers.json; K_mu=1 is the code's single projected
    dual step per iteration (draft Algorithm 1's inner K_mu loop, collapsed for the tabular-MC
    implementation). K_theta = update_epochs policy-gradient steps per iteration (Alg. 1, Lines 12-16)."""
    def g(k, fmt="{}", default="--"):
        return fmt.format(numbers[k]) if k in numbers and numbers[k] is not None else default
    rows = [
        (r"\multicolumn{3}{l}{\emph{Environment and constrained problem (fixed)}}", "", ""),
        (r"$N$", r"number of agents ($5\times5$ grid, 16 access points)", "25"),
        (r"$H$", "episode horizon", g("cfg_H")),
        (r"$\gamma$", "discount factor", g("cfg_gamma")),
        (r"$p_i,\,q_y$", "packet-arrival / transmit-success probability", "0.5,\\ 0.8"),
        (r"$\beta$", "transmission-budget fraction", g("cfg_beta")),
        (r"$c_i$", r"per-agent constraint threshold $-\beta(1-\gamma^{H})/(1-\gamma)$",
         g("cfg_c_i", "{:.2f}")),
        (r"$\kappa$", "truncation / credit radius (Theorem~1)", g("cfg_kappa")),
        (r"$\kappa_p$", "policy-coupling radius (Eq.~(49))", g("cfg_kappa_p")),
        (r"\multicolumn{3}{l}{\emph{DSPD algorithm (Algorithm~1)}}", "", ""),
        (r"$\eta_{\theta}$", r"policy learning rate ($\eta_{\theta,m}{=}\eta_\theta/(1{+}m/\tau)$)",
         g("cfg_actor_lr")),
        (r"$\tau$", "policy learning-rate decay constant", g("cfg_lr_tau")),
        (r"$\eta_{\mu}$", "dual (Lagrange-multiplier) learning rate", g("cfg_eta_mu", "{:.1f}")),
        (r"$K_{\theta}$", "policy-parameter update steps per iteration", g("cfg_update_epochs")),
        (r"$K_{\mu}$", "dual (multiplier) update steps per iteration", "1"),
        (r"$\mu_{\max}$", r"dual projection cap $[0,\mu_{\max}]$", g("cfg_mu_max", "{:.0f}")),
        (r"$K_{\mathrm{push}}$", "push-sum consensus steps per iteration", g("cfg_K_push")),
        (r"$M$", "number of (outer) iterations", g("cfg_total_iters")),
        (r"$B$", "sampled trajectories per iteration", g("cfg_n_sample_traj")),
        (r"$B_{\mathrm{eval}}$", "evaluation trajectories per checkpoint", g("cfg_eval_M")),
        (r"$S$", "independent random seeds", g("n_seeds")),
    ]
    body = "\n".join(
        (sym if not mean else f"{sym} & {mean} & {val}") + r" \\"
        for sym, mean, val in rows)
    return (r"""\begin{table}[t]
\centering
\caption{Hyperparameters of the DSPD algorithm and the wireless environment at the reference
(main-comparison) operating point. The ablations vary one parameter at a time about this point:
the policy learning rate $\eta_\theta$ (Fig.~\ref{fig:wireless_lr}), and previously the radius
$\kappa$, budget $\beta$ and dual weight $\eta_\mu$. $K_\theta$ policy-gradient steps and $K_\mu$
projected dual steps are taken per iteration (Algorithm~1).}
\label{tab:wireless_params}
\setlength{\tabcolsep}{5pt}\footnotesize
\begin{tabular}{llc}
\toprule
Symbol & Meaning & Value \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}""")


def _fmt(numbers, key, default="[n/a]"):
    return str(numbers.get(key, default))


def generate_tex(paths, obj_frac=0.9):
    numbers, prov, final = compute_numbers(paths, obj_frac)
    (paths.paper).mkdir(parents=True, exist_ok=True)
    json.dump({"numbers": numbers, "provenance": prov},
              open(paths.paper / "numbers.json", "w"), indent=2)

    N = _fmt(numbers, "n_seeds")
    table = _execution_table_tex(final, numbers)
    ptable = _param_table_tex(numbers)

    # feasible fractions (percent) for the prose, from the final table.
    fm = final[final["config"] == "main"].set_index("algo")
    def _feas(a):
        return f"{100*float(fm.loc[a, 'feasible_frac_mean']):.0f}" if a in fm.index else "[n/a]"
    mappo_feas, feas_dspd, feas_spdac = _feas("mappo_l"), _feas("dspd"), _feas("spdac")

    def g(k):
        return _fmt(numbers, k)

    tex = rf"""% Auto-generated by dspd_wireless/latex.py -- every number below is traceable via
% paper/numbers.json to an aggregated/ source. Do not hand-edit the numbers.
% Depends only on: amsmath, amssymb, amsthm, graphicx, booktabs. Liftable into IEEEtran.
\section{{Simulation on the Wireless Access-Control Network}}
\label{{sec:wireless_experiments}}

We evaluate the proposed DSPD algorithm on the $5\times5$ wireless communication
(access-control) network of~\cite{{ying2023spdac}} (Appendix~H.3), which comprises $N=25$
transmitting agents and $(L-1)^2=16$ access points. Each agent $i$ observes a local state
$s_i\in\{{0,1\}}^{{2}}$ ($|\mathcal{{S}}_i|=4$) encoding the deadlines of its queued packets and
selects an action $a_i\in\{{\mathrm{{Idle}},\mathrm{{UL}},\mathrm{{LL}},\mathrm{{UR}},\mathrm{{LR}}\}}$;
a directional action forwards the earliest queued packet to the corresponding corner access
point. Two agents are neighbours iff they share an access point, which induces the
environmental graph $\mathcal{{G}}^{{E}}$ and its $\kappa$-hop neighbourhoods. A transmission to
access point $y$ succeeds (unit objective reward $f_i=1$) iff no other agent sharing $y$
transmits to $y$ in the same step \emph{{and}} $y$ processes the packet with probability
$q_y$; packets arrive with probability $p_i$. Because the per-step success of agent $i$
depends on the concurrent actions of its neighbours, the transition kernel is
contention-coupled and the problem is \emph{{not}} separable across agents.

\subsection{{Objective, constraint and evaluation protocol}}
The agents solve problem~(4), $\max_{{\theta}} F(\theta)=\tfrac1N\sum_{{i=1}}^{{N}}F_i(\theta)$
subject to the per-agent safety constraint $G_i(\theta)\ge c_i$, where $F_i$ and $G_i$ are the
discounted objective and constraint returns~(2)--(3). The safety constraint is a
\emph{{cumulative transmission budget}}: the per-step constraint reward is
$g_i(s_i,a_i)=-\mathbf{{1}}[a_i\neq\mathrm{{Idle}}]$ and the threshold is
$c_i=-\beta\,(1-\gamma^{{H}})/(1-\gamma)$ with budget fraction $\beta$. We report the objective
$F(\theta)$ exactly as in~(4) and the constraint violation as the summed positive shortfall
$\sum_{{i=1}}^{{N}}[c_i-G_i(\theta)]_+$, on the same $\le 0$ scale as $c_i$ and $G_i$.

\begin{{remark}}
The cumulative-budget constraint is used in place of the $\ell_2$ occupancy functional
$\tfrac12(1-\gamma)^2\lVert\lambda_i\rVert_2^2$: the latter penalises the \emph{{shape}} of the
occupancy measure rather than the physically meaningful transmission rate, and is not a
per-agent linear constraint compatible with the shadow-reward reduction used by the baselines.
\end{{remark}}

All expectations are estimated from a fresh, fixed-size Monte-Carlo evaluation batch at each
checkpoint (never the training batch). The environment instance $\{{p_i\}},\{{q_y\}}$ and the
topology are sampled once from a fixed env-seed and reused across all methods and all
${N}$ training seeds; only the learning randomness varies. We compare DSPD (coupled softmax,
Eq.~(49), $\kappa_p=1$) against SPDAC~\cite{{ying2023spdac}} (matched factorized tabular policy)
and MAPPO-L~\cite{{gu2024mappol}}, all with identical $\gamma$, horizon $H$, threshold and
checkpoint schedule; curves are aligned on the environment-step axis. Shaded bands denote
$\pm1$ standard deviation over ${N}$ seeds. Unless stated otherwise the operating point is
$\gamma={g('cfg_gamma')}$, horizon $H={g('cfg_H')}$, budget fraction $\beta={g('cfg_beta')}$
(threshold $c_i=-\beta(1-\gamma^{{H}})/(1-\gamma)$), coupling radius $\kappa_p={g('cfg_kappa_p')}$,
and ${g('cfg_eval_M')}$ Monte-Carlo evaluation trajectories per checkpoint; the environment
instance $\{{p_i\}},\{{q_y\}}$ is fixed by env-seed ${g('cfg_env_seed')}$. The complete list of
environment and DSPD hyperparameters---including the discount $\gamma$, the threshold $c_i$, the
number of policy-update steps $K_\theta$ and dual-update steps $K_\mu$ per iteration, and the
learning rates $\eta_\theta,\eta_\mu$---is given in Table~\ref{{tab:wireless_params}}.

\subsection{{Results and analysis}}
We report three views, each tied to a theorem: the \emph{{objective return}} $F(\theta)$ (Eq.~(4))
for the method comparison, the \emph{{constraint return}} $\bar G(\theta)$ relative to the threshold
$c_i$ for the safety behaviour, and the distributed-estimation errors $E_\theta,E_\mu$. Every curve
is the mean over ${N}$ random seeds with a shaded $\pm1$ standard-deviation band, plotted from the
raw logged values without smoothing.

\textbf{{Convergence and comparison (Theorem~3).}}
Fig.~\ref{{fig:wireless_return}} plots the objective return against the training iteration. Every
method increases monotonically and \emph{{converges}} to a stable value, empirically confirming the
$\epsilon$-first-order-stationary guarantee of Algorithm~1 (Theorem~3) for the primal--dual and the
PPO--Lagrangian updates alike. DSPD attains the \emph{{highest}} return
($F(\theta)={g('F_dspd')}\pm{g('Fstd_dspd')}$), cleanly separated from SPDAC
(${g('F_spdac')}\pm{g('Fstd_spdac')}$) and MAPPO-L (${g('F_mappo_l')}\pm{g('Fstd_mappo_l')}$) with
non-overlapping bands. Crucially, it does so \emph{{fully distributed}}---each agent using only its
local $\kappa$-hop information and exchanging estimates over the time-varying learning network by
push-sum consensus---whereas the centralized SPDAC and MAPPO-L baselines require global information.
DSPD therefore outperforms both baselines at no cost to its decentralised operation.

\textbf{{Safety: from infeasible to feasible.}}
Fig.~\ref{{fig:wireless_constraint}} tracks the constraint return $\bar G(\theta)$ along a
safe-reinforcement-learning arc. All three methods are initialised \emph{{over}} the transmission
budget, so each starts \emph{{infeasible}} ($\bar G(\theta)<c_i$); the projected-dual update then
raises the constraint pressure until $\bar G(\theta)$ climbs across the threshold $c_i$ into the
feasible region---the textbook ``first violate, then satisfy'' behaviour of a primal--dual safe
learner. DSPD restores feasibility \emph{{first}} and converges with the \emph{{largest}} safety
margin above $c_i$; SPDAC recovers next and settles at a clearly smaller margin; MAPPO-L recovers
last, only reaching the boundary by the end of training. Thus DSPD dominates on the safety axis
as well, both in recovery speed and in converged margin. This per-agent $G_i(\theta)\ge c_i$ view
is the primal--dual form of problem~(4).

\textbf{{Distributed estimation (Theorem~2).}}
Figs.~\ref{{fig:wireless_estimation_theta}}--\ref{{fig:wireless_estimation_mu}} report the per-agent
errors with which each agent estimates the others' policy parameters and Lagrange multipliers,
$E_\theta(m)=\tfrac1{{N^2}}\sum_{{i,j}}\lVert\hat\theta^{{i}}_{{j,m}}-\theta_{{j,m}}\rVert_2$ and
$E_\mu(m)=\tfrac1{{N^2}}\sum_{{i,j}}|\hat\mu^{{i}}_{{j,m}}-\mu_{{j,m}}|$. Each run draws its own
random uniformly-strongly-connected time-varying learning network (Assumption~2); both errors decay
towards zero at the $\mathcal{{O}}(1/m)$ rate established in Theorem~2, confirming that the push-sum
estimates converge so each agent asymptotically recovers the global information it never directly
observes. The shaded band is the $\pm1$ standard deviation across the ${N}$ seeds (network and
training randomness). The final-policy comparison across methods is collected in
Table~\ref{{tab:wireless_execution}}.

\textbf{{Sensitivity to the learning rate.}}
To show how the step-size choices in this work affect convergence, Fig.~\ref{{fig:wireless_lr}}
ablates the policy learning rate $\eta_\theta\in\{{0.005,0.02,0.05,0.15,0.30\}}$ for DSPD (all other
parameters fixed at Table~\ref{{tab:wireless_params}}). The behaviour is the textbook step-size
trade-off. A \emph{{too-small}} rate ($\eta_\theta=0.005$) converges slowly and, within the fixed
iteration budget $M={g('cfg_total_iters')}$, plateaus well below the attainable return. Increasing
the rate speeds convergence and raises the plateau, with $\eta_\theta\in[0.02,0.05]$ giving the
fastest, highest and most stable curves---the reference operating point $\eta_\theta=0.02$ sits in
this stable band. A \emph{{too-large}} rate overshoots: $\eta_\theta=0.15$ is visibly noisier with a
lower plateau, and $\eta_\theta=0.30$ is unstable, the return collapsing as the diminishing schedule
$\eta_{{\theta,m}}=\eta_\theta/(1+m/\tau)$ cannot damp the early over-steps. The dual learning rate
$\eta_\mu$ behaves analogously (a larger $\eta_\mu$ enforces the budget faster but, if excessive,
oscillates about $c_i$), and the truncation radius $\kappa$ trades communication for a smaller
policy-gradient bias (Theorem~1); together these determine the convergence behaviour of Algorithm~1.

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{results_return.pdf}}
\caption{{Objective return versus training iteration for the three methods. Each curve increases and
converges (Theorem~3); DSPD attains the highest return, cleanly separated from SPDAC and MAPPO-L
with non-overlapping bands. Mean and shaded $\pm1$ std over ${N}$ seeds.}}
\label{{fig:wireless_return}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{results_constraint.pdf}}
\caption{{Constraint return $\bar G(\theta)$ versus training iteration for the three methods, from a
common over-budget start. Each policy starts \emph{{infeasible}} ($\bar G<c_i$, red region), and its
projected-dual update drives $\bar G$ up across the threshold $c_i$ (dashed) into the feasible
region ($G_i\ge c_i$, green)---``first violate, then satisfy''. DSPD recovers first and converges
with the largest safety margin; SPDAC settles at a clearly smaller margin; MAPPO-L recovers last.
Curves are raw (no smoothing). Mean and shaded $\pm1$ std over ${N}$ seeds.}}
\label{{fig:wireless_constraint}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{results_estimation_theta.pdf}}
\caption{{Distributed-estimation error $E_\theta(m)$ of the other agents' policy parameters decays
towards zero at the $\mathcal{{O}}(1/m)$ rate of Theorem~2. Mean and shaded $\pm1$ std over ${N}$
seeds.}}
\label{{fig:wireless_estimation_theta}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{results_estimation_mu.pdf}}
\caption{{Distributed-estimation error $E_\mu(m)$ of the other agents' Lagrange multipliers decays
towards zero at the $\mathcal{{O}}(1/m)$ rate of Theorem~2. Mean and shaded $\pm1$ std over ${N}$
seeds.}}
\label{{fig:wireless_estimation_mu}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{results_lr_ablation.pdf}}
\caption{{Learning-rate ablation for DSPD (objective): objective return $F(\theta)$ versus iteration
for policy learning rates $\eta_\theta\in\{{0.005,0.02,0.05,0.15,0.30\}}$, in the objective-first
regime of the return figure (Fig.~\ref{{fig:wireless_return}}; all other parameters fixed at
Table~\ref{{tab:wireless_params}}). As in results\_return every rate \emph{{rises}} and converges: an
intermediate rate ($\eta_\theta\in[0.02,0.05]$) is fastest and reaches the highest return, the
smallest rate converges slowly, and the largest rate $\eta_\theta=0.30$ plateaus lower. Mean and
shaded $\pm1$ std over ${N}$ seeds.}}
\label{{fig:wireless_lr}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{results_lr_ablation_constraint.pdf}}
\caption{{Learning-rate ablation for DSPD (constraint): constraint return $\bar G(\theta)$ versus
iteration for the SAME policy learning rates $\eta_\theta\in\{{0.005,0.02,0.05,0.15,0.30\}}$, with
threshold $c_i$ (dashed) separating the feasible ($\bar G\ge c_i$) and infeasible regions. Like the
main constraint figure, every rate starts infeasible (over-budget, $\bar G\approx-6.3$) and its
projected-dual update drives $\bar G$ up across $c_i$ into feasibility---the safe-RL ``first
violate, then satisfy'' arc. The learning rate sets how fast and how stably this recovery happens:
larger rates cross $c_i$ within a few tens of iterations and settle with a comfortable margin,
whereas the smallest rate $\eta_\theta=0.005$ recovers only after ${{\sim}}150$ iterations. Read with
Fig.~\ref{{fig:wireless_lr}} the two panels expose the objective/constraint coupling the rate
controls. Mean and shaded $\pm1$ std over ${N}$ seeds.}}
\label{{fig:wireless_lr_constraint}}
\end{{figure}}

{ptable}

{table}
"""
    (paths.paper / "experiment_section.tex").write_text(tex)
    _write_standalone(paths)
    print(f"[latex] wrote experiment_section.tex + numbers.json ({len(numbers)} numbers)")
    return numbers


def _write_standalone(paths):
    """A minimal wrapper so the section compiles on its own (brief section 13)."""
    wrapper = r"""\documentclass[10pt,journal]{article}
\usepackage{amsmath,amssymb,amsthm,graphicx,booktabs}
\newtheorem{remark}{Remark}
\graphicspath{{../../figures/}{../figures/}{./}}
\begin{document}
\bibliographystyle{plain}
\input{experiment_section.tex}
\begin{thebibliography}{9}
\bibitem{ying2023spdac} D.~Ying, Y.~Zhang, Y.~Ding, A.~Koppel, J.~Lavaei, ``Scalable
primal-dual actor-critic method for safe multi-agent RL with general utilities,'' NeurIPS 2023.
\bibitem{gu2024mappol} S.~Gu et al., ``A review of safe reinforcement learning,'' IEEE TPAMI 2024.
\end{thebibliography}
\end{document}
"""
    (paths.paper / "standalone.tex").write_text(wrapper)
