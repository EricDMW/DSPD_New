"""Experiment orchestration: expand the matrix, train, log, checkpoint (brief sections 9-11).

One invocation == one ``runs/run_XXXX`` directory (auto-incrementing id). Inside it the full
brief section-11 layout is produced, plus a ``data_csv/`` folder holding every raw, unsmoothed
time series for the record (user requirement).
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .env.wireless import WirelessEnv
from .metrics import threshold_c
from .algos.dspd import DSPD
from .algos.spdac import SPDAC
from .algos.mappo_l import MAPPOL
from . import results as R

PKG_ROOT = Path(__file__).resolve().parents[2]  # dspd_wireless/
CONFIG_DIR = PKG_ROOT / "configs"


def build_trainer(algo, env, cfg, rng):
    kappa = int(cfg["kappa"])
    if algo == "dspd":
        return DSPD(env, cfg, rng, credit_kappa=kappa)
    if algo == "spdac":
        return SPDAC(env, cfg, rng, credit_kappa=kappa)
    if algo in ("mappo_l", "mappo_l_dec", "mappo_l_decagg"):
        return MAPPOL(env, cfg, rng, variant=algo, credit_kappa=kappa)
    raise ValueError(f"unknown algo {algo}")


def make_env(cfg):
    env = WirelessEnv.fixed_instance(
        env_seed=int(cfg["env_seed"]), L=int(cfg["L"]), ddl=int(cfg["ddl"]), H=int(cfg["H"]),
        pkg_p=float(cfg["pkg_p"]), success_p=float(cfg["success_p"]),
        heterogeneous=bool(cfg["heterogeneous"]),
        rng=np.random.default_rng(int(cfg["seed"])),
    )
    return env


def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True


def load_profile(profile_path: Path) -> tuple[dict, list]:
    prof = yaml.safe_load(Path(profile_path).read_text())
    base = yaml.safe_load((CONFIG_DIR / prof["base"]).read_text())
    base.update(prof.get("overrides", {}))
    return base, prof["experiments"]


def resolve_cfg(base: dict, exp_overrides: dict, seed: int) -> dict:
    cfg = dict(base)
    cfg.update(exp_overrides)
    cfg["seed"] = int(seed)
    cfg["c_i"] = threshold_c(float(cfg["beta"]), float(cfg["gamma"]), int(cfg["H"]))
    return cfg


def expand_matrix(base, experiments):
    """Yield (algo, config_name, cfg) for every run in the matrix."""
    for exp in experiments:
        config_name = exp["config"]
        for algo in exp["methods"]:
            for seed in base["seeds"]:
                cfg = resolve_cfg(base, exp.get("overrides", {}), seed)
                yield algo, config_name, cfg


def train_one(algo, config_name, cfg, paths: R.RunPaths, manifest: R.IndexManifest):
    """Train a single (algo, config, seed) run end-to-end, logging + checkpointing."""
    seed = cfg["seed"]
    seed_everything(seed, cfg.get("torch_deterministic", True))
    env = make_env(cfg)
    prov = R.provenance()
    # run_id folds in the invocation scope (paths.run_id_str) and the resolved cfg, so it
    # updates across separate runnings and across parameter changes (see results.run_id_hash).
    rid = R.run_id_hash(algo, config_name, seed, cfg["env_seed"], prov["git_hash"],
                        run_scope=paths.run_id_str, cfg=cfg)
    seed_dir = paths.seed_dir(algo, config_name, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    # persist fully-resolved config + provenance (brief section 11.2)
    R.atomic_write_json(seed_dir / "config.json", {"algo": algo, "config": config_name,
                                                   "resolved": cfg, "provenance": prov})
    R.save_env_instance(seed_dir, env.instance_dict())
    manifest.upsert(dict(run_id=rid, algo=algo, config=config_name, seed=seed,
                         env_seed=cfg["env_seed"], status="running", env_steps=0,
                         wall_clock_s=0, git_hash=prov["git_hash"],
                         created_utc=prov["timestamp_utc"], path=str(seed_dir)))

    trainer = build_trainer(algo, env, cfg, np.random.default_rng(seed))
    writer = R.MetricsWriter(seed_dir, rid, algo, config_name, seed)
    logf = open(seed_dir / "logs_stdout.log", "w")
    t0 = time.time()
    train_info = {}
    T = int(cfg["total_iters"])
    # Gradient tracking (observability only; does NOT touch params/RNG/optimisation). The
    # pre-clip policy-gradient norm ||grad_theta L|| is the first-order-stationarity measure
    # of Theorem 3; we log both the instantaneous value at each test point and the mean over
    # every iteration since the last test point (full-run coverage) to assess convergence.
    grad_window: list[float] = []
    for m in range(1, T + 1):
        train_info = trainer.iterate(m)
        grad_window.append(float(train_info.get("actor_grad_norm", 0.0)))
        if m % int(cfg["test_every"]) == 0 or m == 1 or m == T:
            ev = trainer.evaluate(int(cfg["eval_M"]))
            ev.update({k: train_info[k] for k in ("policy_loss", "actor_grad_norm") if k in train_info})
            ev["actor_grad_norm_mean"] = float(np.mean(grad_window)) if grad_window else 0.0
            grad_window = []
            # Large-batch gradient probe (Theorem-3 validation): measure ||grad_theta L|| at each
            # listed batch size, RNG-isolated so training is unperturbed. Logged as grad_probe_m{B}.
            for B in cfg.get("grad_probe_batches", []):
                ev[f"grad_probe_m{int(B)}"] = trainer.probe_grad_norm(int(B))
            ev["wall_clock_s"] = time.time() - t0
            ev["env_steps"] = trainer.env_steps
            writer.log(trainer.env_steps, m, ev)
            line = (f"[{algo}/{config_name}/seed{seed}] m={m} step={trainer.env_steps} "
                    f"F={ev['objective_F']:.3f} viol={ev['violation_total']:.2f} "
                    f"feas={ev['feasible_frac']:.2f} mu={ev.get('dual_mu',0):.2f}")
            logf.write(line + "\n"); logf.flush()
        if m % int(cfg["checkpoint_every"]) == 0:
            R.save_checkpoint(seed_dir, f"iter_{m:06d}.pt",
                              dict(trainer.state_dict(), iter=m, run_id=rid))
    R.save_checkpoint(seed_dir, "final.pt", dict(trainer.state_dict(), iter=T, run_id=rid))
    writer.close()
    logf.close()
    wall = time.time() - t0
    R.atomic_write_json(seed_dir / "run_meta.json",
                        dict(status="complete", wall_clock_s=wall, env_steps=trainer.env_steps,
                             iters=T, git_hash=prov["git_hash"], run_id=rid))
    manifest.upsert(dict(run_id=rid, algo=algo, config=config_name, seed=seed,
                         env_seed=cfg["env_seed"], status="complete",
                         env_steps=trainer.env_steps, wall_clock_s=round(wall, 1),
                         git_hash=prov["git_hash"], created_utc=prov["timestamp_utc"],
                         path=str(seed_dir)))
    return rid, wall, trainer.env_steps


def run_profile(profile: str = "pilot", base_runs_dir: Path | None = None, verbose=True):
    """Run a whole profile into a fresh runs/run_XXXX and return its RunPaths."""
    profile_path = CONFIG_DIR / f"{profile}.yaml"
    base, experiments = load_profile(profile_path)
    base_runs_dir = Path(base_runs_dir) if base_runs_dir else (PKG_ROOT / "runs")
    paths = R.next_run_dir(base_runs_dir)
    R.atomic_write_json(paths.root / "config.json",
                        {"profile": profile, "base": base, "experiments": experiments,
                         "provenance": R.provenance()})
    manifest = R.IndexManifest(paths.index_csv)
    runs = list(expand_matrix(base, experiments))
    if verbose:
        print(f"== {paths.run_id_str}: {len(runs)} runs ==")
    t0 = time.time()
    for i, (algo, config_name, cfg) in enumerate(runs, 1):
        seed_dir = paths.seed_dir(algo, config_name, cfg["seed"])
        meta = seed_dir / "run_meta.json"
        if meta.exists() and '"status": "complete"' in meta.read_text():
            if verbose:
                print(f"  ({i}/{len(runs)}) skip complete {algo}/{config_name}/seed{cfg['seed']}")
            continue
        rid, wall, steps = train_one(algo, config_name, cfg, paths, manifest)
        if verbose:
            print(f"  ({i}/{len(runs)}) {algo}/{config_name}/seed{cfg['seed']} "
                  f"done in {wall:.1f}s ({steps} steps)")
    if verbose:
        print(f"== training done in {time.time()-t0:.1f}s ==")
    return paths
