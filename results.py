"""Results & data management (brief section 11) + run/run_id run isolation.

Layout (per the brief, wrapped in an auto-incrementing ``runs/run_XXXX`` per invocation, as
requested by the user):

    runs/
      run_0001/                       <- one full experiment invocation, id auto +1 each run
        config.json                   experiment-level resolved config + provenance
        results/
          index.csv                   run manifest (one row per (algo,config,seed))
          runs/<algo>/<config>/seed<k>/
            config.json  env_instance.npz  metrics.csv  metrics.parquet  run_meta.json
            checkpoints/iter_XXXXXX.pt  final.pt
            logs/stdout.log
          aggregated/  paper/
        data_csv/                     <- ALL raw, UNSMOOTHED time series, for the record
        figures/  logs/

All writes are atomic (temp file + os.replace, brief section 11.7).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Long/tidy metrics schema (brief section 11.4).
METRIC_COLUMNS = ["run_id", "algo", "config", "seed", "env_step", "iter",
                  "scope", "agent", "metric", "value"]

# Metric vocabulary (brief section 11.4) -- keys allowed in the tidy table.
METRIC_VOCAB = {
    "objective_F", "objective_F_undisc", "constraint_G", "constraint_G_min", "constraint_G_max",
    "constraint_G_std_agents", "constraint_slack", "feasible_frac",
    "violation_total", "violation_mean", "violation_max", "dual_mu", "lagrangian_L",
    "transmit_freq", "occupancy_lambda", "policy_loss", "critic1_loss", "critic2_loss",
    "critic1_score", "critic2_score", "E_theta", "E_mu", "E_theta_sq",
    "pushsum_disagreement", "pushsum_invariant_err",
    "end_step", "n_sample_traj", "wall_clock_s", "env_steps", "samples",
    # first-order-stationarity tracking (Theorem 3): pre-clip policy-gradient norm
    "actor_grad_norm", "actor_grad_norm_mean",
}


def atomic_write_text(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def git_provenance():
    def _run(args):
        try:
            return subprocess.check_output(args, cwd=Path(__file__).parent,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {
        "git_hash": _run(["git", "rev-parse", "HEAD"]) or "no-git",
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def provenance(cmdline=None):
    import numpy as _np
    try:
        import torch
        torch_v, cuda_v = torch.__version__, (torch.version.cuda or "cpu")
    except Exception:
        torch_v, cuda_v = None, None
    p = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": _np.__version__,
        "torch": torch_v,
        "cuda": cuda_v,
        "platform": platform.platform(),
        "cmdline": cmdline if cmdline is not None else " ".join(sys.argv),
    }
    p.update(git_provenance())
    return p


# ---------------------------------------------------------------------------------------

@dataclass
class RunPaths:
    root: Path            # runs/run_XXXX
    run_id_str: str       # "run_0001"

    @property
    def results(self): return self.root / "results"
    @property
    def runs(self): return self.results / "runs"
    @property
    def aggregated(self): return self.results / "aggregated"
    @property
    def paper(self): return self.results / "paper"
    @property
    def figures(self): return self.root / "figures"
    @property
    def data_csv(self): return self.root / "data_csv"
    @property
    def logs(self): return self.root / "logs"
    @property
    def index_csv(self): return self.results / "index.csv"

    def seed_dir(self, algo, config, seed):
        return self.runs / algo / config / f"seed{seed}"


def next_run_dir(base: Path) -> RunPaths:
    """Create runs/run_XXXX with the id = (max existing id) + 1 (auto-increment)."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    ids = [int(m.group(1)) for p in base.iterdir()
           if (m := re.fullmatch(r"run_(\d+)", p.name)) and p.is_dir()]
    nid = (max(ids) + 1) if ids else 1
    run_id_str = f"run_{nid:04d}"
    root = base / run_id_str
    for d in ("results/runs", "results/aggregated", "results/paper",
              "figures", "data_csv", "logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    return RunPaths(root=root, run_id_str=run_id_str)


def run_id_hash(algo, config, seed, env_seed, git_hash,
                run_scope: str = "", cfg: dict | None = None) -> str:
    """Short id for a single (algo,config,seed) run (brief section 11.3).

    Bug fix: the previous id hashed only ``(algo, config, seed, env_seed, git_hash)``. Because
    ``config`` is a *name* (e.g. ``main``) and ``git_hash`` is a constant (``no-git`` when the
    tree is not a git repo), the id never changed across separate invocations, nor when the
    *actual* parameter values changed under the same config name -- so two different
    experimental runs collided on one id. We now also fold in

      * ``run_scope`` -- the invocation's ``run_XXXX`` id, so a fresh invocation gets a fresh id
        (the user's "update the run id in different runnings" requirement); it is *stable*
        within a run directory, so resuming a crashed run keeps its id (brief section 11.7),
      * a canonical hash of the fully-resolved ``cfg`` -- so any change to the actual parameter
        values (lr, gamma, beta, H, ...) yields a different id even under the same config name.

    The id is therefore reproducible for a given (run directory, resolved parameters) and
    genuinely distinct otherwise.
    """
    import hashlib
    parts = [str(algo), str(config), str(seed), str(env_seed), str(git_hash), str(run_scope)]
    if cfg is not None:
        # canonical, order-independent digest of the resolved parameters that define the run
        canon = json.dumps(cfg, sort_keys=True, default=_json_default)
        parts.append(hashlib.sha1(canon.encode()).hexdigest())
    key = "|".join(parts)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------

class MetricsWriter:
    """Append-only tidy time-series writer for one (algo,config,seed) run (brief section 11.4).

    Flushes to ``metrics.csv`` incrementally (brief section 11.7 append-only), and writes
    ``metrics.parquet`` at :meth:`close`. Raw, UNSMOOTHED values only.
    """

    def __init__(self, seed_dir: Path, run_id, algo, config, seed):
        self.seed_dir = Path(seed_dir)
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.ids = dict(run_id=run_id, algo=algo, config=config, seed=int(seed))
        self.rows = []
        self.csv_path = self.seed_dir / "metrics.csv"
        self._fh = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(METRIC_COLUMNS)

    def log(self, env_step: int, itr: int, metrics: dict):
        """Add all metrics from one checkpoint. Scalars -> scope='mean'; *_per_agent -> 'agent'."""
        base = dict(self.ids, env_step=int(env_step), iter=int(itr))
        for key, val in metrics.items():
            if key.endswith("_per_agent"):
                metric = key[: -len("_per_agent")]
                metric = {"occupancy_lambda": "occupancy_lambda"}.get(metric, metric)
                if metric not in METRIC_VOCAB:
                    continue
                arr = np.asarray(val).reshape(-1)
                for a, v in enumerate(arr):
                    self._emit(base, "agent", a, metric, float(v))
            else:
                if key not in METRIC_VOCAB and not key.startswith("grad_probe_m"):
                    continue
                scope = "agg" if key in ("violation_total",) else "mean"
                self._emit(base, scope, None, key, float(val))
        self._fh.flush()

    def _emit(self, base, scope, agent, metric, value):
        row = dict(base, scope=scope, agent=("" if agent is None else agent),
                   metric=metric, value=value)
        self.rows.append(row)
        self._w.writerow([row[c] for c in METRIC_COLUMNS])

    def close(self):
        self._fh.close()
        try:
            import pandas as pd
            df = pd.DataFrame(self.rows, columns=METRIC_COLUMNS)
            df.to_parquet(self.seed_dir / "metrics.parquet", index=False)
        except Exception:
            pass  # parquet is a convenience; metrics.csv is authoritative


def save_env_instance(seed_dir: Path, instance: dict):
    # np.savez appends ".npz" to the filename, so name the temp file accordingly.
    tmp = Path(seed_dir) / "_env_instance_tmp.npz"
    np.savez(tmp, **{k: np.asarray(v) for k, v in instance.items()})
    os.replace(tmp, Path(seed_dir) / "env_instance.npz")


def save_checkpoint(seed_dir: Path, name: str, payload: dict):
    import torch
    ck = Path(seed_dir) / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    tmp = ck / (name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, ck / name)


class IndexManifest:
    """results/index.csv, updated atomically as runs finish (brief section 11.3)."""

    COLUMNS = ["run_id", "algo", "config", "seed", "env_seed", "status",
               "env_steps", "wall_clock_s", "git_hash", "created_utc", "path"]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows = []
        if self.path.exists():
            import pandas as pd
            self.rows = pd.read_csv(self.path).to_dict("records")

    def upsert(self, row: dict):
        self.rows = [r for r in self.rows if not (r["run_id"] == row["run_id"])]
        self.rows.append({c: row.get(c, "") for c in self.COLUMNS})
        self._flush()

    def _flush(self):
        tmp = self.path.with_suffix(".csv.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.COLUMNS)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)
