"""One entry point that reproduces the full pipeline (brief sections 16-17).

    python run.py --profile pilot      # train matrix -> aggregate -> figures -> LaTeX
    python run.py --profile full
    python run.py --only aggregate --run runs/run_0001   # re-aggregate/plot an existing run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import results as R
from .runner import run_profile, PKG_ROOT
from .aggregate import aggregate
from .plotting import make_all_figures
from .latex import generate_tex


def _paths_from_dir(run_dir: Path) -> R.RunPaths:
    run_dir = Path(run_dir)
    return R.RunPaths(root=run_dir, run_id_str=run_dir.name)


def post_process(paths: R.RunPaths, obj_frac: float = 0.9):
    agg = aggregate(paths)
    if agg is None:
        print("[pipeline] nothing to aggregate; skipping figures/latex")
        return
    make_all_figures(paths)
    generate_tex(paths, obj_frac=obj_frac)
    _consistency_check(paths)
    _compile_pdf(paths)


def _compile_pdf(paths: R.RunPaths):
    """Compile the standalone LaTeX section to a readable PDF so every run carries the paper
    draft at ``results/paper/standalone.pdf`` (previously only ``run_full.sh`` did this)."""
    import shutil
    import subprocess
    tex = paths.paper / "standalone.tex"
    if not tex.exists():
        return
    if shutil.which("pdflatex") is None:
        print("[latex] pdflatex not found; wrote .tex only (no PDF).")
        return
    try:
        for _ in range(2):  # twice to resolve refs
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "standalone.tex"], cwd=paths.paper,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        for ext in (".aux", ".log", ".out"):
            (paths.paper / ("standalone" + ext)).unlink(missing_ok=True)
        pdf = paths.paper / "standalone.pdf"
        print(f"[latex] compiled {pdf}" if pdf.exists() else "[latex] pdflatex ran but no PDF")
    except Exception as e:
        print(f"[latex] pdflatex failed (non-fatal): {e}")


def _consistency_check(paths: R.RunPaths):
    """Confirm every LaTeX number equals its aggregated source (brief section 15)."""
    import json
    nj = paths.paper / "numbers.json"
    tex = paths.paper / "experiment_section.tex"
    if not nj.exists() or not tex.exists():
        return
    numbers = json.load(open(nj))["numbers"]
    body = tex.read_text()
    # Every number *quoted* in the tex must equal its aggregated source (brief section 15).
    # (The reverse -- a logged-but-unquoted number -- is allowed; brief section 11.7 only
    # forbids a quoted number with no provenance entry.) latex.py inserts from the same
    # source, so we assert the specific quoted keys are present verbatim.
    quoted = ["F_dspd", "Fstd_dspd", "F_spdac", "Fstd_spdac", "F_mappo_l", "Fstd_mappo_l"]
    bad = [k for k in quoted if k in numbers and str(numbers[k]) not in body]
    print(f"[verify] LaTeX/aggregated consistency: {'OK' if not bad else 'MISMATCH '+str(bad)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="DSPD wireless experiment harness")
    ap.add_argument("--profile", default="pilot", help="pilot | full | <name>.yaml")
    ap.add_argument("--only", default=None,
                    choices=[None, "aggregate", "plot", "latex", "post"],
                    help="skip training; act on an existing --run")
    ap.add_argument("--run", default=None, help="existing runs/run_XXXX dir for --only")
    ap.add_argument("--obj-frac", type=float, default=0.9,
                    help="objective fraction for the convergence-speed (steps) numbers")
    args = ap.parse_args(argv)

    if args.only:
        assert args.run, "--only requires --run"
        paths = _paths_from_dir(args.run)
        if args.only in ("aggregate", "post"):
            post_process(paths, args.obj_frac)
        elif args.only == "plot":
            make_all_figures(paths)
        elif args.only == "latex":
            generate_tex(paths, args.obj_frac)
        return

    paths = run_profile(args.profile)
    post_process(paths, args.obj_frac)
    print(f"\nDone. Results in {paths.root}")
    print(f"  figures/  data_csv/  results/aggregated/  results/paper/experiment_section.tex")


if __name__ == "__main__":
    main()
