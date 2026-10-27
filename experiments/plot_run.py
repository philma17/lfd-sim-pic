"""Re-create the plots of an existing run from its saved data (no simulation).

    python experiments/plot_run.py results/runs/<batch>/<run>
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    from pic import plots
    run = plots.RunData(sys.argv[1])
    out = os.path.join(run.path, "plots_regenerated")      # does not overwrite plots/
    os.makedirs(out, exist_ok=True)
    print(*plots.make_run_plots(run, out), sep="\n")
