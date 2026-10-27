"""Build a small S2NNDS directory with the extra datasets (Five_Obstacle_DS, Cshape_bottom, ...)
from a full S2NNDS checkout, WITHOUT modifying the checkout and without committing anything.

    git clone https://github.com/allemmbinn/S2NNDS.git ../S2NNDS_src        # revision 3636a91 was used
    python experiments/prepare_s2nnds.py --src ../S2NNDS_src --dst data/S2NNDS_extra
    # then, in the shell where you run experiments (PowerShell: $env:S2NNDS_DIR="data\\S2NNDS_extra"):
    export S2NNDS_DIR=data/S2NNDS_extra

Why a copy: the upstream NNModels.py imports GUI/training packages (pyqtgraph, ...). The copy uses the
adapted torch-only NNModels.py from vendor/S2NNDS (see vendor/S2NNDS/SOURCE.txt). Only the files needed
for the listed models are copied (model_f.pth, config json, demonstration data, robot recordings).
The S2NNDS files were included in this repository with the owner's permission only for the Sine model,
so the extra data are deliberately NOT vendored (data/ is git-ignored).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = [("LASA", "Sine"), ("2D_Shapes", "Five_Obstacle_DS"), ("3D_Shapes", "Cshape_bottom")]
RECORDINGS = ["Robot_Data/2D_Shapes/Five_Obstacle_DS.csv",
              "Robot_Data/2D_Shapes/Five_Obstacle_DS_perturbed.csv",
              "robot_demonstrations/3D_Shapes/Recording_Five_Obstacle_DS.csv"]


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", required=True, help="full S2NNDS checkout")
    ap.add_argument("--dst", default=os.path.join(ROOT, "data", "S2NNDS_extra"))
    a = ap.parse_args(argv)
    for dataset, shape in DEFAULT:
        m = os.path.join("models_verified", dataset, shape, "model_f.pth")
        c = os.path.join("config_files", dataset, f"{shape}_config.json")
        for rel in (m, c):
            if not os.path.exists(os.path.join(a.src, rel)):
                print(f"missing in source: {rel}", file=sys.stderr)
                return 2
            cp(os.path.join(a.src, rel), os.path.join(a.dst, rel))
        if dataset == "2D_Shapes":
            cp(os.path.join(a.src, "Datasets", "2D_Shapes", shape), os.path.join(a.dst, "Datasets", "2D_Shapes", shape))
        elif dataset == "3D_Shapes":
            f = os.path.join("Datasets", "3D_Shapes", f"3D_{shape}.mat")
            cp(os.path.join(a.src, f), os.path.join(a.dst, f))
    for rel in RECORDINGS:
        if os.path.exists(os.path.join(a.src, rel)):
            cp(os.path.join(a.src, rel), os.path.join(a.dst, rel))
    cp(os.path.join(ROOT, "vendor", "S2NNDS", "NNModels.py"), os.path.join(a.dst, "NNModels.py"))
    print(f"prepared {a.dst}\nset S2NNDS_DIR={os.path.abspath(a.dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
