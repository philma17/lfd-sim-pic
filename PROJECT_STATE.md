# Project state

**Last updated:** 2026-10-26
**Latest log entry:** [Current state audit](EXPERIMENT_LOG.md#current-state-audit-2026-10-22)

## Update 2026-10-26 — experiment framework (uncommitted)

- **Corrections to the 2026-10-22 text below:** the committed checkout (`75cbb4e`) does not contain `common/experiment_manifest.py`, `audit/`, `tests/test_provenance.py` or a `ds_speed_factor` implementation (apparently uncommitted local changes at that time); it has 19 tests, now 35. Working directory in the cloud session: `/projects/sandbox/lfd-sim-pic`.
- **New (uncommitted, awaiting approval):** config-driven runner `experiments/run_experiment.py` (+ `compare_runs.py`, `plot_run.py`), `pic/{plant,analysis,plots,experiment}.py`, `configs/experiments/*.yaml`, `tests/test_experiment.py`, `HOWTO_EXPERIMENTS.md`, `requirements-tested.txt`; `pic/sim_pic.py:build_case` extended. Outputs go to `results/runs/` and `results/compare/` (new run directories; `results/pic/` untouched).
- **Support matrix changes:** `ds_speed_factor` (α) implemented and run (0.5/1/2); friction, damping, gravity, mass, inertia and torque-limit scales implemented, unit-tested for physical effect, and run once each (see `EXPERIMENT_LOG.md`, `HOWTO_EXPERIMENTS.md` §8). Not implemented: COM offsets, armature, gravity direction. UR5e: executes in a throwaway test, **not integrated**.
- **Tests:** 35 run; 33 pass, 1 skipped, 1 fails (`test_vendored_files_unchanged`). The cause is now localized: `ds.py` and `robots.py` differ from `a06ed55` only in path defaults (+ `MENAGERIE_COMMIT` constant in `robots.py`). Hashes were not changed.
- **Open:** Windows execution unverified; provenance fix needs approval; baseline-vs-vendored Menagerie revision difference (`d7b481b` vs `cda5236`) not investigated; UR5e integration; repeatability/multistart; `RESEARCH_PLAN.md` unchanged (plan not altered).

## Objective

This project investigates how physical robot properties and actuator constraints affect execution of a learned dynamical system (DS). PIC-DS and the conventional joint-position servo are execution strategies used to expose those effects; the scientific target is not a controller contest. Outcomes of interest are inaccurate or slow execution, geometric deviation, dynamic infeasibility, constraint violations, and failure to converge.

## Repositories and revision state

| Item | Verified value |
|---|---|
| Active repository | `https://github.com/philma17/lfd-sim-pic` |
| Working directory / Git root | `/workspaces/lfd-sim-pic` |
| Branch | `main` |
| Current committed revision | `104107e` (`Integrated Models nd requirements`) |
| Working tree | Dirty before this documentation change; existing edits and generated results are preserved |
| Reference repository | `https://github.com/philma17/LfD_baseline` (read-only) |
| Baseline revision inspected by the audit | `67a2a84995c264fbb52d2ca01d01cf985da17512` (`main`) |
| Vendored provenance revision | `a06ed55463047b9b9c42ff3e0c8075a90d34c999`, recorded in [`common/VENDORED.txt`](common/VENDORED.txt) |
| Local baseline path | `/home/philipp/lfd_baseline` is unavailable in this environment |

The two baseline revisions are not silently reconciled: `67a2a849...` is the audit checkout revision, while `a06ed554...` is the provenance recorded for the copied `common/` files. The baseline was not modified by this task.

## Implemented architecture

- **Learned model and ideal rollout:** [`common/ds.py`](common/ds.py) loads the bundled S2NNDS LASA/Sine PTH model and provides the ideal DS rollout.
- **Scenario/configuration:** [`common/scenario.py`](common/scenario.py) loads [`configs/picds.yaml`](configs/picds.yaml), builds the planar embedding, and applies validated positive `ds_speed_factor` scaling to the DS rate.
- **Kinematic/dynamic baseline:** [`common/sim.py`](common/sim.py) contains shared kinematic execution; [`common/sim_dynamic.py`](common/sim_dynamic.py) runs the MuJoCo joint-position servo and exposes diagnostic dynamics and metrics.
- **PIC-DS:** [`pic/controller.py`](pic/controller.py) implements the task-space damping/coupling law and energy tank; [`pic/sim_pic.py`](pic/sim_pic.py) executes direct joint torques through `qfrc_applied`, with gravity compensation, null-space damping, optional TCP force pulses, torque clipping, and logging.
- **Experiment entry point:** [`experiments/run_pic.py`](experiments/run_pic.py).
- **Reproducibility support:** [`common/experiment_manifest.py`](common/experiment_manifest.py) records Git revision, config/model hashes, controller, robot, and seed.
- **Audit data:** [`audit/repository_inventory.json`](audit/repository_inventory.json) and [`audit/scenario_coverage.csv`](audit/scenario_coverage.csv).
- **Tests:** [`tests/test_pic.py`](tests/test_pic.py).

## Current support matrix

| Dimension | Implemented / verified | Not implemented or unverified |
|---|---|---|
| Robot | FR3 MuJoCo Menagerie asset | UR5e dynamic asset/validation |
| Task | 3D TCP position | Full-pose/orientation control |
| Learned model | Bundled LASA/Sine PTH model | Other model assets and broad scenario coverage |
| Execution | Position servo and direct-torque PIC-DS | PIC decompositions `linear` and `lyapunov` |
| Plant studies | Friction scale 0 diagnostic and 1 nominal stored runs | Systematic parameter sweeps, multistart campaigns, statistical repeatability |
| Disturbance | One configured 5 N TCP pulse | Comparable servo pulse campaign |

The current PIC configuration is FR3, LASA/Sine, `dt=0.002 s`, `T_max=10 s`, `lambda=(20,200,200) Ns/m`, `s_max=10 J`, `s0=5 J`, quintic coupling, `d_q=5 Nms/rad`, documented FR3 torque limits of `(87,87,87,87,12,12,12) Nm`, and `ds_speed_factor=1.0`. The null-space damping and `s0=5 J` are chosen deviations requiring scientific validation. The nominal friction scale is `1.0`; scale `0.0` is explicitly diagnostic.

## Existing evidence

[`results/pic/metrics.json`](results/pic/metrics.json) and the corresponding trajectory/figure files are pre-existing working-tree artifacts. The stored single-start results show:

- Nominal Menagerie friction: servo converges (`0.0002 mm`, `6.21 s`); PIC and PIC+pulse do not converge within 10 s (`94.1 mm` final error).
- Friction-free diagnostic: servo converges in `6.13 s`; PIC in `6.92 s`; PIC+pulse in `6.80 s`.
- Stored PIC runs report no torque clipping, joint-limit violations, or contacts; this is evidence for those runs only.

The baseline audit reports 288 completed **kinematic** Phase-1 rows: 259 `OK`, 15 `FEHLER_SINGULAER`, 8 `UNGUELTIG_START`, and 6 `FEHLER_LIMIT`, across FR3, FR3-null, and UR5e position/full-pose LASA/Sine placements. These rows cannot establish effects of friction, inertia, gravity, actuator saturation, or PIC passivity.

## Validation status

**Tested:** `python -B -m unittest discover -s tests -v` ran 21 tests: 19 passed, 1 skipped, 1 failed. The skip is the unavailable baseline-results comparison. The failure is `TestProvenance.test_vendored_files_unchanged`: the current SHA-256 of [`common/ds.py`](common/ds.py) is `dc368c53...`, while [`common/VENDORED.txt`](common/VENDORED.txt) records `39466697...`. This is unresolved and is not labelled pre-existing without further investigation.

**Tested:** `git diff --check` passed before this documentation change.
**Unverified:** formal passivity, discrete-time stability, broad repeatability, speed-scaling conclusions, and all unsupported scenarios.

## Priority next actions

1. Resolve the vendored `common/ds.py` provenance discrepancy by comparing the intended baseline revision, without overwriting current work.
2. Make the nominal FR3 PIC friction/gain limitation a controlled experiment with captured manifests and repeatable initial conditions.
3. Run a preregistered `ds_speed_factor` study, separating geometric path error from time lag and convergence.
4. Add controlled friction, gravity, inertia, torque-limit, and initial-condition sweeps with commanded/applied torque and saturation metrics.
5. Validate dynamic FR3 results and only then integrate a UR5e asset and extend scenarios.
