# Step 1 — Stall diagnosis, PIC-DS on the FR3 (friction x1, nominal)

Reproduce: `python -B experiments/diagnose_stall.py` → `results/summary/stall_diagnosis.json`
(single start, seed 0, `T_max` 10 s unless stated; labels at the top are used throughout; everything except λ1 as in `configs/picds.yaml`).
Labels: **[V]** verified by a run/test in this session, **[H]** hypothesis, **[N]** not checked.

## 1a. Paper equation → code, and parameters

| Paper (RA-L 2016) | Code |
|---|---|
| Eq. 3 `D = Q Λ Qᵀ`, `e1 = f/|f|` | `pic/controller.py:9` (doc), `DampingBasis.update` `:131-140` (basis, kept if `|f| < η`), `damping_matrix` `:142` |
| Eq. 5 | **[N]** not referenced anywhere in code or docs and the PDF was not accessible from this environment; no mapping claimed |
| Eq. 10 tank `ṡ = α(s) ẋᵀDẋ − β_s λ1 z`, `z = ẋ·f_R` | `pic/controller.py:204` (z), `:211` (sdot), `:212` (explicit Euler), `:213-219` (clamp + count) |
| Eq. 13 force `F = −Dẋ + λ1 (f_C + β_R f_R)` | `pic/controller.py:209`; torque `τ = g + JpᵀF + τ_null` at `pic/sim_pic.py:228` |
| Eq. 11, 12, 14 (coupling conditions) | grid test `tests/test_pic.py::test_conditions_eq_11_12_14` (exists, passes **[V]**) |
| Eq. 25-28 (quintic step, α, β_s, β_R) | `pic/controller.py:41`, `:75`, `:79`, `:87` |

| Parameter | Value | Source (from `configs/picds.yaml` tags) |
|---|---|---|
| λ1, λ2, λ3 | 20, 200, 200 N·s/m | [PAPER] Sec. IV-A, **a different robot (Kuka)**; not validated for the Menagerie FR3 |
| η | 1e-6 | [REPO] `MINSPEED` of the reference implementation |
| s0 | 5 J | **CHOSEN** (paper text: 10 J = s_max → deadlock for f_C = 0, reproduced by a test) |
| s_max | 10 J | [PAPER] Sec. IV-A |
| δs | 0.1·s_max = 1 J | [PAPER] Appendix A |
| δz | 0.01 (m/s)² | [PAPER] Appendix A |
| decomposition | none (f_C = 0, f_R = f) | config comment cites paper Sec. III-D as allowing it **[N: not re-read]** |
| null-space damping d_q | 5 N·m·s/rad | **CHOSEN, unvalidated** |

## 1b. State at t = 0 (qd = 0, nominal friction x1) **[V]**

`|f| = 0.0117 m/s` at the start (the DS starts slowly), tank `s = 5 J`, `β_R = 1`, torque clipping 0.

| joint | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| `|τ_drive|/frictionloss` at λ1 = 20 | 0.083 | 0.003 | 0.089 | 0.003 | 0.040 | 0.007 | 0.000 |

`τ_drive = Jpᵀ λ1 f` (gravity is compensated exactly by the nominal model). Every joint is far below its Coulomb limit
(frictionloss 1.137/1.137/1.137/1.137/0.763/0.44/0.248 Nm). **Cause at t = 0: static (Coulomb) friction.**
Not saturation (0 clipped steps, peak commanded 0.22 of the limit), not an empty tank (`s = 5 J`, `β_R = 1`), not the termination criterion.
Measured with MuJoCo's soft `frictionloss`: the arm *creeps* (5.5 mm TCP travel in 10 s at λ1 = 20) instead of being perfectly stuck.

## 1c. λ1 sweep, friction x1 **[V]** (λ1* is the predicted breakaway threshold)

Predicted from the start state: `λ1* = min_i frictionloss_i/|(Jpᵀ f)_i| = 224.9` (joint 3; joint 1: 240.6, joint 5: 501.9; joints 2, 4, 6: > 3000).
Explicit-damping stability `ρ = max|eig(dt·Jp M⁻¹ Jpᵀ D)|` along the path (stable if < 2; a necessary check, not a proof) and clipping:

| λ1 | converged (10 s) | final error [mm] | TCP travel [mm] | `|ẋ|/|f|` first 3 s | ρ_max | clipped steps | n_dW>0 |
|---|---|---|---|---|---|---|---|
| 20 | no | 94.1 | 5.5 | 0.03 | 0.10 | 0 | 1 |
| 40 | no | 95.2 | 16.9 | 0.06 | 0.10 | 0 | 1 |
| 80 | no | 103.7 | 44.7 | 0.11 | 0.11 | 0 | 1 |
| 160 | no | 85.4 | 65.7 | 0.35 | 0.12 | 0 | 1 |
| 320 | no | 46.4 | 194.3 | 0.46 | 0.18 | 0 | 1 |
| 640* | no | 100.5 | 270.9 | 0.78 | 0.36 | 0 | 1 |
| 1280* | no | 96.7 | 276.6 | 0.94 | 0.73 | 0 | 1 |

\* extension beyond the requested sweep. With `T_max = 40 s` (diagnostic duration only): λ1 = 160/320/640/1280 → 141.6/147.8/168.4/205.0 mm, still not converged.

Findings:
- **[V]** Breakaway is gradual, not a step: the tracking ratio `|ẋ|/|f|` rises monotonically with λ1 (0.03 → 0.94). The predicted `λ1* = 225` is **not** visible as a sharp transition in the measurement (soft constraint creep below it, partial motion above it). The prediction is therefore only a *lower-bound estimate for the onset of sliding of the first joint*.
- **[V]** Damping stays stable for all λ1 tested (ρ_max ≤ 0.73 < 2), and there is **no torque saturation** (0 clipped steps, peak commanded torque < limit).
- **[V]** No tested λ1 ≥ λ1* converges, neither in 10 s nor in 40 s. At λ1 = 320 (40 s run) the arm ends at rest with *all* joints below their stiction limit again (`|τ_drive|/frictionloss` = 0.05/0.96/0.07/0.12/0.01/0.61/0.00), i.e. it is held by Coulomb friction again, further along the path where `|f|` or the joint-space projection is smaller; at λ1 = 1280 joints 2 and 6 sit exactly at their limit and the TCP creeps at ~3 mm/s.
- **[H]** Mechanism: joint-space Coulomb friction acts as a dead band on the joint torques; the resulting TCP motion deviates from `f`, the deviation feeds back through the DS evaluated at the actual position, and the arm settles where `Jpᵀ λ1 f` falls below the local stiction limits. Not isolated by a separate experiment.
- **[N]** λ2 = λ3 = 200 were left at the paper values (only λ1 was swept as requested). Whether larger perpendicular damping contributes was not tested.

## 1d. Energy check **[V]**

| run | success | n_dW_positive_steps | sum of positive dW [J] | max dW [J] | tank s range [J] |
|---|---|---|---|---|---|
| friction x0 (the converged run), λ1 = 20 | yes, 6.924 s | 5 | 1.66e-7 | 8.1e-8 | 4.989 … 5.0 |
| friction x1, λ1 = 20 (stalled) | no | 1 | 4.2e-9 | (see JSON) | 4.998 … 5.0 |

These are numerical diagnostics of the discrete tank bookkeeping, **not** a passivity proof (README, "Passivity").

## 1e. MJCF commits `cda5236` (vendored) vs `d7b481b` (baseline) **[V]**

`diff -rq` of the whole `franka_fr3/` directory: **identical** (all files). frictionloss `[1.137, 1.137, 1.137, 1.137, 0.763, 0.44, 0.248]`,
damping 0.21 for all joints, armature `[0.195 ×4, 0.074 ×3]` in both. The earlier open question "does the model differ from the baseline's" is closed for the FR3.

## D. Do the paper defaults explain the stall?

- **[V]** The *initial* stall is explained by λ1 = 20: the start drive is 8-9 % of the stiction limit; ~225 N·s/m would be needed.
- **[V]** s0 does not cause it: `β_R = 1` during the whole nominal run, tank never empty (`s_min` ≥ 3.3 J in all sweeps).
- **[V]** The paper defaults do **not** explain the failure to converge: with λ1 ≥ λ1* the arm moves but still does not converge (above).

## Decision: base condition for all further experiments

**Friction x0 (DIAGNOSTIC plant), paper gains λ = (20, 200, 200), s0 = 5 J.** Reason: friction x1 with λ1 ≥ λ1* (stable, 320…1280) did not converge in
any tested case, so no nominal-friction base condition exists without changing the algorithm or other gains, which was ruled out. All results
that use this base are diagnostic-plant results and are labelled so; nominal-friction (x1) runs are kept only for the servo and as the documented PIC-DS failure.

---

## Addendum (2026-10-27): UR5e and the gains of the real-robot recordings

**UR5e, same derivation (`experiments/diagnose_stall.py --robot UR5e`, SYNTHETIC friction = same fraction of the torque limit, see `pic/plant.py`) [V]:**
`λ1* = 414.8` (first joint), start drive/frictionloss ≤ 0.048. λ1 = 20…1280: none converged in 10 s; `T_max = 40 s` with λ1 = 160/320/640/1280: 90.3/172.4/151.4/145.0 mm, none converged; stability ρ_max ≤ 0.73, no clipping. Same pattern as the FR3
(file `stall_diagnosis_UR5e.json`). Because the UR5e friction is an assumption, this is weaker evidence than the FR3 result. **Base condition for both robots: friction x0, paper gains.**

**Recorded real-robot gains [V, file header]:** the S2NNDS robot recordings (`Robot_Data/2D_Shapes/Five_Obstacle_DS.csv`,
`robot_demonstrations/3D_Shapes/Recording_Five_Obstacle_DS.csv`) store `eigen_0 / eigen_1 = 150 / 300` and `200 / 400`, `max_tank_level = 100000`,
`translational_stiffness = [0, 0, 500]`, `rotational_stiffness = [80,80,80]` resp. `[50,50,50]`, `damping_ratio = 1.414`.
The recorded λ1 (150-200) is of the same order as the predicted FR3 breakaway threshold (≈ 225) and ~10x the paper value 20. **[H]** this is
why the paper value does not move the simulated arm; it was **not** used to tune anything here (no λ was changed outside the stall sweep). The recorded
control law and units were not verified to be identical to the implemented one (e.g. the recording also uses a z-stiffness and a huge tank).

---

## Addendum 2 (2026-10-27): final distance over λ1 (log-log), friction x1

Reproduce: `python -B experiments/final_distance_vs_lambda.py` and `... --start-norm 0.3`.
Figures `stall_final_distance.png`, `stall_final_distance_near0.3.png`; data `stall_final_distance*.json`. 13 values λ1 = 20…1280 (factor √2), both robots,
T_max 10 s (**the success criterion, unchanged**) and 40 s (extra information). UR5e friction is SYNTHETIC. Only λ1 changes.

**Expectation tested:** final distance ∝ 1/λ1 (Coulomb offset at standstill). **Result [V]: not observed.**
- Demo-mean start, T_max 10 s: final distance is flat and non-monotonic (FR3 94 → 105 → 46 (λ1 = 320) → 97 mm; UR5e 94 → 42 (λ1 = 452) → 114 mm); log-log slope −0.02 (FR3) / −0.015 (UR5e) instead of −1. Converged runs: **0 of 13 per robot** (criterion). At T_max 40 s the slope is +0.20 / +0.15 (the arm drifts further away).
- Reason it is not a standstill offset [V]: at the end the TCP is mostly still moving or re-stalled at a *different place on the path* (`|f_end|` 0.011-0.05 m/s; `λ1·|f_end|` grows from 0.7 N to 12-16 N for λ1 = 20 → 450), i.e. the stopping point is not set by `λ1 |f| = friction` at a point close to the goal. The 1/λ1 law would need the arm to arrive near the goal first.
- Extra measurement starting ON the ideal path at ‖x_n‖ = 0.3 (30 mm, same DS path): criterion at T_max 10 s still 0/13 per robot. At **T_max 40 s** the final distance drops by orders of magnitude for large λ1 (FR3 λ1 ≥ 640: converged, 3/13 runs; 452: 4 mm; UR5e λ1 = 1280: converged, 1/13), with *no* 1/λ1 trend below the threshold (22-157 mm). This is a threshold-like transition, not a power law.

**[H]** There is a λ1 threshold (about 2-3 × the predicted breakaway λ1\*: FR3 ≈ 450-640 vs 225, UR5e ≈ 900-1280 vs 415) above which the arm can finish the path; below it joint friction re-stalls it somewhere along the path. Not isolated by a controlled experiment; single start; not used to change the base condition (success criterion not met at T_max 10 s from the demo start).
