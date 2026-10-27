# Report — FR3 vs UR5e under physical variations (branch `pic-physics`, 2026-10-27)

Scope: LASA/Sine, 3D TCP position task with orientation `none | hold`, one start (demo mean, seed 0) unless
stated, **friction x0 diagnostic plant**, paper PIC gains λ = (20, 200, 200), s0 = 5 J. Single runs are observations,
not statements about generality. Tables/figures: `break_matrix.{csv,png}`, `matrix_long.csv`, `alpha_starts.csv`,
`cmp_{alpha,placement,com_inertia}_{hold,none}.png`, `stall_diagnosis.md`, `datasets_nominal_*.csv`.

## Verified (run or test in this session)
- **Base condition:** friction x1 stalls PIC-DS at the start (drive 8 % of stiction, λ1\* ≈ 225 predicted) and **no tested λ1 ≥ λ1\*
  (320…1280) converged**, stable (ρ ≤ 0.73), no clipping → base = friction x0 for both robots. UR5e friction is SYNTHETIC. FR3 MJCF `cda5236` ≡ `d7b481b`.
- **Runner:** robot, dataset/shape/start, orientation, plant (friction, damping, gravity, mass, inertia, COM, torque limit) from the config.
  FR3 + UR5e execute servo and PIC-DS on LASA/Sine, Five_Obstacle_DS (2D, unsafe-set clearance), Cshape_bottom (3D); all PIC runs converged,
  UR5e servo on Cshape_bottom ended `diverged` (not analysed). Historic FR3 numbers are reproduced (regression test).
- **Tests:** 59 run. With the extra S2NNDS data prepared: 57 pass, 1 skip (baseline results file), **1 fail (known vendored-hash mismatch of `ds.py`/`robots.py`, unchanged)**; without them 54 pass, 4 skip.
- **Matrix:** 312 runs (2 robots x 2 orientation modes x 7 variation configs + α with 5 starts), 0 execution errors, 0 obstacle penetrations,
  torque saturation in 1 run (limits were not binding; converged runs peak at 0.59 of the limit).
- **α ∈ {0.5,1,2,4}:** PIC-DS converged in all 4 combos (single start) and for 5/5 starts in each of the 16 (robot, orientation, α) cells. The servo diverged at α=4 (all combos, 0/5 starts), at α=2 on the UR5e
  (hold 0/5, none 1/5), and in one UR5e α=1 start. In the 2 diverged servo cases I inspected (UR5e α=2 hold, FR3 α=4 hold) the TCP left the model's training domain (x_n2 > 1 after 2.8 s / 1.4 s); no joint/torque limit was violated in any servo run of the matrix.
- **Mass (all links) with nominal g-compensation:** PIC-DS `diverged` at ×0.5, ×1.5, ×2 in all 4 combos (joint position/velocity limit first in 6 of 12), the servo converged. With the **true** model
  in the compensation (`gravity_comp: plant`) all PIC-DS runs converged → the compensation mismatch is the necessary cause here.
- **COM ±5 cm (last link, nominal comp):** PIC-DS with `orientation: none` stalled/slow (both robots, 5 of 6), with `hold` converged; true-comp: all converged. Servo converged.
- **Inertia ×0.5…×2** (masses unchanged): nothing broke.
- **Placement:** r = 0.3 and 0.5 m converge everywhere. r = 0.8 m: FR3 `hold` stalls (servo and PIC, 47 mm; σ_min(J6)=1e-4, cond(J6)≈3·10⁴), UR5e `hold` PIC `slow` (cond(J6)≈6·10⁴), servo converges; `none` converges. (Baseline's kinematic grid also marks FR3 full-pose r=0.8 singular.)
- **Orientation `hold` (nominal placement):** max orientation error 0.0012 (FR3) / 0.0015 rad (UR5e) vs 0.195 / 0.068 rad for `none`; null dimension FR3 1, UR5e 0.
- **Energy bookkeeping (incl. spring energy):** converged PIC runs: max per-step W increase ≤ 1e-4 J; non-converged: up to 0.08 J. Numerical diagnostic only, no passivity claim.

## Hypotheses (consistent with the data, not isolated)
- Servo divergence at high α is caused by servo lag leaving the learned model's domain.
- Joint Coulomb friction re-stalls the arm along the path for λ1 ≥ λ1\* (stall sweep, `T_max` 10 and 40 s).
- `hold` rescues the COM-mismatch cases because the orientation spring absorbs the wrist moment.

## Not verified / not done
- Windows execution; repeatability beyond 1 seed/start (α has 5 starts only); statistics.
- Nominal-friction PIC-DS (no working gain found), UR5e friction value (**open question** for the supervisor: relative-to-limit vs absolute, `plant.synthetic_friction`), K_r = 50 Nm/rad as isotropic world-frame stiffness (taken from a recording header).
- Acceleration/jerk limits (no verified documented values here); servo `qacc` is derived, not logged; torque-limit sweep only for the FR3 (`torque_limit.yaml`, earlier batch).
- Stretch items (`null_d_q` > 0 study, passive rotational PD for full6) were not done.

## Next steps
1. Decide on the UR5e friction convention and the provenance fix (`VENDORED.txt` "local patches").
2. Repeat key cells with more starts/seeds; isolate the servo-divergence mechanism (lag vs. model domain).
3. Torque-limit sweep for both robots on the base condition; accel/jerk limits from a verified source.
