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


---

# Addendum (2026-10-27, second request): stall law, extreme speeds, UR5e Cshape servo, torque limits

Same base condition (friction x0, paper gains), single start, seed 0 unless stated; matrix re-run once with the new columns
(456 runs, 0 errors); the 312 earlier cells reproduce **exactly** (same classes, identical final distances).

## Verified
- **Final distance vs λ1 (friction x1):** *not* ∝ 1/λ1 — flat/non-monotonic, log-log slope ≈ 0 (details and near-goal variant: `stall_diagnosis.md` Addendum 2). Success criterion unchanged: 0/13 converged per robot at T_max 10 s.
- **α up to 16, L ∈ {0.1, 0.2, 0.3} m** (`speed_L_torque_binding.csv/.png`, 120 runs): PIC-DS converged in **all 60** (2 robots × 2 orientation modes × 15), servo in 18 of 60 (diverged in 38, slow in 4 = UR5e hold). The documented torque limit binds in only **1 of 60 PIC runs**: FR3 α=16, L=0.3, first at joint 5 after 1.06 s (peak commanded 1.30 × limit). UR5e PIC never saturates (peak commanded ≤ 0.43 × limit). The servo saturates in 4 runs, all FR3 and all `diverged`: joint 2 (none; α=8/16, L ≥ 0.2; first at 1.1-1.8 s) and joint 5 (hold, α=16, L=0.3, already at 0.078 s, peak commanded 4.2 × limit). Obstacle penetration (clearance < 0) is flagged in two runs: UR5e PIC α=16, L=0.1 (none; converged) and UR5e servo α=2, L=0.2 (none; diverged).
- **Domain exit:** 88 of 89 `diverged` runs in the matrix left the model's training domain [−1,1]² (servo 77/77, PIC 11/12); the one exception is UR5e mass ×0.5 PIC.
- **UR5e servo on Cshape_bottom:** the TCP leaves the training domain after **0.206 s** (axis 3 reaches −1.0; start x_n3 = −0.946, i.e. 5.4 % from the edge), then runs away (final 1371 mm), no joint/torque limit violated. The servo sags 8.7 mm (= 0.087 normalized) at the start pose without gravity feed-forward (FR3: 5.5 mm = 0.054), more than the 0.054 margin. With the baseline's own option `gravity_ff=True` the exit is delayed to 4.4 s and the final distance is 375 mm (still not converged). So the exit is triggered by the initial sag and a second mechanism remains (not analysed).
- **Torque-limit sweep, both robots** (`torque_limit_sweep.csv`; ×1, ×0.5, ×0.25 of 87/12 resp. 150/28 Nm): UR5e converged everywhere (peak commanded 0.43-0.47 of the *scaled* limit at ×0.25, no saturation). FR3 `none` ×0.25: both PIC and servo `saturated` (joint 2 first, at 3.6 s resp. 3.3 s), ×0.5 and ×1 converge. FR3 `hold` ×0.25 converged, but only just: peak commanded 0.975 of the scaled limit (no saturation).
- **Columns in `matrix_long.csv`:** Savitzky-Golay jerk (TCP and joint, peak and after the 0.1 s start-up window), SG joint acceleration, and **direct MuJoCo `qacc`** for PIC *and* servo (servo via a recording wrapper around `mj_step`, bit-identical states, test). Over converged runs the SG acceleration peak is 0.17 / 0.61 / 0.93 (p10 / median / p90) of the direct `qacc` peak, i.e. SG smoothing underestimates peaks; jerk values are indicators only.

## Hypotheses (not isolated)
- High-α servo failures: servo lag pushes the TCP out of the learned model's domain (association verified, cause not).
- FR3 `hold` survives ×0.25 because its full-pose start posture needs less joint-2 holding torque (not checked).
- The λ1 threshold (≈ 2-3 × λ1\*) above which friction-x1 PIC finishes the path within 40 s (see stall addendum).

## Not verified / not done
- Windows, more seeds/starts for the α×L and torque-limit sweeps (single start each), accel/jerk limits (no verified documented values), statistics.
- UR5e friction stays "relative" (SYNTHETIC) as instructed; no check against a real UR5e.
