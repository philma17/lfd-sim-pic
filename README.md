# lfd_sim_pic

First prototype of Passive Interaction Control with Dynamical Systems (PIC-DS, Kronander & Billard,
RA-L 2016) as an execution mode in MuJoCo: **FR3 only, 3D TCP position task, LASA/Sine**, compared
with the baseline joint position servo. No UR5e, no orientation control, no sweeps, no multistart.

`/home/philipp/lfd_baseline` is only a reference (commit `a06ed55`, never modified). S2NNDS
(`~/S2NNDS`, read-only) and `~/lfd_env` are reused, not copied. No git repo, branch or worktree here.

## Files

| path | origin |
|---|---|
| `common/{__init__,ds,ik,metrics,robots,scenario,sim,sim_dynamic}.py` | copied verbatim from the baseline, hashes in `common/VENDORED.txt` (checked by a test) |
| `pic/controller.py` | new: coupling functions, damping matrix, energy tank, force law (pure numpy) |
| `pic/sim_pic.py` | new: MuJoCo loop (torque interface, clipping, TCP pulses, logging) |
| `configs/picds.yaml` | new: all parameters with a source tag per value |
| `experiments/run_pic.py` | new: servo vs. PIC vs. PIC + pulse, `results/pic/`, `figures/pic/` |
| `tests/test_pic.py` | new: 19 validations |

Run (`-B` and `sys.dont_write_bytecode` keep `__pycache__` out of the tree):

    cd ~/lfd_sim_pic
    ~/lfd_env/bin/python -B -m unittest discover -s tests -v     # ~40 s
    ~/lfd_env/bin/python -B experiments/run_pic.py                # ~1 min

## Control law as implemented (checked against the paper PDF and the reference repo)

    D = Q diag(l1,l2,l3) Q^T, e1 = f/|f|          [P] Eq. 3  (previous basis kept if |f| < eta)
    F = -D xdot + l1 (f_C + beta_R(z,s) f_R)      [P] Eq. 13
    z = xdot . f_R,  sdot = alpha(s) xdot^T D xdot - beta_s(z,s) l1 z      [P] Eq. 10
    tau = g(q) + Jp^T F - d_q (I - Jp^+ Jp) qd    (null-space term only if null(Jp) is not empty)

Decomposition: `none` only (f_C = 0, f_R = f). `linear`/`lyapunov` raise `NotImplementedError`.
`f = v_peak E f_n(x_n) - k_perp d_perp n`, exactly as `sim.py` (the out-of-plane term is part of f).

**Differences between my draft, the paper and the reference code (verified, not assumed):**

1. Coupling functions: I use paper Appendix A (Eq. 25-28, quintic step). The reference code uses a
   cosine step, `alpha(0) = 0`, a simpler `beta_R` (no upper band), and a dead band `|z| < 0.01`.
   None of that is used, except the cosine step as option `step: cosine`.
2. The PDF text drops over/underlines. I read the underlined `s` in Eq. 27/28 as the lower bound 0
   (needed for Eq. 12 and consistent with the reference code). Test `test_conditions_eq_11_12_14`
   checks Eq. 11, 12 and 14 numerically on a grid for both step types.
3. **Full tank deadlock.** Eq. 12 and Eq. 14 force `beta_R(0, s_bar) = 0`. With `f_C = 0` and
   `s(0) = s_bar` the arm cannot leave rest (measured: zero motion for 10 s). The paper's
   experiment uses an extracted `f_C`, which avoids this. Hence `s(0) = 5 J` (Theorem 1 only
   needs `0 < s(0) <= s_bar`). This is a deviation from "s(0) = 10" in the paper text.
4. The paper values for `delta_z` (0.01) are in (m/s)^2. Here `|f| <= 0.05 m/s`, so `|z| <= 0.0026`
   is far below `delta_z`; the notch of `beta_R` around `z = 0` is wide compared to the signal. It
   only matters when the tank is nearly full, which does not occur in these runs.

## Parameter sources (`configs/picds.yaml`)

| value | source |
|---|---|
| lambda = (20, 200, 200) N s/m, s_bar = 10 J, delta_s = 0.1 s_bar, delta_z = 0.01 | paper Sec. IV-A / Appendix A |
| eta = 1e-6 | reference code `MINSPEED` |
| task cell, DS, L = 0.10 m, goal (0.5, 0, 0.5), dt = 0.002, T_max = 10 s, success < 1 % L = 1 mm, k_perp, start IK | baseline `dynamic_baseline.yaml` (pos mode, mean demo start x0) |
| FR3 torque limits 87/12 Nm | baseline `TAU_MAX_DOC` (franka_description), clipped manually |
| s(0) = 5 J | **chosen**, see deviation 3 |
| d_q = 5 Nms/rad (null-space damping) | **chosen, unvalidated.** Paper has no null-space term. Orthogonal projector, dissipative by construction. |
| pulse 5 N = (0, 3, 4) N, 0.05 s at t = 1 s | **chosen, unvalidated.** Impulse 0.25 N s: expected displacement up to impulse/lambda1 = 12 mm along f, 1 mm across. Measured peak deviation from the undisturbed run: 5.7 mm. |

The start configuration equals the baseline FR3 pos start (checked against the baseline `metrics.json`),
and the baseline servo run reproduces the stored baseline error (`2.0e-7 m`, `t_reach 6.21 s`).

## Results (`results/pic/metrics.json`, `figures/pic/fig_overview.png`)

| plant | controller | final error | t_reach (<1 mm) | geom. dev. max | peak applied tau/limit |
|---|---|---|---|---|---|
| unmodified Menagerie model | servo | 0.0002 mm | 6.21 s | 5.7 mm | 0.31 |
| | PIC | **94.1 mm, no convergence** | - | 1.0 mm | 0.22 |
| | PIC + pulse | 94.1 mm | - | 1.2 mm | 0.22 |
| DIAGNOSTIC: dof_frictionloss = 0 | servo | ~0 | 6.13 s | 5.2 mm | 0.29 |
| | PIC | 0.001 mm | 6.92 s | 1.7 mm | 0.30 |
| | PIC + pulse | < 0.001 mm | 6.80 s | 2.3 mm | 0.30 |

No torque clipping, no joint limit violation, no contacts, tank never clamped, in all PIC runs.
Servo and PIC differ in what they do: the servo tracks the open-loop time-indexed ideal rollout
less closely in shape (5-6 mm), PIC follows the path shape closer (1-2 mm) but lags in time.

### Gains and friction (main limitation)

The Menagerie FR3 has Coulomb joint friction (`dof_frictionloss` 1.137 / 0.763 / 0.44 / 0.248 Nm).
With the paper gains the TCP force is `l1 |f| <= 1 N`, which is below stiction, so the arm does not
move. `lambda1 = 200` (one diagnostic run, not kept) moved it but stalled at 68 mm. The paper
itself says lambda1 was set to the minimum that overcomes static joint friction. Even a large
lambda1 would stall a few mm from the goal, because `|f| -> 0` there. With friction switched off
in memory the controller converges (table). This is a property of the plant/gain combination, not
of the control law, but it is **not solved here**.

## Passivity: what is and is not claimed

No passivity guarantee is claimed. Theorem 1 of the paper is continuous-time with exact gravity
compensation, a rigid-body model, unconstrained torques and a joint input `tau_e`. This
implementation has explicit-Euler tank integration, a one-step control delay, MuJoCo's implicit
integrator, torque clipping (never active in the runs shown), Coulomb friction, a null-space
term outside the proof, and the proof for `f_C = 0` relies only on the tank. What is checked:

- coupling functions satisfy Eq. 11, 12, 14 (grid test),
- the tank stayed in `[0, s_bar]` without clamping (counted: `tank.n_clamp_steps`),
- `W = KE + s` is logged. In the undisturbed run its per-step increase is at most 8e-8 J
  (`energy.W_dW_max_J`, a numerical diagnostic of Eq. 17, not a proof),
- with `f = 0` kinetic energy decreases monotonically from an initial TCP speed of 0.2 m/s.

## Interface notes

- `mjDSBL_ACTUATION` set on `robot.model.opt` and restored in `finally`; torques via `qfrc_applied`.
- The pulse is applied with `xfrc_applied` at the body centre of mass plus the moment `(p_tcp - xipos) x F`.
  Test: identical `qacc` to `qfrc_applied = Jp^T F`.
- `g(q)` = `sim_dynamic.gravity_torque` of the nominal model (same as the simulated model here, so exact).
- Clipped and unclipped torques are separate arrays (`tau_applied`, `tau_cmd`) and counters.

## Open questions

1. Friction: accept a friction-free diagnostic plant, add friction feed-forward (outside the paper,
   chatter near zero velocity, affects the passivity argument), or raise lambda1 (stall radius stays)?
2. `s(0)`: keep 5 J, or implement `linear`/`lyapunov` decomposition (an `f_C` removes the deadlock)?
3. Null-space damping `d_q` and the pulse size/direction are my choices. Fine, or do you have values?
4. Orientation is uncontrolled (damped in the null space only). Check the drift before adding PD.
5. Servo vs. PIC under the pulse: the baseline `sim_dynamic.run` has no `xfrc` hook, so the servo
   was not pushed. Would need a copy of that function (vendored files stay unchanged).
6. Success criterion: 1 mm at T = 10 s is taken from the baseline. PIC reaches it at 6.9 s on the
   diagnostic plant, but it is a position error of a time-free DS, not time-synchronous tracking.
