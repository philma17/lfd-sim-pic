# Research plan

## Research question and scope

How do physical robot properties and actuator constraints affect execution of a learned dynamical system, and under which conditions does execution become inaccurate, slow, dynamically infeasible, or unable to converge? The study uses learned DSs as the reference behavior and compares execution strategies only as instruments for isolating plant and actuator effects.

The current scope is a validated FR3, LASA/Sine, 3D TCP-position cell. Orientation, UR5e, other model families, and broad scenarios are later stages, not current evidence.

## Conceptual separation

1. **Ideal DS rollout:** integrates the learned vector field without a robot. It tests the model/reference and its numerical integration.
2. **Kinematic execution:** maps a desired task-space motion through IK or joint velocity commands. It reveals geometry, conditioning, limits, and discretization, but not physical friction, inertia, gravity, or actuator saturation.
3. **Dynamic execution:** advances a MuJoCo plant. The position servo and direct-torque PIC-DS have different actuator interfaces and must be analysed with their plant, gains, and constraints recorded.

For PIC-DS, the current law uses `D = Q Lambda Q^T`, `F = -D xdot + lambda1 beta_R f` for decomposition `none`, an explicit-Euler energy-tank update, gravity compensation, and projected null-space damping. The implementation does not establish the continuous-time theorem under its discrete-time integrator, clipping, friction, or null-space term.

## Hypotheses (not established facts)

- **H1 — execution speed:** increasing `ds_speed_factor` changes execution time and may increase dynamic tracking error without changing the learned vector-field geometry.
- **H2 — friction:** friction and stiction can create a non-convergence region when available task force falls below the plant's resisting torque; the threshold depends on configuration and gain.
- **H3 — actuator limits:** torque saturation increases path/timing error and can convert convergence into failure; the effect is configuration- and task-dependent.
- **H4 — gravity/inertia:** changing gravity, mass distribution, or inertia changes required actuator effort and transient error, but the direction and size must be measured rather than assumed.
- **H5 — conditioning:** proximity to a kinematic singularity amplifies joint commands and may expose different failure boundaries for the servo and PIC-DS.
- **H6 — disturbance response:** a fixed external pulse can change geometric deviation, recovery time, and tank diagnostics; neither controller is assumed in advance to be superior.

## Variables

**Independent variables:** friction scale and stiction model; gravity and configuration; masses, centres of mass, and inertia; torque limits/saturation; `ds_speed_factor`; robot model; initial condition; task geometry; execution strategy; PIC gains and tank initialization where explicitly controlled.

**Dependent metrics:** final goal error, convergence success and time, timeout rate, geometric path deviation separate from time-synchronous error, TCP/joint velocity/acceleration/jerk, commanded versus applied torque, saturation count/duration/magnitude, joint-limit/contact violations, Jacobian singular values/manipulability, tank state, numerical finiteness, and repeatability.

**Controlled variables:** model weights, task definition, timestep, horizon, success threshold, IK/start-selection rule, nominal controller settings, disturbance definition, seed policy, and output naming. Each change must be recorded in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## Experimental design

For each condition, capture a manifest with Git revision, config/model hashes, robot, controller, scenario, seed, initial condition, physical parameters, and actuator limits. Use unique run identifiers and preserve prior results. Run ideal, kinematic, and dynamic controls where applicable. Use fixed initial conditions for paired comparisons, then a separately declared multistart sample for robustness. Repeat conditions across seeds/starts and report failure classes, not only successful means.

Classify failures as model/rollout, IK/conditioning, controller numerical, plant/dynamic infeasibility, actuator saturation, joint/contact constraint, timeout, or infrastructure/data failure. Record partial/interrupted runs as incomplete.

To separate causes, first compare ideal rollouts; then compare kinematic execution with identical geometry; then change one plant/controller variable at a time. Log both commanded and applied torque and distinguish actual saturation from a documented reporting limit. Compare geometric error against time-synchronous tracking error so a time-free DS is not penalized for speed alone.

## Validity criteria

A valid experiment has a fixed declared configuration, reproducible start and seed, complete trajectory/metrics/manifest, finite state, known termination status, and no silent fallback. A conclusion requires repeated conditions, an explicit comparison, an uncertainty or variability report, and a failure classification. A single diagnostic run can motivate a hypothesis but cannot establish a general controller or plant claim.

## Staged roadmap

1. Reproduce and archive the existing baseline and PIC FR3 cell; resolve the vendored provenance discrepancy.
2. Validate dynamic FR3 servo and PIC-DS interfaces and nominal/friction-free diagnostics.
3. Study positive DS speed scaling while holding the vector field and plant fixed.
4. Sweep friction, gravity, mass/inertia, configuration, and initial conditions with repeatability.
5. Sweep actuator limits and quantify saturation and constraint effects.
6. Integrate UR5e only after the FR3 model and metrics are validated.
7. Extend to additional scenarios/models, including Five_Obstacle and 3D tasks, only when assets and loaders are verified.

## Limitations and threats to validity

The current study is position-only, FR3-only, single-cell, and largely single-start. The PIC decomposition is `none`; `linear` and `lyapunov` are unimplemented. The nominal PIC failure may be a gain/plant mismatch rather than a fundamental PIC limitation. The friction-free plant is diagnostic, not a physical baseline. Torque limits are manually clipped in PIC but documented as not enforced by the FR3 Menagerie servo asset. MuJoCo integration, explicit tank integration, finite precision, and one-step control timing complicate theorem transfer. The existing 288-row baseline results are kinematic and do not answer dynamic questions.
