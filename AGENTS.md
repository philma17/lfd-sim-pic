# Instructions for coding agents

## Research integrity

- Never fabricate results, test outcomes, dates, seeds, or experimental evidence.
- Label statements as hypotheses, observations, or conclusions.
- Do not silently change parameters or tune after seeing results. Record every parameter change and its reason.
- Separate diagnostic runs from nominal or preregistered controlled experiments.
- Never interpret kinematic results as dynamic evidence.
- Do not claim formal passivity, stability, safety, or convergence guarantees without checking the applicable assumptions for this discrete implementation.

## Repository protection

- `lfd-sim-pic` is the active development repository.
- `LfD_baseline` is read-only. Do not modify, commit to, or push it.
- Inspect Git status, branch, revision, and existing changes before editing.
- Never overwrite, delete, regenerate, or silently reinterpret existing result files without explicit authorization.
- Keep code, configuration, manifests, and [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) consistent.
- Preserve vendored provenance checks; investigate mismatches rather than weakening the test.

## Implementation discipline

- Read existing code and reuse its helpers and naming before adding abstractions.
- Make small, coherent changes and add/update tests for behavior changes.
- Run targeted tests first, then the full suite when practical; report exact commands and outputs.
- Investigate failures instead of suppressing, skipping, or swallowing them.
- Avoid hidden fallbacks, silent tuning, broad exception handling, and partial-success reporting.
- Do not modify source code or results for a documentation-only task.

## Experiment reproducibility

- Each run must capture commit, configuration/model hashes, robot/model, controller, scenario, seed, initial condition, and physical parameters.
- Use unique run identifiers/directories; never overwrite prior runs.
- Distinguish ideal rollout, kinematic execution, and dynamic simulation.
- Record commanded and applied torques separately where possible.
- Record convergence, path geometry, timing, velocity/acceleration/jerk, saturation, constraints, and energy-tank diagnostics where relevant.
- Mark interrupted or partially completed runs as incomplete.

## Handover

- At the end of a task update [`PROJECT_STATE.md`](PROJECT_STATE.md), [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), and [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) when the scientific plan changes.
- Report changed files, commits, tests, experiments, artifacts, and unresolved issues.
- Do not rely on conversational memory as the only project record.
