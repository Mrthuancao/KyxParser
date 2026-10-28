# KYX Parser Repository

This repository contains the source code and related assets for the KYX Parser project. Its primary goal is to parse KYX files and generate Abstract Syntax Trees (ASTs), ultimately turning them into Python gym environment.

## Repository Structure

Below is an overview of the main directories and their contents:

*   **`AST_parser/`**
    *   **Purpose:** This is the most critical folder in the repository, housing the core logic for parsing, AST generation, and related utilities.
    *   **Key Files:**
        *   `ast_config.py` (or `ast_config.ipynb`): Containing PyAST Configuration nodes.
        *   `gym_generator.py` (or `gym_generator.ipynb`): Responsible for generating "gym" environments, based on the parsed PyASTs.
        *   `keymaerax_parser.py`: The main AST parser implementation specifically designed for Keymaerax syntax.
        *   `PyAST_test.py` (or `PyAST_test.ipynb`): Manual end to end testing code for the system from KeymaeraX_file to Gym environment

*   **`documents/`**
    *   **Purpose:** Stores documentation and notes related to the project.
    *   **Contents:** Includes Markdown (`.md`) files explaining how the system works, design decisions, and meeting notes.

*   **`kyxs/`**
    *   **Purpose:** Contains sample or test KYX files.
    *   **Contents:** `.kyx` files that can be used for future testing, development, and validation of the parser.

*   **`sandbox/`**
    *   **Purpose:** A dedicated area for experimental code, proofs-of-concept, or temporary scripts that are not yet integrated into the main codebase.

*   **`test/`**
    *   **Purpose:** Houses all testing-related assets for the system.
    *   **Contents:**
        *   `input/`: Sample input data used for testing.
        *   `output_AST/`: Expected or generated AST outputs for comparison.
        *   `output_PyAST/`: Expected or generated Python AST outputs.
        *   `test_AST.py`: Test scripts specifically for the AST generation process.
        *   `test_PyAST.py`: Test scripts for the Python AST components.

*   **`.venv/`**
    *   **Purpose:** A virtual environment for Python dependencies, ensuring project isolation and reproducible environments.

*   **`_pycache__/`**
    *   **Purpose:** Python's bytecode cache directory, automatically generated.

*   **`archived/`**
    *   **Purpose:** Contains older or deprecated files/folders that are kept for reference but are no longer actively used.

*   **`logs/`**
    *   **Purpose:** Stores log files generated during the execution of various scripts or the system itself.

*   **`README.md`**
    *   **Purpose:** This file, providing a high-level overview and explanation of the repository structure and project.

---

This README aims to provide a quick guide to navigating the repository. For detailed information on specific components, please refer to the documentation within the `documents/` folder or the comments within the source code.

## Quantified Box-Modal Simulation

`AST_parser_2` can generate a simulator for quantified problems containing exactly one
box modality and a hybrid program with one top-level loop at the end of the program.
For example, the `CruiseControlStability.txt` problem's `forall`/`exists` formula is
preserved in the generated config as verification metadata; the quantifiers are not
sampled or treated as a per-step reward or termination condition.

The modal program's prefix (such as cruise-control initialization) runs during
`reset()`. If the loop calls a named `controlFunc` (or `ctrl`), each complete choice
path in that controller is exposed as a discrete Gym action and checked by
`action_masks()` against its guards; the chosen controller path runs before the
remaining loop body. Without such a controller, the environment uses a single no-op
discrete action. Other enabled nondeterministic choices are sampled uniformly, ODE
branches advance by the configured `time_step`, and integration stops at detected
evolution-domain boundaries. When predicates carry `@safety` or `@progress`
annotations, their shaping/reward/penalty and success/failure actions are applied
after each simulated loop iteration; otherwise the reward is `0.0`. Episodes end at
the configured step limit or if the hybrid program has no enabled execution. This is
a simulator for sampling executions, not a proof checker for the quantified stability
claim.

The simulation currently rejects quantified modal formulas with multiple or diamond
modalities, programs without exactly one final top-level loop, recursive hybrid-program
references, nested loops inside a simulation step, and unbounded nondeterministic
assignments.

### Training a Cruise Control Policy

Train and evaluate a masked PPO policy over the enabled `controlFunc` paths with:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_cruise_control.py --timesteps 10000
```

The policy is saved as `models/ppo_cruise_control.zip`; training and evaluation
diagnostics are written under `logs/cruise_control/`. The generated Python
environment is retained at `generated_cruise_control_env.py` for inspection and is
overwritten on the next run; pass `--generated-env <path>` to choose another location.
Evaluation displays a step-level progress bar for the PPO policy and the baseline.
Set the per-episode step limit for both training and evaluation with
`--max-episode-steps` (default: `200`), for example:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_cruise_control.py --timesteps 10000 --max-episode-steps 1000
```

To evaluate the saved policy without training, compare it with the first-legal-action
baseline on identical episode seeds and record mode-transition steps/times:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_cruise_control.py --eval-only --eval-episodes 10 --seed 11 --max-episode-steps 5000 --out-dir logs\cruise_control\horizon_5000
```

This writes policy and baseline episode CSVs, including transition events and the
first step/time where `|v| <= 25` is reached, the number and fraction of steps
inside the zone, the longest consecutive in-zone run, sampled simulation time
spent inside the zone, and whether the episode finishes inside it. In-zone
simulation-time totals include only intervals whose consecutive sampled states
are both inside the zone.

The evaluation band can be changed with `--target-speed-min` and
`--target-speed-max`. The dedicated fixed-tolerance entry point uses isolated
artifacts and defaults to `eps=30`, `del=25`, target band `[25, 30]`,
initial velocity in `[25, 29.9]`, and an `|v| <= 30` safety limit:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_cruise_control_fixed_tol.py --timesteps 10000 --eval-episodes 20
```

For this experiment only, the fixed-tolerance model has an added `holdSpeed`
mode with zero acceleration inside `[del, eps]`, and the `ebrakeact` controller
can transition into it. This changes the simulated plant/controller relative
to the original cruise-control model; the result evaluates learned behavior on
the modified simulator and is not evidence that the original or fixed-tolerance
modal theorem is proved. `targetSpeedBand` rewards proximity to `del`, and
`holdingTargetSpeed` adds a per-step bonus while the controller is in the
hold mode inside the target band. `epsilonSafe` tracks episodes exceeding the
absolute speed limit. The bounded initial-speed override intentionally starts
episodes in the target range; it does not test the theorem's quantified
initialization premise.

Training also writes a reward/episode-length curve. Each evaluation run writes
a paired PPO-vs-baseline summary plot for reward, target-band occupancy, longest
consecutive in-band duration, finishing inside the band, controller paths,
mode transitions, and (for the dedicated fixed-tolerance trainer) epsilon
violations. Up to two PPO rollout plots are saved for episodes that reach the
configured target band, prioritizing the longest sustained in-zone runs, and
up to two for episodes that never reach it. Trajectory plots show velocity and
the target band, position error, controller mode, selected path, rewards, and
the optional epsilon limit. Reaching the band at least once is not equivalent
to holding it; use occupancy, consecutive duration, final-in-band, and safety
metrics to assess the policy.

To compare on a separate fixed seed range from the original `11` evaluation,
use a new base seed; the evaluator starts at `--seed + 1000`. For example, this
evaluates seeds `1100` through `1119`:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_cruise_control.py --eval-only --eval-episodes 20 --seed 100 --max-episode-steps 5000 --out-dir logs\cruise_control\unseen_1100_1119
```

This trains on the generated simulation and is not evidence that the quantified
stability theorem is proved.

### Training the Cruise-Control Model Centered at 25

The separate `CruiseControlStabilityAround25.txt` model uses
`nearTargetSpeed()`, equivalent to `abs(v - 25) <= 1`. Train and evaluate PPO
with:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_cruise_control_around25.py --timesteps 10000 --eval-episodes 20
```

By default, the trainer evaluates the `[24, 26]` target band and lets the
generated environment sample initial states from the modal's `@var` bounds
(including `v` in `[5, 65]`). It does not override initial velocity, position,
or mode sampling. The target band corresponds to the model's
`nearTargetSpeed()` predicate. The trainer stores its policy, generated
environment, CSVs, and plots separately under `models/cruise_control_around25/`,
`generated_cruise_control_around25_env.py`, and `logs/cruise_control_around25/`.
Use `--eval-only` to evaluate the saved policy; `--target-speed-min` and
`--target-speed-max` adjust the evaluation target band only.

The pipeline plots training reward/episode length, paired PPO versus first-legal
baseline metrics, representative rollouts, and the annotated `cruiseSafe`
invariant rate. The default target band is derived from `nearTargetSpeed`; the
safety invariant is reported independently. The modal theorem's quantified
premise is not a simulation proof. The trainer prints legal-choice counts
observed over sample resets and warns when the sampled resets offer no
meaningful controller choice for PPO to learn.

### Training the Test 1 Static-Safety Controller

Train and evaluate a masked PPO policy on the planar robot model in
`test/input/withBound/test1.txt`:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_test1.py --timesteps 10000 --eval-episodes 20
```

The script saves the policy under `models/test1/`, the generated environment
under `generated_test1_ppo_env.py`, and training/evaluation plots and an
episode-level CSV under `logs/test1/`. Use `--eval-only` to evaluate the saved
policy without training; `--seed`, `--max-episode-steps`, `--model-dir`, and
`--out-dir` control reproducibility, episode length, and artifact locations.

The trainer samples initial states satisfying Test 1's exact premise
(`v=0`, unit direction, distinct robot/obstacle positions, and `r!=0`); the
unmodified variable-bound samplers almost never satisfy its equality
constraints. The model's `initialState`/`assumptions` signatures now explicitly
include `r`, which is referenced by the initial-state predicate. Evaluation
uses the same deterministic test scenario for both policies: the robot starts
from rest, facing an obstacle placed 70 distance units ahead along its
constant-radius path. Repeated evaluation episodes reuse this scenario, so
their rates are repeatability checks, not independent randomized trials. In
Test 1, `ensure` only checks that robot and obstacle
positions differ, so it is not a useful standalone measure of learned safety.
Evaluation also checks `loopinv` and reports termination reasons; these
simulation metrics are not a formal proof.

The comparison baseline repeatedly requests controller path 2 and supplies
`r=512`, `w=v/r`, and the fixed obstacle coordinates through the simulator's
nondeterministic-assignment hook. The obstacle is placed 70 distance units
along the resulting planned arc. Unlike the prior externally stopped
experiment, this baseline does not override the generated termination
condition, bypass the controller's `admissibleSeparation` guard, or apply a
custom collision radius. Termination reasons therefore come from generated
environment behavior (for example, a failed controller test, a safety
postcondition, or a loop invariant), or from the episode time limit. A
`generated_safety_termination` result specifically means the environment
reported `model_loop_invariant_violation` or
`model_safety_postcondition_violation`; a failed controller guard is reported
separately. The output includes a PPO-versus-baseline plot and CSV plus
representative trajectory plots.

Test 1 episode plots include the remaining stopping-distance safety margin
`max(|x-xo|, |y-yo|) - stopDist(v)`; values at or below zero violate the
distance disjunction in `isSafe`. Direction normalization is evaluated with a
small numerical tolerance in the episode metrics because the ODE solver uses
floating-point integration.

### Training the ETCS Safety Controller

Train and evaluate the five masked discrete acceleration choices in
`train_safety.kyx` with:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_train_safety.py --timesteps 20000 --eval-episodes 20 --out-dir logs\train_safety_controlfunc --model-dir models\train_safety_controlfunc
```

Training and evaluation show progress bars. The run saves the policy and a
two-panel training reward/episode-length plot in the specified output
directories; the `em` program variable starts at zero to satisfy the model's
initial condition while retaining its `[0, 1]` range for subsequent controller
updates. Evaluation saves a summary plot of reward, episode length, minimum
stopping margin, flow duration, controller action counts, and cumulative
safety-pass rate. It also saves up to two full-trajectory plots for each
safety-passing and safety-failing category, showing position, velocity,
stopping margin, acceleration, action, and actual ODE flow duration. A
safety-passing rollout means only that no tracked safety/integration failure
occurred; reaching the episode limit is not proof of safety. If evaluation has
no examples of one category, no plot for that category is created.

Evaluation prints each episode's termination reason and final state, and saves
per-episode diagnostics to JSON, including complete state/action trajectories
and a configurable pre-failure trace of stopping margin, legal-action mask,
selected branch, and state before/after each step. Evaluation disables the
diverse-state buffer so `--eval-seed` produces reproducible initial-state
samples. To inspect the saved policy without retraining and capture the same
20-step trace window for each invariant failure, run:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_train_safety.py --eval-only --eval-episodes 20 --eval-seed 1000 --trace-window-steps 20 --out-dir logs\train_safety_controlfunc\failure_traces_seed1000 --model-dir models\train_safety_controlfunc
```

The finite action set uses `a=-b`, `a=0`, or `a=A` when `v <= vdes`, and
`a=-b` or `a=0` when `v > vdes`. This replaces the model's previous continuous
nondeterministic acceleration choices with a finite controller for RL; safety
results for this discrete controller do not establish the original continuous
nondeterministic model's safety.

For conjunctive ODE evolution domains made from inequalities, integration stops
at the first domain boundary and snaps directly constrained state variables to
that boundary. For example, braking under `v >= 0` ends the flow at `v=0`
instead of integrating to a negative velocity; the next control cycle can then
choose a new action.