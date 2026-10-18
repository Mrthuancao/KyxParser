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
first step/time where `|v| <= 1` is reached, the number and fraction of steps
inside the zone, the longest consecutive in-zone run, sampled simulation time
spent inside the zone, and whether the episode finishes inside it. In-zone
simulation-time totals include only intervals whose consecutive sampled states
are both inside the zone.

To compare on a separate fixed seed range from the original `11` evaluation,
use a new base seed; the evaluator starts at `--seed + 1000`. For example, this
evaluates seeds `1100` through `1119`:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_cruise_control.py --eval-only --eval-episodes 20 --seed 100 --max-episode-steps 5000 --out-dir logs\cruise_control\unseen_1100_1119
```

This trains on the generated simulation and is not evidence that the quantified
stability theorem is proved.

### Training the ETCS Safety Controller

Train and evaluate the five masked discrete acceleration choices in
`train_safety.kyx` with:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_train_safety.py --timesteps 20000 --eval-episodes 20 --out-dir logs\train_safety_controlfunc --model-dir models\train_safety_controlfunc
```

Training and evaluation show progress bars. The run saves the policy and plots in
the specified output directories; the `em` program variable starts at zero to
satisfy the model's initial condition while retaining its `[0, 1]` range for
subsequent controller updates. Evaluation prints each episode's termination
reason and final state, and saves per-episode diagnostics to JSON, including a
configurable pre-failure trace of stopping margin, legal-action mask, selected
branch, and state before/after each step. Evaluation disables the diverse-state
buffer so `--eval-seed` produces reproducible initial-state samples. To inspect
the saved policy without retraining and capture the same 20-step trace window
for each invariant failure, run:

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