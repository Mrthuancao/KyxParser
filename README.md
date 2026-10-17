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
Set the per-episode step limit for both training and evaluation with
`--max-episode-steps` (default: `200`), for example:

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_cruise_control.py --timesteps 10000 --max-episode-steps 1000
```

This trains on the generated simulation and is not evidence that the quantified
stability theorem is proved.