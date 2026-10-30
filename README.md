# KYX Parser and Reinforcement-Learning Experiments

## 1. Description

This project parses KeYmaera X differential dynamic logic (dL) models and generates Gymnasium simulation environments. It includes PPO training and evaluation experiments for robot safety, waypoint reaching, and cruise control.

## 2. Repository Structure

```text
AST_parser_2/                    # Lexer, parser, AST, generator, and Gymnasium template
scripts/
  train_ppo_test1.py             # Test 1 robot-safety experiment
  train_ppo_car2.py              # Car 2 waypoint-reaching experiment
  train_ppo_cruise_control_around25.py  # Cruise control around v=25
  train_ppo_cruise_control.py    # Shared cruise-control evaluation/training utilities
test/input/withBound/
  test1.txt                      # Test 1 dL model
  car2.txt                       # Test 2 dL model
  CruiseControlStabilityAround25.txt  # Test 3 dL model
models/                          # Saved PPO checkpoints
logs/                            # Generated plots, CSVs, and run diagnostics
requirements.txt                 # Pinned Python dependencies
LICENSE                          # MIT license for project-authored code
README.md                        # Setup and experiment instructions
```

Generated environment modules are written to the repository root by default (or to the path provided to the script). Training outputs are saved under the selected model and log directories.

## 3. Prerequisites & Installation

- **Hardware:** No robot hardware is required; the experiments run in simulation.
- **Software:** Python 3.13.1 was tested. Other Python versions have not been validated.
- **Install (PowerShell):**

```powershell
git clone https://github.com/Mrthuancao/KyxParser.git
cd KyxParser
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run commands below from the repository root. `requirements.txt` pins direct dependencies; pip resolves their transitive dependencies during installation. The MIT license covers project-authored code; third-party dependencies and materials remain subject to their own terms. The pinned PyTorch package uses the default PyPI build; install a hardware-specific PyTorch build if needed.

## 4. How to Recreate the Experiments

Each script reads the corresponding dL model, generates a Gymnasium environment, trains PPO, and evaluates the resulting policy. The results are empirical simulation results, not formal proofs of the dL models.

### Test 1 — Robot safety

Input: [`test1.txt`](test/input/withBound/test1.txt). The script samples initial states satisfying the model's initial premise, then evaluates the policy against a separate forced-acceleration safety-monitor stress-test baseline.

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_test1.py --timesteps 10000 --eval-episodes 20 --seed 11 --out-dir logs\paper\test1 --model-dir models\paper\test1
```

The policy and training curve are saved under `models/paper/test1/` and `logs/paper/test1/`. Evaluation saves PPO and baseline rollout plots, episode CSVs, and a policy-versus-baseline plot in `logs/paper/test1/`. The generated environment defaults to `generated_test1_ppo_env.py`.

### Test 2 — Car 2 waypoint reaching

Input: [`car2.txt`](test/input/withBound/car2.txt). The script uses the model's initial-state buffer and controller choices to train a policy to reach the waypoint.

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_car2.py --timesteps 10000 --eval-episodes 20 --out-dir logs\paper\car2 --model-dir models\paper\car2
```

The policy is saved under `models/paper/car2/`; the training curve, up to five successful and five failed rollouts, validation plot, and per-episode CSV are saved under `logs/paper/car2/`. The generated environment defaults to `generated_car2_env.py`. This script currently has no seed or evaluate-only option.

### Test 3 — Cruise control around 25

Input: [`CruiseControlStabilityAround25.txt`](test/input/withBound/CruiseControlStabilityAround25.txt), with target speed band `[24, 26]`. The default maximum episode length is 3,000 steps. Evaluation compares PPO with the first-legal-action baseline on matching seeds.

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_cruise_control_around25.py --timesteps 10000 --eval-episodes 20 --seed 11 --max-episode-steps 3000 --out-dir logs\paper\cruise_control_around25 --model-dir models\paper\cruise_control_around25
```

The policy is saved under `models/paper/cruise_control_around25/`; training and evaluation plots, PPO/baseline CSVs, safety metrics, and up to ten PPO rollout plots per reachability category are saved under `logs/paper/cruise_control_around25/`. The generated environment defaults to `generated_cruise_control_around25_env.py`.

To re-evaluate that saved policy without training, run the same command with `--eval-only` and the same model directory. Use a new `--out-dir` to retain the original plots and CSVs. To evaluate lower initial velocities, provide both `--eval-initial-velocity-min` and `--eval-initial-velocity-max`; this changes evaluation sampling only.

## 5. Data & Plotting

The three commands above generate the figures used for the experiments; there is no separate plotting script. The plot files are saved in each experiment's `--out-dir`:

- **Test 1:** PPO and forced-acceleration-baseline episode rollouts, including square trajectory plots and separate control-signal plots, plus the PPO/baseline comparison.
- **Test 2:** PPO training reward/episode-length curve, waypoint-reaching episode rollouts, and validation plot.
- **Test 3:** PPO training curve, representative PPO episode rollouts, and the PPO-versus-baseline evaluation summary and safety plot.

The corresponding episode-level CSVs and diagnostics are saved alongside the figures. Use a distinct `--out-dir` for each run to avoid overwriting prior results. To reproduce a trained-policy evaluation, keep the checkpoint, input model, and relevant evaluation options the same. Test 1 and Test 3 accept a seed; Test 2 does not currently expose one.
