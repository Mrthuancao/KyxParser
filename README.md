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
  train_ppo_car1.py              # Car 1 passive-safety experiment
  train_ppo_train_essential.py   # Train-approach essential-property experiment
  train_ppo_train_safety.py      # Train-safety experiment
test/input/withBound/
  test1.txt                      # Test 1 dL model
  car2.txt                       # Test 2 dL model
  CruiseControlStabilityAround25.txt  # Test 3 dL model
  car1.txt                       # Car 1 passive-safety model
  train_essential_fixed_authority.kyx  # Train-approach model
  train_safety.kyx               # Train-safety model
models/                          # Saved PPO checkpoints
logs/                            # Generated plots, CSVs, and run diagnostics
requirements.txt                 # Pinned Python dependencies
LICENSE                          # MIT license for project-authored code
README.md                        # Setup and experiment instructions
```

Generated environment modules are written to the repository root by default (or to the path provided to the script). Training outputs are saved under the selected model and log directories.

## 3. Prerequisites & Installation

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

### Test 4 — Car 1 passive safety

Input: [`car1.txt`](test/input/withBound/car1.txt), a passive-safety model with nested controller choices and continuous dynamics. The training wrapper exposes a path choice plus discretized acceleration and curve-radius parameters. This experiment is a partially validated design: the learned policy may favor the stationary braking path, so safe episodes alone do not demonstrate useful motion or progress.

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_car1.py --timesteps 10000 --eval-episodes 20 --seed 11 --out-dir logs\paper\car1 --model-dir models\paper\car1
```

The script saves `ppo_car1.zip` under `models/paper/car1/`, and the training curve and validation plot under `logs/paper/car1/`. The generated environment is written to the repository root as `generated_car1_env.py`. This script trains and evaluates in one run; it does not currently provide an evaluate-only option.

### Test 5 — Train approach (essential property)

Input: [`train_essential_fixed_authority.kyx`](test/input/withBound/train_essential_fixed_authority.kyx). The controller selects between guarded braking and acceleration choices. Evaluation reports both masked and unmasked policy behavior, so the results show whether performance depends on action masking.

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_train_essential.py --timesteps 10000 --eval-episodes 20 --max-episode-steps 200 --plot-episodes 5 --out-dir logs\paper\train_essential --model-dir models\paper\train_essential
```

The checkpoint is saved as `ppo_train_essential_fixed_authority_masked.zip` under `models/paper/train_essential/`. The output directory contains the training curve, masked/unmasked validation summary, and representative episode plots. The generated environment is saved under the model directory. This script currently does not expose a seed or evaluate-only option.

### Test 6 — Train safety

Input: [`train_safety.kyx`](test/input/withBound/train_safety.kyx). The script generates a Gymnasium environment, trains a mask-aware PPO policy, and evaluates safety outcomes over the requested number of episodes.

```powershell
.\.venv\Scripts\python.exe scripts\train_ppo_train_safety.py --timesteps 20000 --eval-episodes 20 --eval-seed 1000 --out-dir logs\paper\train_safety --model-dir models\paper\train_safety
```

The policy is saved under `models/paper/train_safety/` using the input filename stem. The output directory contains the evaluation summary, representative episode plots, a training curve (when training episodes complete), and per-episode diagnostics in JSON. The generated environment is written to the repository root. To evaluate the saved policy without retraining, repeat the command with `--eval-only` and the same `--model-dir`; use a different `--out-dir` to preserve previous results.

## 5. Containerized Execution Environment

The Docker image packages the checked-out source and the pinned direct Python dependencies. Build it from a recorded commit so the image contains the same source revision used for the experiment. Use a clean worktree to ensure the image matches that commit. The image uses the tested Python 3.13.1 base and installs the CPU-only PyTorch 2.9.1 wheel; this avoids downloading CUDA libraries that are not used in the container. The standard local installation instructions above still use the PyTorch package resolved from PyPI.

From PowerShell, build the image from the checked-out revision:

```powershell
git status --short
$commit = (git rev-parse HEAD).Trim()
docker build --build-arg VCS_REF=$commit -t kyxparser-thesis:$commit .
docker image inspect --format='{{.Id}}' "kyxparser-thesis:$commit"
```

Record the printed commit and image ID with the experiment results. For example, run Train Safety while mounting host folders for its outputs:

```powershell
New-Item -ItemType Directory -Force logs\paper\train_safety, models\paper\train_safety
docker run --rm `
  --mount "type=bind,source=$($PWD.Path)\logs\paper\train_safety,target=/outputs" `
  --mount "type=bind,source=$($PWD.Path)\models\paper\train_safety,target=/models" `
  "kyxparser-thesis:$commit" `
  scripts/train_ppo_train_safety.py --timesteps 20000 --eval-episodes 20 --eval-seed 1000 --out-dir /outputs --model-dir /models
```

Use the corresponding script and arguments from Section 4 to run another experiment; set its `--out-dir` and `--model-dir` to mounted container paths to retain plots, diagnostics, and checkpoints on the host. The generated environment is a derived file and may be regenerated by the script.


## 6. Data & Plotting

The experiment commands above generate their figures; there is no separate plotting script. The plot files are saved in each experiment's `--out-dir`:

- **Test 1:** PPO and forced-acceleration-baseline episode rollouts, including square trajectory plots and separate control-signal plots, plus the PPO/baseline comparison.
- **Test 2:** PPO training reward/episode-length curve, waypoint-reaching episode rollouts, and validation plot.
- **Test 3:** PPO training curve, representative PPO episode rollouts, and the PPO-versus-baseline evaluation summary and safety plot.
- **Car 1:** PPO training curve and validation plot.
- **Train approach:** training curve, masked/unmasked validation summary, and representative episodes for both evaluation modes.
- **Train safety:** evaluation summary, representative episode plots, and per-episode termination diagnostics.

Episode-level CSVs or diagnostics are saved alongside the figures where supported. Use a distinct `--out-dir` for each run to avoid overwriting prior results. To reproduce a trained-policy evaluation, keep the checkpoint, input model, and relevant evaluation options the same. Test 1, Test 3, and Car 1 expose a training seed; Test 2, Train approach, and Train safety do not expose a training seed (Train safety does expose an evaluation seed).
