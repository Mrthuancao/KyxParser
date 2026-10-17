"""Generate and smoke-test the quantified Cruise Control Gym simulator.

Run from the repository root with the project virtual environment:

    python scripts/end_to_end_cruise_control_test.py
    python scripts/end_to_end_cruise_control_test.py --steps 100 --seed 7
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AST2 = ROOT / "AST_parser_2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AST2) not in sys.path:
    sys.path.insert(0, str(AST2))

from generator_core import generate_gym_environment_file


def load_generated_env(module_path: Path):
    module_name = "generated_cruise_control_smoke_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.KeYmaeraGymEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Cruise Control environment and run a short simulation."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "test" / "input" / "withBound" / "CruiseControlStability.txt",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if not args.model.is_file():
        raise FileNotFoundError(f"Cruise Control model not found: {args.model}")

    with tempfile.TemporaryDirectory(prefix="cruise_control_smoke_") as temp_dir:
        module_path = Path(temp_dir) / "generated_cruise_control_env.py"
        model_text = args.model.read_text(encoding="utf-8")
        if not generate_gym_environment_file(
            model_text, "CruiseControlSmokeTest", str(module_path)
        ):
            raise RuntimeError("Cruise Control environment generation failed")

        env_class = load_generated_env(module_path)
        env = env_class()
        try:
            if not env.simulation_mode:
                raise AssertionError("Generated environment did not enable modal simulation.")
            if getattr(env.action_space, "n", None) != len(env.control_choice_paths):
                raise AssertionError(
                    "Action space must expose each controlFunc path: "
                    f"{env.action_space} for {len(env.control_choice_paths)} paths."
                )

            observation, info = env.reset(seed=args.seed)
            if not env.observation_space.contains(observation):
                raise AssertionError(f"Reset observation is outside its space: {observation}")

            initial_state = dict(info["state"])
            total_reward = 0.0
            terminated = False
            truncated = False
            steps_run = 0
            actions_seen = set()
            for steps_run in range(1, args.steps + 1):
                action_mask = env.action_masks()
                legal_actions = action_mask.nonzero()[0]
                if len(legal_actions) == 0:
                    raise AssertionError("Controller action mask contains no legal action.")
                action = int(legal_actions[0])
                actions_seen.add(action)
                observation, reward, terminated, truncated, info = env.step(action)
                if info["control_action"] != action:
                    raise AssertionError(
                        f"Environment did not execute selected control path {action}."
                    )
                if not env.observation_space.contains(observation):
                    raise AssertionError(f"Step observation is outside its space: {observation}")
                if not math.isfinite(float(reward)):
                    raise AssertionError(f"Non-finite reward returned: {reward}")
                if not math.isfinite(float(info["simulation_elapsed_time"])):
                    raise AssertionError("Simulation elapsed time is not finite.")
                total_reward += float(reward)
                if terminated or truncated:
                    break

            final_state = info["state"]
            print("Cruise Control simulator smoke test passed.")
            print(f"Model: {args.model}")
            print(f"Observation shape: {observation.shape}")
            print(f"Initial state: {initial_state}")
            print(f"Final state: {final_state}")
            print(f"Steps run: {steps_run}")
            print(
                f"Controller actions: {env.action_space.n}; "
                f"legal actions selected: {sorted(actions_seen)}"
            )
            print(f"Simulation time: {info['simulation_elapsed_time']:.6f}")
            print(f"Total reward: {total_reward:.6f}")
            print(f"Terminated: {terminated}; truncated: {truncated}")
            print(f"Termination reason: {info['termination_reason']}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
