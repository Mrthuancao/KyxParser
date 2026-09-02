#!/usr/bin/env python3
"""Generate a Gym environment from a KeYmaera model and evaluate diversity in the
initial-state buffer.

Usage:
    python scripts/test_generated_env_diversity.py --model test/input/withBound/train2.txt
    python scripts/test_generated_env_diversity.py --model path/to/model.txt --env-name MyEnv --capacity 25 --samples 25
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AST2 = ROOT / "AST_parser_2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AST2) not in sys.path:
    sys.path.insert(0, str(AST2))

from generator_core import generate_gym_environment_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a KeYmaera Gym env and measure diversity of its initial state buffer.")
    parser.add_argument("--model", type=str, default=str(ROOT / "test" / "input" / "withBound" / "train2.txt"), help="Formal model file to parse")
    parser.add_argument("--env-name", type=str, default="GeneratedDiversityEnv", help="Generated environment class name")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "tmp_generated_envs"), help="Directory to write generated env Python file")
    parser.add_argument("--capacity", type=int, default=20, help="Target initial state buffer capacity")
    parser.add_argument("--samples", type=int, default=20, help="How many states to inspect from the buffer")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "z3_diverse"], help="Initial-state buffer strategy; random is the most robust default, z3_diverse is optional for diversity-focused testing")
    return parser.parse_args()


def load_model(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def generate_env_file(model_text: str, env_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env_name}.py"
    ok = generate_gym_environment_file(model_text, env_name, str(out_path))
    if not ok:
        raise RuntimeError(f"Failed to generate environment file at {out_path}")
    return out_path


def import_generated_module(module_path: Path):
    module_name = f"generated_env_{module_path.stem}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def extract_bound_info(env) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for var in env.config.get("variables", []):
        name = var.get("name")
        low = var.get("lower_bound")
        high = var.get("upper_bound")
        if name is None:
            continue
        if low is None and high is None:
            bounds[name] = (-100.0, 100.0)
        else:
            try:
                low_val = float(low) if low is not None else -100.0
                high_val = float(high) if high is not None else 100.0
            except (TypeError, ValueError):
                low_val = -100.0
                high_val = 100.0
            bounds[name] = (low_val, high_val)
    return bounds


def summarize_states(states: Iterable[Dict[str, Any]], bounds: Dict[str, Tuple[float, float]]):
    grouped: Dict[str, List[float]] = {}
    for state in states:
        for key in bounds:
            if key in state:
                grouped.setdefault(key, []).append(float(state[key]))

    summary = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=np.float64)
        low, high = bounds.get(key, (float(values.min()), float(values.max())))
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        min_v = float(np.min(values))
        max_v = float(np.max(values))

        if math.isfinite(low) and math.isfinite(high) and high > low:
            center = (low + high) / 2.0
            span = high - low
            normalized_mean = (mean - center) / (span / 2.0)
            normalized_std = std / (span / 2.0)
        else:
            normalized_mean = 0.0
            normalized_std = 0.0

        summary.append(
            {
                "variable": key,
                "mean": mean,
                "std": std,
                "min": min_v,
                "max": max_v,
                "low_bound": low,
                "high_bound": high,
                "mean_vs_center": normalized_mean,
                "std_vs_span": normalized_std,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_text = load_model(model_path)
    out_dir = Path(args.out_dir).resolve()
    env_file = generate_env_file(model_text, args.env_name, out_dir)
    module = import_generated_module(env_file)

    config_override = {
        "initial_state_buffer_config": {
            "enabled": True,
            "capacity": args.capacity,
            "populate_on_init": True,
            "max_populate_attempts": max(1000, args.capacity * 50),
            "strategy": args.strategy,
        }
    }

    try:
        env = module.KeYmaeraGymEnv(config_override=config_override)
    except Exception as exc:
        if args.strategy == "z3_diverse":
            print(f"Z3-diverse strategy failed for model: {exc}\nFalling back to the random strategy for measurement.")
            config_override["initial_state_buffer_config"]["strategy"] = "random"
            env = module.KeYmaeraGymEnv(config_override=config_override)
        else:
            raise

    buffer_states = list(env.initial_state_buffer)
    if not buffer_states:
        if args.strategy == "z3_diverse":
            print("Z3-diverse strategy produced no states; retrying with the random strategy so the buffer can still be measured.")
            config_override["initial_state_buffer_config"]["strategy"] = "random"
            env = module.KeYmaeraGymEnv(config_override=config_override)
            buffer_states = list(env.initial_state_buffer)
        if not buffer_states:
            raise RuntimeError("No states were generated in the initial-state buffer. Check the model or sampling bounds.")

    if len(buffer_states) > args.samples:
        buffer_states = buffer_states[: args.samples]

    bounds = extract_bound_info(env)
    summary = summarize_states(buffer_states, bounds)

    print(f"Generated environment: {env_file.name}")
    print(f"Buffer strategy: {args.strategy}")
    print(f"Buffer size: {len(buffer_states)}")
    print()
    print("Per-variable diversity summary:")
    print("- mean: average value across sampled states")
    print("- std: standard deviation across sampled states")
    print("- mean_vs_center: mean offset relative to midpoint of [low, high]")
    print("- std_vs_span: std relative to the total bound span")
    print()
    for row in summary:
        print(
            f"{row['variable']:>8s} | mean={row['mean']:.4f} | std={row['std']:.4f} | "
            f"min={row['min']:.4f} | max={row['max']:.4f} | "
            f"bounds=[{row['low_bound']:.4f}, {row['high_bound']:.4f}] | "
            f"mean_vs_center={row['mean_vs_center']:.4f} | std_vs_span={row['std_vs_span']:.4f}"
        )

    print()
    print("First 3 sampled states:")
    for idx, state in enumerate(buffer_states[:3], 1):
        print(f"  State {idx}: {json.dumps(state, sort_keys=True, default=str)}")


if __name__ == "__main__":
    main()
