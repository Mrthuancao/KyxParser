"""Compare Z3 initial-state sampling strategies on the Test 1 model.

Strategies (one valid initial state per call, N states each):
  seed_solver      Z3 Solver, hard constraints + variable bounds, a different
                   random seed for every state (the "built-in seed" approach).
  seed_optimize    Same, but through z3.Optimize with no soft constraints.
  soft_targets     The production path: env._populate_initial_state_buffer_diverse_z3
                   (hard constraints + bounds + random soft targets per state).
  soft_blocking    Soft targets plus a blocking clause that actually persists
                   (the production code adds one inside push()/pop(), which
                   discards it; this variant shows what it would change).
  soft_greedy      Soft targets on a plain Solver: variables are visited in a
                   random order and ``var == target`` is kept only if the
                   problem stays satisfiable. Unlike Optimize it also works on
                   nonlinear preconditions.
  rejection        env._generate_valid_initial_state (uniform sampling + test).

Z3's Optimize engine answers ``unknown (incomplete (theory arithmetic))`` when
soft constraints meet a nonlinear precondition (Test 1's dx^2 + dy^2 = 1), so
soft_targets / soft_blocking only produce states on linear models such as car2.

Run from the repository root (use the project venv, which has z3):

    .\\.venv\\Scripts\\python.exe scripts\\compare_initial_state_sampling.py --samples 30
    .\\.venv\\Scripts\\python.exe scripts\\compare_initial_state_sampling.py \\
        --model test\\input\\withBound\\car2.txt --env-name KeYmaeraCar2 \\
        --out-dir logs\\initial_state_sampling_car2
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import z3

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "AST_parser_2", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generator_core import generate_gym_environment_file  # noqa: E402

SEED_PARAMS = ("smt.random_seed", "sat.random_seed", "nlsat.seed", "sls.random_seed")
BLOCK_EPSILON = 0.1


def build_env(model_path: Path, generated_path: Path, seed: int, env_name: str = "KeYmaeraTest1"):
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    if not generate_gym_environment_file(
        model_path.read_text(encoding="utf-8"), env_name, str(generated_path)
    ):
        raise RuntimeError(f"Could not generate an environment from {model_path}")
    spec = importlib.util.spec_from_file_location("generated_sampling_env", generated_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    env = module.KeYmaeraGymEnv(
        render_mode=None,
        config_override={
            "seed": seed,
            "initial_state_buffer_config": {"enabled": False},
        },
    )
    env._np_random, _ = gym.utils.seeding.np_random(seed)
    return env


class Problem:
    """The Z3 problem the environment itself builds: initial condition + bounds."""

    def __init__(self, env):
        self.env = env
        self.ranges = env._build_initial_state_sample_ranges()
        self.symbols = {}
        self.constants = {}
        for var in env.config.get("variables", []):
            self.symbols[var["name"]] = z3.Real(var["name"])
        for const in env.config.get("constants", []):
            try:
                self.constants[const["name"]] = float(const.get("value"))
            except (TypeError, ValueError):
                self.symbols[const["name"]] = z3.Real(const["name"])
        self.initial_condition = env.config["problem"]["initial_condition"]
        self.hard = [env._expr_to_z3(self.initial_condition, self.symbols, self.constants)]
        for name, symbol in self.symbols.items():
            low, high = self.ranges.get(name, (-100.0, 100.0))
            self.hard += [symbol >= low, symbol <= high]
        self.variables = [v["name"] for v in env.config["variables"]]

    def read_model(self, model):
        values = {}
        for name, symbol in self.symbols.items():
            value = model.evaluate(symbol, model_completion=True)
            if z3.is_algebraic_value(value):
                value = value.approx(30)
            values[name] = float(value.as_fraction())
        # Floats cannot satisfy dx^2 + dy^2 == 1 exactly, so record Z3's own verdict.
        values["__exact_valid"] = z3.is_true(model.evaluate(z3.And(*self.hard), model_completion=True))
        return values

    def is_valid(self, values):
        if "__exact_valid" in values:
            return values["__exact_valid"]
        state = dict(self.env.current_state)
        state.update(values)
        state["t"] = values.get("t", 0.0)
        state.setdefault("T", 10.0)
        return bool(self.env._evaluate_expression(self.initial_condition, state))


def set_z3_seed(seed: int):
    accepted = []
    for param in SEED_PARAMS:
        try:
            z3.set_param(param, seed)
            accepted.append(param)
        except z3.Z3Exception:
            pass
    return accepted


def sample_seed_solver(problem, count, base_seed, optimize=False, stats=None):
    states, times = [], []
    for index in range(count):
        started = time.perf_counter()
        set_z3_seed(base_seed + index)
        solver = z3.Optimize() if optimize else z3.Solver()
        solver.set(timeout=10000) if optimize else solver.set("timeout", 10000)
        solver.add(*problem.hard)
        result = solver.check()
        if stats is not None:
            stats[str(result)] += 1
        if result == z3.sat:
            states.append(problem.read_model(solver.model()))
        times.append(time.perf_counter() - started)
    return states, times


def sample_soft_targets(problem, count, rng, block=False, stats=None):
    """Optimize-based soft targets; ``block`` keeps the blocking clause across states."""
    opt = z3.Optimize()
    opt.set(timeout=10000)
    opt.add(*problem.hard)
    states, times = [], []
    attempts = 0
    while len(states) < count and attempts < count * 2:
        attempts += 1
        started = time.perf_counter()
        opt.push()
        for name, symbol in problem.symbols.items():
            low, high = problem.ranges.get(name, (-100.0, 100.0))
            if low != high:
                opt.add_soft(symbol == rng.uniform(low, high), weight=1)
        result = opt.check()
        if stats is not None:
            stats[str(result)] += 1
        if result == z3.sat:
            values = problem.read_model(opt.model())
            states.append(values)
            opt.pop()
            if block:
                opt.add(z3.Or([
                    z3.Or(symbol < values[name] - BLOCK_EPSILON,
                          symbol > values[name] + BLOCK_EPSILON)
                    for name, symbol in problem.symbols.items()
                ]))
        else:
            if stats is not None and result == z3.unknown:
                stats["reason: " + opt.reason_unknown()] += 1
            opt.pop()
        times.append(time.perf_counter() - started)
    return states, times


def sample_soft_greedy(problem, count, rng, stats=None):
    """Soft targets on a plain Solver: keep ``var == target`` only while still sat."""
    states, times = [], []
    for _ in range(count):
        started = time.perf_counter()
        solver = z3.Solver()
        solver.set("timeout", 10000)
        solver.add(*problem.hard)
        result = solver.check()
        if result == z3.sat:
            order = list(problem.symbols)
            rng.shuffle(order)
            for name in order:
                low, high = problem.ranges.get(name, (-100.0, 100.0))
                if low == high:
                    continue
                solver.push()
                solver.add(problem.symbols[name] == z3.RealVal(rng.uniform(low, high)))
                if solver.check() != z3.sat:
                    solver.pop()
            solver.check()
            states.append(problem.read_model(solver.model()))
        if stats is not None:
            stats[str(result)] += 1
        times.append(time.perf_counter() - started)
    return states, times


def sample_production(env, problem, count, seed, stats=None):
    env._np_random, _ = gym.utils.seeding.np_random(seed)
    env.initial_state_buffer.clear()
    env.initial_state_buffer_config.update({"capacity": count, "max_populate_attempts": count * 2})
    captured = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(captured):
            env._populate_initial_state_buffer_diverse_z3()
    except ValueError as error:
        # Z3 returned an algebraic (irrational) value the template cannot convert.
        if stats is not None:
            stats["crash: " + str(error).split(":")[0]] += 1
    elapsed = time.perf_counter() - started
    if stats is not None:
        stats["unknown"] += captured.getvalue().count("Z3 UNKNOWN")
        stats["unsat"] += captured.getvalue().count("Z3 UNSAT")
    states = [{name: float(s[name]) for name in problem.variables} for s in env.initial_state_buffer]
    return states, [elapsed / max(len(states), 1)] * len(states)


def sample_rejection(env, problem, count, attempts_per_state=100):
    states, times = [], []
    for _ in range(count):
        started = time.perf_counter()
        try:
            update = env._generate_valid_initial_state(max_attempts=attempts_per_state)
            states.append({name: float(update.get(name, 0.0)) for name in problem.variables})
        except RuntimeError:
            pass
        times.append(time.perf_counter() - started)
    return states, times


def summarize(name, problem, states, times, requested, stats):
    columns = [v for v in problem.variables if v in states[0]] if states else []
    matrix = np.array([[s[v] for v in columns] for s in states]) if states else np.empty((0, 0))
    spans = np.array([max(problem.ranges[v][1] - problem.ranges[v][0], 1e-12) for v in columns])
    result = {
        "method": name,
        "requested": requested,
        "returned": len(states),
        "valid": sum(problem.is_valid(s) for s in states),
        "unique": len({tuple(np.round(row, 9)) for row in matrix}) if len(states) else 0,
        "mean_seconds_per_state": float(np.mean(times)) if times else float("nan"),
        "z3_status": dict(stats),
    }
    if len(states) > 1:
        normalized = matrix / spans
        result["mean_norm_std"] = float(np.mean(normalized.std(axis=0)))
        bins = [
            len(set(np.clip(((matrix[:, i] - problem.ranges[v][0]) / spans[i] * 10).astype(int), 0, 9)))
            for i, v in enumerate(columns)
        ]
        result["mean_bin_coverage_of_10"] = float(np.mean(bins))
        distances = np.linalg.norm(normalized[:, None] - normalized[None, :], axis=2)
        result["min_pairwise_norm_dist"] = float(distances[np.triu_indices(len(states), 1)].min())
    return result


def save_outputs(out_dir, problem, results, all_states, plot_vars=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "index", *problem.variables])
        for method, states in all_states.items():
            for index, state in enumerate(states):
                writer.writerow([method, index, *[state[v] for v in problem.variables]])

    fig, axes = plt.subplots(1, len(all_states), figsize=(3.2 * len(all_states), 3.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    x_name, y_name = plot_vars or problem.variables[:2]
    for axis, (method, states) in zip(axes, all_states.items()):
        if states:
            axis.scatter([s[x_name] for s in states], [s[y_name] for s in states], s=14)
        axis.set_title(f"{method}\n(n={len(states)})", fontsize=9)
        axis.set_xlabel(x_name)
        axis.set_xlim(*problem.ranges[x_name])
        axis.set_ylim(*problem.ranges[y_name])
    axes[0].set_ylabel(y_name)
    fig.tight_layout()
    fig.savefig(out_dir / "sampling_scatter.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "test" / "input" / "withBound" / "test1.txt")
    parser.add_argument("--env-name", default="KeYmaeraTest1")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "logs" / "initial_state_sampling")
    parser.add_argument("--plot-vars", nargs=2, metavar=("X", "Y"),
                        help="Two variables for the scatter plot (default: first two declared).")
    args = parser.parse_args()

    env = build_env(args.model, args.out_dir / "generated_env.py", args.seed, args.env_name)
    problem = Problem(env)
    print(f"Initial condition: {problem.initial_condition}")
    print(f"Hard constraint: {problem.hard[0]}")

    def fresh(fn):
        # Z3 parameters are process-global; keep one method's seeds from leaking into the next.
        def wrapped(stats):
            z3.reset_params()
            return fn(stats)
        return wrapped

    runs = {
        "seed_solver": lambda s: sample_seed_solver(problem, args.samples, args.seed, stats=s),
        "seed_optimize": lambda s: sample_seed_solver(problem, args.samples, args.seed, optimize=True, stats=s),
        "soft_targets": lambda s: sample_production(env, problem, args.samples, args.seed, stats=s),
        "soft_blocking": lambda s: sample_soft_targets(
            problem, args.samples, np.random.default_rng(args.seed), block=True, stats=s
        ),
        "soft_greedy": lambda s: sample_soft_greedy(
            problem, args.samples, np.random.default_rng(args.seed), stats=s
        ),
        "rejection": lambda s: sample_rejection(env, problem, args.samples),
    }
    results, all_states = [], {}
    for name, run in runs.items():
        print(f"Running {name} ...", flush=True)
        stats = Counter()
        states, times = fresh(run)(stats)
        all_states[name] = states
        results.append(summarize(name, problem, states, times, args.samples, stats))

    if args.plot_vars and not set(args.plot_vars) <= set(problem.variables):
        parser.error(f"--plot-vars must be among {problem.variables}")
    save_outputs(args.out_dir, problem, results, all_states, args.plot_vars)
    header = ("method", "returned", "valid", "unique", "norm_std", "bins/10", "min_dist", "s/state")
    print("\n" + " ".join(f"{h:>14}" for h in header))
    for r in results:
        print(" ".join(f"{v:>14}" for v in (
            r["method"], r["returned"], r["valid"], r["unique"],
            f"{r.get('mean_norm_std', float('nan')):.3f}",
            f"{r.get('mean_bin_coverage_of_10', float('nan')):.1f}",
            f"{r.get('min_pairwise_norm_dist', float('nan')):.3f}",
            f"{r['mean_seconds_per_state']:.3f}",
        )))
    print("\nZ3 status counts:")
    for r in results:
        print(f"  {r['method']:>14}: {r['z3_status']}")
    print(f"\nSaved results to {args.out_dir}")


if __name__ == "__main__":
    main()
