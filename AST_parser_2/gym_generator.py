"""Thin wrapper for the generator core.

The heavy lifting was moved to `generator_core.py`. This module re-exports
the main API and provides a small CLI for convenience.
"""

from generator_core import GymnasiumEnvironmentGenerator, generate_gym_environment_file

__all__ = ["GymnasiumEnvironmentGenerator", "generate_gym_environment_file"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python gym_generator.py <input_dl_file> <output_env_py> [environment_name]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    env_name = sys.argv[3] if len(sys.argv) > 3 else "KeYmaeraEnvironment"
    with open(input_path, 'r', encoding='utf-8') as f:
        dl_text = f.read()
    ok = generate_gym_environment_file(dl_text, env_name, output_path)
    if ok:
        print(f"Generated environment written to {output_path}")
    else:
        print("Generation failed")
        sys.exit(2)
