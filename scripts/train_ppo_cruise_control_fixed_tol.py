"""Train and evaluate PPO on the fixed-tolerance cruise-control model.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe scripts\\train_ppo_cruise_control_fixed_tol.py

This dedicated entry point shares the PPO implementation with
``train_ppo_cruise_control.py`` but defaults to the fixed-tolerance model,
target band, initial-speed range, safety limit, and isolated output paths.
"""

from train_ppo_cruise_control import main


if __name__ == "__main__":
    main(fixed_tolerance=True)
