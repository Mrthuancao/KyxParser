# ast_config.py (or your main training script)

import gymnasium as gym
import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy # <--- NEW IMPORT

# Import your CarEnv (assuming it's in ideal_car_env.py or defined above)
from ideal_car_env import CarEnv # If CarEnv is in a separate file

# --- Configuration ---
MODEL_PATH = "car_env_ppo_model.zip"
LOG_DIR = "./logs/"
TOTAL_TIMESTEPS_PER_RUN = 200_000 # Number of timesteps to train in each session
EVAL_FREQ = 10_000 # Evaluate every 10,000 timesteps
N_EVAL_EPISODES = 5 # Number of episodes for evaluation
REWARD_THRESHOLD_TO_STOP = 100.0 # Stop training if mean reward reaches this

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    # Create the environment
    # Use make_vec_env for vectorized environments, which is common with SB3
    # Pass render_mode to the environment constructor if you want rendering during training
    env = make_vec_env(CarEnv, n_envs=1)

    # Setup the StopTrainingOnRewardThreshold callback first
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESHOLD_TO_STOP,
        verbose=1
    )

    # Now, setup EvalCallback and pass the stop_callback to its callback_after_eval parameter
    eval_callback = EvalCallback(
        env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False, # Set to True if you want to see rendering during evaluation
        callback_after_eval=stop_callback
    )

    # Check if a model already exists to continue training
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH} for continued training...")
        model = PPO.load(MODEL_PATH, env=env, print_system_info=True)
        # It's important to set reset_num_timesteps=False to continue the global timestep count
        # and learning rate schedules.
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS_PER_RUN,
            callback=eval_callback, # Pass only eval_callback, as stop_callback is now nested
            reset_num_timesteps=False,
            progress_bar=True
        )
    else:
        print("No existing model found. Starting new training run...")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS_PER_RUN,
            callback=eval_callback, # Pass only eval_callback
            progress_bar=True
        )

    # Save the final model after training
    model.save(MODEL_PATH)
    print(f"Training finished. Model saved to {MODEL_PATH}")

    # Optional: Load the best model found during training and evaluate it
    best_model_path = os.path.join(LOG_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final evaluation...")
        best_model = PPO.load(best_model_path, env=env)

        print("Evaluating best model...")
        # <--- THIS IS THE KEY CHANGE: Call evaluate_policy directly from the module
        mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=10, deterministic=True)
        print(f"Best model mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    else:
        print(f"No best model found at {best_model_path}. This might happen if training stopped before an evaluation saved a best model, or if the reward threshold was met very early.")
        print("Evaluating the final trained model instead.")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True) # Use the final model
        print(f"Final model mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()