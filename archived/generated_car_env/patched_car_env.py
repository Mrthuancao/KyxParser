import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os # Import the os module to check for file existence

# --- YOUR ACTUAL CAR ENV CLASS (from ideal_car_env.py) ---
# Make sure your CarEnv is defined here or imported correctly.
# This CarEnv is a placeholder. If your actual CarEnv is in a separate file
# like 'ideal_car_env.py', you should import it:
# from ideal_car_env import CarEnv
class CarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Example observation space: x_pos, y_pos, velocity, angle, fuel
        # Ensure this matches your actual CarEnv's observation space
        self.observation_space = spaces.Box(low=np.array([-100, -100, -10, -np.pi, 0]),
                                            high=np.array([100, 100, 10, np.pi, 100]),
                                            dtype=np.float64)
        # Example action space: 0=brake, 1=turn_left, 2=turn_right, 3=reverse, 4=accelerate
        # Ensure this matches your actual CarEnv's action space
        self.action_space = spaces.Discrete(5)

        self.state = {
            'x_position': np.float64(0.0),
            'y_position': np.float64(0.0),
            'velocity': np.float64(0.0),
            'angle': np.float64(0.0), 
            'fuel': np.float64(100.0)
        }
        self.current_step = 0
        self.max_steps = 200 # Increased max steps for a longer episode

    def _get_obs(self):
        # Ensure this returns the observation array in the correct format
        return np.array([self.state['x_position'], self.state['y_position'],
                         self.state['velocity'], self.state['angle'],
                         self.state['fuel']], dtype=np.float64)

    def _get_info(self):
        return {} # You can add more info here if needed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = {
            'x_position': np.float64(0.0),
            'y_position': np.float64(0.0),
            'velocity': np.float64(0.0),
            'angle': np.float64(0.0),
            'fuel': np.float64(100.0)
        }
        self.current_step = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Placeholder dynamics - REPLACE WITH YOUR ACTUAL CarEnv DYNAMICS
        reward = 0.01 # Base reward
        terminated = False
        truncated = False

        # Apply action
        if action == 4: # Accelerate
            self.state['velocity'] += 0.2
            self.state['x_position'] += self.state['velocity'] * np.cos(self.state['angle']) * 0.1
            self.state['y_position'] += self.state['velocity'] * np.sin(self.state['angle']) * 0.1
            self.state['fuel'] -= 0.1
            reward += 0.05 # Bonus for accelerating
        elif action == 0: # Brake
            self.state['velocity'] = max(0, self.state['velocity'] - 0.3)
            self.state['fuel'] -= 0.05
            reward -= 0.02 # Small penalty for braking
        elif action == 1: # Turn Left
            self.state['angle'] += 0.1
            self.state['fuel'] -= 0.05
            reward -= 0.01 # Small penalty for turning
        elif action == 2: # Turn Right
            self.state['angle'] -= 0.1
            self.state['fuel'] -= 0.05
            reward -= 0.01 # Small penalty for turning
        elif action == 3: # Reverse
            self.state['velocity'] -= 0.1
            self.state['x_position'] += self.state['velocity'] * np.cos(self.state['angle']) * 0.1
            self.state['y_position'] += self.state['velocity'] * np.sin(self.state['angle']) * 0.1
            self.state['fuel'] -= 0.1
            reward -= 0.03 # Penalty for reversing

        # Clamp values
        self.state['velocity'] = np.clip(self.state['velocity'], -5, 5) # Adjusted velocity range
        self.state['fuel'] = np.clip(self.state['fuel'], 0, 100)
        self.state['angle'] = (self.state['angle'] + np.pi) % (2 * np.pi) - np.pi # Keep angle in [-pi, pi]

        # Check for termination/truncation
        if self.state['fuel'] <= 0:
            terminated = True
            reward = -1.0 # Penalize running out of fuel
        
        # Example: Terminate if car goes too far (e.g., out of bounds)
        if abs(self.state['x_position']) > 50 or abs(self.state['y_position']) > 50:
            terminated = True
            reward = -5.0 # Large penalty for going out of bounds

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 0.5 # Small penalty for truncating without finishing

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        # Implement your rendering logic here if you want to visualize
        print(f"Render: X={self.state['x_position']:.2f}, Y={self.state['y_position']:.2f}, "
              f"Vel={self.state['velocity']:.2f}, Angle={np.degrees(self.state['angle']):.2f}, Fuel={self.state['fuel']:.1f}")

    def close(self):
        pass
# --- END OF CarEnv PLACEHOLDER ---


if __name__ == "__main__":
    # Define model path and log directory
    model_path = "car_env_ppo_model.zip"
    log_dir = "./logs/"

    # 1. Create the environment
    # It's good practice to wrap your environment in a VecEnv for parallelization,
    # even if you only use one environment for simplicity.
    # make_vec_env automatically handles this.
    # If your CarEnv is in ideal_car_env.py, you would import it:
    # from ideal_car_env import CarEnv

    # For this example, we'll directly create the environment instances
    env = make_vec_env(lambda: CarEnv(), n_envs=1) # Use a lambda to create multiple instances if n_envs > 1
    eval_env = CarEnv() # Evaluation environment for callbacks

    # 2. Load existing model or create a new one
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} for continued training...")
        # When loading, it's good practice to pass the env to the model
        model = PPO.load(model_path, env=env)
        # Ensure the model's environment is set correctly if it was saved without it
        model.set_env(env)
    else:
        print("No existing model found. Starting new training run...")
        # Define the PPO model with hyperparameters
        # MlpPolicy means a Multi-layer Perceptron (feedforward neural network) policy
        # for both actor and critic.
        # verbose=1 will print training progress.
        model = PPO("MlpPolicy", env, verbose=1,
                    learning_rate=0.0003, # Common learning rate
                    n_steps=2048,        # Number of steps to run for each environment per update
                    batch_size=64,       # Minibatch size for SGD
                    gamma=0.99,          # Discount factor
                    gae_lambda=0.95,     # Factor for trade-off between bias and variance for Generalized Advantage Estimator
                    ent_coef=0.01,       # Entropy coefficient for exploration
                    seed=42)             # For reproducibility

    # Optional: Callbacks for evaluation and saving the best model
    # Stop training if the average reward over 5 episodes reaches 100
    # Note: A reward threshold of 100 might be very high for this placeholder env.
    # Adjust based on your actual environment's reward scale.
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
    # Evaluate the agent every 10000 timesteps and save the best model
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=log_dir,
                                 log_path=log_dir,
                                 eval_freq=10000, # Evaluate every 10,000 steps
                                 deterministic=True,
                                 render=False, # Set to True if you want to see rendering during eval
                                 callback_on_new_best=stop_callback)


    # 3. Train the agent
    print("\n--- Starting Training ---")
    # Define how many *additional* timesteps you want to train for.
    # This is a crucial parameter. RL often requires millions of steps.
    additional_timesteps = 200_000 # Example: Train for 200,000 more steps
    
    # The 'reset_num_timesteps=False' argument is crucial here!
    # It tells Stable Baselines3 to continue the internal timestep counter
    # from where the loaded model left off, rather than starting from 0.
    model.learn(total_timesteps=additional_timesteps,
                callback=eval_callback,
                reset_num_timesteps=False)
    print("--- Training Finished ---")

    # 4. Save the trained model (overwriting the old one or saving with a new name)
    # It's good practice to save the model after continued training.
    model.save(model_path)
    print(f"Updated model saved as {model_path}")

    # 5. (Optional) Load and test the trained model
    print("\n--- Testing Trained Model (After Continued Training) ---")
    # Delete the current model instance to ensure we load from disk.
    del model
    loaded_model = PPO.load(model_path)

    obs, info = eval_env.reset()
    total_test_reward = 0
    test_steps = 0
    # Run for a maximum of 2 episodes' worth of steps for testing
    for i in range(eval_env.max_steps * 2): 
        action, _states = loaded_model.predict(obs, deterministic=True) # deterministic=True for exploitation
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_test_reward += reward
        test_steps += 1
        eval_env.render() # See the agent in action
        if terminated or truncated:
            print(f"Episode finished after {test_steps} steps with total reward: {total_test_reward:.2f}")
            break
    if not (terminated or truncated): # If loop finishes without termination/truncation
        print(f"Episode truncated after {test_steps} steps with total reward: {total_test_reward:.2f}")

    eval_env.close()
    env.close()