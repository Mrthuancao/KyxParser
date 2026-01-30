import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

# Import stable-baselines3 components
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# --- YOUR ACTUAL CAR ENV CLASS (from ideal_car_env.py) ---
# # Make sure your CarEnv is defined here or imported correctly.
# # This CarEnv is a placeholder. If your actual CarEnv
# # like 'ideal_car_env.py', you should import it:
# from ideal_car_env import CarEnv # This line would be in your training script

# For the purpose of this explanation, we'll define CarEnv here
# as if it were in ideal_car_env.py
class CarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # 1. Define Observation Space: x pos, y pos, velocity, angle, fuel
        # Ensure this matches your actual CarEnv's observation space
        self.observation_space = spaces.Box(
            low=np.array([-100, -100, -10, -np.pi, 0], dtype=np.float64),
            high=np.array([100, 100, 10, np.pi, 100], dtype=np.float64),
            dtype=np.float64
        )

        # 2. Define Action Space: 0=brake, 1=turn_left, 2=turn_right, 3=reverse, 4=accelerate
        # Ensure this matches your actual CarEnv's action space
        self.action_space = spaces.Discrete(5)

        # Environment parameters (you'll need to tune these)
        self.max_episode_steps = 200 # Max steps per episode, matches your terminal output
        self.current_step = 0
        self.target_position = np.array([50.0, 50.0]) # Example target

        # State variables (these will be initialized in reset())
        self.x = 0.0
        self.y = 0.0
        self.velocity = 0.0
        self.angle = 0.0 # Radians
        self.fuel = 100.0

        self.render_mode = render_mode

    def _get_obs(self):
        """Helper to return the current observation based on internal state."""
        return np.array([self.x, self.y, self.velocity, self.angle, self.fuel], dtype=np.float64)

    def _get_info(self):
        """Helper to return additional info (optional)."""
        return {
            "distance_to_target": np.linalg.norm(np.array([self.x, self.y]) - self.target_position),
            "current_position": np.array([self.x, self.y])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for Gymnasium's seeding

        # Initialize car's state to a starting position
        self.x = self.np_random.uniform(low=-10.0, high=10.0) # Random start x
        self.y = self.np_random.uniform(low=-10.0, high=10.0) # Random start y
        self.velocity = 0.0
        self.angle = self.np_random.uniform(low=-np.pi, high=np.pi) # Random start angle
        self.fuel = 100.0 # Full tank

        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.current_step += 1

        # --- 3. Implement Car Dynamics (This is where the magic happens!) ---
        # These are very basic physics. You'll need to replace them with your actual car model.

        # Constants for movement
        ACCELERATION_RATE = 0.5
        BRAKE_RATE = 1.0
        TURN_RATE = np.pi / 16 # Radians per step
        FUEL_CONSUMPTION_PER_STEP = 0.1
        FUEL_CONSUMPTION_PER_ACCELERATION = 0.5
        FUEL_CONSUMPTION_PER_TURN = 0.05

        # Apply action
        if action == 0: # Brake
            self.velocity = max(0, self.velocity - BRAKE_RATE)
        elif action == 1: # Turn Left
            self.angle += TURN_RATE
            self.fuel -= FUEL_CONSUMPTION_PER_TURN
        elif action == 2: # Turn Right
            self.angle -= TURN_RATE
            self.fuel -= FUEL_CONSUMPTION_PER_TURN
        elif action == 3: # Reverse
            self.velocity = max(-5, self.velocity - ACCELERATION_RATE) # Max reverse speed -5
            self.fuel -= FUEL_CONSUMPTION_PER_ACCELERATION
        elif action == 4: # Accelerate
            self.velocity = min(10, self.velocity + ACCELERATION_RATE) # Max forward speed 10
            self.fuel -= FUEL_CONSUMPTION_PER_ACCELERATION

        # Ensure angle wraps around
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi

        # Update position based on velocity and angle
        self.x += self.velocity * np.cos(self.angle)
        self.y += self.velocity * np.sin(self.angle)

        # Basic fuel consumption
        self.fuel = max(0, self.fuel - FUEL_CONSUMPTION_PER_STEP)

        # --- 4. Define Reward Function ---
        # This is CRUCIAL for guiding the agent.
        # Example: Reward for getting closer to a target, penalize fuel consumption, penalize going out of bounds.

        reward = 0.0
        # Reward for moving towards target
        old_distance = np.linalg.norm(np.array([self.x - self.velocity * np.cos(self.angle),
                                                self.y - self.velocity * np.sin(self.angle)]) - self.target_position)
        current_distance = np.linalg.norm(np.array([self.x, self.y]) - self.target_position)

        # Reward for reducing distance to target
        reward += (old_distance - current_distance) * 0.5

        # Small penalty for each step to encourage efficiency
        reward -= 0.01

        # Penalty for fuel consumption
        reward -= (FUEL_CONSUMPTION_PER_STEP + (FUEL_CONSUMPTION_PER_ACCELERATION if action in [3,4] else 0) + (FUEL_CONSUMPTION_PER_TURN if action in [1,2] else 0)) * 0.1

        # --- 5. Define Termination Conditions ---
        terminated = False
        truncated = False

        # Terminate if car reaches target
        if current_distance < 5.0: # Within 5 units of target
            reward += 100.0 # Big reward for reaching target
            terminated = True
            print(f"Target reached! Final reward: {reward}")

        # Terminate if out of bounds (e.g., x or y outside [-100, 100])
        if not self.observation_space.contains(self._get_obs()):
             # Check if x, y are out of bounds (simplified for example)
            if not (-100 <= self.x <= 100 and -100 <= self.y <= 100):
                reward -= 50.0 # Penalty for going out of bounds
                terminated = True
                print(f"Out of bounds! Final reward: {reward}")

        # Terminate if fuel runs out
        if self.fuel <= 0:
            reward -= 20.0 # Penalty for running out of fuel
            terminated = True
            print(f"Out of fuel! Final reward: {reward}")

        # Truncate if max steps reached
        if self.current_step >= self.max_episode_steps:
            truncated = True
            print(f"Max steps reached. Final reward: {reward}")

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # This is a placeholder for actual rendering.
        # In a real scenario, you'd use a library like Pygame, Matplotlib, or OpenGL.
        # For now, we'll just print the state, which is what your terminal output shows.
        if self.render_mode == "human":
            print(f"Render: X={self.x:.2f}, Y={self.y:.2f}, Vel={self.velocity:.2f}, "
                  f"Angle={np.degrees(self.angle):.2f}, Fuel={self.fuel:.1f}")
        # If render_mode is "rgb_array", you would return an actual image (numpy array) here.
        # For now, returning a dummy array if in rgb_array mode.
        if self.render_mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8) # Dummy image

    def close(self):
        pass # No resources to close for this simple example