from typing import Optional
import numpy as np
import gymnasium as gym


class TempEnv(gym.Env):

    def __init__(self):
      pass

    def _get_obs(self):
      """Convert internal state to observation format."""
      pass

    def _get_info(self):
      """Compute auxiliary information for debugging."""
      pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
      """Start a new episode."""
      # IMPORTANT: Must call this first to seed the random number generator
      super().reset(seed=seed)

      pass

    def step(self, action):
      """Execute one timestep within the environment."""

      observation = self._get_obs()
      info = self._get_info()
      terminated = False
      truncated = False
      reward = False

      return observation, reward, terminated, truncated, info


