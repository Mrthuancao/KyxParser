import gymnasium as gym
import numpy as np

class HeuristicAgent:
    """
    A heuristic agent for the GeneratedKXEnv environment.
    This agent prioritizes safety by braking if too close to the obstacle or if acceleration
    would violate safety constraints. Otherwise, it attempts to accelerate and turn
    away from the obstacle.
    """
    def __init__(self, env: gym.Env):
        # Store the unwrapped environment to access its specific attributes
        self.env = env.unwrapped 
        # Constants will be populated in env.reset(). We'll access them dynamically.

    def choose_action(self, observation: np.ndarray) -> dict:
        """
        Chooses an action based on the current state and a set of heuristics.

        Args:
            observation (np.ndarray): The current observation of the environment.

        Returns:
            dict: A dictionary representing the chosen action, matching the environment's
                  action space structure.
        """
        # Access named state variables directly from self.env.state for readability.
        # The environment's `reset` method ensures `self.env.state` and `self.env.constants` are populated.
        state = self.env.state

        # Retrieve necessary constants. Use .get() with a default in case a constant
        # is unexpectedly missing or zero (e.g., due to Z3 failure or initial defaults).
        # Access constants via self.env.constants
        A = self.env.constants.get('A', 1.0) 
        b = self.env.constants.get('b', 1.0) 
        W = self.env.constants.get('W', 1.0) 
        # ep = self.env.constants.get('ep', 1.0) # ep is used in ODEs, not directly in action choice logic

        # Initialize action with default values for non-chosen options.
        # Defaulting to braking (Choice 0) as a safe initial guess.
        action = {
            'choice_ChoiceLevel_0': 0, 
            'assign_w': 0.0,
            'assign_r': 1.0, # Default non-zero for r!=0 constraint in Choice 2
            'assign_xo': state['xo'], # Assume agent doesn't control obstacle position
            'assign_yo': state['yo'], # Assume agent doesn't control obstacle position
        }

        current_v = state['v']
        current_x = state['x']
        current_y = state['y']
        current_dx = state['dx']
        current_dy = state['dy']
        current_xo = state['xo']
        current_yo = state['yo']

        # --- Evaluate safety conditions ---

        # 1. Check `isSafe` predicate (part of the loop invariant).
        # If `b` is extremely small or zero, `stopDist` (and `admissibleSeparation`) can become problematic
        # (division by zero or extremely large values). We handle this by setting a floor.
        # Access equality_tolerance and functions via self.env
        if b < self.env.equality_tolerance:
            # If braking is impossible, assume stop_dist is effectively zero for safety checks
            # or handle as a critical failure condition. For now, treat as unable to brake.
            effective_stop_dist = 0.0 
            effective_admissible_sep = 0.0
        else:
            effective_stop_dist = self.env.functions['stopDist'](current_v)
            effective_admissible_sep = self.env.functions['admissibleSeparation'](current_v)

        # `isSafe` is `abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v)`.
        # If this is FALSE, the agent is in an unsafe region according to the loop invariant.
        is_currently_safe_by_loop_inv = (np.abs(current_x - current_xo) > effective_stop_dist) or \
                                        (np.abs(current_y - current_yo) > effective_stop_dist)

        # 2. Check the specific constraint for Choice 2 (acceleration).
        # Choice 2 requires `abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v)`.
        can_accelerate_safely_by_choice2_constraint = \
            (np.abs(current_x - current_xo) > effective_admissible_sep) or \
            (np.abs(current_y - current_yo) > effective_admissible_sep)

        # --- Agent Decision Logic ---

        # Prioritize safety: If the loop invariant's safety part is violated OR
        # if we cannot accelerate safely according to Choice 2's constraint, then BRAKE.
        if not is_currently_safe_by_loop_inv or not can_accelerate_safely_by_choice2_constraint:
            action['choice_ChoiceLevel_0'] = 0 # Action 0: a = -b (Brake)
            # The other assignment values (w, r, xo, yo) are ignored for Choice 0, but must be present.
            return action
        
        # If we are safe and can accelerate, then choose to accelerate (Action 2) and attempt to turn.
        action['choice_ChoiceLevel_0'] = 2 # Action 2: a = A (Accelerate)

        # Determine `w` (angular velocity) and `r` (radius of turn) for turning.
        target_w = 0.0
        target_r = 1.0 # Default for when w=0 (straight line)

        # If the vehicle is stopped or cannot turn (W is zero), then w must be 0.
        # Access equality_tolerance via self.env
        if current_v < self.env.equality_tolerance or W < self.env.equality_tolerance:
            target_w = 0.0 
            # If v is 0, then r*w=v implies r*w=0. Since r!=0, w must be 0.
            # If W is 0, no turning is possible.
        else:
            # Calculate vector from agent to obstacle
            vec_to_obstacle_x = current_xo - current_x
            vec_to_obstacle_y = current_yo - current_y
            
            # Calculate current direction vector
            current_dir_x = current_dx
            current_dir_y = current_dy

            # Dot product to check if obstacle is ahead: current_dir . vec_to_obstacle
            dot_product = current_dir_x * vec_to_obstacle_x + current_dir_y * vec_to_obstacle_y
            
            # Cross product (2D equivalent) to check which side the obstacle is on: current_dir x vec_to_obstacle
            # Positive cross_product_z means obstacle is to the left (relative to current direction)
            # Negative means obstacle is to the right
            cross_product_z = current_dir_x * vec_to_obstacle_y - current_dir_y * vec_to_obstacle_x

            distance_to_obstacle = np.sqrt(vec_to_obstacle_x**2 + vec_to_obstacle_y**2)

            # If the obstacle is generally ahead (dot_product > 0) and relatively close,
            # attempt to turn away from it.
            if dot_product > 0 and distance_to_obstacle < 15.0: # Threshold distance for active turning
                if cross_product_z > 0: # Obstacle is to the left, turn right (negative w)
                    target_w = -0.5 * W # Turn with half max angular velocity
                else: # Obstacle is to the right, turn left (positive w)
                    target_w = 0.5 * W
            else:
                # If obstacle is not directly ahead or far, try to maintain a straight path
                # or a very slight random turn to explore.
                target_w = 0.0 
                # Optional: Add a small random wiggle to encourage exploration if needed
                # target_w = 0.1 * W * self.env.np_random.uniform(-1, 1)

            # Ensure target_w is within the allowed bounds [-W, W]
            target_w = np.clip(target_w, -W, W)

            # Calculate `r` such that `r*w = v`.
            # If `w` is effectively zero, `r` can be any non-zero value (straight line).
            # Access equality_tolerance via self.env
            if np.abs(target_w) > self.env.equality_tolerance:
                target_r = current_v / target_w
                # Clip `r` to prevent extremely large or small values that might cause numerical instability
                target_r = np.clip(target_r, -1000.0, 1000.0) # Arbitrary reasonable bounds
            else:
                target_r = 1.0 # Default to 1.0 if w is zero (effectively infinite radius for straight motion)

        action['assign_w'] = target_w
        action['assign_r'] = target_r
        action['assign_xo'] = current_xo # Keep obstacle fixed
        action['assign_yo'] = current_yo # Keep obstacle fixed
        
        return action