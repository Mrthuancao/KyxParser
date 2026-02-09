import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import random
from collections import deque
import sys
import os
import time
import matplotlib.pyplot as plt # UNCOMMENT THIS LINE

# Add project root and AST_parser to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root: {project_root} to sys.path.")

ast_parser_path = os.path.join(project_root, 'AST_parser')
if ast_parser_path not in sys.path:
    sys.path.insert(0, ast_parser_path)
    print(f"Added AST_parser directory: {ast_parser_path} to sys.path for ast_config.")

test_path = os.path.join(project_root, 'test')
if test_path not in sys.path:
    sys.path.insert(0, test_path)
    print(f"Added test directory: {test_path} to sys.path for environment import.")

# Import the environment class
try:
    # Assuming the environment is in ideal_output_python/test1.py
    from ideal_output_python.test1 import IJRR17_Theorem_1__Static_safety
    print("Successfully imported environment: IJRR17_Theorem_1__Static_safety")
except ImportError as e:
    print(f"Error importing environment: {e}")
    print("Please ensure 'ideal_output_python/test1.py' exists and is correctly structured.")
    sys.exit(1)

# --- Configuration ---
# Hyperparameters for the RL agent
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
CRITIC_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
PPO_EPOCHS = 10
MINI_BATCH_SIZE = 64
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.01 # For early stopping PPO epochs
LOG_STD_MAX = 2 # Max value for log_std to prevent very large std
LOG_STD_MIN = -20 # Min value for log_std to prevent very small std

# Environment specific constant (will be updated from env.constants['W'] in run_training_for_env)
W_VAL = 0.5 

# --- Policy Network (Actor-Critic) ---
class PolicyNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_dict, W_val):
        super(PolicyNetwork, self).__init__()

        self.W_val = W_val # Max angular velocity for 'w'

        # Determine input size for the policy network
        self.obs_dim = observation_space_shape[0]

        # Determine output sizes for discrete and continuous actions
        self.num_discrete_choices = action_space_dict['choice_ChoiceLevel_0'].n
        
        # Continuous actions: w, r, xo, yo
        self.num_continuous_actions = sum(1 for k in action_space_dict if k.startswith('assign_'))

        # Shared layers for actor and critic
        self.shared_layers = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Actor (policy) head for discrete actions
        self.actor_discrete_head = nn.Linear(256, self.num_discrete_choices)

        # Actor (policy) head for continuous actions (mean and log_std)
        self.actor_continuous_mean_head = nn.Linear(256, self.num_continuous_actions)
        self.actor_continuous_log_std_head = nn.Linear(256, self.num_continuous_actions)

        # Critic (value) head
        self.critic_head = nn.Linear(256, 1)

    def forward(self, obs):
        # DEBUG: Check for NaN in input observation
        if torch.isnan(obs).any():
            print(f"DEBUG: PolicyNetwork input 'obs' contains NaN: {obs}")
            # Consider returning NaNs or zeros to propagate the issue for easier debugging
            return torch.full((obs.shape[0], self.num_discrete_choices), float('nan'), device=obs.device), \
                   torch.full((obs.shape[0], self.num_continuous_actions), float('nan'), device=obs.device), \
                   torch.full((obs.shape[0], self.num_continuous_actions), float('nan'), device=obs.device), \
                   torch.full((obs.shape[0], 1), float('nan'), device=obs.device)

        shared_output = self.shared_layers(obs)
        if torch.isnan(shared_output).any():
            print(f"DEBUG: PolicyNetwork 'shared_output' contains NaN: {shared_output}")

        # Discrete action logits
        discrete_logits = self.actor_discrete_head(shared_output)
        if torch.isnan(discrete_logits).any():
            print(f"DEBUG: PolicyNetwork 'discrete_logits' contains NaN: {discrete_logits}")

        # Continuous action mean and log_std
        mean_continuous = self.actor_continuous_mean_head(shared_output)
        log_std_continuous = self.actor_continuous_log_std_head(shared_output)
        
        # Clamp log_std to avoid numerical issues and ensure stability
        log_std_continuous = torch.clamp(log_std_continuous, LOG_STD_MIN, LOG_STD_MAX)
        if torch.isnan(mean_continuous).any() or torch.isnan(log_std_continuous).any():
            print(f"DEBUG: PolicyNetwork 'mean_continuous' or 'log_std_continuous' contains NaN: mean={mean_continuous}, log_std={log_std_continuous}")


        # Value estimate
        value = self.critic_head(shared_output)
        if torch.isnan(value).any():
            print(f"DEBUG: PolicyNetwork 'value' contains NaN: {value}")

        return discrete_logits, mean_continuous, log_std_continuous, value

# --- Agent Implementation (PPO) ---
class PPOAgent:
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        # W_val will be updated from the environment's constants during training setup
        self.W_val = W_VAL 

        self.policy_network = PolicyNetwork(
            env.observation_space.shape,
            env.action_space,
            W_val=self.W_val # Pass W_val to the policy network
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)

        self.buffer = [] # Stores (state, action, log_prob, reward, next_state, done)
        self.gamma = GAMMA
        self.gae_lambda = GAE_LAMBDA
        self.ppo_epsilon = PPO_EPSILON
        self.critic_loss_coef = CRITIC_LOSS_COEF
        self.entropy_coef = ENTROPY_COEF
        self.ppo_epochs = PPO_EPOCHS
        self.mini_batch_size = MINI_BATCH_SIZE
        self.max_grad_norm = MAX_GRAD_NORM
        self.target_kl = TARGET_KL

        # Store the mapping from action space keys to their indices for continuous actions
        self.continuous_action_keys = [k for k in env.action_space.keys() if k.startswith('assign_')]
        self.discrete_action_key = 'choice_ChoiceLevel_0'
        
        # Set the agent's reference in the environment for constant updates
        # This allows the environment to update agent.W_val if env.constants['W'] changes
        self.env.agent = self 

    def choose_action(self, observation):
        state = torch.FloatTensor(observation).to(self.device)
        
        # Forward pass through the policy network
        discrete_logits, mean_continuous, log_std_continuous, _ = self.policy_network(state)

        # Discrete action sampling
        discrete_dist = Categorical(logits=discrete_logits)
        discrete_action = discrete_dist.sample()
        discrete_log_prob = discrete_dist.log_prob(discrete_action)

        # Continuous action sampling
        std_continuous = torch.exp(log_std_continuous)
        continuous_dist = Normal(mean_continuous, std_continuous)
        continuous_actions_raw = continuous_dist.sample()
        # Sum log_probs for all continuous actions, as they are independent
        continuous_log_prob = continuous_dist.log_prob(continuous_actions_raw).sum(axis=-1) 

        # Combine log probabilities
        total_log_prob = discrete_log_prob + continuous_log_prob

        # Convert continuous actions to environment's expected format
        # Apply any necessary clipping or scaling based on environment's action space definition.
        # For 'w', it's constrained by -W <= w <= W.
        
        # Create the action dictionary for the environment
        env_action = {
            self.discrete_action_key: discrete_action.item()
        }
        
        # Populate continuous actions in the correct order
        for i, key in enumerate(self.continuous_action_keys):
            val = continuous_actions_raw[i].item()
            if key == 'assign_w':
                # Clip 'w' to [-W_val, W_val]
                env_action[key] = np.clip(val, -self.W_val, self.W_val)
            else:
                env_action[key] = val

        return env_action, total_log_prob.item() # Return action dict and its log_prob

    def store_transition(self, state, action_dict, log_prob, reward, next_state, done):
        # Convert action_dict back to a flat tensor/array for batching in the buffer.
        # The first element will be the discrete choice.
        # Subsequent elements will be the continuous assignments in a fixed order.
        
        action_array = [action_dict[self.discrete_action_key]]
        for key in self.continuous_action_keys:
            action_array.append(action_dict[key])
        
        # Ensure all elements are floats for consistency
        action_tensor = torch.FloatTensor(action_array).to(self.device)

        self.buffer.append((
            torch.FloatTensor(state).to(self.device),
            action_tensor,
            torch.FloatTensor([log_prob]).to(self.device),
            torch.FloatTensor([reward]).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor([done]).to(self.device)
        ))

    def compute_advantages_and_returns(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0

        # Iterate backwards to compute GAE and returns
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # If episode ends, next value is 0 (or value of next_state if not done)
                next_value = 0.0 
            else:
                next_value = values[t+1]

            # GAE formula: delta = R_t + gamma * V(S_{t+1}) * (1 - D_t) - V(S_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            # A_t = delta_t + gamma * lambda * (1 - D_t) * A_{t+1}
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lam
        
        # Returns = Advantages + Values
        returns = advantages + values
        return advantages, returns

    def learn(self):
        if not self.buffer:
            return

        # Stack all experiences from the buffer
        batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_dones = zip(*self.buffer)
        
        batch_states = torch.stack(batch_states)
        batch_actions = torch.stack(batch_actions)
        batch_log_probs = torch.stack(batch_log_probs).squeeze(1) # Remove extra dimension
        batch_rewards = torch.stack(batch_rewards).squeeze(1)
        batch_dones = torch.stack(batch_dones).squeeze(1)

        # Clear buffer after collecting a batch
        self.buffer = []

        # DEBUG: Check batch_states before passing to network
        if torch.isnan(batch_states).any():
            print(f"DEBUG: learn(): batch_states contains NaN before policy_network call: {batch_states}")
            return # Skip learning for this batch if input is bad

        # Get initial values from the current policy network for GAE calculation
        _, _, _, batch_values = self.policy_network(batch_states)
        if torch.isnan(batch_values).any():
            print(f"DEBUG: learn(): batch_values from policy_network contains NaN: {batch_values}")
            return # Skip learning if network already output NaN

        batch_values = batch_values.squeeze(1)

        # Compute advantages and returns
        advantages, returns = self.compute_advantages_and_returns(batch_rewards, batch_values, batch_dones)
        
        # DEBUG: Check advantages and returns
        if torch.isnan(advantages).any() or torch.isnan(returns).any():
            print(f"DEBUG: learn(): advantages or returns contain NaN: advantages={advantages}, returns={returns}")
            return # Skip learning if GAE calculation produced NaN

        # --- CRITICAL FIX: Detach advantages and returns from the graph ---
        # These values are calculated using the OLD policy and should be treated as constants
        # for the policy and value updates in the PPO epochs.
        advantages = advantages.detach()
        returns = returns.detach()
        
        # Normalize advantages for better training stability
        if advantages.numel() <= 1: # Check if advantages has 0 or 1 element
            print(f"WARNING: advantages tensor has {advantages.numel()} element(s). std() may be 0 or undefined. Normalization might be trivial.")
            # If advantages has only one element, std() is 0.
            # Normalization would be division by 0, even with 1e-8 it's not meaningful.
            # In this case, we allow the 1e-8 to prevent division by zero, resulting in 0.0 for advantages.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create a DataLoader for mini-batching
        dataset = torch.utils.data.TensorDataset(
            batch_states, batch_actions, batch_log_probs, advantages, returns
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)

        for epoch in range(self.ppo_epochs):
            for mini_batch_states, mini_batch_actions, mini_batch_old_log_probs, mini_batch_advantages, mini_batch_returns in loader:
                
                # DEBUG: Check mini_batch_states
                if torch.isnan(mini_batch_states).any():
                    print(f"DEBUG: learn(): mini_batch_states contains NaN inside PPO epoch: {mini_batch_states}")
                    continue # Skip this mini-batch

                # Forward pass for the minibatch
                discrete_logits, mean_continuous, log_std_continuous, values_pred = self.policy_network(mini_batch_states)
                
                # DEBUG: Check outputs of policy_network
                if torch.isnan(discrete_logits).any() or torch.isnan(mean_continuous).any() or \
                   torch.isnan(log_std_continuous).any() or torch.isnan(values_pred).any():
                    print(f"DEBUG: learn(): Policy network output contains NaN in PPO epoch. discrete_logits={discrete_logits}, mean_continuous={mean_continuous}, log_std_continuous={log_std_continuous}, values_pred={values_pred}")
                    continue # Skip this mini-batch
                
                values_pred = values_pred.squeeze(1)

                # --- Calculate new log probabilities ---
                # Discrete action log prob
                discrete_dist = Categorical(logits=discrete_logits)
                # mini_batch_actions[:, 0] contains the discrete action indices
                new_log_prob_discrete = discrete_dist.log_prob(mini_batch_actions[:, 0].long())

                # Continuous action log prob
                std_continuous = torch.exp(log_std_continuous)
                
                # FIX: Define continuous_dist here
                continuous_dist = Normal(mean_continuous, std_continuous)
                
                # mini_batch_actions[:, 1:] contains the continuous action values
                new_log_prob_continuous = continuous_dist.log_prob(mini_batch_actions[:, 1:]).sum(axis=-1)
                
                # Combined new log probability
                new_log_probs = new_log_prob_discrete + new_log_prob_continuous

                # --- PPO Loss Calculation ---
                # Policy Loss (Actor Loss)
                ratio = torch.exp(new_log_probs - mini_batch_old_log_probs)
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    print(f"DEBUG: learn(): ratio contains NaN or Inf: {ratio}")
                    # Optionally, clip ratio to prevent explosion, or skip this mini-batch
                    # ratio = torch.clamp(ratio, 1e-10, 1e10) # Example clipping
                    continue # Skip this mini-batch if ratio is problematic

                surr1 = ratio * mini_batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * mini_batch_advantages
                
                if torch.isnan(surr1).any() or torch.isnan(surr2).any():
                    print(f"DEBUG: learn(): surr1 or surr2 contains NaN: surr1={surr1}, surr2={surr2}")
                    continue # Skip this mini-batch

                policy_loss = -torch.min(surr1, surr2).mean()
                if torch.isnan(policy_loss).any():
                    print(f"DEBUG: learn(): policy_loss contains NaN: {policy_loss}")
                    continue # Skip this mini-batch

                # Value Loss (Critic Loss)
                value_loss = self.critic_loss_coef * nn.functional.mse_loss(values_pred, mini_batch_returns)
                if torch.isnan(value_loss).any():
                    print(f"DEBUG: learn(): value_loss contains NaN: {value_loss}")
                    continue # Skip this mini-batch

                # Entropy Loss (for exploration)
                entropy_discrete = discrete_dist.entropy().mean()
                # Sum entropies for all continuous actions
                entropy_continuous = continuous_dist.entropy().sum(axis=-1).mean() 
                entropy_loss = -self.entropy_coef * (entropy_discrete + entropy_continuous)
                if torch.isnan(entropy_loss).any():
                    print(f"DEBUG: learn(): entropy_loss contains NaN: {entropy_loss}")
                    continue # Skip this mini-batch

                # Total Loss
                total_loss = policy_loss + value_loss + entropy_loss
                if torch.isnan(total_loss).any():
                    print(f"DEBUG: learn(): total_loss contains NaN: {total_loss}")
                    continue # Skip this mini-batch

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # DEBUG: Check gradients for NaNs
                for name, param in self.policy_network.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"DEBUG: learn(): NaN gradient found in {name} before optimizer step: {param.grad}")
                        # Optionally, skip optimizer step or reset gradients
                        # self.optimizer.zero_grad()
                        # break # Break from parameter loop, continue to next mini-batch
                
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Early stopping check for PPO (optional but good practice)
            # Calculate KL divergence for the entire batch after an epoch
            with torch.no_grad():
                # DEBUG: Check batch_states before policy_network call for KL divergence
                if torch.isnan(batch_states).any():
                    print(f"DEBUG: learn(): batch_states contains NaN before KL divergence policy_network call: {batch_states}")
                    break # Break from epoch loop if input is bad

                discrete_logits_full, mean_continuous_full, log_std_continuous_full, _ = self.policy_network(batch_states)
                
                # DEBUG: Check outputs for KL divergence calculation
                if torch.isnan(discrete_logits_full).any():
                    print(f"DEBUG: learn(): discrete_logits_full contains NaN during KL divergence check: {discrete_logits_full}")
                    break # Break from epoch loop if output is bad

                discrete_dist_full = Categorical(logits=discrete_logits_full)
                new_log_prob_discrete_full = discrete_dist_full.log_prob(batch_actions[:, 0].long())

                std_continuous_full = torch.exp(log_std_continuous_full)
                continuous_dist_full = Normal(mean_continuous_full, std_continuous_full)
                new_log_prob_continuous_full = continuous_dist_full.log_prob(batch_actions[:, 1:]).sum(axis=-1)
                
                new_log_probs_full = new_log_prob_discrete_full + new_log_prob_continuous_full

                kl_divergence = (batch_log_probs - new_log_probs_full).mean()
                if kl_divergence > 1.5 * self.target_kl:
                    print(f"Early stopping PPO epoch {epoch} due to large KL divergence: {kl_divergence:.4f}")
                    break

        # After all epochs, the buffer should be empty, but good to ensure
        self.buffer = []

# --- Main Training Loop ---
def run_training_for_env(
    env_class,
    num_episodes=500,
    max_steps_per_episode=200,
    evaluate_every_n_episodes=50,
    EVALUATE_ONLY=False
):
    print(f"Starting training for {env_class.__name__}...")
    env = env_class()
    
    # Update the global W_VAL for the agent from the environment's constants
    global W_VAL
    # This will now correctly get 0.5 because env.constants['W'] is initialized to 0.5
    W_VAL = env.constants.get('W', 0.5) 
    print(f"Agent's W_VAL set to: {W_VAL}")

    agent = PPOAgent(env, device='cpu') # Using CPU for simplicity

    if EVALUATE_ONLY:
        print("Running in EVALUATE_ONLY mode.")
        # Load a pre-trained model if available
        # agent.policy_network.load_state_dict(torch.load("path_to_model.pth"))
        # agent.policy_network.eval()
        evaluate_agent(env, agent, num_episodes=10) # Run a few evaluation episodes
        return

    total_rewards = []
    episode_rewards = deque(maxlen=100) # For average over last 100 episodes

    for episode in range(num_episodes):
        observation, info = env.reset()
        # DEBUG: Check initial observation for NaNs
        if np.isnan(observation).any():
            print(f"DEBUG: Initial observation contains NaN in episode {episode+1}: {observation}")
            # Skip episode or handle
            continue

        done = False
        truncated = False
        current_episode_reward = 0

        for step in range(max_steps_per_episode):
            # Agent chooses action
            action_dict, log_prob = agent.choose_action(observation)
            
            # Environment takes a step
            next_observation, reward, done, truncated, info = env.step(action_dict)
            
            # DEBUG: Check next_observation for NaNs
            if np.isnan(next_observation).any():
                print(f"DEBUG: next_observation contains NaN at step {step+1} in episode {episode+1}: {next_observation}")
                # Terminate episode immediately if state becomes NaN
                reward -= 500.0 # Heavy penalty for NaN state
                done = True # Mark as done
            
            # Store transition
            agent.store_transition(observation, action_dict, log_prob, reward, next_observation, done or truncated)
            
            observation = next_observation
            current_episode_reward += reward

            if done or truncated:
                break
        
        # Agent learns after each episode (or after a batch of episodes, depending on PPO setup)
        agent.learn() 

        total_rewards.append(current_episode_reward)
        episode_rewards.append(current_episode_reward)

        print(f"Episode {episode+1}/{num_episodes}, Reward: {current_episode_reward:.2f}, Avg100: {np.mean(episode_rewards):.2f}")

        if (episode + 1) % evaluate_every_n_episodes == 0:
            print(f"--- Evaluating agent after {episode+1} episodes ---")
            evaluate_agent(env, agent)
            print(f"--- End Evaluation ---")
            # Optionally save model
            # torch.save(agent.policy_network.state_dict(), f"ppo_agent_ep{episode+1}.pth")

    env.close()
    return total_rewards

def evaluate_agent(env, agent, num_episodes=5):
    agent.policy_network.eval() # Set policy to evaluation mode
    eval_rewards = []
    for _ in range(num_episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            # In evaluation, we often just take the mean of the continuous distribution
            # or sample from it without exploration noise. For simplicity, let's sample.
            action_dict, _ = agent.choose_action(observation) 
            observation, reward, done, truncated, _ = env.step(action_dict)
            episode_reward += reward
        eval_rewards.append(episode_reward)
    
    print(f"Evaluation over {num_episodes} episodes: Mean Reward = {np.mean(eval_rewards):.2f}, Std Dev = {np.std(eval_rewards):.2f}")
    agent.policy_network.train() # Set policy back to training mode

# --- Entry Point ---
if __name__ == "__main__":
    # Define W_VAL globally for the agent's initialization
    # It will be updated from the environment in run_training_for_env
    W_VAL = 0.5 

    # Example of how to run training
    # Set EVALUATE_ONLY to True to only run evaluation with a (potentially pre-trained) agent
    EVALUATE_ONLY = False 

    rewards = run_training_for_env(
        IJRR17_Theorem_1__Static_safety,
        num_episodes=50, # Reduced for quicker test runs
        max_steps_per_episode=50, # Reduced for quicker test runs
        evaluate_every_n_episodes=10,
        EVALUATE_ONLY=EVALUATE_ONLY
    )

    # You can plot rewards here if matplotlib is installed
    # UNCOMMENT THE FOLLOWING LINES TO PLOT THE REWARDS
    plt.plot(rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()