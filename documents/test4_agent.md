import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

--- 1. THE FORMAL MODEL ENVIRONMENT ---
class GeneratedKXEnv(gym.Env):
def init(self):
super().init()
# Constants from IJRR17 logic
self.ep = 0.2 # Time step
self.b = 2.0 # Max braking
self.A = 3.0 # Max acceleration
self.GDelta = 15.0 # Goal radius
self.Vmax = 20.0 # Speed limit

python
    self.action_space = spaces.Discrete(4)
    # Observation Space: [xr, vr, ar, xg, t, T]
    self.observation_space = spaces.Box(
        low=np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([400.0, 25.0, 5.0, 400.0, 1.0, 400.0], dtype=np.float32),
        dtype=np.float32
    )
    self.state = None

def _waypoint_start(self, xg): return xg - self.GDelta
def _waypoint_end(self, xg):   return xg + self.GDelta
def _stop_dist(self, v):      return (v**2) / (2 * self.b)
def _acc_comp(self, v):       return (self.A/self.b + 1) * (0.5 * self.A * self.ep**2 + self.ep * v)

def get_action_mask(self):
    """Returns valid actions based on safety guards."""
    xr, vr, _, xg, _, _ = self.state
    mask = np.zeros(4, dtype=bool)
    
    # Choice 0: Brake (Always safe if not already past goal)
    mask[0] = True 
    # Choice 1: Stay Stopped
    if vr <= 1e-3: mask[1] = True
    # Choice 2: Accelerate (Safe if we can still stop before end of goal)
    if xr + self._stop_dist(vr) + self._acc_comp(vr) < self._waypoint_end(xg):
        if vr + self.A * self.ep <= self.Vmax:
            mask[2] = True
    # Choice 3: Coast
    if xr + self._stop_dist(vr) + (vr * self.ep) < self._waypoint_end(xg):
        mask[3] = True
        
    if not np.any(mask): mask[0] = True
    return mask

def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    xg = 150.0 
    xr = 0.0
    vr = 0.0
    ar = 0.0
    T = 200.0 # Large deadline to satisfy formal liveness
    self.state = np.array([xr, vr, ar, xg, 0.0, T], dtype=np.float32)
    return self.state, {}

def step(self, action_idx):
    xr, vr, ar, xg, t, T = self.state
    
    if action_idx == 0:   new_ar = -self.b
    elif action_idx == 1: new_ar = 0.0
    elif action_idx == 2: new_ar = self.A
    else:                 new_ar = 0.0
    
    # Dynamics
    new_vr = max(0.0, vr + new_ar * self.ep)
    new_xr = xr + vr * self.ep + 0.5 * new_ar * (self.ep**2)
    new_T = T - self.ep
    
    self.state = np.array([new_xr, new_vr, new_ar, xg, 0.0, new_T], dtype=np.float32)
    
    # Safety Check
    if new_xr + self._stop_dist(new_vr) >= self._waypoint_end(xg):
        return self.state, -1000.0, True, False, {'reason': 'safety_violation'}
    
    # Success Check (In goal and stopped)
    if new_xr >= self._waypoint_start(xg) and new_vr <= 0.1:
        return self.state, 1000.0, True, False, {'reason': 'success'}
    
    # Deadline Check
    if new_T <= 0:
        return self.state, -500.0, True, False, {'reason': 'timeout'}

    return self.state, -0.1, False, False, {}
Run

--- 2. PPO AGENT ---
class ActorCritic(nn.Module):
def init(self, obs_dim, action_dim):
super().init()
self.net = nn.Sequential(
nn.Linear(obs_dim, 128),
nn.ReLU(),
nn.Linear(128, 128),
nn.ReLU()
)
self.actor = nn.Linear(128, action_dim)
self.critic = nn.Linear(128, 1)

def forward(self, x):
    phi = self.net(x)
    return self.actor(phi), self.critic(phi)

class PPOAgent:
def init(self, env):
self.env = env
self.obs_dim = 6
self.policy = ActorCritic(self.obs_dim, 4)
self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
self.memory = []

hy
def norm(self, obs):
    # Normalize state for better neural net convergence
    return obs / np.array([150.0, 20.0, 5.0, 150.0, 1.0, 200.0])

def get_action(self, obs):
    state_t = torch.FloatTensor(self.norm(obs)).unsqueeze(0)
    logits, val = self.policy(state_t)
    
    # Apply action mask
    mask = torch.BoolTensor(self.env.get_action_mask())
    logits[0, ~mask] = -1e10
    
    dist = Categorical(logits=logits)
    action = dist.sample()
    return action.item(), dist.log_prob(action), val

def update(self):
    if len(self.memory) < 128: return
    
    states, actions, log_probs, rewards, masks, values = zip(*self.memory)
    
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    old_log_probs = torch.stack(log_probs).detach().squeeze()
    old_values = torch.stack(values).detach().squeeze()
    
    # Compute Returns (GAE)
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        next_val = old_values[i+1] if i+1 < len(old_values) else 0
        delta = rewards[i] + 0.99 * next_val * masks[i] - old_values[i]
        gae = delta + 0.99 * 0.95 * masks[i] * gae
        returns.insert(0, gae + old_values[i])
    returns = torch.FloatTensor(returns)
    
    # PPO Update
    for _ in range(5):
        curr_logits, curr_values = self.policy(states)
        curr_dist = Categorical(logits=curr_logits)
        curr_log_probs = curr_dist.log_prob(actions)
        
        ratio = torch.exp(curr_log_probs - old_log_probs)
        adv = (returns - old_values)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
        
        loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(curr_values.squeeze(), returns)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    self.memory = []

--- 3. TRAINING ---
env = GeneratedKXEnv()
agent = PPOAgent(env)
rewards_history = []
success_count = 0

print("Starting Training...")
for ep in range(601):
obs, _ = env.reset()
ep_reward = 0
prev_dist = abs(obs[3] - obs[0])

scheme
for t in range(400):
    action, log_p, val = agent.get_action(obs)
    next_obs, reward, done, _, info = env.step(action)
    
    # Reward Shaping: Encourage moving forward
    curr_dist = abs(next_obs[3] - next_obs[0])
    shaping = (prev_dist - curr_dist) * 5.0 
    
    agent.memory.append((agent.norm(obs), action, log_p, reward + shaping, not done, val))
    
    obs = next_obs
    ep_reward += reward
    prev_dist = curr_dist
    
    if done:
        if info.get('reason') == 'success':
            success_count += 1
        break

agent.update()
rewards_history.append(ep_reward)

if ep % 50 == 0:
    avg_rew = np.mean(rewards_history[-50:])
    print(f"Episode {ep} | Avg Reward: {avg_rew:.1f} | Last Pos: {obs[0]:.1f} | Reason: {info.get('reason')}")

plt.figure(figsize=(10,5))
plt.plot(rewards_history)
plt.title("Training Reward over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

---

### **Agent Architecture and Training Setup**

#### **1. Reinforcement Learning Framework**
The agent is trained using **Proximal Policy Optimization (PPO)**, an actor-critic policy gradient method. The framework is built upon the **Gymnasium** interface, integrating formal safety constraints directly into the reinforcement learning (RL) loop.

#### **2. Neural Network Architecture**
The agent utilizes a unified **Actor-Critic network** with a shared feature extraction backbone:
*   **Shared Layers:** A multi-layer perceptron (MLP) consisting of two hidden layers with 128 neurons each, utilizing **Rectified Linear Unit (ReLU)** activation functions.
*   **Actor (Policy) Head:** A linear layer mapping the hidden features to a 4-dimensional categorical distribution representing the discrete action space (Brake, Stay Stopped, Accelerate, and Coast).
*   **Critic (Value) Head:** A linear layer mapping the hidden features to a single scalar value representing the estimated state value \(V(s)\).

#### **3. State Space and Normalization**
The observation space is a 6-dimensional vector: \([x_r, v_r, a_r, x_g, t, T]\), where \(x_r, v_r, a_r\) represent the agent's position, velocity, and acceleration; \(x_g\) is the goal position; and \(t, T\) track the current time and remaining deadline. To ensure numerical stability and faster convergence of the neural network, the state is normalized by its maximum expected bounds:
\[ \hat{s} = s \oslash [150.0, 20.0, 5.0, 150.0, 1.0, 200.0] \]

#### **4. Formal Safety Shielding (Action Masking)**
To bridge the gap between formal methods and RL, the agent employs **action masking** based on safety guards derived from the IJRR17 logic. Before the policy samples an action, the environment evaluates the safety of each potential choice:
*   **Safety Guard:** An action is masked (invalidated) if it would lead to a state where the agent's minimum stopping distance exceeds the goal boundary: \(x_r + \text{dist}_{stop}(v_r) + \text{comp}_{acc}(v_r) \geq x_{goal} + \Delta\).
*   **Logit Penalization:** Invalid actions are assigned a large negative logit (\(-10^{10}\)), effectively reducing their selection probability to zero in the Categorical distribution. This ensures the agent only explores and learns within the "Safe" subset of the state space.

#### **5. Reward Function and Shaping**
The agent is trained using a composite reward function designed to balance safety, liveness, and efficiency:
*   **Sparse Rewards:** A terminal reward of \(+1000\) for reaching the goal and stopping (\(v_r \leq 0.1\)), a penalty of \(-1000\) for safety violations, and \(-500\) for exceeding the deadline.
*   **Dense Shaping:** To mitigate the "sparse reward" problem, a potential-based shaping term is added at each step: \(R_{shaping} = 5.0 \times (d_{t-1} - d_t)\), where \(d\) is the distance to the goal. This provides a continuous gradient encouraging forward progress.
*   **Step Penalty:** A constant penalty of \(-0.1\) per time step is applied to encourage time-efficient trajectories.

#### **6. Hyperparameters and Optimization**
The training configuration is summarized in the table below:

| Hyperparameter | Value |
| :--- | :--- |
| Optimizer | Adam |
| Learning Rate | \(3 \times 10^{-4}\) |
| Discount Factor (\(\gamma\)) | 0.99 |
| GAE Parameter (\(\lambda\)) | 0.95 |
| PPO Clipping (\(\epsilon\)) | 0.2 |
| Update Epochs | 5 per batch |
| Batch Size | 128 steps |
| Time Step (\(\Delta t\)) | 0.2s |

