# 5. Evaluation

The primary objective of this evaluation is to demonstrate that our proposed parser system can successfully bridge the gap between formal \(\text{d}\mathcal{L}\) models and practical reinforcement learning. We evaluate whether the RL agent, trained in an environment synthesized directly from the KeYmaera X AST, can internalize formal safety invariants while simultaneously achieving liveness goals.

## 5.1 Experimental Setup
All experiments were conducted using the environment architecture defined by our transpiler. 
*   **Physics Engine:** Continuous dynamics were simulated using the `scipy.integrate.solve_ivp` solver, governed by the ODEs extracted from the `stepFunc` block.
*   **Initial State Sampling:** Initial states were drawn from a Z3-populated, diverse state buffer subject to the model's `assumptions(...)` predicate.
*   **Reward Mapping:** For Car 2, reward and termination behavior came from the model's `@safety` and `@progress` annotations. The training script did not add separate reward-shaping terms or an extra ODE-failure penalty.
*   **MDP Formulation:** The environment was modeled as a Markov Decision Process (MDP), where the state space \(S\) and action space \(A\) are derived directly from the \(\text{d}\mathcal{L}\) program variables and the `controlFunc` choices, respectively.

For the Car 2 waypoint-and-deadline experiment (Test 3), the observation contains the six model variables \((x_r, v_r, a_r, x_g, t, T)\); the training script adds no derived features or normalization. It uses Stable-Baselines3 PPO with `MaskablePPO` by default, masking `controlFunc` branches whose guards are false. Training defaults to 10,000 environment timesteps, the SB3 default constant learning rate of \(3 \times 10^{-4}\), and an entropy coefficient of \(0.01\). Evaluation uses deterministic actions over 20 episodes by default. The training count is timesteps, not episodes.

## 5.2 Test 1: Validating Safety via the "Safe-Lock" Phenomenon
Our first experiment utilized a purely safety-oriented model (*Theorem 1: Static Safety*). The objective was to verify if the parser correctly translated the collision avoidance invariant \(\textit{isSafe}\) into a termination signal.

**Model Snippet:**
```cpp
// Formal Guard in controlFunc
if (v + A*ts < maxV && dist > stopDist(v + A*ts)) { a := A; }
else { a := -b; } // Fallback to braking
```

**Results:** The agent achieved a high mean reward (\(\approx 1.97 \times 10^5\)) and maximum episode lengths. However, state evolution plots (Fig. 5.1) revealed that the agent learned to maintain a constant velocity of \(v=0\). 

**Analysis:** This result validates the parser's accuracy. The agent discovered that by choosing the braking/stopped branch of the `controlFunc`, it could satisfy the safety invariant indefinitely. While the agent was **formally safe**, it lacked **liveness** (movement toward a goal). This confirmed that a direct mapping of safety logic is insufficient for functional control and necessitated the introduction of progress-oriented annotations.

![Test 1 states](test1_states.png)

## 5.3 Test 2: System Validation via Forced Dynamics
In the second stage, we conducted a diagnostic stress test to verify the integration of the ODE solver and the safety monitoring system. To ensure the environment was functioning correctly beyond the "safe-lock" state, we manually restricted the agent’s action space, forcing it to exclusively select the acceleration branch (\(a := A\)).

**Results:** As expected, the agent transitioned to a linearly increasing velocity profile. However, the episodes were consistently truncated after approximately 10 steps. 

**Analysis:** This test provided critical validation for two components of our framework:
1.  **Physics Integration:** The smooth, linear increase in velocity confirmed that the transpiler correctly mapped the symbolic ODEs to the numerical solver.
2.  **Safety Monitor Accuracy:** The rapid truncation demonstrated that the safety monitor was functioning as intended. As velocity \(v\) increased, the required stopping distance—\(\text{stopDist}(v) = \frac{v^2}{2b}\)—expanded quadratically. The environment correctly triggered a violation the moment the "safety buffer" exceeded the available separation distance.

![Test 1 forced states](test1_forced_states.png)

## 5.4 Test 3: Car 2 Waypoint Reachability with a Deadline
Test 3 uses the Car 2 model for *Theorem 14: Reach waypoint with deadline*. The robot moves along one dimension toward the interval defined by `waypointStartDist(xg)` and `waypointEndDist(xg)`. Its loop invariant constrains velocity, stopping distance, and the remaining deadline \(T\).

The policy selects among four guarded `controlFunc` branches: brake in the waypoint region, remain stopped there, accelerate when the guard permits, or coast when acceleration is not permitted. In the default training configuration, `MaskablePPO` applies these guards as an action mask. The environment monitors the loop invariant and the `ensure` predicate; the model annotations define the safety penalty and success/failure termination behavior.

The training script records episode rewards and lengths, then evaluates the policy deterministically and reports waypoint successes and termination reasons. Success rate is an empirical result for a particular trained model and evaluation run, not a fixed property of the environment. Report it with the checkpoint and evaluation episode count; the default is 20 episodes. Use the corresponding training and trajectory plots to support any claim about a learned velocity profile or deadline behavior rather than presenting a particular profile as guaranteed.


## 5.5 Quantitative Summary

| Experiment | Strategy Observed | Safety | Liveness | Success Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Test 1** | Static (Stay Still) | Absolute | None | 100% (Trivial) |
| **Test 2** | Aggressive (Forced Accel) | Failed | High | 0% (Safety Violation) |
| **Test 3** | Guard-masked PPO over Car 2 controller branches | Invariant monitored at runtime | Waypoint success measured per evaluation run | Run-specific (20 episodes by default) |

**Note on Success Rates:** Success is measured by evaluating the trained policy against the model's `ensure` predicate. Report the result for the exact checkpoint and evaluation run; a runtime success rate is not itself a formal verification result.

## 5.6 Conclusion of Evaluation
These experiments assess the generated Gymnasium environment's ability to expose model dynamics, controller choices, rewards, and runtime checks to an RL policy. In Test 3, the policy is evaluated for waypoint success under the model's deadline and invariant conditions. These empirical results demonstrate runtime integration of the formal model; they do not establish a formal safety proof or guarantee a particular learned trajectory.