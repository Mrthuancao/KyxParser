# Documentation: RL-Annotation Syntax for KeYmaera X

## 1. Overview
The **RL-Annotation Syntax** is a comment-based Domain-Specific Language (DSL) designed to bridge KeYmaera X formal Differential Dynamic Logic (\(dL\)) models with Gymnasium Reinforcement Learning environments. By annotating definitions with category-specific tags like `@safety`, `@progress`, `@stability`, `@const`, or `@var`, developers can define complex reward structures, termination logic, and episode status directly within their formal models, without altering the underlying proof logic.

The transpiler parses these annotations and attaches them as metadata to the corresponding Abstract Syntax Tree (AST) nodes. This structured configuration is then interpreted by the Gymnasium environment wrapper, which evaluates the annotated logic during the step() and reset() cycles to calculate rewards and state transitions..

## 2. General Syntax
Annotations must be placed on the line immediately preceding a `Bool`, `Real`, or `HP` definition.

**Template:**
```java
// @Category: key1=value1, key2=value2, ...
Definition name(...) <-> ...
```
Where `Category` is one of `safety`, `progress`, `stability`, `const`, or `var` and `Definition` is one of `Bool`, `Real`, or `HP`.

---

## 3. Reward and Termination Logic Annotations (`@safety`, `@progress`, `@stability`)

This section details annotations used to define the reward structure, episode termination conditions, and success/failure status based on the evaluation of KeYmaera X predicates.

### 3.1. The Reward Decomposition Model
To effectively address the "Sparse Reward" problem often encountered in reinforcement learning with formal models, this syntax decomposes the step-wise reward into three distinct components. The total reward for a given step is calculated based on whether the annotated predicate's condition evaluates to `True` or `False`:

$$
R_{total} = \text{shape} + 
\begin{cases} 
\text{reward} & \text{if condition is True} \\
\text{penalty} & \text{if condition is False}
\end{cases}
$$

This decomposition allows for fine-grained control over the reward signal, separating discrete bonuses/costs from continuous guidance.

| Attribute | Logic Type | When is it applied? | Purpose |
| :-------- | :--------- | :------------------ | :------ |
| **`reward`** | Discrete   | Only if the predicate's condition is **True**. | A "bonus" for satisfying the condition, encouraging desired states. |
| **`penalty`** | Discrete   | Only if the predicate's condition is **False**. | A "cost" for violating the condition, discouraging undesired states. |
| **`shape`** | Continuous | **Always** applied, regardless of the condition's truth value. | A "gradient" signal to guide the agent toward the goal or desired state. |

### 3.2. Category Tags for Reward and Termination Logic

The primary tag (`@safety`, `@progress`, `@stability`) implicitly categorizes the associated logic, enhancing readability and providing default behaviors for reward and termination.

#### `@safety`
Used for logic that defines the "safe operating envelope" of the agent. These predicates are critical for preventing undesirable or dangerous states.
*   **Primary Use Cases:** Invariants (e.g., `loopinv`), collision avoidance, physical validity.
*   **Default Behavior:** Typically uses `on_failure=truncate:status=failure` to immediately stop the episode and mark it as a failure if safety is violated. `on_success` usually defaults to `continue:status=none` as simply being safe doesn't mean success.
*   **Example:** Penalizing proximity to an obstacle and truncating the episode if a critical safety distance is violated.
    ```java
    // @safety: penalty=-500, shape={dist_to_obstacle}, on_failure=truncate:status=failure
    Bool safeDist(Real dist_to_obstacle) <-> dist_to_obstacle > 0;
    ```
    *   If `dist_to_obstacle > 0` (True): Reward = `dist_to_obstacle`. `on_success` is not specified, so it defaults to `continue:status=none`. No termination, and `info['is_success']` is not set by this predicate.
    *   If `dist_to_obstacle <= 0` (False): Reward = `-500 + dist_to_obstacle`. `on_failure=truncate:status=failure` sets `truncated=True` and `info['is_success']=False`.

#### `@progress`
Used for logic that encourages the agent to achieve a specific goal, maintain a desired state, or make forward progress in a task.
*   **Primary Use Cases:** Post-conditions (e.g., `ensure`), velocity maintenance, reaching target coordinates.
*   **Default Behavior:** Often uses `on_success=terminate:status=success` for goal achievement, and `on_failure=continue:status=none` to allow the agent to keep trying without explicitly marking failure.
*   **Example:** Rewarding the agent for reaching a goal and shaping based on the inverse distance to the goal.
    ```java
    // @progress: reward=1000, penalty={-((x-gx)^2 + (y-gy)^2) + 0.1}, on_success=terminate:status=success, on_failure=continue:status=none
    Bool atGoal(Real x, Real y, Real gx, Real gy) <-> (x-gx)^2 + (y-gy)^2 < 0.1;
    ```
    *   If `(x-gx)^2 + (y-gy)^2 < 0.1` (True): Reward = `1000 - ((x-gx)^2 + (y-gy)^2)`. `on_success=terminate:status=success` sets `terminated=True` and `info['is_success']=True`.
    *   If `(x-gx)^2 + (y-gy)^2 >= 0.1` (False): Reward = `-((x-gx)^2 + (y-gy)^2)`. `on_failure=continue:status=none` means no termination, and `info['is_success']` is not set by this predicate.

#### `@stability`
Used for logic derived from **Lyapunov functions** or attractivity proofs, guiding the agent towards a stable equilibrium point.
*   **Primary Use Cases:** Parking maneuvers, maintaining equilibrium, reducing energy.
*   **Default Behavior:** Typically uses `shape` to provide a potential-based reward and `on_success=continue:status=none`, `on_failure=continue:status=none` as these predicates usually do not directly terminate an episode or define overall success/failure.
*   **Example:** Guiding the agent to a point where a Lyapunov function \(V\) is minimized.
    ```java
    // @stability: shape={-V}, on_success=continue:status=none, on_failure=continue:status=none
    Real V(Real x, Real y) = x^2 + y^2;
    ```
    *   The reward is always `-V`, encouraging the agent to move towards the origin (where \(V\) is minimal). Neither termination nor `info['is_success']` is affected by this predicate.

### 3.3. Control Attributes for Reward and Termination
These attributes determine the episode's state and metadata in the Gymnasium environment. They explicitly define the action and outcome based on whether the predicate's condition evaluates to `True` or `False`.

*   **`on_success=[terminate | truncate | continue][:status=[success | failure | none]]`**: Specifies the action to take and the outcome to report if the predicate's condition evaluates to **True**.
    *   **Action (`terminate`, `truncate`, `continue`):**
        *   `terminate`: Sets `terminated = True`.
        *   `truncate`: Sets `truncated = True`.
        *   `continue`: The episode continues.
    *   **Optional `status` (`success`, `failure`, `none`):** This part is optional. If specified, it explicitly sets `info['is_success']`. If omitted, or set to `none`, this predicate does not explicitly modify `info['is_success']`.
        *   `status=success`: Sets `info['is_success'] = True`.
        *   `status=failure`: Sets `info['is_success'] = False`.
        *   `status=none`: Does not modify `info['is_success']`.
    *   **Default if omitted:** `on_success=continue:status=none`.
    *   **Example:** `on_success=terminate:status=success` (ends episode, marks as success). `on_success=continue:status=none` (continues, does not affect overall success status).

*   **`on_failure=[terminate | truncate | continue][:status=[success | failure | none]]`**: Specifies the action to take and the outcome to report if the predicate's condition evaluates to **False**.
    *   **Action (`terminate`, `truncate`, `continue`):** Same as `on_success` actions.
    *   **Optional `status` (`success`, `failure`, `none`):** Same as `on_success` statuses.
    *   **Default if omitted:** `on_failure=continue:status=none`.
    *   **Example:** `on_failure=truncate:status=failure` (truncates episode, marks as failure). `on_failure=continue:status=none` (continues, does not affect overall success status).

### 3.4. Expression Syntax
The expressions for `reward`, `penalty`, and `shape` can be simple values, variable mappings, or complex mathematical formulas.
*   **Operators**: The transpiler automatically converts KeYmaera X operators to their Python equivalents:
    *   `^` \(\rightarrow\) `**` (for exponentiation)
    *   `&` \(\rightarrow\) `and` (for logical AND)
    *   `|` \(\rightarrow\) `or` (for logical OR)
    *   `if A then B else C` \(\rightarrow\) `B if A else C` (for conditional expressions). The Python-style `B if A else C` is also directly supported within the annotation expressions.

---

## 4. Environment Definition Annotations (`@const`, `@var`)

This section describes annotations used to define the fixed parameters and state variables of the Gymnasium environment, which are crucial for configuring the observation space and initial conditions.

### 4.1. Category Tags for Environment Definition

#### `@const`
Used for `Real` definitions that represent fixed parameters of the environment or system. These values might be sampled from a nondeterministic range at the start of an episode.
*   **Primary Use Cases:** Physical constants, environmental parameters, configuration settings.
*   **Attributes:**
    *   **`bound=[min:max]`**: (Optional) Specifies a nondeterministic range from which the constant's value will be sampled at the beginning of each episode. `min` and `max` should be real numbers.
    *   **`def="description"`**: (Optional) A descriptive string explaining the constant's purpose.
*   **Example:**
    ```java
    // @const: bound=[0.1:0.5], def="time limit for control decisions"
    Real ep;
    ```

#### `@var`
Used for `Real` or `Bool` variables that define the state space of the environment. These annotations help in configuring the Gymnasium observation space.
*   **Primary Use Cases:** Robot position, velocity, orientation, time, obstacle positions.
*   **Attributes:**
    *   **`bound=[min:max]`**: (Optional) Specifies the valid numerical range for a `Real` variable. `min` and `max` should be real numbers. This is used to define the observation space limits.
    *   **`def="description"`**: (Optional) A descriptive string explaining the variable's purpose.
*   **Example:**
    ```java
    // @var: bound=[0:100], def="robot position: x"
    Real x;
    ```

### 4.2. Plotting and Diagnostics
To facilitate training diagnosis, `@const` and `@var` support a `plot` attribute. This tells the transpiler/logger which variables to track and how to visualize them.

*   **`plot=[type]`**: Specifies the visualization style for this variable.
    *   `line`: Standard time-series plot (Value vs. Step).
    *   `hist`: Distribution of the variable's values across the episode.
    *   `scatter`: Used when tracking relationships between variables.
    *   `none` (default): Variable is not plotted.

**Example:**
```java
// @var: bound=[-10:10], plot=line, def="lateral error"
Real e_y;
```

---

## 5. Comprehensive Example

This example illustrates how to annotate a robot model navigating to a point while maintaining a safe distance from an obstacle, ensuring its orientation is well-formed, and defining its constants and variables.

```java
// Physical Constants
// @const: bound=[0.1:0.5], def="time limit for control decisions"
Real ep; 
// @const: bound=[0.5:5.0], def="minimum braking capability of the robot"
Real b;       
// @const: bound=[1.0:10.0], def="maximum acceleration"
Real A;       
// @const: bound=[0.1:1.0], def="maximum steering (rotational velocity)"
Real W;       

// Program Variables
// @var: bound=[0:100], def="robot position: x"
Real x;    
// @var: bound=[0:100], def="robot position: y"
Real y;    
// @var: bound=[0:20], def="robot translational velocity"
Real v;   
// @var: bound=[-10:10], def="robot translational acceleration"
Real a;    
// @var: bound=[-1:1], def="robot orientation: x component"
Real dx;   
// @var: bound=[-1:1], def="robot orientation: y component"
Real dy;   
// @var: bound=[-2:2], def="robot rotational velocity"
Real w;    
// @var: bound=[-1000:1000], def="robot curve radius (r!=0)"
Real r;    
// @var: bound=[0:100], def="position of closest obstacle on curve: x"
Real xo;  
// @var: bound=[0:100], def="position of closest obstacle on curve: y"
Real yo;   
// @var: bound=[0:1], def="time elapsed in current control step"
Real t;  

// Directional Integrity:
// Ensures the orientation vector (dx, dy) remains normalized.
// @progress: reward=10, shape={-((dx**2 + dy**2) - 1)**2}, on_success=continue:status=none, on_failure=continue:status=none
Bool isWellformedDir(Real dx, Real dy) <-> dx^2 + dy^2 = 1;

// Physical Validity:
// Prevents the agent from entering physically impossible states (negative velocity).
// @safety: penalty=-500, on_failure=truncate:status=failure, on_success=continue:status=none
Bool isPhysicallyValid(Real v, Real r) <-> (
  v >= 0 & r != 0
);

// Collision Avoidance (Core Safety):
// The primary safety invariant. The shape encourages maximizing the gap
// between current distance and required stopping distance.
// @safety: penalty=-1000, shape={(abs(x-xo) - stopDist(v)) + (abs(y-yo) - stopDist(v))}, on_failure=truncate:status=failure, on_success=continue:status=none
Bool isSafe(Real x, Real y, Real v, Real xo, Real yo) <-> (abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v));

// Velocity Efficiency:
// Encourages the agent to maintain a productive velocity.
// @progress: shape={1/v if v > 0 else v}, on_success=continue:status=none, on_failure=continue:status=none
Bool isEfficient(Real v) <-> ( v >= 0 );

// Goal Condition:
// This predicate defines the goal of avoiding the obstacle.
// If the condition is true (robot is not at the obstacle), it's a success and the episode terminates.
// If the condition is false (robot is at the obstacle), it's a failure and the episode terminates.
// @progress: reward=500, penalty=-1000, on_success=terminate:status=success, on_failure=terminate:status=failure
Bool ensure(Real x, Real y, Real xo, Real yo) <-> (
  (x - xo)^2 + (y - yo)^2 > 0
);
```
---

## 6. Why this System is Superior

1.  **Semantic Clarity:** Category-first tags (`@safety`, `@progress`, `@stability`, `@const`, `@var`) provide immediate implicit meaning. The explicit `on_success` and `on_failure` attributes, now with distinct `action` and `status` components, further enhance readability by clearly stating the consequences of a predicate's truth value on both episode flow and overall success.
2.  **Reward Decomposition Power:** Separating `reward`, `penalty`, and `shape` allows for highly nuanced reward engineering. This directly combats the "sparse reward" problem by providing continuous `shape` signals while retaining discrete `reward` bonuses or `penalty` costs for specific logical outcomes.
3.  **Decoupling Logic:** It clearly distinguishes between the formal verification logic (the predicate itself) and the RL training strategy (the annotations), yet keeps them co-located for maintainability.
4.  **Granular Control over `is_success`:** The new `status` sub-attribute within `on_success` and `on_failure` provides precise control over when `info['is_success']` is set, and to what value, independent of whether the predicate itself is true or false. This directly addresses the concern that a predicate's success doesn't always imply agent success.
5.  **Symmetric and Explicit Control:** The `on_success` and `on_failure` attributes offer a symmetrical and comprehensive way to define episode termination and success/failure outcomes for *both* true and false predicate conditions, eliminating ambiguity.
6.  **Robustness for Safety:** Explicit `on_failure=truncate:status=failure` for `@safety` conditions provides a strong, unambiguous signal for safety violations, crucial in safety-critical systems.
7.  **Flexibility for Progress:** `on_success=terminate:status=success` clearly marks successful task completion, while `on_success=continue:status=none` allows for tracking sub-goals or intermediate conditions without immediate termination or defining overall success.
8.  **Extensibility:** The structured key-value pair format makes it easy to add new attributes in the future without breaking existing parsing logic.
9.  **Direct Mapping to Gymnasium:** The design directly maps to Gymnasium's `step()` function return values (`reward`, `terminated`, `truncated`, `info`), simplifying the transpilation process and ensuring correct `is_success` reporting.