# Documentation: RL-Annotation Syntax for KeYmaera X

## 1. Overview
The **RL-Annotation Syntax** is a comment-based Domain-Specific Language (DSL) designed to bridge KeYmaera X formal Differential Dynamic Logic (\(dL\)) models with Gymnasium Reinforcement Learning environments. By annotating definitions with category-specific tags like `@safety`, `@progress`, `@stability`, `@const`, or `@var`, developers can define complex reward structures, termination logic, and episode status directly within their formal models, without altering the underlying proof logic.

The transpiler parses these annotations, attaches them to the corresponding Abstract Syntax Tree (AST) nodes, and maps them to the `step()` and `reset()` functions of a Gymnasium environment, facilitating a seamless transition from formal verification to RL training.

## 2. General Syntax
Annotations must be placed on the line immediately preceding a `Bool`, `Real`, or `HP` definition.

**Template:**
```java
// @Category: key1=value1, key2=value2, ...
Definition name(...) <-> ...
```
Where `Category` is one of `safety`, `progress`, `stability`, `const`, or `var`.

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

## 6. Transpiler Implementation Notes

The Python transpiler is designed to process these annotations and generate robust Gymnasium environment code.

### 6.1. Pre-processing (Regex Matching)
The tool scans the KeYmaera X model file for annotation comments. The updated regex logic should identify the category-first tags and extract the associated attributes.

```python
import re

def parse_annotations(file_content):
    # Matches @category and the following definition.
    # Captures the category, the attribute string, and the definition name.
    pattern = r"// @(safety|progress|stability|const|var): (.*?)\n\s*(?:Bool|Real|HP)\s+(\w+)"
    matches = re.findall(pattern, file_content)
    
    annotations = {}
    for category, attr_string, def_name in matches:
        attrs = {}
        # This regex captures key=value pairs, handling curly braces, square brackets,
        # and quoted strings for values.
        attr_pairs_raw = re.findall(
            r'(\w+)\s*=\s*'
            r'(?:(\{.*?\})|'   # Group 1: curly braces
            r'(\[.*?\])|'   # Group 2: square brackets
            r'(\".*?\")|'   # Group 3: quoted string
            r'([^,]*))',    # Group 4: anything else (non-comma)
            attr_string
        )
        for match_tuple in attr_pairs_raw:
            key = match_tuple # The actual key
            # Find the first non-empty value from the capturing groups
            value_raw = next((v for v in match_tuple[1:] if v), '').strip()
            
            # Remove surrounding delimiters for the final dict value
            if (value_raw.startswith('{') and value_raw.endswith('}')) or \
               (value_raw.startswith('[') and value_raw.endswith(']')) or \
               (value_raw.startswith('"') and value_raw.endswith('"')):
                attrs[key] = value_raw[1:-1].strip()
            else:
                # Handle on_success/on_failure values with optional status
                if key in ['on_success', 'on_failure']:
                    parts = value_raw.split(':', 1) # Split only on the first colon
                    action = parts.strip()
                    status = 'none' # Default status if not specified
                    if len(parts) > 1:
                        status_part = parts.strip()
                        if status_part.startswith('status='):
                            status = status_part[len('status='):].strip()
                    attrs[key] = {'action': action, 'status': status}
                else:
                    attrs[key] = value_raw
        
        attrs['type'] = category  # Automatically assign type based on the tag used
        annotations[def_name] = attrs
            
    return annotations
```
*Note: The regex for parsing `attr_string` is designed to handle nested curly braces to a certain degree, but highly complex nested expressions might require a more sophisticated parser (e.g., a stack-based approach for brace matching).*

### 6.2. AST Linking
During the traversal of the KeYmaera X Abstract Syntax Tree (AST), when a `FunctionDefinition` node (for `Bool`, `Real`, or `HP` definitions) is encountered, the pre-parsed metadata from the `annotations` dictionary is injected directly into that node's properties. This ensures that each relevant AST node carries its associated RL configuration.

### 6.3. Code Generation (`env.step()`)
The Python code generator iterates through all annotated AST nodes. For each node, it generates Python code that will be part of the Gymnasium `env.step()` method. This involves:
*   Evaluating the boolean predicate of the definition to determine `condition_met`.
*   Evaluating the `shape` expression.
*   Conditionally evaluating `reward` or `penalty` based on `condition_met`.
*   Summing these components to compute the `total_reward`.
*   Applying `on_success` or `on_failure` logic to update `terminated`, `truncated`, and `info['is_success']`.

The generated Python logic for each annotated predicate will resemble the following structure. Note that `is_terminated`, `is_truncated`, and `info_is_success` are typically aggregated across all predicates in a single `step()`.

```python
def calculate_predicate_reward_and_state(predicate_name, state_variables, current_global_state, annotations):
    # current_global_state would be a dictionary holding the accumulated
    # terminated, truncated, and is_success from previous predicates
    # Initial values for current_global_state might be:
    # {'terminated': False, 'truncated': False, 'is_success': None}
    total_reward_for_predicate = 0
    
    # Evaluate the predicate's boolean condition
    # (This part is specific to how your transpiler evaluates KeYmaera X predicates)
    # Placeholder for actual predicate evaluation
    condition_met = evaluate_keymaera_predicate(predicate_name, state_variables) 

    # Retrieve annotations for this predicate
    annotation_data = annotations.get(predicate_name, {})

    # 1. Apply the continuous shape (Always)
    if 'shape' in annotation_data:
        shape_expr = annotation_data['shape']
        total_reward_for_predicate += evaluate_expression(shape_expr, state_variables)

    # 2. Apply discrete logic (reward or penalty) and control attributes
    if condition_met:
        if 'reward' in annotation_data:
            reward_expr = annotation_data['reward']
            total_reward_for_predicate += evaluate_expression(reward_expr, state_variables)
        
        # Apply on_success logic
        on_success_config = annotation_data.get('on_success', {'action': 'continue', 'status': 'none'})
        if on_success_config['action'] == 'terminate':
            current_global_state['terminated'] = True
        elif on_success_config['action'] == 'truncate':
            current_global_state['truncated'] = True
        
        if on_success_config['status'] == 'success':
            current_global_state['is_success'] = True
        elif on_success_config['status'] == 'failure':
            current_global_state['is_success'] = False
        # If status is 'none', it doesn't modify current_global_state['is_success']
    else: # condition_met is False
        if 'penalty' in annotation_data:
            penalty_expr = annotation_data['penalty']
            total_reward_for_predicate += evaluate_expression(penalty_expr, state_variables)
        
        # Apply on_failure logic
        on_failure_config = annotation_data.get('on_failure', {'action': 'continue', 'status': 'none'})
        if on_failure_config['action'] == 'terminate':
            current_global_state['terminated'] = True
        elif on_failure_config['action'] == 'truncate':
            current_global_state['truncated'] = True
        
        if on_failure_config['status'] == 'success':
            current_global_state['is_success'] = True
        elif on_failure_config['status'] == 'failure':
            current_global_state['is_success'] = False
        # If status is 'none', it doesn't modify current_global_state['is_success']
    
    return total_reward_for_predicate, current_global_state

# Placeholder functions for demonstration purposes
def evaluate_keymaera_predicate(predicate_name, state_variables):
    # This function would contain the transpiled KeYmaera X predicate logic
    # For example:
    if predicate_name == "safeDist":
        return state_variables['dist_to_obstacle'] > 0
    elif predicate_name == "atGoal":
        x, y, gx, gy = state_variables['x'], state_variables['y'], state_variables['gx'], state_variables['gy']
        return (x-gx)**2 + (y-gy)**2 < 0.1
    return False # Default

def evaluate_expression(expression_str, state_variables):
    # This function would safely evaluate the expression string using the state_variables
    # WARNING: Using eval() directly with untrusted input is a security risk.
    # A safer approach would be to parse the expression into an AST and evaluate it.
    try:
        # For demonstration, we'll use a simplified eval. In a real transpiler,
        # you'd replace KeYmaera X operators and then evaluate safely.
        # Example: Replace '^' with '**', '&' with 'and', '|' with 'or'
        python_expression = expression_str.replace('^', '**').replace('&', 'and').replace('|', 'or')
        # Handle 'if A then B else C' -> 'B if A else C'
        if 'if' in python_expression and 'then' in python_expression and 'else' in python_expression:
            parts = re.match(r'if (.*?) then (.*?) else (.*)', python_expression)
            if parts:
                condition, true_val, false_val = parts.groups()
                python_expression = f"({true_val} if {condition} else {false_val})"

        # Create a dictionary for local variables to pass to eval
        eval_locals = {k: v for k, v in state_variables.items()}
        # Add common math functions if needed
        eval_locals['abs'] = abs 
        
        return eval(python_expression, {}, eval_locals)
    except Exception as e:
        print(f"Error evaluating expression '{expression_str}': {e}")
        return 0 # Or handle error appropriately

```
*The `evaluate_keymaera_predicate` and `evaluate_expression` functions would be generated code that translates KeYmaera X logic into executable Python.*

**Aggregation of `terminated`, `truncated`, and `is_success`:**
When multiple predicates are evaluated in a single `step()`, the `terminated` and `truncated` flags are typically combined using a logical OR. If *any* predicate sets `terminated` or `truncated` to `True`, the episode ends.
For `info['is_success']`, the aggregation needs careful consideration:
*   If any predicate sets `is_success` to `False`, the overall `info['is_success']` for the step should be `False`. This takes precedence (a single failure means overall failure).
*   Otherwise, if any predicate sets `is_success` to `True`, the overall `info['is_success']` for the step is `True`.
*   If no predicate explicitly sets `is_success` (i.e., all `status` were `none`), the overall `info['is_success']` might remain `None` or default to `False` (depending on the environment's general policy).

---

## 7. Why this System is Superior

1.  **Semantic Clarity:** Category-first tags (`@safety`, `@progress`, `@stability`, `@const`, `@var`) provide immediate implicit meaning. The explicit `on_success` and `on_failure` attributes, now with distinct `action` and `status` components, further enhance readability by clearly stating the consequences of a predicate's truth value on both episode flow and overall success.
2.  **Reward Decomposition Power:** Separating `reward`, `penalty`, and `shape` allows for highly nuanced reward engineering. This directly combats the "sparse reward" problem by providing continuous `shape` signals while retaining discrete `reward` bonuses or `penalty` costs for specific logical outcomes.
3.  **Decoupling Logic:** It clearly distinguishes between the formal verification logic (the predicate itself) and the RL training strategy (the annotations), yet keeps them co-located for maintainability.
4.  **Granular Control over `is_success`:** The new `status` sub-attribute within `on_success` and `on_failure` provides precise control over when `info['is_success']` is set, and to what value, independent of whether the predicate itself is true or false. This directly addresses the concern that a predicate's success doesn't always imply agent success.
5.  **Symmetric and Explicit Control:** The `on_success` and `on_failure` attributes offer a symmetrical and comprehensive way to define episode termination and success/failure outcomes for *both* true and false predicate conditions, eliminating ambiguity.
6.  **Robustness for Safety:** Explicit `on_failure=truncate:status=failure` for `@safety` conditions provides a strong, unambiguous signal for safety violations, crucial in safety-critical systems.
7.  **Flexibility for Progress:** `on_success=terminate:status=success` clearly marks successful task completion, while `on_success=continue:status=none` allows for tracking sub-goals or intermediate conditions without immediate termination or defining overall success.
8.  **Extensibility:** The structured key-value pair format makes it easy to add new attributes in the future without breaking existing parsing logic.
9.  **Direct Mapping to Gymnasium:** The design directly maps to Gymnasium's `step()` function return values (`reward`, `terminated`, `truncated`, `info`), simplifying the transpilation process and ensuring correct `is_success` reporting.