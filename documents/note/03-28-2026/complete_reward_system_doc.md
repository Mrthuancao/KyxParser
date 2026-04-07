# Documentation: RL-Annotation Syntax for KeYmaera X

## 1. Overview
The **RL-Annotation Syntax** is a comment-based Domain-Specific Language (DSL) designed to bridge KeYmaera X formal Differential Dynamic Logic (\(dL\)) models with Gymnasium Reinforcement Learning environments. By annotating definitions with category-specific tags like `@safety`, `@progress`, or `@stability`, developers can define complex reward structures, termination logic, and episode status directly within their formal models, without altering the underlying proof logic.

The transpiler parses these annotations, attaches them to the corresponding Abstract Syntax Tree (AST) nodes, and maps them to the `step()` and `reset()` functions of a Gymnasium environment, facilitating a seamless transition from formal verification to RL training.

## 2. General Syntax
Annotations must be placed on the line immediately preceding a `Bool`, `Real`, or `HP` definition.

**Template:**
```java
// @Category: reward={expr}, penalty={expr}, shape={expr}, on_success=action[:status=outcome], on_failure=action[:status=outcome]
Definition name(...) <-> ...
```
Where `Category` is one of `safety`, `progress`, or `stability`.

## 3. The Reward Decomposition Model
To effectively address the "Sparse Reward" problem often encountered in reinforcement learning with formal models, this syntax decomposes the step-wise reward into three distinct components. The total reward for a given step is calculated based on whether the annotated predicate's condition evaluates to `True` or `False`:

[
R_{total} = \text{shape} + 
\begin{cases} 
\text{reward} & \text{if condition is True} \\
\text{penalty} & \text{if condition is False}
\end{cases}
]

This decomposition allows for fine-grained control over the reward signal, separating discrete bonuses/costs from continuous guidance.

| Attribute | Logic Type | When is it applied? | Purpose |
| :-------- | :--------- | :------------------ | :------ |
| **`reward`** | Discrete   | Only if the predicate's condition is **True**. | A "bonus" for satisfying the condition, encouraging desired states. |
| **`penalty`** | Discrete   | Only if the predicate's condition is **False**. | A "cost" for violating the condition, discouraging undesired states. |
| **`shape`** | Continuous | **Always** applied, regardless of the condition's truth value. | A "gradient" signal to guide the agent toward the goal or desired state. |

## 4. Category Tags & Termination Logic

The primary tag (`@safety`, `@progress`, `@stability`) implicitly categorizes the associated logic, enhancing readability and providing default behaviors.

### `@safety`
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

### `@progress`
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

### `@stability`
Used for logic derived from **Lyapunov functions** or attractivity proofs, guiding the agent towards a stable equilibrium point.
*   **Primary Use Cases:** Parking maneuvers, maintaining equilibrium, reducing energy.
*   **Default Behavior:** Typically uses `shape` to provide a potential-based reward and `on_success=continue:status=none`, `on_failure=continue:status=none` as these predicates usually do not directly terminate an episode or define overall success/failure.
*   **Example:** Guiding the agent to a point where a Lyapunov function \(V\) is minimized.
    ```java
    // @stability: shape={-V}, on_success=continue:status=none, on_failure=continue:status=none
    Real V(Real x, Real y) = x^2 + y^2;
    ```
    *   The reward is always `-V`, encouraging the agent to move towards the origin (where \(V\) is minimal). Neither termination nor `info['is_success']` is affected by this predicate.

## 5. Attribute Reference

### Control Attributes
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

### Expression Syntax
The expressions for `reward`, `penalty`, and `shape` can be simple values, variable mappings, or complex mathematical formulas.
*   **Mappings**: Can use `{var -> expression}` for clarity or direct expressions `{expression}`.
*   **Operators**: The transpiler automatically converts KeYmaera X operators to their Python equivalents:
    *   `^` \(\rightarrow\) `**` (for exponentiation)
    *   `&` \(\rightarrow\) `and` (for logical AND)
    *   `|` \(\rightarrow\) `or` (for logical OR)
    *   `if A then B else C` \(\rightarrow\) `B if A else C` (for conditional expressions)

## 6. Comprehensive Example
This example illustrates how to annotate a robot model navigating to a point while maintaining a safe distance from an obstacle, and ensuring its orientation is well-formed.

```java
/** 
 * Safety: Penalize being close to obstacle, truncate if distance <= 0.
 * The 'shape' encourages increasing distance.
 * If safe, the episode continues and overall success is not affected by this predicate.
 */
// @safety: penalty=-500, shape={dist}, on_failure=truncate:status=failure, on_success=continue:status=none
Bool safe(Real dist) <-> dist > 0;

/** 
 * Progress: Reward reaching the goal, with a continuous shape based on squared distance.
 * The episode terminates successfully upon reaching the goal.
 * If not at goal, the episode continues and overall success is not affected by this predicate.
 */
// @progress: reward=1000, shape={-((x-gx)^2 + (y-gy)^2)}, on_success=terminate:status=success, on_failure=continue:status=none
Bool goal(Real x, Real y, Real gx, Real gy) <-> (x-gx)^2 + (y-gy)^2 < 0.1;

/** 
 * Progress: Ensure the robot's orientation vector is well-formed (unit vector).
 * The 'shape' continuously pulls the vector towards a unit length, and a bonus is given when it's perfect.
 * This predicate does not terminate the episode, nor does it directly determine overall success,
 * but it provides a reward signal.
 */
// @progress: reward=10, shape={-((dx^2 + dy^2) - 1)^2}, on_success=continue:status=none, on_failure=continue:status=none
Bool isWellformedDir(Real dx, Real dy) <-> dx^2 + dy^2 = 1;

/** 
 * Progress: Encourage maintaining a productive velocity.
 * The 'shape' provides a reward inversely proportional to velocity (for efficiency) or a penalty if stopped.
 * This is a continuous guidance predicate and does not affect termination or success status directly.
 */
// @progress: shape={v -> if v > 0 then 1/v else v}, on_success=continue:status=none, on_failure=continue:status=none
Bool moving(Real v) <-> v > 0;
```

## 7. Transpiler Implementation Notes

The Python transpiler is designed to process these annotations and generate robust Gymnasium environment code.

1.  **Pre-processing (Regex Matching):**
    The tool scans the KeYmaera X model file for annotation comments. The updated regex logic should identify the category-first tags and extract the associated attributes.

    ```python
    import re

    def parse_annotations(file_content):
        # Matches @safety, @progress, or @stability and the following definition
        # It captures the category, the attribute string, and the function name.
        pattern = r"// @(safety|progress|stability): (.*?)\n\s*(?:Bool|Real|HP)\s+(\w+)"
        matches = re.findall(pattern, file_content)
        
        annotations = {}
        for category, attr_string, func_name in matches:
            attrs = {}
            # This regex captures key=value pairs, handling curly braces for expressions.
            # It's robust to commas inside curly braces.
            # It also handles the 'action:status=outcome' format for on_success/on_failure
            attr_pairs_raw = re.findall(r'(\w+)=(?:({(?:[^{}]|{[^{}]*})*}|[^,]*))', attr_string)
            for key, value_raw in attr_pairs_raw:
                # Remove surrounding curly braces for expression values
                if value_raw.startswith('{') and value_raw.endswith('}'):
                    attrs[key] = value_raw[1:-1]
                else:
                    # Handle on_success/on_failure values with optional status
                    if key in ['on_success', 'on_failure']:
                        parts = value_raw.split(':', 1) # Split only on the first colon
                        action = parts[0]
                        status = 'none' # Default status if not specified
                        if len(parts) > 1:
                            status_part = parts[1]
                            if status_part.startswith('status='):
                                status = status_part[len('status='):]
                        attrs[key] = {'action': action, 'status': status}
                    else:
                        attrs[key] = value_raw
            
            attrs['type'] = category  # Automatically assign type based on the tag used
            annotations[func_name] = attrs
            
        return annotations
    ```
    *Note: The regex for parsing `attr_string` is designed to handle nested curly braces to a certain degree, but highly complex nested expressions might require a more sophisticated parser (e.g., a stack-based approach for brace matching).*

2.  **AST Linking:**
    During the traversal of the KeYmaera X Abstract Syntax Tree (AST), when a `FunctionDefinition` node (for `Bool`, `Real`, or `HP` definitions) is encountered, the pre-parsed metadata from the `annotations` dictionary is injected directly into that node's properties. This ensures that each relevant AST node carries its associated RL configuration.

3.  **Code Generation (`env.step()`):**
    The Python code generator iterates through all annotated AST nodes. For each node, it generates Python code that will be part of the Gymnasium `env.step()` method. This involves:
    *   Evaluating the boolean predicate of the definition to determine `condition_met`.
    *   Evaluating the `shape` expression.
    *   Conditionally evaluating `reward` or `penalty` based on `condition_met`.
    *   Summing these components to compute the `total_reward`.
    *   Applying `on_success` or `on_failure` logic to update `terminated`, `truncated`, and `info['is_success']`.

    The generated Python logic for each annotated predicate will resemble the following structure. Note that `is_terminated`, `is_truncated`, and `info_is_success` are typically aggregated across all predicates in a single `step()`.

    ```python
    def calculate_predicate_reward_and_state(predicate_name, state_variables, current_global_state):
        # current_global_state would be a dictionary holding the accumulated
        # terminated, truncated, and is_success from previous predicates
        # Initial values for current_global_state might be:
        # {'terminated': False, 'truncated': False, 'is_success': None}
        total_reward_for_predicate = 0
        
        # Evaluate the predicate's boolean condition
        # (This part is specific to how your transpiler evaluates KeYmaera X predicates)
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
    ```
    *The `evaluate_keymaera_predicate` and `evaluate_expression` functions would be generated code that translates KeYmaera X logic into executable Python.*

    **Aggregation of `terminated`, `truncated`, and `is_success`:**
    When multiple predicates are evaluated in a single `step()`, the `terminated` and `truncated` flags are typically combined using a logical OR. If *any* predicate sets `terminated` or `truncated` to `True`, the episode ends.
    For `info['is_success']`, the aggregation needs careful consideration:
    *   If any predicate sets `is_success` to `False`, the overall `info['is_success']` for the step should be `False`. This takes precedence (a single failure means overall failure).
    *   Otherwise, if any predicate sets `is_success` to `True`, the overall `info['is_success']` for the step is `True`.
    *   If no predicate explicitly sets `is_success` (i.e., all `status` were `none`), the overall `info['is_success']` might remain `None` or default to `False` (depending on the environment's general policy).

## 8. Why this System is Superior

1.  **Semantic Clarity:** Category-first tags (`@safety`, `@progress`, `@stability`) provide immediate implicit meaning. The explicit `on_success` and `on_failure` attributes, now with distinct `action` and `status` components, further enhance readability by clearly stating the consequences of a predicate's truth value on both episode flow and overall success.
2.  **Reward Decomposition Power:** Separating `reward`, `penalty`, and `shape` allows for highly nuanced reward engineering. This directly combats the "sparse reward" problem by providing continuous `shape` signals while retaining discrete `reward` bonuses or `penalty` costs for specific logical outcomes.
3.  **Decoupling Logic:** It clearly distinguishes between the formal verification logic (the predicate itself) and the RL training strategy (the annotations), yet keeps them co-located for maintainability.
4.  **Granular Control over `is_success`:** The new `status` sub-attribute within `on_success` and `on_failure` provides precise control over when `info['is_success']` is set, and to what value, independent of whether the predicate itself is true or false. This directly addresses the concern that a predicate's success doesn't always imply agent success.
5.  **Symmetric and Explicit Control:** The `on_success` and `on_failure` attributes offer a symmetrical and comprehensive way to define episode termination and success/failure outcomes for *both* true and false predicate conditions, eliminating ambiguity.
6.  **Robustness for Safety:** Explicit `on_failure=truncate:status=failure` for `@safety` conditions provides a strong, unambiguous signal for safety violations, crucial in safety-critical systems.
7.  **Flexibility for Progress:** `on_success=terminate:status=success` clearly marks successful task completion, while `on_success=continue:status=none` allows for tracking sub-goals or intermediate conditions without immediate termination or defining overall success.
8.  **Extensibility:** The structured key-value pair format makes it easy to add new attributes in the future without breaking existing parsing logic.
9.  **Direct Mapping to Gymnasium:** The design directly maps to Gymnasium's `step()` function return values (`reward`, `terminated`, `truncated`, `info`), simplifying the transpilation process and ensuring correct `is_success` reporting.