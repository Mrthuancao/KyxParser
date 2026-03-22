*Thinking...*

> **Developing Documentation**
> 
> \n\n
> 
> I've begun outlining the documentation structure for the annotation system. I'm focusing on clarity, ensuring the comment-based approach is readily understandable. Currently, I'm working through examples demonstrating how the system will interface with the KeYmaera X safety models. The goal is to provide a seamless transition from the structured comments to the Gymnasium environment via the Python transpiler.
> 
> 
> **Outlining Attribute Explanations**
> 
> \n\n
> 
> I'm now diving into the detailed explanations of each annotation attribute. I'm prioritizing the `reward` attribute first, providing clear examples on how to map variables to expressions. Then, I'll move on to `type`, `on_fail`, and `status`, ensuring each is clearly defined with examples. Considering the target audience, I'll emphasize concise definitions, and the key-value syntax. I'm aiming for practical utility.
> 
> 
> **Constructing the Outline**
> 
> \n\n
> 
> My current focus is solidifying the documentation structure. I've broken it down into key sections: a title and overview, syntax details with the `@RL:` prefix and key-value pairs, and a comprehensive attribute reference. I'm prioritizing clear explanations of `reward` mappings and `type`, `on_fail`, and `status` attribute definitions. I'll provide a mapping to the `step()` function for Gymnasium, and will leverage examples from the robot safety model, culminating in a short note on the AST parsing for implementation context.
> 
> 


This documentation describes the **RL-Annotation Syntax**, a comment-based domain-specific language (DSL) designed to bridge KeYmaera X formal models with Gymnasium reinforcement learning environments. 

By annotating Differential Dynamic Logic ($dL$) definitions with `@RL` metadata, the transpiler identifies specific Abstract Syntax Tree (AST) nodes to automatically generate reward functions, termination logic, and episode status.

---

# Documentation: RL-Annotation Syntax for KeYmaera X

## 1. Overview
The **RL-Annotation Syntax** allows developers to embed Reinforcement Learning (RL) training logic directly into a formal safety model. The transpiler parses these annotations, attaches them to the corresponding AST nodes, and maps them to the `step()` and `reset()` functions of a Gymnasium environment.

## 2. General Syntax
Annotations must be placed on the line immediately preceding a `Bool`, `Real`, or `HP` definition.

**Template:**
```java
// @RL: reward={mapping}, type=category, on_fail=action, status=outcome
Definition name(...) <-> ...
```

## 3. Attribute Reference

### `reward={var -> expression}`
Defines how the environment calculates the step-wise reward. 
- **Mapping:** Maps a state variable to a mathematical expression.
- **Shaping:** Supports conditional logic for reward shaping (e.g., `if v > 0 then 1/v else v`).
- **Transpilation:** The parser converts KeYmaera X operators (e.g., `^`, `&`, `|`) into Python equivalents (`**`, `and`, `or`).

### `type=[safety | progress | stability]`
Categorizes the logic for organizational and logging purposes.
- `safety`: Logic related to avoiding collisions or unsafe states.
- `progress`: Logic related to reaching a goal or maintaining velocity.
- `stability`: Logic derived from Lyapunov or attractivity proofs.

### `on_fail=[truncate | terminate | none]`
Determines the Gymnasium episode state if the predicate evaluates to **False**.
- `truncate`: Sets `truncated = True`. Used for safety violations (e.g., leaving the safe invariant).
- `terminate`: Sets `terminated = True`. Used for reaching a terminal state (e.g., a goal or a crash).
- `none`: The episode continues regardless of the predicate value.

### `status=[success | failure | none]`
Defines the semantic outcome of the episode termination.
- `success`: The agent achieved the goal (e.g., `ensure` condition met).
- `failure`: The agent failed (e.g., safety invariant violated).
- `none`: Default; used for intermediate shaping logic.

---

## 4. Mapping to Gymnasium (AST to Code)

The transpiler follows a specific mapping logic when converting AST nodes to Python:

| Annotation Attribute | Gymnasium Implementation |
| :--- | :--- |
| `reward={...}` | Added to the cumulative `reward` variable in `env.step()`. |
| `on_fail=truncate` | `truncated = not (predicate_logic)` |
| `on_fail=terminate` | `terminated = not (predicate_logic)` |
| `status=success` | Included in the `info` dict: `info['is_success'] = True`. |

---

## 5. Concrete Example
Using the "Static Safety" robot model, we annotate the loop invariant and the goal condition:

### Example 1: Safety & Truncation
If the robot violates the safe distance (`loopinv`), the episode is truncated and marked as a failure.
```java
// @RL: type=safety, on_fail=truncate, status=failure
Bool loopinv(Real x, Real y, Real v, Real xo, Real yo) <-> (
  v >= 0 & (abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v))
);
```

### Example 2: Goal & Success
If the robot satisfies the `ensure` condition, the episode terminates successfully.
```java
// @RL: type=progress, on_fail=terminate, status=success
Bool ensure(Real x, Real y, Real xo, Real yo) <-> (
  (x - xo)^2 + (y - yo)^2 > 0
);
```

### Example 3: Reward Shaping
Providing a dense reward for maintaining a specific orientation and velocity.
```java
// @RL: reward={v -> (1/v if v>0 else v), dx -> dx}, type=progress
Bool isWellformedDir(Real dx, Real dy) <-> dx^2 + dy^2 = 1;
```

---

## 6. Parser Implementation Notes
1. **Pre-processing:** The tool scans for the `// @RL:` prefix and stores the metadata in a dictionary keyed by the function name.
2. **AST Linking:** During the walk of the KeYmaera X AST, when a `FunctionDefinition` node is encountered, the metadata is injected into the node's properties.
3. **Code Generation:** The Python generator iterates through all annotated nodes to build the `reward = ...` and `done = ...` strings for the final `.py` environment file.

---

*Thinking...*

> **Considering Tag Replacements**
> 
> I'm currently focused on refactoring the `@RL` tag. The goal is to enhance clarity by replacing it with more specific tags. I'm exploring the nuances of `@safety`, `@progress`, and `@stability` tags. The initial plan involves modifying the reward parameter and incorporating `on_fail` directives within the `@safety` tag.
> 
> 
> **Implementing Tag Refinements**
> 
> I've made progress on the tag replacements. Replacing `@RL` with `@safety`, `@progress`, and `@stability` appears cleaner and improves readability. Parsing becomes simpler with tag-specific categorization during the initial scan. I'm also planning default behaviors for each tag, such as `on_fail=truncate` for `@safety`.
> 
> 


That is a much cleaner and more intuitive design. By using the category as the primary tag (e.g., `@safety`), you make the code more readable and reduce the number of key-value pairs you have to type.

Here is the updated documentation reflecting this **Category-First** syntax.

---

# Documentation: RL-Annotation Syntax for KeYmaera X

## 1. Overview
The **RL-Annotation Syntax** uses category-specific tags (`@safety`, `@progress`, `@stability`) to bridge KeYmaera X formal models with Gymnasium RL environments. These tags allow the transpiler to automatically extract reward shaping and termination logic directly from the model's Abstract Syntax Tree (AST).

## 2. General Syntax
Annotations must be placed on the line immediately preceding a `Bool`, `Real`, or `HP` definition.

**Template:**
```java
// @category: reward={mapping}, on_fail=action, status=outcome
Definition name(...) <-> ...
```

---

## 3. The Three Categories

### `@safety`
Used for logic that defines the "safe operating envelope" of the agent.
*   **Default Behavior:** Usually maps to `on_fail=truncate`.
*   **Use Case:** If the safety invariant (e.g., `loopinv`) is violated, the episode is truncated and marked as a failure.
*   **Example:**
    ```java
    // @safety: penalty=-100, on_fail=truncate, status=failure
    Bool loopinv(Real x, Real v) <-> v >= 0 & x > stopDist(v);
    ```

### `@progress`
Used for logic that encourages the agent to achieve a specific goal or maintain a state.
*   **Default Behavior:** Usually maps to a step-wise reward.
*   **Use Case:** Rewarding the agent for moving forward, maintaining velocity, or reaching a target coordinate.
*   **Example:**
    ```java
    // @progress: reward={v -> 1/v}, status=success
    Bool ensure(Real x, Real y) <-> (x - xo)^2 + (y - yo)^2 > 0;
    ```

### `@stability`
Used for logic derived from **Lyapunov functions** or attractivity proofs.
*   **Default Behavior:** Maps to a "potential-based" reward.
*   **Use Case:** Guiding the agent toward a stable equilibrium point (e.g., a "parking" spot) where the reward increases as a Lyapunov value $V$ decreases.
*   **Example:**
    ```java
    // @stability: reward={V -> -V}
    Real V(Real x, Real y) = x^2 + y^2;
    ```

---

## 4. Attribute Reference

| Attribute | Description |
| :--- | :--- |
| `reward={var -> expr}` | Maps a variable to a math expression (e.g., `1/v`). Supports `if-then-else`. |
| `on_fail=[truncate\|terminate\|none]` | Action taken if the condition is **False**. `truncate` for safety, `terminate` for goals. |
| `status=[success\|failure\|none]` | Metadata for logging the episode outcome in the `info` dictionary. |

---

## 5. Parser Implementation (Regex Update)

To support this new syntax, your Python pre-parser should look for multiple tag types. Here is a snippet of how the regex logic changes:

```python
import re

def parse_annotations(file_content):
    # Matches @safety, @progress, or @stability and the following definition
    pattern = r"// @(safety|progress|stability): (.*?)\n\s*(?:Bool|Real|HP)\s+(\w+)"
    matches = re.findall(pattern, file_content)
    
    annotations = {}
    for category, attr_string, func_name in matches:
        # Store the category + attributes
        attrs = dict(re.findall(r"(\w+)=({.*?}|[\w.-]+)", attr_string))
        attrs['type'] = category  # Automatically assign type based on the tag used
        annotations[func_name] = attrs
        
    return annotations
```

---

## 6. Why this is better
1.  **Implicit Meaning:** Just by seeing `// @safety`, the developer knows this logic is about preventing crashes.
2.  **Cleaner Code:** You no longer need to write `type=safety` inside the brackets.
3.  **Semantic Clarity:** It separates the **Safety Model** (what must be true) from the **Training Strategy** (how to reward the agent) while keeping them in the same file.