This refined documentation integrates your new **Reward Decomposition** model (Reward, Penalty, Shape) with the **Category-First** tagging system. This structure provides a clear bridge between formal KeYmaera X predicates and the continuous feedback loops required for Reinforcement Learning.

---

# Documentation: RL-Annotation Syntax for KeYmaera X

## 1. Overview
The **RL-Annotation Syntax** is a comment-based DSL designed to map formal Differential Dynamic Logic ($dL$) models to Gymnasium Reinforcement Learning environments. By annotating definitions with `@safety`, `@progress`, or `@stability`, developers can define complex reward structures and episode termination logic without modifying the underlying formal proof.

## 2. General Syntax
Annotations must be placed on the line immediately preceding a `Bool`, `Real`, or `HP` definition.

**Template:**
```java
// @Category: reward={expr}, penalty={expr}, shape={expr}, on_fail=action, status=outcome
Definition name(...) <-> ...
```

---

## 3. The Reward Decomposition Model
To solve the "Sparse Reward" problem in formal models, this syntax decomposes the reward into three components. The total reward for a step is calculated as:

\[
R_{total} = \text{shape} + 
\begin{cases} 
\text{reward} & \text{if condition is True} \\
\text{penalty} & \text{if condition is False}
\end{cases}
\]

| Attribute | Logic Type | Application | Purpose |
| :--- | :--- | :--- | :--- |
| **`reward`** | Discrete | Only if **True** | A "bonus" for satisfying the condition. |
| **`penalty`** | Discrete | Only if **False** | A "cost" for violating the condition. |
| **`shape`** | Continuous | **Always** | A "gradient" to guide the agent toward the goal. |

---

## 4. Category Tags & Termination Logic

### `@safety`
Defines the safe operating envelope. 
*   **Primary Use:** Invariants (e.g., `loopinv`).
*   **Default Action:** `on_fail=truncate` (The agent "timed out" or "broke" the safety proof).
*   **Example:**
    ```java
    // @safety: penalty=-100, shape={dist}, on_fail=truncate, status=failure
    Bool safeDist(Real dist) <-> dist > 0;
    ```

### `@progress`
Defines goal-reaching or task-performance logic.
*   **Primary Use:** Post-conditions (e.g., `ensure`).
*   **Default Action:** `on_fail=none` (Keep trying until the goal is met).
*   **Example:**
    ```java
    // @progress: reward=500, shape={-dist_to_goal}, on_fail=terminate, status=success
    Bool atGoal(Real x, Real y) <-> (x-xo)^2 + (y-yo)^2 < 1;
    ```

### `@stability`
Defines convergence toward an equilibrium point.
*   **Primary Use:** Lyapunov functions or attractivity logic.
*   **Example:**
    ```java
    // @stability: shape={-V}
    Real V(Real x, Real y) = x^2 + y^2;
    ```

---

## 5. Attribute Reference

### Control Attributes
*   **`on_fail=[truncate | terminate | none]`**: 
    *   `truncate`: Sets `truncated=True` (Safety violation).
    *   `terminate`: Sets `terminated=True` (Goal reached/Crash).
*   **`status=[success | failure | none]`**: Sets the metadata in the Gymnasium `info` dictionary for logging.

### Expression Syntax
*   **Mappings**: Use `{var -> expression}` or direct expressions `{expression}`.
*   **Operators**: The transpiler converts KeYmaera X syntax to Python:
    *   `^` $\rightarrow$ `**`
    *   `&` $\rightarrow$ `and`
    *   `|` $\rightarrow$ `or`
    *   `if A then B else C` $\rightarrow$ `B if A else C`

---

## 6. Comprehensive Example
This example demonstrates a robot navigating to a point while maintaining a safe distance from an obstacle.

```java
/** 
 * Safety: Penalize being close to obstacle, truncate if distance <= 0 
 */
// @safety: penalty=-500, shape={dist}, on_fail=truncate, status=failure
Bool safe(Real dist) <-> dist > 0;

/** 
 * Progress: Reward reaching the goal, shape by distance to goal 
 */
// @progress: reward=1000, shape={-((x-gx)^2 + (y-gy)^2)}, on_fail=terminate, status=success
Bool goal(Real x, Real y, Real gx, Real gy) <-> (x-gx)^2 + (y-gy)^2 < 0.1;

/** 
 * Shaping: Encourage high velocity but penalize instability 
 */
// @progress: shape={v -> if v > 0 then 1/v else v}
Bool moving(Real v) <-> v > 0;
```

---

## 7. Transpiler Implementation Note
The Python generator iterates through all annotated AST nodes. For each `step()`, it evaluates the boolean predicate of the definition. If the predicate is `True`, it sums the `reward` and `shape`. If `False`, it sums the `penalty` and `shape`. It then checks the `on_fail` attribute to determine if the episode should end.