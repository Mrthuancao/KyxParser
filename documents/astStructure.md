This document provides a technical specification for the **KeYmaera X AST (Abstract Syntax Tree)** design. This AST is specifically architected to bridge the gap between formal Differential Dynamic Logic ($\text{d}\mathcal{L}$) and Reinforcement Learning (RL) environments.

---

### 1. Architectural Overview
The AST serves as the intermediate representation (IR) that allows the synthesis engine to perform two critical tasks:
1.  **Symbolic Execution:** Parsing the Hybrid Program (HP) to determine state transitions and action spaces.
2.  **Logic-to-Reward Mapping:** Extracting metadata-tagged predicates to automatically construct reward functions and safety monitors.

The hierarchy is rooted in `ASTConfigNode` and organized into six functional packages.

---

### 2. Node Package Specifications

#### A. Structural & Definition Nodes
These nodes represent the high-level organization of a KeYmaera X model.
*   **`GymEnvironmentConfig`**: The root node containing the environment name and links to definitions, variables, and the problem statement.
*   **`DefinitionConfig`**: A container for constants, functions, and predicates.
*   **`HybridProgramDefinitionConfig`**: Specifically handles the `HP` blocks.
    *   `ControlFuncConfig`: Maps to the discrete/continuous action selection logic.
    *   `StepFuncConfig`: Maps to the continuous ODE system.

#### B. Variable & Type Nodes
Handles the declaration of the state space.
*   **`VariableDeclarationConfig`**: Captures the name and type (Real/Bool). In the synthesis phase, these are used to define the `gym.spaces.Box` dimensions.
*   **`IdentifierConfig`**: Used for cross-referencing variables within expressions to ensure type safety.

#### C. Expression Nodes (The Math Engine)
This is the most granular part of the new AST, allowing the generator to translate math into Python/NumPy lambdas.
*   **`LiteralConfig`**: Represents hardcoded values (e.g., `0.5`, `10`, `True`).
*   **`FunctionCallConfig`**: Enables the reuse of defined logic (e.g., calling `stopDist(v)` inside a safety invariant).
*   **`Binary/UnaryOperatorConfig`**: Supports full arithmetic ($+, -, *, /, \text{^}$) and logical ($\wedge, \vee, \neg, \rightarrow, \leftrightarrow$) operations.
*   **`Modal Operators`**:
    *   `BoxModalOperatorConfig` ($[\alpha]\phi$): Represents "In all executions of $\alpha$, $\phi$ holds." Used to define safety requirements.
    *   `DiamondModalOperatorConfig` ($\langle\alpha\rangle\phi$): Represents "There exists an execution of $\alpha$ where $\phi$ holds." Used for reachability/goal logic.

#### D. HP Statement Nodes (The Execution Logic)
These nodes define how the system state changes over time.
*   **`AssignmentConfig` ($:=$)**: Deterministic state updates.
*   **`NondeterministicAssignmentConfig` ($:= *$)**: Represents an environment or agent choice. This is the primary trigger for creating an **Action Space**.
*   **`GuardConfig` ($?$)**: Conditional execution. In RL, these are converted into "Action Masks" or "Validity Checks."
*   **`DifferentialEquationBlockConfig`**: Contains a list of `ODEConfig` objects (e.g., $x' = f(x)$) and an evolution domain (e.g., $v \geq 0$).

---

### 3. Mapping Logic to RL Components
Using the provided `IJRR17` example, the AST facilitates the following automated mapping:

| KeYmaera X Construct | AST Node Type | RL Implementation |
| :--- | :--- | :--- |
| `Real x;` | `VariableDeclarationConfig` | `obs_space['x']` |
| `a := -b ++ {?v=0; a:=0}` | `NondeterministicChoiceOperatorConfig` | `Discrete(3)` Action Space |
| `isSafe(...)` | `PredicateDefinitionConfig` | Safety Monitor / Termination Logic |
| `@progress: shape={...}` | `Metadata` in `PredicateDefinition` | Reward Shaping Function |
| `x' = v, t <= ep` | `DifferentialEquationBlockConfig` | `scipy.integrate.solve_ivp` |
| `assumptions(...)` | `PreconditionConfig` | `env.reset()` via Z3 SMT Solver |

---

### 4. Metadata Enrichment
The AST is designed to parse comments as metadata to guide the RL synthesis. For example, the node for `isSafe` would contain a `Metadata` object:
```json
{
  "type": "safety",
  "penalty": -1000,
  "shape": "(abs(x-xo) - stopDist(v)) + (abs(y-yo) - stopDist(v))",
  "on_failure": "truncate"
}
```
This allows the **Expression Nodes** to not only represent the boolean truth of the safety statement but also the "distance to violation" used for dense reward signals.

---

### 5. Implementation Strategy (Visitor Pattern)
To process this AST, the framework utilizes a **Recursive Visitor Pattern**:
1.  **`LogicToPythonVisitor`**: Traverses `AbstractExpressionConfig` to generate string-based Python expressions for NumPy.
2.  **`ActionSpaceVisitor`**: Traverses the `controlFunc` to count non-deterministic branches and assignments to determine the `gym.ActionSpace`.
3.  **`ODEVisitor`**: Extracts equations from `DifferentialEquationBlockConfig` to build the derivative function for the numerical solver.
4.  **`SMTVisitor`**: Translates `assumptions` into SMT-LIB format for initial state sampling.