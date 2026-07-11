### 4. Approach

Our framework bridges the gap between formal verification and practical control by treating the KeYmaera X Abstract Syntax Tree (AST) as a blueprint for a **Markov Decision Process (MDP)**. In this context, the MDP provides the mathematical framework for the RL agent's world: the $\text{d}\mathcal{L}$ state variables define the **State Space ($S$)**, the non-deterministic choices in the Hybrid Program define the **Action Space ($A$)**, the ODEs define the **Transition Function ($P$)**, and our annotated predicates define the **Reward Function ($R$)**.

We implement a syntax-directed translation that extracts physical dynamics into a high-fidelity ODE solver and logical invariants into a dense reward function.

#### 4.1 Extended Formal Syntax and RL-Annotations

The RL-Annotation Syntax is a comment-based DSL embedded within KeYmaera X files. By separating these into variable-level and predicate-level annotations, we provide a clean mapping to the Gymnasium environment's **Observation Space** and **Reward/Termination Logic**, respectively.

##### 4.1.1 Variable Annotations (`@var`, `@const`)
These annotations are placed above `Real` declarations in the `ProgramVariables` or `Definitions` sections. They define the numerical boundaries and metadata for the environment's state.

**Grammar:**
\[
\begin{array}{lcl}
\text{VarAnnotation} & ::= & \text{"//"} \text{ ("@var" } \mid \text{ "@const")} \text{ ":" } \text{ VarAttrList} \\
\text{VarAttrList}   & ::= & \text{VarAttr} \mid \text{VarAttr} \text{ "," } \text{ VarAttrList} \\
\text{VarAttr}       & ::= & \text{"bound=[" } r \text{ ":" } r \text{ "]"} \mid \text{"def=" } \text{STRING} \mid \text{"plot=" } \text{PlotType} \\
\text{PlotType}      & ::= & \text{"line"} \mid \text{"hist"} \mid \text{"scatter"} \mid \text{"none"}
\end{array}
\]

*   **`@var`**: Maps a `Real` program variable to an element in the Gymnasium **Observation Space**. The `bound` defines the `Box` limits.
*   **`@const`**: Maps a `Real` definition to a fixed parameter. If a `bound` is provided, the value is sampled once at `reset()`.

##### 4.1.2 Predicate Annotations (`@safety`, `@progress`, `@stability`)
These annotations are placed above `Bool` or `Real` definitions. They define how the truth value of a $\text{d}\mathcal{L}$ formula translates into rewards and episode transitions.

**Grammar:**
\[
\begin{array}{lcl}
\text{PredAnnotation} & ::= & \text{"//"} \text{ Category} \text{ ":" } \text{ PredAttrList} \\
\text{Category}       & ::= & \text{"@safety"} \mid \text{"@progress"} \mid \text{"@stability"} \\
\text{PredAttrList}   & ::= & \text{PredAttr} \mid \text{PredAttr} \text{ "," } \text{ PredAttrList} \\
\text{PredAttr}       & ::= & \text{RewardAttr} \mid \text{ControlAttr} \\
\text{RewardAttr}     & ::= & \text{"reward=" } \text{Expr} \mid \text{"penalty=" } \text{Expr} \mid \text{"shape=" } \text{"\{"} \text{Expr} \text{"\}"} \\
\text{ControlAttr}    & ::= & (\text{"on\_success"} \mid \text{"on\_failure"}) \text{"=" } \text{Action} [\text{":"} \text{Status}] \\
\text{Action}         & ::= & \text{"terminate"} \mid \text{"truncate"} \mid \text{"continue"} \\
\text{Status}         & ::= & \text{"success"} \mid \text{"failure"} \mid \text{"none"}
\end{array}
\]

*   **`@safety`**: Used for invariants. Typically triggers `truncate:failure` when the formula is false.
*   **`@progress`**: Used for goal conditions. Typically triggers `terminate:success` when the formula is true.
*   **`@stability`**: Used for Lyapunov-like functions to provide continuous `shape` rewards without necessarily terminating the episode.

##### 4.1.3 The Reward Decomposition Model
The transpiler interprets **Predicate Annotations** using a decomposition model to solve the "Sparse Reward" problem. For any annotated predicate $\phi$, the step-wise reward $R_{total}$ is calculated as:

\[
R_{total} = \text{shape} + 
\begin{cases} 
\text{reward} & \text{if } \phi \text{ is True} \\
\text{penalty} & \text{if } \phi \text{ is False}
\end{cases}
\]

This ensures that even if the agent has not yet achieved the discrete `reward` (e.g., reaching a goal), the `shape` expression (e.g., $-\text{distance\_to\_goal}$) provides a continuous gradient to guide learning.

##### 4.1.4 Why use Comment-based DSL?
1.  **Non-Invasive Integration**: By using comments, the model remains a valid `.kyx` file. The KeYmaera X prover ignores the RL metadata, ensuring that the **Formal Proof** is never compromised by training-specific heuristics.
2.  **Single Source of Truth**: The safety logic used in the proof (e.g., `Bool isSafe <-> ...`) is the exact same code used to penalize the RL agent. This prevents "specification drift" where the verified model and the trained agent operate on different assumptions.
3.  **Simplified Transpilation**: Splitting into **Variable** and **Predicate** types allows the transpiler to build the Gymnasium `ObservationSpace` and `step()` logic in two clean, independent passes.



#### 4.2 Structural AST Redesign
The transpiler parses the KeYmaera X model into a specialized AST designed for RL synthesis. This intermediate representation allows us to map symbolic logic to imperative Python code.
*   **Variable & Type Nodes:** `VariableDeclarationConfig` nodes capture the name, type, and physical bounds. These are used to programmatically synthesize the `gym.spaces.Box` dimensions and define the sampling volume for the SMT solver.
*   **Formula Nodes:** These nodes translate symbolic logic into numerical execution. The generator traverses these nodes to convert binary and unary operators into Python lambda functions, enabling the Gymnasium environment to evaluate guards and invariants at microsecond scales during runtime.
*   **HP Statement Nodes:** These map the "Control" and "Plant" of the Hybrid Program. Deterministic assignments ($:=$) are translated into state updates. Crucially, non-deterministic choices ($++$) and assignments ($:=*$) are identified as "Decision Points," triggering the automatic synthesis of `Discrete` and `Box` action spaces, respectively.

#### 4.3 Refinement from $\text{d}\mathcal{L}$ to Python
The transition from symbolic logic to an executable Gymnasium environment is handled by four primary translation engines:

**A. Translation of Hybrid Dynamics (The Physics Engine)**
The generator identifies `DifferentialEquationBlockConfig` nodes and transforms symbolic ODEs into Python functions. Evolution domain constraints (e.g., $v \geq 0$) are converted into "Termination Events" for the `scipy.integrate.solve_ivp` solver. This ensures the simulation respects the physical boundaries defined in the formal model.

**B. Synthesis of the Action and State Spaces**
The MDP interface is automated through structural analysis. The generator maps every `Real` variable to a dimension in a continuous observation space. For control, the generator traverses the Hybrid Program; every non-deterministic choice operator increments the size of a `Discrete` action space, while guards ($?$) are mapped to "Action Masks" that prevent the RL agent from selecting branches that are logically invalid in the current state.

**C. Formal Specification as Reward and Monitor**
We utilize the formal proof obligations as runtime monitors. The **Loop Invariant** (the logic that must remain true throughout the control loop) is synthesized into a safety monitor. If the invariant is violated, the environment applies a safety penalty and truncates the episode. Similarly, **Postconditions** (defined via `ensure` predicates) are mapped to termination logic, providing a clear "Success" signal when the agent achieves the verified goal.

**D. SMT-Seeded Initial State Sampling**
To ensure the agent is trained on a distribution consistent with the formal proof, we automate initialization using the model's **Preconditions**. During environment construction, we use the Z3 SMT solver to pre-populate a **Precondition-Satisfying Buffer**. This ensures that every training episode starts in a state that the formal proof recognizes as valid. To ensure diversity, we use a **Soft-Constraint Targeting** strategy: Z3 solves for the formal preconditions (hard constraints) while simultaneously optimizing to minimize the distance to a randomized target point.


#### 4.4 Mapping Logic to RL Components
The following table summarizes the syntax-directed translation from $\text{d}\mathcal{L}$ constructs to Gymnasium components:

| KeYmaera X Construct | AST Node Type | RL Implementation |
| :--- | :--- | :--- |
| `Real x;` | `VariableDeclarationConfig` | `obs_space['x']` |
| `a := -b ++ {?v=0; a:=0}` | `NondeterministicChoiceOperatorConfig` | `Discrete(n)` Action Space |
| `isSafe(...)` | `PredicateDefinitionConfig` | Safety Monitor / Termination |
| `@progress: shape={...}` | `Metadata` in `PredicateDefinition` | Reward Shaping Function |
| $x' = v, t \leq ep$ | `DifferentialEquationBlockConfig` | `scipy.integrate.solve_ivp` |
| `assumptions(...)` | `PreconditionConfig` | `env.reset()` via Z3 Solver |

By extracting the physical dynamics into a high-precision ODE solver and the logical invariants into a dense reward function, we ensure that the RL agent is actively guided by the formal safety properties of the original $\text{d}\mathcal{L}$ model.
