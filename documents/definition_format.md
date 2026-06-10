Reserved Names and Naming Conventions for KeYmaera X Models
This section outlines the specific naming conventions and reserved identifiers that your KeYmaera X models must adhere to for successful parsing and interpretation by the system. This is particularly crucial for configuring the reinforcement learning environment, where specific Hybrid Program (HP) names are reserved, and metadata annotations are used to interpret the role of Bool predicates.

1. General Definition Structure
Your parser recognizes definitions within the Definitions block with the following general structures:

Real and Bool Definitions:

Syntax: (Real|Bool) <name>(<optional_parameters>) <optional_assignment_or_equivalence_operator> <optional_value>;
Example:
keymaerax
Real myConstant = 10;
Bool isPositive(Real x) <-> (x > 0);

Metadata: Optional metadata comments (// @category: key=value) placed immediately before the definition line are parsed and associated with the definition. These are crucial for Bool predicates, as explained in Section 3.
Hybrid Program (HP) Definitions:

Syntax: HP <name> ::= { <program_body> }; (or ;; for double semicolon termination)
Example:
keymaerax
HP myProgram ::= {
  x := x + 1;
  {x' = v, v' = a & x >= 0}
};

Metadata: Optional metadata comments (// @category: key=value) placed immediately before the definition line are parsed and associated with the HP.
2. Reserved Hybrid Program (HP) Names
The following HP names are specially recognized by the parser and the environment configuration logic due to their distinct roles in defining the system's behavior. These names must be used as specified:

controlFunc:

Purpose: Defines the discrete actions available to the agent. Its body is parsed as a sequence of choices, assignments, and tests, forming the action space that the RL agent can interact with.
Location: Must be defined within the Definitions block.
Example:
keymaerax
HP controlFunc ::= {
  (x := x + 1) ++ (x := x - 1)
};

stepFunc:

Purpose: Defines the continuous dynamics (Ordinary Differential Equations) and evolution domain constraints of the system. This HP describes how the system evolves over time between discrete actions.
Location: Must be defined within the Definitions block.
Example:
keymaerax
HP stepFunc ::= {
  {x' = v, y' = w & x >= 0 & y >= 0}
};

3. Metadata Annotations for Bool Predicates
For Bool predicates, the system does not reserve specific names. Users are free to name their Bool predicates as they see fit. Instead, the metadata annotations (// @category: key=value) are the critical mechanism for the system to understand the role and purpose of each Bool predicate in the context of the reinforcement learning environment.

The parser extracts these comments, and the subsequent environment configuration logic uses them to set up rewards, penalties, and termination conditions. This significantly influences the behavior of the reinforcement learning agent.

It is essential to correctly format and place these metadata annotations immediately preceding the Bool definition they apply to.

Here are examples of how Bool predicates, regardless of their names, are interpreted based on their metadata:

Safety Conditions:

keymaerax
// @safety: penalty=-500, on_failure=truncate:status=failure
Bool physicalConstraints(Real v, Real r) <-> ( // User-defined name
  v >= 0 & r != 0
);

// @safety: penalty=-1000, shape={(abs(x-xo) - stopDist(v)) + (abs(y-yo) - stopDist(v))}, on_failure=truncate:status=failure
Bool collisionAvoidance(Real x, Real y, Real v, Real xo, Real yo) <-> ( // User-defined name
  (abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v))
);

Progress and Goal Conditions:

keymaerax
// @progress: reward=10, shape={-((dx**2 + dy**2) - 1)**2}, on_success=continue:status=none
Bool directionNormalized(Real dx, Real dy) <-> dx^2 + dy^2 = 1; // User-defined name

// @progress: shape={1/v if v > 0 else v}, on_success=continue:status=none
Bool maintainSpeed(Real v) <-> ( v >= 0 ); // User-defined name

// @progress: reward=500, penalty=-1000, on_success=terminate:status=success, on_failure=terminate:status=failure
Bool targetReached(Real x, Real y, Real xo, Real yo) <-> ( // User-defined name
  (x - xo)^2 + (y - yo)^2 > 0
);
