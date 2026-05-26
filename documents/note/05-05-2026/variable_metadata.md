Theorem "IJRR17/Theorem 1: Static safety"

Definitions
  // Physical Constants
  Real ep;      // @var_meta: description="time limit for control decisions", range=[0.01,1.0], unit="s", plot_label="Control Decision Time Limit"
  Real b;       // @var_meta: description="minimum braking capability of the robot", range=[1.0,10.0], unit="m/s^2", plot_label="Braking Capability"
  Real A;       // @var_meta: description="maximum acceleration -b <= a <= A", range=[0.0,5.0], unit="m/s^2", plot_label="Maximum Acceleration"
  Real W;       // @var_meta: description="maximum steering (rotational velocity)", range=[0.1,1.0], unit="rad/s", plot_label="Maximum Steering Rate"

  // Distance Calculations
  Real stopDist(Real v) = v^2 / (2*b);
  Real accelComp(Real v) = ((A/b + 1) * (A/2 * ep^2 + ep*v));
  Real admissibleSeparation(Real v) = stopDist(v) + accelComp(v);

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

  // Global Bounds
  Bool bounds() <-> (
      A >= 0
    & b > 0
    & ep > 0
  );

  // Initial Conditions
  Bool initialState(Real x, Real y, Real v, Real dx, Real dy, Real xo, Real yo) <-> (
    v = 0
    & (x-xo)^2 + (y-yo)^2 > 0
    & isWellformedDir(dx, dy)
    & r != 0
  );

  Bool assumptions(Real x, Real y, Real v, Real dx, Real dy, Real xo, Real yo) <->
    bounds() & initialState(x, y, v, dx, dy, xo, yo);

  // Safety Invariant (Aggregate):
  // Combines sub-predicates for the formal proof.
  Bool loopinv(Real x, Real y, Real v, Real dx, Real dy, Real xo, Real yo, Real r) <-> (
    isPhysicallyValid(v, r)
    & isWellformedDir(dx, dy)
    & isSafe(x, y, v, xo, yo)
    & isEfficient(v)
  );

  // Goal Condition:
  // This predicate defines the goal of avoiding the obstacle.
  // If the condition is true (robot is not at the obstacle), it's a success and the episode terminates.
  // If the condition is false (robot is at the obstacle), it's a failure and the episode terminates.
  // @progress: reward=500, penalty=-1000, on_success=terminate:status=success, on_failure=terminate:status=failure
  Bool ensure(Real x, Real y, Real xo, Real yo) <-> (
    (x - xo)^2 + (y - yo)^2 > 0
  );

  // Control Logic
  HP controlFunc ::= {
      // Option 1: Brake on current curve or remain stopped
      { a := -b; }
      ++
      { ?v = 0; a := 0; w := 0; }
      ++
      // Option 2: Choose a new safe curve
      { a := A;
        w := *; ?-W<=w & w<=W;
        r := *;
        xo := *; yo := *;
        ?r!=0 & r*w = v;
        ? abs(x-xo) > admissibleSeparation(v)
        | abs(y-yo) > admissibleSeparation(v);
      }
    };;

    // Continuous Dynamics
    HP stepFunc ::= {
        x' = v * dx, y' = v * dy, v' = a,
        dx' = -w * dy, dy' = w * dx, w' = a/r,
        t' = 1 & t <= ep & v >= 0
      };
End.

ProgramVariables
  Real x;    // @var_meta: description="robot position: x", range=[-1000.0,1000.0], unit="m", plot_label="Robot X Position"
  Real y;    // @var_meta: description="robot position: y", range=[-1000.0,1000.0], unit="m", plot_label="Robot Y Position"
  Real v;    // @var_meta: description="robot translational velocity", range=[0.0,50.0], unit="m/s", plot_label="Robot Velocity"
  Real a;    // @var_meta: description="robot translational acceleration", range=[-10.0,5.0], unit="m/s^2", plot_label="Robot Acceleration"
  Real dx;   // @var_meta: description="robot orientation: x component", range=[-1.0,1.0], unit="unitless", plot_label="Orientation X Component"
  Real dy;   // @var_meta: description="robot orientation: y component", range=[-1.0,1.0], unit="unitless", plot_label="Orientation Y Component"
  Real w;    // @var_meta: description="robot rotational velocity", range=[-1.0,1.0], unit="rad/s", plot_label="Robot Rotational Velocity"
  Real r;    // @var_meta: description="robot curve radius", range=[-1000.0,1000.0], unit="m", plot_label="Turning Radius"
  Real xo;   // @var_meta: description="position of closest obstacle on curve: x", range=[-1000.0,1000.0], unit="m", plot_label="Obstacle X Position"
  Real yo;   // @var_meta: description="position of closest obstacle on curve: y", range=[-1000.0,1000.0], unit="m", plot_label="Obstacle Y Position"
  Real t;    // @var_meta: description="time", range=[0.0,100.0], unit="s", plot_label="Time Elapsed"
End.

Problem
assumptions(x, y, v, dx, dy, xo, yo) ->
  [
    {
      {
        controlFunc;
        t := 0;
      }

      // dynamics with local invariants
      {
      stepFunc;@invariant(t>=0, isWellformedDir(dx, dy))
      }
    }*@invariant(loopinv(x, y, v, dx, dy, xo, yo, r))
  ]ensure(x, y, xo, yo)
End.