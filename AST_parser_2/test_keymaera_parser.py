import unittest
import re
from typing import List, Optional, Union, Dict, Any
import os # Import os module for file path manipulation

# Assuming keymaera_parser.py, keymaera_lexer.py, and ast_config.py are in the same directory
from keymaera_parser import KeYmaeraParser
from keymaera_lexer import parse_annotations
from ast_config import (
    ASTNode, KeYmaeraModel, ArchiveEntry, DefinitionsBlock, Definition, FuncDef, PredDef, HPDef,
    ProgramVariablesBlock, ProgramVariableDecl, ProblemBlock, TacticBlock,
    Term, Literal, VarRef, BinaryTermOp, UnaryTermOp, FunctionCall,
    Formula, BooleanLiteral, BinaryFormulaOp, UnaryFormulaOp, Comparison, Quantifier,
    ModalFormula, PredicateCall, C_Formula,
    HybridProgram, Assignment, NondetAssignment, Test, Sequence, Choice, Iteration,
    ODE, ODEEquation, HPBlock
)


class TestKeYmaeraParser(unittest.TestCase):

    # --- Toggle this flag to True to see the AST output for each test ---
    # Set to True to write full ASTs to files
    _print_ast_output = True 

    # --- Sample KeYmaera Model Texts ---
    SAMPLE_MODEL_IJRR17_T14 = """
    Theorem "IJRR17/Theorem 14: Reach waypoint with deadline"

/*
 *	Robot must stop within distance delta at goal.
 *
 *  Robot
 * 	- must stop within distance delta of goal
 *  - can only drive straight and forward
 *  - ensures progress towards goal
 *
 *  Liveness property:
 *  - Robot can stop at goal
 *
 */

Definitions
    // @const: bound=[0.1:0.5]; def="time limit for control decisions"
    Real ep;      /* time limit for control decisions */
    // @const: bound=[0.5:5.0]; def="minimum braking capability of the robot"
    Real b;       /* minimum braking capability of the robot */
    // @const: bound=[1.0:10.0]; def="maximum acceleration"
    Real A;       /* maximum acceleration -b <= a <= A */
    // @const: bound=[1.0:10.0]; def="goal area size"
    Real GDelta;	 /* goal area size */
    // @const: bound=[5.0:30.0]; def="maximum robot velocity"
    Real Vmax;    /* robot cannot go faster than this */

    Real waypointStartDist(Real xg) = xg-GDelta;
    Real waypointEndDist(Real xg)   = xg+GDelta;

    Real minV = A*ep;
    Real maxTravelTime(Real g) = waypointStartDist(g)/minV;
    Real stopTime(Real v)      = v/b;
    Real speedUpTime(Real v)   = ep-v/A;

    Real stopDist(Real v) = v^2/(2*b);
    Real accComp(Real v)  = ( (A/b + 1)*(A/2*ep^2 + ep*v) );

    Bool bounds() <-> (                                                                 
        A > 0                                                                                         /* Working engine */
        & b > 0                                                                                         /* Working brakes */
        & ep > 0                                                                              /* Controller reaction time */
        & Vmax >= 2*A*ep
        & GDelta > Vmax*ep + Vmax^2/(2*b)  /* waypoint is large enough that robot can start driving and still stop inside */
    );

    Bool initialState(Real xr, Real vr, Real xg, Real T) <-> (                      
        vr = 0
        & xr < waypointStartDist(xg)
        & T > ep + maxTravelTime(xg-xr) + ep + stopTime(Vmax)
    );

    Bool assumptions(Real xr, Real vr, Real xg, Real T) <->              
        bounds() & initialState(xr, vr, xg, T);

    // @safety: penalty=-1000, on_failure=truncate:status=failure
    Bool loopinv(Real xr, Real vr, Real xg, Real T) <-> ( // Added T to parameters as it's used in the invariant
        0 <= vr & vr <= Vmax & xr + stopDist(vr) < waypointEndDist(xg)
                & (waypointStartDist(xg) < xr -> (vr = 0 | T >= stopTime(vr)))
                & (xr <= waypointStartDist(xg) ->
                    (vr >= minV & T > maxTravelTime(xg-xr) + ep + stopTime(Vmax))
                    | (vr <= minV & T > speedUpTime(vr) + maxTravelTime(xg-xr) + ep + stopTime(Vmax)))
    );

    HP controlFunc ::= {
        // Choice 1: If in the goal area, decide to brake
        { ? (xr > waypointStartDist(xg));  ar := -b; } 
        ++
        // Choice 2: If in the goal area AND already stopped, decide to stay stopped
        { ? (xr > waypointStartDist(xg) & vr = 0);  ar := 0; }
        ++
        // Choice 3: If outside the goal area AND can accelerate, decide to accelerate
        { ? (xr <= waypointStartDist(xg) & (xr + stopDist(vr) + accComp(vr) < waypointEndDist(xg) & vr+A*ep <= Vmax));  ar := A; }
        ++
        // Choice 4: If outside the goal area AND cannot accelerate (must coast), decide to coast
        { ? (xr <= waypointStartDist(xg) & !(xr + stopDist(vr) + accComp(vr) < waypointEndDist(xg) & vr+A*ep <= Vmax));  ar := 0; }
    };

    HP stepFunc ::= {xr' = vr, vr' = ar, t' = 1, T'=-1 & t <= ep & vr >= 0};

    HP dwwpdl ::= {
        {	{ controlFunc; t := 0; }
            { stepFunc; }
        }*@invariant(loopinv(xr, vr, xg, T))
    };

    // @progress: reward=1000, on_success=terminate:status=success, on_failure=terminate:status=failure
    Bool ensure(Real xr, Real vr, Real xg) <-> (
        waypointStartDist(xg) < xr & vr = 0
    );

End.

ProgramVariables
    // @var: bound=[0:100], def="robot position: x"
    Real xr;
    // @var: bound=[0:20], def="robot translational velocity"
    Real vr;
    // @var: bound=[-10:10], def="robot translational acceleration"
    Real ar;
    // @var: bound=[0:100], def="goal position: x"
    Real xg;
    // @var: bound=[0:1], def="time elapsed in current control step"
    Real t;
    // @var: bound=[0:1000], def="global time remaining until deadline"
    Real T;
End.

Problem
    assumptions(xr, vr, xg, T)
        -> [ dwwpdl; ](xr < waypointEndDist(xg) & (T <= 0 -> ensure(xr, vr, xg))) // Safety postcondition
        & < dwwpdl; > ensure(xr, vr, xg)                                         // Liveness postcondition
End.
    """

    SAMPLE_MODEL_IJRR17_T15 = """
Theorem "IJRR17/Theorem 15: Cross intersection with deadline"

Definitions
  // @const: bound=[0.01:0.1]; def="time limit for control decisions"
  Real ep;      /* time limit for control decisions */
  // @const: bound=[0.5:5.0]; def="minimum braking capability of the robot"
  Real b;       /* minimum braking capability of the robot */
  // @const: bound=[1.0:10.0]; def="maximum acceleration"
  Real A;       /* maximum acceleration -b <= a <= A */
  // @const: bound=[0.1:2.0]; def="obstacle minimum speed"
  Real Vmin;    /* obstacle minimum speed */
  Real ixr = 0; /* position of intersection on path of robot */
  Real ixo = 0; /* position of intersection on path of obstacle */
  // @const: bound=[1.0:10.0]; def="deadline for robot to cross intersection"
  Real D;       /* Deadline */

  Real minV = A*ep;

  Real stopDist(Real v) = v^2/(2*b);
  Real accComp(Real v)  = ( (A/b + 1)*(A/2*ep^2 + ep*v) );

  Bool OAfterX(Real xo) <-> xo>ixo;
  Bool RAfterX(Real xr) <-> xr>ixr;

  // Global Bounds
  Bool bounds() <-> (
      A > 0 
    & b > 0 
    & ep > 0 
    & Vmin > 0
    & D >= ep
  );

  // Initial Conditions
  Bool initialState(Real xr, Real vr, Real ixr, Real xo, Real vo, Real ixo, Real T) <-> (
    vr = 0
    & vo>=Vmin
    & xr < ixr
    & ixr-xr < A/2*(D-ep)^2
    & T = min(0, (xo-ixo)/Vmin) - ep
    & RAfterX(xr + A/2*(D - ep)^2)
  );

  Bool assumptions(Real xr, Real vr, Real ixr, Real xo, Real vo, Real ixo, Real T) <->
    bounds() & initialState(xr, vr, ixr, xo, vo, ixo, T);

  // Safety Invariant
  // @safety: penalty=-1000, on_failure=truncate:status=failure, on_success=continue:status=none
  Bool safetyloopinv(Real xr, Real vr, Real ixr, Real xo, Real vo, Real ixo) <-> (
    0 <= vr & Vmin<=vo & (RAfterX(xr) | OAfterX(xo) | xr + stopDist(vr) < ixr |
      (vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo
              | ixo < xo + Vmin*(ixr-xr)/vr))
    )
  );

  // Deadline Invariant
  // @progress: reward=10, on_success=continue:status=none, on_failure=continue:status=none, shape={-T}
  Bool deadlineloopinv(Real xr, Real vr, Real ixr, Real xo, Real vo, Real ixo, Real T) <-> (
    0 <= vr & Vmin<=vo
    & T <= (xo-ixo)/Vmin
    & ( T <= 0 & RAfterX(xr + A/2*(D - ep)^2)
      | T > 0 & RAfterX(xr + vr*(max(0,D-T)) + A/2*(max(0,D-T))^2) )
  );

  HP controlFunc ::= {
    // Obstacle's action
    ao := *; ?-b<=ao & ao<=A;
    
    // Robot's action rewritten from if-else to {P?; A;} ++ {!P?; B;}
    {
      ?OAfterX(xo); 
      ar := A;
    } ++ {
      ?!OAfterX(xo);
      {
        ?RAfterX(xr);
        ar := *; ?-b<=ar & ar<=A;
      } ++ {
        ?!RAfterX(xr);
        {
          ?(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) ));
          ar := *; ?0<=ar & ar<=A;
        } ++ {
          ?!(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) ));
          {
            ?(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr);
            ar := 0;
          } ++ {
            ?!(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr);
            { ar := -b; } 
            ++ { ?vr = 0; ar := 0; }
            ++ { ?xr + stopDist(vr) + accComp(vr) < ixr; ar := A; }
          }
        }
      }
    };
    t := 0;
  };

  HP stepFunc ::= { {xr' = vr, vr' = ar, xo'=vo, vo'=ao, t' = 1, T'=-1 & t <= ep & vr >= 0 & vo>=Vmin } };

  HP dwwcxd ::= {
    {
      controlFunc;
      stepFunc;
    }*@invariant(safetyloopinv(xr, vr, ixr, xo, vo, ixo) & deadlineloopinv(xr, vr, ixr, xo, vo, ixo, T))
  };

  // @safety: penalty=-1000, on_failure=terminate:status=failure, on_success=continue:status=none
  Bool finalSafety(Real xr, Real xo, Real T) <-> ( (xr=ixr -> xo!=ixo) & (T>=D -> RAfterX(xr)) );

  // @progress: reward=500, on_success=terminate:status=success, on_failure=terminate:status=failure
  Bool ensure(Real xr) <-> RAfterX(xr);

End.

ProgramVariables
  // @var: bound=[0:100]; def="robot position: x"
  Real xr;
  // @var: bound=[0:20]; def="robot translational velocity"
  Real vr;
  // @var: bound=[-10:10]; def="robot translational acceleration"
  Real ar;
  // @var: bound=[0:100]; def="obstacle position: x"
  Real xo;
  // @var: bound=[0:20]; def="obstacle velocity"
  Real vo;
  // @var: bound=[-10:10]; def="obstacle acceleration"
  Real ao;
  // @var: bound=[0:1]; def="time elapsed in current control step"
  Real t;
  // @var: bound=[-100:100]; def="remaining time until deadline"
  Real T;
End.

Problem
  assumptions(xr, vr, ixr, xo, vo, ixo, T)
    ->
    [ dwwcxd; ]finalSafety(xr, xo, T)
    &
    < dwwcxd; >ensure(xr)
End.
    """

    SAMPLE_MODEL_IJRR17_T1 = """
Theorem "IJRR17/Theorem 1: Static safety"

Definitions
  // Physical Constants
  // @const: bound=[0.1:0.5], def="time limit for control decisions"
  Real ep; 
  // @const: bound=[0.5:5.0], def="minimum braking capability of the robot"
  Real b;       
  // @const: bound=[1.0:10.0], def="maximum acceleration"
  Real A;       
  // @const: bound=[0.1:1.0], def="maximum steering (rotational velocity)"
  Real W;       

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
    """

    # --- Additional Sample KeYmaera Model Texts ---

    SAMPLE_MODEL_IJRR17_T13 = """
Theorem "IJRR17/Theorem 13: Passive safety with trajectory distance measurement"

Definitions
  // @const: bound=[0.1:0.5], def="time limit for control decisions"
  Real ep;      /* time limit for control decisions */
  // @const: bound=[0.5:5.0], def="minimum braking capability of the robot"
  Real b;       /* minimum braking capability of the robot */
  // @const: bound=[1.0:10.0], def="maximum acceleration"
  Real A;       /* maximum acceleration -b <= a <= A */
  // @const: bound=[0.1:1.0], def="maximum steering (rotational velocity)"
  Real W;       /* maximum steering */
  // @const: bound=[0.1:5.0], def="maximum obstacle velocity"
  Real V;       /* maximum obstacle velocity */

  Real stopDist(Real v) = v^2 / (2*b) + V*v/b;
  Real accelComp(Real v, Real a) = ((a/b + 1) * (A/2 * ep^2 + ep*(v+V)));
  Real admissibleSeparationG(Real v, Real a) = stopDist(v) + accelComp(v,a);
  Real admissibleSeparationL(Real v, Real a) = -v^2/(2*a)-V*v/a;
  Real trajectoryDist(Real r, Real xo, Real xc, Real yo, Real yc) = abs(abs(r) - ((xo-xc)^2 + (yo-yc)^2)^(1/2));

  // @progress: reward=10, shape={-((dx**2 + dy**2) - 1)**2 - ((r**2 - ((x-xc)**2 + (y-yc)**2)))**2}, on_success=continue:status=none, on_failure=continue:status=none
  Bool isWellformedDir(Real x, Real y, Real v, Real w, Real r, Real dx, Real dy, Real xc, Real yc) <->
    r!=0 & r^2 = (x-xc)^2 + (y-yc)^2 & dx^2 + dy^2 = 1 & dx = -(y-yc)/r & dy = (x-xc)/r & w*r=v;

  Bool bounds() <-> (                                                                  /* Bounds for global constants */
      A >= 0                                                                                        /* Working engine */
    & b > 0                                                                                         /* Working brakes */
    & ep > 0                                                                              /* Controller reaction time */
    & V >= 0
  );
  /* Stopped somewhere safe initially */
  Bool initialState(Real x, Real y, Real v, Real w, Real r, Real dx, Real dy, Real xo, Real yo, Real xc, Real yc) <-> (
    v = 0
    & (x-xo)^2 + (y-yo)^2 > 0
    & isWellformedDir(x, y, v, w, r, dx, dy, xc, yc)
  );

  /* Under these assumptions we guarantee safety */
  Bool assumptions(Real x, Real y, Real v, Real w, Real r, Real dx, Real dy, Real xo, Real yo, Real xc, Real yc) <->
    bounds() & initialState(x, y, v, w, r, dx, dy, xo, yo, xc, yc);

  // @safety: penalty=-1000, on_failure=truncate:status=failure, on_success=continue:status=none
  /* Conditions that are true on each loop iteration */
  Bool loopinv(Real x, Real y, Real v, Real w, Real r, Real dx, Real dy, Real xo, Real yo, Real xc, Real yc) <-> (
    v >= 0
    & isWellformedDir(x, y, v, w, r, dx, dy, xc, yc)
    & (v>0 -> abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v) | abs(abs(r) - ((xo-xc)^2 + (yo-yc)^2)^(1/2)) > V*v/b)
  );

  // @progress: reward=1000, on_success=terminate:status=success, on_failure=terminate:status=failure
  Bool ensure(Real v, Real x, Real y, Real xo, Real yo) <-> (v>0 -> (x - xo)^2 + (y - yo)^2 > 0);

End.

ProgramVariables
  // @var: bound=[0:100], def="robot position: x"
  Real x;    /* robot position: x */
  // @var: bound=[0:100], def="robot position: y"
  Real y;    /* robot position: y */
  // @var: bound=[0:20], def="robot translational velocity"
  Real v;    /* robot translational velocity */
  // @var: bound=[-10:10], def="robot translational acceleration"
  Real a;    /* robot translational acceleration */
  // @var: bound=[-1:1], def="robot orientation: x component"
  Real dx;   /* robot orientation: x */
  // @var: bound=[-1:1], def="robot orientation: y component"
  Real dy;   /* robot orientation: y */
  // @var: bound=[-2:2], def="robot rotational velocity"
  Real w;    /* robot rotational velocity */
  // @var: bound=[-1000:1000], def="robot curve radius (r!=0)"
  Real r;    /* robot curve radius */
  // @var: bound=[0:100], def="center of curvature: x"
  Real xc;
  // @var: bound=[0:100], def="center of curvature: y"
  Real yc;
  // @var: bound=[0:100], def="position of closest obstacle on curve: x"
  Real xo;   /* position of closest obstacle on curve */
  // @var: bound=[0:100], def="position of closest obstacle on curve: y"
  Real yo;
  // @var: bound=[-5:5], def="obstacle velocity: x component"
  Real vxo;  /* velocity vector of obstacle */
  // @var: bound=[-5:5], def="obstacle velocity: y component"
  Real vyo;
  // @var: bound=[0:1], def="time elapsed in current control step"
  Real t;    /* time */
End.

Problem
assumptions(x, y, v, w, r, dx, dy, xo, yo, xc, yc) ->
  [
    {
      {
        /* obstacle control */
        {
          vxo := *; vyo := *;
          ?vxo^2+vyo^2<=V^2;
        }
        /* robot control */
        {
          /* brake on current curve or remain stopped */
          { a := -b; }
          ++
          { ?v = 0; a := 0; w := 0; { dx:=-dx;dy:=-dy; ++ dx:=dx;dy:=dy; }; r := *; xc := *; yc := *;
            ?(r!=0 & r^2 = (x-xc)^2+(y-yc)^2 & dx=-(y-yc)/r & dy=(x-xc)/r & r*w=v); }
          ++
          /* or choose a new safe curve */
          { a := *; ?-b<=a & a<=A;
            w := *; ?-W<=w & w<=W;       /* choose steering */
            r := *;
            xc := *; yc := *;

            xo := *; yo := *;            /* measure closest obstacle on the curve */

            /* admissible curve */
            ?(r!=0 & r^2 = (x-xc)^2+(y-yc)^2 & dx=-(y-yc)/r & dy=(x-xc)/r & r*w=v);

            /* use that curve, if it is a safe one (admissible velocities) */
            {
              ?(v+a*ep>=0);
              ?abs(x-xo) > admissibleSeparationG(v,a) | abs(y-yo) > admissibleSeparationG(v,a) | trajectoryDist(r,xo,xc,yo,yc) > V*(ep + (v+a*ep)/b);
            }
            ++
            {
              ?!(v+a*ep>=0);
              ?abs(x-xo) > admissibleSeparationL(v,a) | abs(y-yo) > admissibleSeparationL(v,a) | trajectoryDist(r,xo,xc,yo,yc) > -V*v/a;
            }
          }
        };
        t := 0;
      }

      /* dynamics */
      { x' = v * dx, y' = v * dy, v' = a,        /* accelerate/decelerate and move */
        dx' = -w * dy, dy' = w * dx, w' = a/r,   /* follow curve */
        xo' = vxo, yo' = vyo,                    /* obstacle moves */
        t' = 1 & t <= ep & v >= 0
      }
    }*@invariant(loopinv(x, y, v, w, r, dx, dy, xo, yo, xc, yc))
  ]ensure(v, x, y, xo, yo)
End.
    """

    SAMPLE_MODEL_IJRR17_C2 = """
Theorem "IJRR17/Corollary 2: Velocity-controlled passive orientation safety"
Definitions
  // @const: bound=[0.1:0.5], def="time limit for control decisions"
  Real ep;      /* time limit for control decisions */
  // @const: bound=[0.1:1.0], def="maximum steering (rotational velocity)"
  Real W;       /* maximum steering */
  // @const: bound=[0.1:5.0], def="maximum obstacle velocity"
  Real V;       /* maximum obstacle velocity */
  // @const: bound=[0.1:PI/2], def="Sensor range in radian"
  Real GammA;   /* Sensor range in radian */

  Real stopDist(Real v)   = 0;
  Real stopMargin(Real v) = stopDist(v);
  Real accelComp(Real v)  = ep*v;
  Real accelMargin(Real v)  = accelComp(v) + ep*V;
  Real admissibleSeparation(Real v) = stopMargin(v) + accelMargin(v);
  Real admissibleTurnLength(Real v) = stopDist(v) + accelComp(v);

  // @progress: reward=10, shape={-((dx**2 + dy**2) - 1)**2}, on_success=continue:status=none, on_failure=continue:status=none
  Bool isWellformedDir(Real dx, Real dy) <-> dx^2 + dy^2 = 1;       /* The orientation of the robot is a unit vector. */

  Bool isVisible(Real x) <-> x>0;  /* Indicates whether or not the obstacle is visible for the robot at the decision. */

  Bool bounds() <-> (                                                                  /* Bounds for global constants */
      ep > 0                                                                              /* Controller reaction time */
    & V >= 0
    & GammA > 0
  );

  /* Stopped somewhere safe initially */
  Bool initialState(Real x, Real y, Real v, Real beta, Real r, Real dx, Real dy, Real xo, Real yo) <-> (
    v = 0
    & beta = 0
    & r != 0
    & (x-xo)^2 + (y-yo)^2 > 0
    & isWellformedDir(dx,dy)
  );

  /* Under these assumptions we guarantee safety */
  Bool assumptions(Real x, Real y, Real v, Real beta, Real r, Real dx, Real dy, Real xo, Real yo) <->
    bounds() & initialState(x, y, v, beta, r, dx, dy, xo, yo);

  // @safety: penalty=-1000, on_failure=truncate:status=failure, on_success=continue:status=none
  Bool loopinv(Real x, Real y, Real v, Real beta, Real visDeg, Real r, Real dx, Real dy, Real xo, Real yo) <-> (                                                /* Conditions that are true on each loop iteration */
    v >= 0
    & isWellformedDir(dx,dy)
    & r != 0
    & (v>0 -> ( (abs(x-xo) > stopMargin(v) | abs(y-yo) > stopMargin(v))
              | (!isVisible(visDeg) & abs(beta) + stopDist(v)/abs(r) < GammA )) )
  );

  // @progress: reward=1000, on_success=terminate:status=success, on_failure=terminate:status=failure
  Bool ensure(Real v, Real x, Real y, Real xo, Real yo, Real visDeg, Real beta) <-> (v > 0 -> ((x - xo)^2 + (y - yo)^2 > 0 | (!isVisible(visDeg) & (abs(beta) < GammA))) );

End.

ProgramVariables
  // @var: bound=[0:100], def="robot position: x"
  Real x;    /* robot position: x */
  // @var: bound=[0:100], def="robot position: y"
  Real y;    /* robot position: y */
  // @var: bound=[0:20], def="robot translational velocity"
  Real v;    /* robot translational velocity */
  // @var: bound=[-10:10], def="robot translational acceleration (not a program variable here, but conceptually exists)"
  Real a;    /* robot translational acceleration */
  // @var: bound=[-1:1], def="robot orientation: x component"
  Real dx;   /* robot orientation: x */
  // @var: bound=[-1:1], def="robot orientation: y component"
  Real dy;   /* robot orientation: y */
  // @var: bound=[-2:2], def="robot rotational velocity"
  Real w;    /* robot rotational velocity */
  // @var: bound=[-1000:1000], def="robot curve radius (r!=0)"
  Real r;    /* robot curve radius */
  // @var: bound=[0:100], def="position of closest obstacle on curve: x"
  Real xo;   /* position of closest obstacle on curve */
  // @var: bound=[0:100], def="position of closest obstacle on curve: y"
  Real yo;
  // @var: bound=[-5:5], def="obstacle velocity: x component"
  Real vxo;  /* velocity vector of obstacle */
  // @var: bound=[-5:5], def="obstacle velocity: y component"
  Real vyo;
  // @var: bound=[0:1], def="time elapsed in current control step"
  Real t;    /* time */
  // @var: bound=[-PI:PI], def="Angle that the robot traveled since it last chose a new curve"
  Real beta; /* Angle that the robot traveled since it last chose a new curve */
  // @var: bound=[-1:1], def="Indicates the 'degree' to which the obstacle is visible to the robot (input to isVisible)"
  Real visDeg; /* Indicates the "degree" to which the obstacle is visible to the robot (input to isVisible) */
End.

Problem
  assumptions(x, y, v, beta, r, dx, dy, xo, yo) -> [
      {
        {
        /* obstacle control */
        {
          vxo := *;
          vyo := *;
          ?vxo^2+vyo^2 <= V^2;
        }

        /* robot control */
        {
           /* brake on current curve. If the velocity is zero this just means that the robot remains stopped */
           {v := 0; w := 0;}
        ++ /* or choose a new safe curve */
           {v := *;
            beta := 0;
            r :=*; ?r!=0;

            /* measure obstacle position and whether or not it is visible to the robot */
            xo := *; yo := *; visDeg := *;
            ?(isVisible(visDeg) ->
               ( abs(x-xo) > admissibleSeparation(v)
               | abs(y-yo) > admissibleSeparation(v)) );
            ?admissibleTurnLength(v) < GammA*abs(r);
            }
        }

        /* Reset the clocks */
        t := 0;
        /* Set w according to physics (rigid body motion) */
        w :=*; ?w*r = v;
        }

        {x' = v * dx, y' = v * dy, dx' = -w * dy, dy' = w * dx, beta'=w,
                      xo' = vxo, yo' = vyo, t' = 1 & (t <= ep & v >= 0)}
      }*@invariant(loopinv(x, y, v, beta, visDeg, r, dx, dy, xo, yo))
    ] ensure(v, x, y, xo, yo, visDeg, beta)
End.
    """

    SAMPLE_MODEL_IJRR17_T4 = """
Theorem "IJRR17/Theorem 4: Passive orientation safety"
Definitions
  // @const: bound=[0.1:0.5], def="time limit for control decisions"
  Real ep;      /* time limit for control decisions */
  // @const: bound=[1.0:5.0], def="minimum braking capability of the robot"
  Real b;       /* minimum braking capability of the robot */
  // @const: bound=[1.0:10.0], def="maximum acceleration -b <= a <= A"
  Real A;       /* maximum acceleration -b <= a <= A */
  // @const: bound=[0.1:1.0], def="maximum steering (rotational velocity)"
  Real W;       /* maximum steering */
  // @const: bound=[0.1:5.0], def="maximum obstacle velocity"
  Real V;       /* maximum obstacle velocity */
  // @const: bound=[0.1:PI/2], def="Sensor range in radian"
  Real GammA;   /* Sensor range in radian */

  Real stopDist(Real v)   = v^2 / (2*b);
  Real stopMargin(Real v) = stopDist(v) + V*v/b;
  Real accelComp(Real v)  = ( (A/b + 1) * (A/2 * ep^2 + ep*v) );
  Real accelMargin(Real v)  = accelComp(v) + (A/b + 1)*ep*V;
  Real admissibleSeparation(Real v) = stopMargin(v) + accelMargin(v);
  Real admissibleTurnLength(Real v) = stopDist(v) + accelComp(v);

  // @progress: reward=10, shape={-((dx**2 + dy**2) - 1)**2}, on_success=continue:status=none, on_failure=continue:status=none
  Bool isWellformedDir(Real dx, Real dy) <-> dx^2 + dy^2 = 1;       /* The orientation of the robot is a unit vector. */

  Bool isVisible(Real x) <-> x>0;  /* Indicates whether or not the obstacle is visible for the robot at the decision. */

  Bool bounds() <-> (                                                                  /* Bounds for global constants */
      A >= 0                                                                                        /* Working engine */
    & b > 0                                                                                         /* Working brakes */
    & ep > 0                                                                              /* Controller reaction time */
    & V >= 0
    & GammA > 0
  );

  /* Stopped somewhere safe initially */
  Bool initialState(Real x, Real y, Real v, Real beta, Real r, Real dx, Real dy, Real xo, Real yo) <-> (
    v = 0
    & beta = 0
    & r != 0
    & (x-xo)^2 + (y-yo)^2 > 0
    & isWellformedDir(dx,dy)
  );

  /* Under these assumptions we guarantee safety */
  Bool assumptions(Real x, Real y, Real v, Real beta, Real r, Real dx, Real dy, Real xo, Real yo) <->
    bounds() & initialState(x, y, v, beta, r, dx, dy, xo, yo);

  // @safety: penalty=-1000, on_failure=truncate:status=failure, on_success=continue:status=none
  Bool loopinv(Real x, Real y, Real v, Real beta, Real visDeg, Real r, Real dx, Real dy, Real xo, Real yo) <-> (                                                /* Conditions that are true on each loop iteration */
    v >= 0
    & isWellformedDir(dx,dy)
    & r != 0
    & (v>0 -> ( (abs(x-xo) > stopMargin(v) | abs(y-yo) > stopMargin(v))
              | (!isVisible(visDeg) & abs(beta) + stopDist(v)/abs(r) < GammA )) )
  );

  // @progress: reward=1000, on_success=terminate:status=success, on_failure=terminate:status=failure
  Bool ensure(Real v, Real x, Real y, Real xo, Real yo, Real visDeg, Real beta) <-> (v > 0 -> ((x - xo)^2 + (y - yo)^2 > 0 | (!isVisible(visDeg) & (abs(beta) < GammA))) );
End.

ProgramVariables
  // @var: bound=[0:100], def="robot position: x"
  Real x;    /* robot position: x */
  // @var: bound=[0:100], def="robot position: y"
  Real y;    /* robot position: y */
  // @var: bound=[0:20], def="robot translational velocity"
  Real v;    /* robot translational velocity */
  // @var: bound=[-10:10], def="robot translational acceleration"
  Real a;    /* robot translational acceleration */
  // @var: bound=[-1:1], def="robot orientation: x component"
  Real dx;   /* robot orientation: x */
  // @var: bound=[-1:1], def="robot orientation: y component"
  Real dy;   /* robot orientation: y */
  // @var: bound=[-2:2], def="robot rotational velocity"
  Real w;    /* robot rotational velocity */
  // @var: bound=[-1000:1000], def="robot curve radius (r!=0)"
  Real r;    /* robot curve radius */
  // @var: bound=[0:100], def="position of closest obstacle on curve: x"
  Real xo;   /* position of closest obstacle on curve */
  // @var: bound=[0:100], def="position of closest obstacle on curve: y"
  Real yo;
  // @var: bound=[-5:5], def="obstacle velocity: x component"
  Real vxo;  /* velocity vector of obstacle */
  // @var: bound=[-5:5], def="obstacle velocity: y component"
  Real vyo;
  // @var: bound=[0:1], def="time elapsed in current control step"
  Real t;    /* time */
  // @var: bound=[-PI:PI], def="Angle that the robot traveled since it last chose a new curve"
  Real beta; /* Angle that the robot traveled since it last chose a new curve */
  // @var: bound=[-1:1], def="Indicates the 'degree' to which the obstacle is visible to the robot (input to isVisible)"
  Real visDeg; /* Indicates the "degree" to which the obstacle is visible to the robot (input to isVisible) */
End.

Problem
  assumptions(x, y, v, beta, r, dx, dy, xo, yo) -> [
      {
        {
        /* obstacle control */
        {
          vxo := *;
          vyo := *;
          ?vxo^2+vyo^2 <= V^2;
        }

        /* robot control */
        {
           /* brake on current curve. If the velocity is zero this just means that the robot remains stopped */
           {a := -b;}
        ++ /* When we are stopped, both translational and rotational acceleration are 0 */
           {?v=0; a := 0; w := 0;}
        ++ /* or choose a new safe curve */
           {a := A;
            beta := 0;
            r :=*; ?r!=0;

            /* measure obstacle position and whether or not it is visible to the robot */
            xo := *; yo := *; visDeg := *;
            ?(isVisible(visDeg) ->
               ( abs(x-xo) > admissibleSeparation(v)
               | abs(y-yo) > admissibleSeparation(v)) );
            ?admissibleTurnLength(v) < GammA*abs(r);
            }
        }

        /* Reset the clocks */
        t := 0;
        /* Set w according to physics (rigid body motion) */
        w :=*; ?w*r = v;
        }

        {x' = v * dx, y' = v * dy, dx' = -w * dy, dy' = w * dx, v' = a, w' = a/r, beta'=w,
                      xo' = vxo, yo' = vyo, t' = 1 & (t <= ep & v >= 0)}
      }*@invariant(loopinv(x, y, v, beta, visDeg, r, dx, dy, xo, yo))
    ] ensure(v, x, y, xo, yo, visDeg, beta)
End.
    """

    SAMPLE_MODEL_IJRR17_T3 = """
Theorem "IJRR17/Theorem 3: Passive Friendly Safety"
Definitions
  // @const: bound=[0.1:0.5], def="time limit for control decisions"
  Real ep;      /* time limit for control decisions */
  // @const: bound=[0.1:0.5], def="time limit for obstacle reaction"
  Real tau;     /* time limit for obstacle reaction */
  // @const: bound=[0.5:5.0], def="minimum braking capability of the robot"
  Real b;       /* minimum braking capability of the robot */
  // @const: bound=[0.5:5.0], def="obstacle brakes"
  Real bo;      /* obstacle brakes */
  // @const: bound=[1.0:10.0], def="maximum acceleration"
  Real A;       /* maximum acceleration -b <= a <= A */
  // @const: bound=[0.1:1.0], def="maximum steering (rotational velocity)"
  Real W;       /* maximum steering */
  // @const: bound=[0.1:5.0], def="maximum obstacle velocity"
  Real V;       /* maximum obstacle velocity */

  Real friendlyMargin(Real v) = v^2/(2*bo) + tau*v;
  Real stopDist(Real v) = v^2 / (2*b) + V*v/b;
  Real accelComp(Real v) = ((A/b + 1) * (A/2 * ep^2 + ep*(v+V)));
  Real admissibleSeparation(Real v) = stopDist(v) + friendlyMargin(V) + accelComp(v);

  // @progress: reward=10, shape={-((dx**2 + dy**2) - 1)**2}, on_success=continue:status=none, on_failure=continue:status=none
  Bool isWellformedDir(Real dx,Real dy) <-> dx^2 + dy^2 = 1;        /* The orientation of the robot is a unit vector. */

  Bool bounds() <-> (                                                                  /* Bounds for global constants */
      A >= 0                                                                                        /* Working engine */
    & b > 0                                                                                         /* Working brakes */
    & ep > 0                                                                              /* Controller reaction time */
    & V >= 0
    & tau >= 0
    & bo > 0
  );

  /* Stopped somewhere safe initially */
  Bool initialState(Real x, Real y, Real v, Real dx, Real dy, Real xo, Real yo, Real vxo, Real vyo) <-> (
    v = 0
    & (abs(x-xo) > friendlyMargin(V) | abs(y-yo) > friendlyMargin(V))
    & vxo^2+vyo^2<=V^2
    & isWellformedDir(dx,dy)
  );

  /* Under these assumptions we guarantee safety */
  Bool assumptions(Real x, Real y, Real v, Real dx, Real dy, Real xo, Real yo, Real vxo, Real vyo) <->
    bounds() & initialState(x, y, v, dx, dy, xo, yo, vxo, vyo);

  // @safety: penalty=-1000, on_failure=truncate:status=failure, on_success=continue:status=none
  /* Conditions that are true on each loop iteration */
  Bool loopinv(Real x, Real y, Real v, Real dx, Real dy, Real xo, Real yo) <-> (
    v >= 0
    & isWellformedDir(dx,dy)
    & (v>0 -> abs(x-xo) > stopDist(v)+friendlyMargin(V) | abs(y-yo) > stopDist(v)+friendlyMargin(V))
  );

  // @safety: penalty=-1000, on_failure=terminate:status=failure, on_success=continue:status=none
  Bool finalSafety(Real xr, Real xo, Real T) <-> ( (xr=ixr -> xo!=ixo) & (T>=D -> RAfterX(xr)) );

  // @progress: reward=1000, on_success=terminate:status=success, on_failure=terminate:status=failure
  Bool ensure(Real v, Real x, Real y, Real xo, Real yo) <-> (v>0 -> (x - xo)^2 + (y - yo)^2 > 0);

End.

ProgramVariables
  // @var: bound=[0:100], def="robot position: x"
  Real x;    /* robot position: x */
  // @var: bound=[0:100], def="robot position: y"
  Real y;    /* robot position: y */
  // @var: bound=[0:20], def="robot translational velocity"
  Real v;    /* robot translational velocity */
  // @var: bound=[-10:10], def="robot translational acceleration"
  Real a;    /* robot translational acceleration */
  // @var: bound=[-1:1], def="robot orientation: x component"
  Real dx;   /* robot orientation: x */
  // @var: bound=[-1:1], def="robot orientation: y component"
  Real dy;   /* robot orientation: y */
  // @var: bound=[-2:2], def="robot rotational velocity"
  Real w;    /* robot rotational velocity */
  // @var: bound=[-1000:1000], def="robot curve radius"
  Real r;    /* robot curve radius */
  // @var: bound=[0:100], def="position of closest obstacle on curve: x"
  Real xo;   /* position of closest obstacle on curve */
  // @var: bound=[0:100], def="position of closest obstacle on curve: y"
  Real yo;
  // @var: bound=[-5:5], def="obstacle velocity: x component"
  Real vxo;  /* velocity vector of obstacle */
  // @var: bound=[-5:5], def="obstacle velocity: y component"
  Real vyo;
  // @var: bound=[0:20], def="refined obstacle speed"
  Real vo;   /* refined obstacle: speed, acceleration, and direction */
  // @var: bound=[-10:10], def="refined obstacle acceleration"
  Real ao;
  // @var: bound=[-1:1], def="refined obstacle orientation: x component"
  Real dxo;
  // @var: bound=[-1:1], def="refined obstacle orientation: y component"
  Real dyo;
  // @var: bound=[0:1], def="time elapsed in current control step"
  Real t;    /* time */
End.

Problem
assumptions(x, y, v, dx, dy, xo, yo, vxo, vyo) ->
  [
    {
      {
        /* obstacle control */
        {
          vxo := *; vyo := *;
          ?vxo^2+vyo^2<=V^2;
        }
        /* robot control */
        {
          /* brake on current curve or remain stopped */
          { a := -b; }
          ++
          { ?v = 0; a := 0; w := 0; }
          ++
          /* or choose a new safe curve */
          { a := A;
            w := *; ?-W<=w & w<=W;       /* choose steering */
            r := *;
            xo := *; yo := *;            /* measure closest obstacle on the curve */

            /* admissible curve */
            ?r!=0 & r*w = v;

            /* use that curve, if it is a safe one (admissible velocities) */
            ? abs(x-xo) > admissibleSeparation(v)
            | abs(y-yo) > admissibleSeparation(v);
          }
        };
        t := 0;
      }

      /* dynamics */
      { x' = v * dx, y' = v * dy, v' = a,        /* accelerate/decelerate and move */
        dx' = -w * dy, dy' = w * dx, w' = a/r,   /* follow curve */
        xo' = vxo, yo' = vyo,                    /* obstacle moves */
        t' = 1 & t <= ep & v >= 0
      }
    }*@invariant(loopinv(x, y, v, dx, dy, xo, yo))
  ](  (v>0 -> (x - xo)^2 + (y - yo)^2 > (friendlyMargin(V))^2)                     /* robot ensures friendly margin */
    & (  0<=vo & vo^2=vxo^2+vyo^2 & dxo*vo=vxo & dyo*vo=vyo                                      /* obstacle can stop */
       & (x - xo)^2 + (y - yo)^2 > (friendlyMargin(V))^2
       ->
        <{
          ao := *; ?-bo <= ao & vo + ao*ep <= V;
          t := 0;
          {xo'=vo*dxo, yo'=vo*dyo, vo'=ao, t'=1 & vo>=0 & t<=ep}
         }*>((x-xo)^2 + (y-yo)^2 > 0 & vo=0))
  )
End.
    """


    def test_parse_annotations(self):
        print("\nTesting parse_annotations...")
        annotation_line1 = '// @const: bound=[0.1:0.5]; def="time limit"; is_active=true'
        parsed_anno1 = parse_annotations(annotation_line1)
        self.assertEqual(parsed_anno1, {'const': {'bound': [0.1, 0.5], 'def': 'time limit', 'is_active': True}})
        print(f"Input: '{annotation_line1}' -> Output: {parsed_anno1}")

        annotation_line2 = '// @param: value=10, type=int'
        parsed_anno2 = parse_annotations(annotation_line2)
        self.assertEqual(parsed_anno2, {'param': {'value': 10, 'type': 'int'}})
        print(f"Input: '{annotation_line2}' -> Output: {parsed_anno2}")

        annotation_line3 = '// Not an annotation'
        parsed_anno3 = parse_annotations(annotation_line3)
        self.assertEqual(parsed_anno3, {})
        print(f"Input: '{annotation_line3}' -> Output: {parsed_anno3}")
        print("parse_annotations tests passed.")

    def _run_full_model_test(self, model_text: str, expected_kind: str, expected_name: str, filename_prefix: str):
        """Helper function to run full model parsing tests."""
        parser = KeYmaeraParser(model_text)
        try:
            model_ast = parser.parse()
            self.assertIsInstance(model_ast, KeYmaeraModel)
            # For ArchiveEntry, the first entry might be a Theorem or an ArchiveEntry
            # We assume the first entry is always the main one we're testing.
            self.assertEqual(model_ast.entries[0].kind, expected_kind)
            self.assertEqual(model_ast.entries[0].name, expected_name)
            self.assertGreater(len(model_ast.entries[0].definitions.defs), 0)
            self.assertGreater(len(model_ast.entries[0].program_variables.vars), 0)
            self.assertIsInstance(model_ast.entries[0].problem.formula, Formula)
            print(f"Full model ({expected_kind}/{expected_name}) parsing successful!")
            if self._print_ast_output:
                filename = f"{filename_prefix}_ast.txt"
                with open(filename, "w") as f:
                    f.write(str(model_ast))
                print(f"--- AST Output for {expected_kind}/{expected_name} written to '{filename}' ---")
                print("----------------------------------------------------")
        except ValueError as e:
            self.fail(f"Error parsing full model ({expected_kind}/{expected_name}): {e}")

    def test_parse_full_model_ijrr17_t14(self):
        print("\n--- Parsing full sample KeYmaera model (IJRR17/T14) ---")
        self._run_full_model_test(
            self.SAMPLE_MODEL_IJRR17_T14,
            "IJRR17",
            "Theorem 14: Reach waypoint with deadline",
            "ijrr17_t14"
        )

    def test_parse_full_model_ijrr17_t15(self):
        print("\n--- Parsing full sample KeYmaera model (IJRR17/T15) ---")
        self._run_full_model_test(
            self.SAMPLE_MODEL_IJRR17_T15,
            "IJRR17",
            "Theorem 15: Cross intersection with deadline",
            "ijrr17_t15"
        )

    def test_parse_full_model_ijrr17_t1(self):
        print("\n--- Parsing full sample KeYmaera model (IJRR17/T1) ---")
        self._run_full_model_test(
            self.SAMPLE_MODEL_IJRR17_T1,
            "IJRR17",
            "Theorem 1: Static safety",
            "ijrr17_t1"
        )

    # --- New tests for the additional models ---

    def test_parse_full_model_ijrr17_t13(self):
        print("\n--- Parsing full sample KeYmaera model (IJRR17/T13) ---")
        self._run_full_model_test(
            self.SAMPLE_MODEL_IJRR17_T13,
            "IJRR17",
            "Theorem 13: Passive safety with trajectory distance measurement",
            "ijrr17_t13"
        )

    def test_parse_full_model_ijrr17_c2(self):
        print("\n--- Parsing full sample KeYmaera model (IJRR17/C2) ---")
        self._run_full_model_test(
            self.SAMPLE_MODEL_IJRR17_C2,
            "IJRR17", # ArchiveEntry still belongs to IJRR17 kind
            "Corollary 2: Velocity-controlled passive orientation safety",
            "ijrr17_c2"
        )

    def test_parse_full_model_ijrr17_t4(self):
        print("\n--- Parsing full sample KeYmaera model (IJRR17/T4) ---")
        self._run_full_model_test(
            self.SAMPLE_MODEL_IJRR17_T4,
            "IJRR17",
            "Theorem 4: Passive orientation safety",
            "ijrr17_t4"
        )

    def test_parse_full_model_ijrr17_t3(self):
        print("\n--- Parsing full sample KeYmaera model (IJRR17/T3) ---")
        self._run_full_model_test(
            self.SAMPLE_MODEL_IJRR17_T3,
            "IJRR17",
            "Theorem 3: Passive Friendly Safety",
            "ijrr17_t3"
        )


    def test_parse_term_expressions(self):
        print("\n--- Testing parse_term ---")
        term_parser = KeYmaeraParser("") # Parser instance for standalone term parsing

        term_expr1 = "x + y * (z - 5)^2 + f(a, b')"
        parsed_term1 = term_parser.parse_term(term_expr1)
        self.assertIsInstance(parsed_term1, Term)
        print(f"Term '{term_expr1}' parsed as: {parsed_term1}")
        if self._print_ast_output:
            print(f"AST for '{term_expr1}':\n{parsed_term1}")
        # Example assertion: check the top-level operator
        self.assertIsInstance(parsed_term1, BinaryTermOp)
        self.assertEqual(parsed_term1.op, '+')

        term_expr2 = "-10 + x''"
        parsed_term2 = term_parser.parse_term(term_expr2)
        self.assertIsInstance(parsed_term2, Term)
        print(f"Term '{term_expr2}' parsed as: {parsed_term2}")
        if self._print_ast_output:
            print(f"AST for '{term_expr2}':\n{parsed_term2}")
        self.assertIsInstance(parsed_term2, BinaryTermOp)
        self.assertEqual(parsed_term2.op, '+')
        self.assertIsInstance(parsed_term2.left, UnaryTermOp)
        self.assertEqual(parsed_term2.left.op, '-')
        self.assertIsInstance(parsed_term2.right, UnaryTermOp)
        self.assertEqual(parsed_term2.right.op, "'")
        self.assertIsInstance(parsed_term2.right.operand, UnaryTermOp)
        self.assertEqual(parsed_term2.right.operand.op, "'")


    def test_parse_formula_expressions(self):
        print("\n--- Testing parse_formula ---")
        formula_parser = KeYmaeraParser("") # Parser instance for standalone formula parsing

        formula_expr1 = "forall(x, y). P(x) -> (x > 0 && y < 10)"
        parsed_formula1 = formula_parser.parse_formula(formula_expr1)
        self.assertIsInstance(parsed_formula1, Formula)
        print(f"Formula '{formula_expr1}' parsed as: {parsed_formula1}")
        if self._print_ast_output:
            print(f"AST for '{formula_expr1}':\n{parsed_formula1}")
        self.assertIsInstance(parsed_formula1, Quantifier)
        self.assertEqual(parsed_formula1.quantifier_type, 'forall')

        formula_expr2 = "![hp_choice] (x = 5) <-> <simple_hp> true"
        parsed_formula2 = formula_parser.parse_formula(formula_expr2)
        self.assertIsInstance(parsed_formula2, Formula)
        print(f"Formula '{formula_expr2}' parsed as: {parsed_formula2}")
        if self._print_ast_output:
            print(f"AST for '{formula_expr2}':\n{parsed_formula2}")
        # FIX: The top-level operator for this formula is '!', which is a UnaryFormulaOp.
        self.assertIsInstance(parsed_formula2, UnaryFormulaOp)
        self.assertEqual(parsed_formula2.op, '!')


    def test_parse_hybrid_program_expressions(self):
        print("\n--- Testing parse_hybrid_program ---")
        hp_parser = KeYmaeraParser("") # Parser instance for standalone HP parsing

        hp_expr1 = "{x' = v, t <= 10} ; ?(x > 0) ++ {y := y + 1}*@invariant(y < 5)"
        parsed_hp1 = hp_parser.parse_hybrid_program(hp_expr1)
        self.assertIsInstance(parsed_hp1, HybridProgram)
        print(f"Hybrid Program '{hp_expr1}' parsed as: {parsed_hp1}")
        if self._print_ast_output:
            print(f"AST for '{hp_expr1}':\n{parsed_hp1}")
        self.assertIsInstance(parsed_hp1, Choice)

        hp_expr2 = "{x := x + 1; y := y - 1} ; complex_ode"
        parsed_hp2 = hp_parser.parse_hybrid_program(hp_expr2)
        self.assertIsInstance(parsed_hp2, HybridProgram)
        print(f"Hybrid Program '{hp_expr2}' parsed as: {parsed_hp2}")
        if self._print_ast_output:
            print(f"AST for '{hp_expr2}':\n{parsed_hp2}")
        self.assertIsInstance(parsed_hp2, Sequence)


if __name__ == '__main__':
    print("--- Starting KeYmaera Parser Test Section ---")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\n--- KeYmaera Parser Test Section Complete ---")