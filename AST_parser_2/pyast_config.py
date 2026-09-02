# file description: This file contains the configuration dataclasses for the Gym environment, including variable declarations, constant definitions, function definitions, hybrid program definitions, and the overall problem configuration.
# name: pyast_config.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union # Added Union for action/observation bounds

@dataclass
class VariableDeclarationConfig:
    name: str
    type: str  # e.g., 'real', 'int', 'bool'
    lower_bound: Optional[str] = None  # Stored as string to allow expressions
    upper_bound: Optional[str] = None  # Stored as string to allow expressions
    initial_value: Optional[str] = None # Stored as string to allow expressions

@dataclass
class ConstantDefinitionConfig:
    name: str
    value: str  # Stored as string to allow expressions or specific types

@dataclass
class FuncDefConfig:
    name: str
    args: List[str]
    body: str  # The body of the function as a string

@dataclass
class HPDefConfig:
    name: str
    args: List[str]
    body: str  # The body of the hybrid program as a string

@dataclass
class ProblemConfig:
    initial_condition: str
    program: str
    goal: str
    loop_invariants: List[str] = field(default_factory=list)
    differential_invariants: List[str] = field(default_factory=list)
    differential_ghosts: List[str] = field(default_factory=list)

@dataclass
class GymEnvironmentConfig:
    environment_name: str
    variables: List[VariableDeclarationConfig] = field(default_factory=list)
    constants: List[ConstantDefinitionConfig] = field(default_factory=list)
    function_definitions: List[FuncDefConfig] = field(default_factory=list)
    hp_definitions: List[HPDefConfig] = field(default_factory=list)
    problem: Optional[ProblemConfig] = None
    action_space_type: str = "continuous" # or "discrete"
    action_space_bounds: Dict[str, Any] = field(default_factory=dict) # e.g., {'x': {'min': -1, 'max': 1}}
    observation_space_type: str = "continuous" # or "discrete"
    observation_space_bounds: Dict[str, Any] = field(default_factory=dict) # e.g., {'x': {'min': -inf, 'max': inf}}
    reward_function_body: str = ""
    termination_condition_body: str = ""
    max_episode_steps: int = 200
    time_step: float = 0.01 # Default simulation time step for differential equations
    ode_solver_method: str = "DOP853" # Default ODE solver, changed from "dopri5" to "DOP853"
    ode_solver_step_size: float = 0.001 # Default step size for ODE solver
    ode_solver_tolerance: float = 1e-6 # Default tolerance for ODE solver
    initial_state_sampling_config: Dict[str, Any] = field(default_factory=dict) # e.g., {'x': {'min': 0, 'max': 1}}
    seed: Optional[int] = None
    render_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    })
    # Add any other configuration parameters needed for the environment
    # e.g., for reward shaping, logging, etc.

    # New field for initial state buffer configuration
    initial_state_buffer_config: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "capacity": 100,
        "populate_on_init": True,
        "max_populate_attempts": 1000,
        "strategy": "random"  # "random" or "z3_diverse"
    })
    # New field to store the serialized controlFunc AST
    control_flow_program_ast_json: Optional[Dict[str, Any]] = None