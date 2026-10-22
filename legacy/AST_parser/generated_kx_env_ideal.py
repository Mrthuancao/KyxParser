# GeneratedKXEnv.py
# This file defines the GeneratedKXEnv class, which is a Gymnasium environment generated from a KeYmaera X problem specification. The environment is designed to simulate the dynamics of the problem, evaluate the specified predicates and formulas, and apply control actions based on the control function's AST configuration. The class extends a base environment defined in kx_env_base.py, which provides core functionality for handling KeYmaera X configurations and logic.
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import re
import math
import textwrap # Needed for dedent
from typing import (Dict, Any, Tuple, List, Optional, Union)

# Import the base class
from gym_generator.kx_env_base import _KeYmaeraXEnvBase

import z3 # Z3 is used in the base class for CISB population

# --- Dummy pyast_config classes (REMOVE if you have the actual library) ---
# These are necessary if pyast_config is not installed as a package,
# so that the eval(config_serialized_str) can work.
class GymEnvironmentConfig:
    def __init__(self, env_name, definitions, variables, problem):
        self.env_name = env_name
        self.definitions = definitions
        self.variables = variables
        self.problem = problem
    def __repr__(self): return f"GymEnvironmentConfig(env_name='{self.env_name}', definitions={self.definitions}, variables={self.variables}, problem={self.problem})"

class DefinitionsConfig:
    def __init__(self, constants, functions, predicates, hybrid_programs):
        self.constants = constants
        self.functions = functions
        self.predicates = predicates
        self.hybrid_programs = hybrid_programs
    def __repr__(self): return f"DefinitionsConfig(constants={self.constants}, functions={self.functions}, predicates={self.predicates}, hybrid_programs={self.hybrid_programs})"

class VariablesConfig:
    def __init__(self, state_variables, auxiliary_variables):
        self.state_variables = state_variables
        self.auxiliary_variables = auxiliary_variables
    def __repr__(self): return f"VariablesConfig(state_variables={self.state_variables}, auxiliary_variables={self.auxiliary_variables})"

class ProblemConfig:
    def __init__(self, precondition, loop_execution, postcondition):
        self.precondition = precondition
        self.loop_execution = loop_execution
        self.postcondition = postcondition
    def __repr__(self): return f"ProblemConfig(precondition={self.precondition}, loop_execution={self.loop_execution}, postcondition={self.postcondition})"

class ConstantDefinitionConfig:
    def __init__(self, name, type, value, metadata):
        self.name = name
        self.type = type
        self.value = value
        self.metadata = metadata
    def __repr__(self): return f"ConstantDefinitionConfig(name='{self.name}', type='{self.type}', value='{self.value}', metadata={self.metadata})"

class FunctionDefinitionConfig:
    def __init__(self, name, return_type, parameters, body, metadata):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.body = body
        self.metadata = metadata
    def __repr__(self): return f"FunctionDefinitionConfig(name='{self.name}', return_type='{self.return_type}', parameters={self.parameters}, body='{self.body}', metadata={self.metadata})"

class PredicateDefinitionConfig:
    def __init__(self, name, parameters, body, metadata):
        self.name = name
        self.parameters = parameters
        self.body = body
        self.metadata = metadata
    def __repr__(self): return f"PredicateDefinitionConfig(name='{self.name}', parameters={self.parameters}, body='{self.body}', metadata={self.metadata})"

class HybridProgramConfig: # Base class for ControlFuncConfig and StepFuncConfig
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata
    def __repr__(self): return f"HybridProgramConfig(name='{self.name}', metadata={self.metadata})"

class ControlFuncConfig(HybridProgramConfig):
    def __init__(self, name, action_space_config, metadata):
        super().__init__(name, metadata)
        self.action_space_config = action_space_config
    def __repr__(self): return f"ControlFuncConfig(name='{self.name}', action_space_config={self.action_space_config}, metadata={self.metadata})"

class StepFuncConfig(HybridProgramConfig):
    def __init__(self, name, odes, evolution_domain, duration, raw_body_snippet, metadata):
        super().__init__(name, metadata)
        self.odes = odes
        self.evolution_domain = evolution_domain
        self.duration = duration
        self.raw_body_snippet = raw_body_snippet
    def __repr__(self): return f"StepFuncConfig(name='{self.name}', odes={self.odes}, evolution_domain={self.evolution_domain}, duration={self.duration}, raw_body_snippet='{self.raw_body_snippet}', metadata={self.metadata})"

class ODEConfig:
    def __init__(self, variable, equation):
        self.variable = variable
        self.equation = equation
    def __repr__(self): return f"ODEConfig(variable='{self.variable}', equation='{self.equation}')"

class EvolutionDomainConfig: # Placeholder
    def __init__(self, formula):
        self.formula = formula
    def __repr__(self): return f"EvolutionDomainConfig(formula='{self.formula}')"

class VariableDeclarationConfig:
    def __init__(self, name, type, lower_bound, upper_bound):
        self.name = name
        self.type = type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    def __repr__(self): return f"VariableDeclarationConfig(name='{self.name}', type='{self.type}', lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"

class PreconditionConfig:
    def __init__(self, formula):
        self.formula = formula
    def __repr__(self): return f"PreconditionConfig(formula='{self.formula}')"

class PostconditionConfig:
    def __init__(self, formula):
        self.formula = formula
    def __repr__(self): return f"PostconditionConfig(formula='{self.formula}')"

class LoopExecutionConfig:
    def __init__(self, iterations, loop_invariant_formula, loop_body):
        self.iterations = iterations
        self.loop_invariant_formula = loop_invariant_formula
        self.loop_body = loop_body
    def __repr__(self): return f"LoopExecutionConfig(iterations='{self.iterations}', loop_invariant_formula='{self.loop_invariant_formula}', loop_body={self.loop_body})"

class HybridProgramStepConfig: # Placeholder
    def __init__(self, name):
        self.name = name
    def __repr__(self): return f"HybridProgramStepConfig(name='{self.name}')"

class ActionType: # Enum-like class
    CHOICE = "choice"
    SEQUENCE = "sequence"
    DET_ASSIGNMENT = "deterministic_assignment"
    NONDET_ASSIGNMENT = "nondeterministic_assignment"
    CONSTRAINT = "constraint"
    CALL_HP = "call_hybrid_program"

class ChoiceActionConfig:
    def __init__(self, level_name, choices):
        self.level_name = level_name
        self.choices = choices
    def __repr__(self): return f"ChoiceActionConfig(level_name='{self.level_name}', choices={self.choices})"

class SequenceActionConfig:
    def __init__(self, actions):
        self.actions = actions
    def __repr__(self): return f"SequenceActionConfig(actions={self.actions})"

class DeterministicAssignmentActionConfig:
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
    def __repr__(self): return f"DeterministicAssignmentActionConfig(variable='{self.variable}', value='{self.value}')"

class NondeterministicAssignmentActionConfig:
    def __init__(self, variable, lower_bound, upper_bound):
        self.variable = variable
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    def __repr__(self): return f"NondeterministicAssignmentActionConfig(variable='{self.variable}', lower_bound='{self.lower_bound}', upper_bound='{self.upper_bound}')"

class ActionConstraintConfig:
    def __init__(self, formula):
        self.formula = formula
    def __repr__(self): return f"ActionConstraintConfig(formula='{self.formula}')"

class CallHybridProgramConfig:
    def __init__(self, hp_name):
        self.hp_name = hp_name
    def __repr__(self): return f"CallHybridProgramConfig(hp_name='{self.hp_name}')"
# --- End Dummy pyast_config classes ---


class GeneratedKXEnv(_KeYmaeraXEnvBase):
    metadata = {"render_modes": ["human", "plot_data", "matplotlib"], "render_fps": 30}

    def __init__(self, render_mode=None, cisb_size: int = 1000, cisb_populate_attempts: int = 5000, seed: Optional[int] = None, debug_mode: bool = False):
        # Define the action space configuration for the controlFunc
        # Here, we allow 'ar' to be a nondeterministic assignment, bounded by -A and A
        ar_nondet_assign_config = NondeterministicAssignmentActionConfig(
            variable='ar',
            lower_bound='-A', # 'A' is a constant, will be evaluated by _build_action_space_from_config
            upper_bound='A'
        )
        # Wrap it in a SequenceActionConfig as the top-level control action
        control_func_action_space_config = SequenceActionConfig(actions=[ar_nondet_assign_config])

        # The original config_serialized_str from the user's prompt, now as a template
        original_config_template = textwrap.dedent('''
            GymEnvironmentConfig(
                env_name='IJRR17_Theorem_14__Reach_waypoint_with_deadline',
                definitions=DefinitionsConfig(
                    constants=[
                        ConstantDefinitionConfig(
                            name='ep',
                            type='Real',
                            value='',
                            metadata={
                                'type': 'const',
                                'bound': '[0.1:0.5]',
                                'def': '"time limit for control decisions"'
                            }
                        ),
                        ConstantDefinitionConfig(
                            name='b',
                            type='Real',
                            value='',
                            metadata={
                                'type': 'const',
                                'bound': '[0.5:5.0]',
                                'def': '"minimum braking capability of the robot"'
                            }
                        ),
                        ConstantDefinitionConfig(
                            name='A',
                            type='Real',
                            value='',
                            metadata={
                                'type': 'const',
                                'bound': '[1.0:10.0]',
                                'def': '"maximum acceleration"'
                            }
                        ),
                        ConstantDefinitionConfig(
                            name='GDelta',
                            type='Real',
                            value='',
                            metadata={
                                'type': 'const',
                                'bound': '[1.0:10.0]',
                                'def': '"goal area size"'
                            }
                        ),
                        ConstantDefinitionConfig(
                            name='Vmax',
                            type='Real',
                            value='',
                            metadata={
                                'type': 'const',
                                'bound': '[5.0:30.0]',
                                'def': '"maximum robot velocity"'
                            }
                        ),
                        ConstantDefinitionConfig(
                            name='minV',
                            type='Real',
                            value='A*ep',
                            metadata={}
                        )
                    ],
                    functions=[
                        FunctionDefinitionConfig(
                            name='waypointStartDist',
                            return_type='Real',
                            parameters=[
                                ('Real', 'xg')
                            ],
                            body='xg-GDelta',
                            metadata={}
                        ),
                        FunctionDefinitionConfig(
                            name='waypointEndDist',
                            return_type='Real',
                            parameters=[
                                ('Real', 'xg')
                            ],
                            body='xg+GDelta',
                            metadata={}
                        ),
                        FunctionDefinitionConfig(
                            name='maxTravelTime',
                            return_type='Real',
                            parameters=[
                                ('Real', 'g')
                            ],
                            body='waypointStartDist(g)/minV',
                            metadata={}
                        ),
                        FunctionDefinitionConfig(
                            name='stopTime',
                            return_type='Real',
                            parameters=[
                                ('Real', 'v')
                            ],
                            body='v/b',
                            metadata={}
                        ),
                        FunctionDefinitionConfig(
                            name='speedUpTime',
                            return_type='Real',
                            parameters=[
                                ('Real', 'v')
                            ],
                            body='ep-v/A',
                            metadata={}
                        ),
                        FunctionDefinitionConfig(
                            name='stopDist',
                            return_type='Real',
                            parameters=[
                                ('Real', 'v')
                            ],
                            body='v^2/(2*b)',
                            metadata={}
                        ),
                        FunctionDefinitionConfig(
                            name='accComp',
                            return_type='Real',
                            parameters=[
                                ('Real', 'v')
                            ],
                            body='( (A/b + 1)*(A/2*ep^2 + ep*v) )',
                            metadata={}
                        )
                    ],
                    predicates=[
                        PredicateDefinitionConfig(
                            name='bounds',
                            parameters=[],
                            body="""A > 0                                                                                                             
        & b > 0                                                                                                             
        & ep > 0                                                                                                            
        & Vmax >= 2*A*ep
        & GDelta > Vmax*ep + Vmax^2/(2*b)""",
                            metadata={}
                        ),
                        PredicateDefinitionConfig(
                            name='initialState',
                            parameters=[
                                ('Real', 'xr'),
                                ('Real', 'vr'),
                                ('Real', 'xg'),
                                ('Real', 'T')
                            ],
                            body="""vr = 0
        & xr < waypointStartDist(xg)
        & T > ep + maxTravelTime(xg-xr) + ep + stopTime(Vmax)""",
                            metadata={}
                        ),
                        PredicateDefinitionConfig(
                            name='assumptions',
                            parameters=[
                                ('Real', 'xr'),
                                ('Real', 'vr'),
                                ('Real', 'xg'),
                                ('Real', 'T')
                            ],
                            body='bounds() & initialState(xr, vr, xg, T)',
                            metadata={}
                        ),
                        PredicateDefinitionConfig(
                            name='loopinv',
                            parameters=[
                                ('Real', 'xr'),
                                ('Real', 'vr'),
                                ('Real', 'xg'),
                                ('Real', 'T')
                            ],
                            body="""0 <= vr & vr <= Vmax & xr + stopDist(vr) < waypointEndDist(xg)
                & (waypointStartDist(xg) < xr -> (vr = 0 | T >= stopTime(vr)))
                & (xr <= waypointStartDist(xg) ->
                       (vr >= minV & T > maxTravelTime(xg-xr) + ep + stopTime(Vmax))
                     | (vr <= minV & T > speedUpTime(vr) + maxTravelTime(xg-xr) + ep + stopTime(Vmax)))""",
                            metadata={}
                        ),
                        PredicateDefinitionConfig(
                            name='ensure',
                            parameters=[
                                ('Real', 'xr'),
                                ('Real', 'vr'),
                                ('Real', 'xg'),
                                ('Real', 'T')
                            ],
                            body='(waypointStartDist(xg) < xr & xr < waypointEndDist(xg) & vr = 0 & T >= 0)',
                            metadata={
                                'type': 'progress',
                                'reward': '500',
                                'penalty': '-1000',
                                'on_success': 'continue:status=success',
                                'on_failure': 'terminate:status=failure'
                            }
                        )
                    ],
                    hybrid_programs=[
                        ControlFuncConfig(
                            name='controlFunc',
                            action_space_config=None, # This will be replaced programmatically
                            metadata={}
                        ),
                        StepFuncConfig(
                            name='stepFunc',
                            odes=[
                                ODEConfig(
                                    variable='{xr',
                                    equation="vr, vr' = ar, t' = 1, T'=-1 & t <= ep & vr >= 0}"
                                )
                            ],
                            evolution_domain=None,
                            duration=None,
                            raw_body_snippet="{ {xr' = vr, vr' = ar, t' = 1, T'=-1 & t <= ep & vr >= 0} }",
                            metadata={}
                        )
                    ]
                ),
                variables=VariablesConfig(
                    state_variables=[
                        VariableDeclarationConfig(
                            name='xr',
                            type='Real',
                            lower_bound=None,
                            upper_bound=None
                        ),
                        VariableDeclarationConfig(
                            name='vr',
                            type='Real',
                            lower_bound=None,
                            upper_bound=None
                        ),
                        VariableDeclarationConfig(
                            name='ar',
                            type='Real',
                            lower_bound=None,
                            upper_bound=None
                        ),
                        VariableDeclarationConfig(
                            name='xg',
                            type='Real',
                            lower_bound=None,
                            upper_bound=None
                        ),
                        VariableDeclarationConfig(
                            name='t',
                            type='Real',
                            lower_bound=None,
                            upper_bound=None
                        ),
                        VariableDeclarationConfig(
                            name='T',
                            type='Real',
                            lower_bound=None,
                            upper_bound=None
                        )
                    ],
                    auxiliary_variables=[]
                ),
                problem=ProblemConfig(
                    precondition=PreconditionConfig(
                        formula='assumptions(xr, vr, xg, T)'
                    ),
                    loop_execution=LoopExecutionConfig(
                        iterations='inf',
                        loop_invariant_formula='loopinv(xr, vr, xg, T)',
                        loop_body=SequenceActionConfig(
                            actions=[]
                        )
                    ),
                    postcondition=PostconditionConfig(
                        formula='ensure(xr, vr, xg, T)'
                    )
                )
            )
        ''').strip()

        # 1. Setup AST Classes for deserialization (re-using from base class init)
        # These classes are defined as dummies above for standalone execution.
        ast_classes_for_eval = {
            cls.__name__: cls for cls in [
                GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig,
                ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig,
                HybridProgramConfig, ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig,
                VariableDeclarationConfig, PreconditionConfig, PostconditionConfig,
                LoopExecutionConfig, HybridProgramStepConfig,
                ActionType, ChoiceActionConfig, SequenceActionConfig,
                DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig,
                ActionConstraintConfig, CallHybridProgramConfig
            ]
        }
        ast_classes_for_eval.update({
            'list': list, 'dict': dict, 'tuple': tuple, 'None': None, 'True': True, 'False': False
        })

        try:
            # Parse the original config string to an AST object
            config_obj: GymEnvironmentConfig = eval(original_config_template, {'__builtins__': None}, ast_classes_for_eval)
        except Exception as e:
            raise ValueError(f"Failed to deserialize original GymEnvironmentConfig template: {e}")

        # Modify the ControlFuncConfig within the AST object
        found_control_func = False
        for i, hp_config in enumerate(config_obj.definitions.hybrid_programs):
            if isinstance(hp_config, ControlFuncConfig) and hp_config.name == 'controlFunc':
                hp_config.action_space_config = control_func_action_space_config
                config_obj.definitions.hybrid_programs[i] = hp_config # Update the list
                found_control_func = True
                break
        
        if not found_control_func:
            raise ValueError("ControlFuncConfig with name 'controlFunc' not found in the provided config template.")

        # Serialize the modified config object back to a string for the base class
        modified_config_serialized_str = repr(config_obj)

        # Call the base class constructor with the modified config string
        super().__init__(config_serialized_str=modified_config_serialized_str, render_mode=render_mode, cisb_size=cisb_size, cisb_populate_attempts=cisb_populate_attempts, seed=seed, debug_mode=debug_mode)

        # After super().__init__ has parsed the config and set self.config
        self._control_func_action_config = None
        for hp_config in self.config.definitions.hybrid_programs:
            if isinstance(hp_config, ControlFuncConfig) and hp_config.name == 'controlFunc':
                self._control_func_action_config = hp_config.action_space_config
                break

        if self._control_func_action_config is None:
            raise ValueError("ControlFuncConfig with name 'controlFunc' not found in definitions.hybrid_programs after super().__init__.")

        # Now, construct the gymnasium action space based on _control_func_action_config
        self.action_space = self._build_action_space_from_config(self._control_func_action_config)

        # Initialize constants, functions, predicates (these lines were already in your code)
        # Note: These values will be overwritten by CISB during reset, but are needed for initial evaluation
        # of expressions like 'minV' and action space bounds.
        self.constants['ep'] = 0.2 # Default for expression evaluation if not in CISB yet
        self.constants['b'] = 1.0
        self.constants['A'] = 5.0
        self.constants['GDelta'] = 1.0
        self.constants['Vmax'] = 20.0
        # Re-evaluate minV as it depends on A and ep. Use _evaluate_expression for this.
        self.constants['minV'] = self._evaluate_expression('A*ep')

        self.functions['waypointStartDist'] = lambda xg: ((xg) - (self.constants['GDelta']))
        self.functions['waypointEndDist'] = lambda xg: ((xg) + (self.constants['GDelta']))
        self.functions['maxTravelTime'] = lambda g: (safe_div(self.functions['waypointStartDist'](g), self.constants['minV'])) # Use safe_div
        self.functions['stopTime'] = lambda v: (safe_div(v, self.constants['b'])) # Use safe_div
        self.functions['speedUpTime'] = lambda v: ((self.constants['ep']) - (safe_div(v, self.constants['A']))) # Use safe_div
        self.functions['stopDist'] = lambda v: (safe_div((v ** 2), (2 * self.constants['b']))) # Use safe_div
        self.functions['accComp'] = lambda v: (((safe_div(self.constants['A'], self.constants['b'])) + 1) * (((safe_div(self.constants['A'], 2)) * (self.constants['ep'] ** 2)) + (self.constants['ep'] * v))) # Use safe_div
        
        self.predicates['bounds'] = lambda : ((((((((((self.constants['A']) > (0))) and (((self.constants['b']) > (0))))) and (((self.constants['ep']) > (0))))) and (((self.constants['Vmax']) >= (((((2) * (self.constants['A']))) * (self.constants['ep']))))))) and (((self.constants['GDelta']) > (((((self.constants['Vmax']) * (self.constants['ep']))) + (((((self.constants['Vmax']) ** (2))) / (((2) * (self.constants['b']))))))))))
        self.predicates['initialState'] = lambda xr, vr, xg, T: ((((self._approx_eq(vr, 0)) and (((xr) < (self.functions['waypointStartDist'](xg)))))) and (((T) > (((((((self.constants['ep']) + (self.functions['maxTravelTime'](((xg) - (xr)))))) + (self.constants['ep']))) + (self.functions['stopTime'](self.constants['Vmax'])))))))
        self.predicates['assumptions'] = lambda xr, vr, xg, T: ((self.predicates['bounds']()) and (self.predicates['initialState'](xr, vr, xg, T)))
        self.predicates['loopinv'] = lambda xr, vr, xg, T: ((((((((((0) <= (vr))) and (((vr) <= (self.constants['Vmax']))))) and (((((xr) + (self.functions['stopDist'](vr)))) < (self.functions['waypointEndDist'](xg)))))) and (((not (((self.functions['waypointStartDist'](xg)) < (xr)))) or (((self._approx_eq(vr, 0)) or (((T) >= (self.functions['stopTime'](vr)))))))))) and (((not (((xr) <= (self.functions['waypointStartDist'](xg))))) or (((((((vr) >= (self.constants['minV']))) and (((T) > (((((self.functions['maxTravelTime'](((xg) - (xr)))) + (self.constants['ep']))) + (self.functions['stopTime'](self.constants['Vmax'])))))))) or (((((vr) <= (self.constants['minV']))) and (((T) > (((((((self.functions['speedUpTime'](vr)) + (self.functions['maxTravelTime'](((xg) - (xr)))))) + (self.constants['ep']))) + (self.functions['stopTime'](self.constants['Vmax'])))))))))))))
        self.predicates['ensure'] = lambda xr, vr, xg, T: ((((((((self.functions['waypointStartDist'](xg)) < (xr))) and (((xr) < (self.functions['waypointEndDist'](xg)))))) and (self._approx_eq(vr, 0)))) and (((T) >= (0))))

        self._state_keys = ['xr', 'vr', 'ar', 'xg', 't', 'T'] # Ensure this matches the config variables
        self.state = {key: 0.0 for key in self._state_keys}

        low_bounds_obs = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds_obs = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds_obs, high_bounds_obs, shape=(len(self._state_keys),), dtype=np.float32)

        self._action_space_keys = [] # This list might be populated if Dict space is used for other things. For now, it's fine.

        self._ode_state_map = {'xr': 0, 'vr': 1, 'ar': 2, 'xg': 3, 't': 4, 'T': 5}
        self._ode_state_idx_map = {0: 'xr', 1: 'vr', 2: 'ar', 3: 'xg', 4: 't', 5: 'T'}

        _ode_func_str_local = '''def ode_func(t_ode, y_array):
    # Extract state variables from y_array
    xr = y_array[0]
    vr = y_array[1]
    ar = y_array[2] # Note: ar is constant during ODE integration, set by agent action
    xg = y_array[3]
    t = y_array[4]
    T = y_array[5]

    # Initialize derivatives for all state variables.
    # For 'ar' and 'xg', their derivatives are 0 as they are constant during evolution.
    dydt = [
        vr, # dxr/dt = vr
        self.state['ar'], # dvr/dt = ar (from environment state, set by agent)
        0.0, # dar/dt = 0 (ar is constant during ODE step)
        0.0, # dxg/dt = 0 (xg is constant during ODE step)
        1.0, # dt/dt = 1
        -1.0, # dT/dt = -1
    ]
    return dydt'''
        _ode_events_str_local = '''def event_t_reaches_ep(t_ode, y_array):
    _t = y_array[4]
    _ep = self.constants['ep']
    return _t - _ep
event_t_reaches_ep.terminal = True
event_t_reaches_ep.direction = 1 # Trigger when (_t - _ep) crosses 0 from negative to positive

def event_vr_non_negative(t_ode, y_array):
    _vr = y_array[1]
    _ar = self.state['ar'] # Get 'ar' from environment's state, as it's constant during ODE step
    # If vr is approximately zero and ar is positive, it means the robot is trying to accelerate from rest.
    # We return a positive value to prevent the event from triggering.
    if self._approx_eq(_vr, 0.0) and _ar > 0:
        return 1.0 # Do not trigger event if vr is 0 and ar > 0 (accelerating from rest)
    return _vr
event_vr_non_negative.terminal = True
event_vr_non_negative.direction = -1 # Trigger when _vr crosses 0 from positive to negative

ode_events = [
    event_t_reaches_ep,
    event_vr_non_negative,
]'''
        self._ode_func_compiled = self._compile_ode_func(_ode_func_str_local)
        self._ode_events_compiled = self._compile_ode_events(_ode_events_str_local)

        # Define sample ranges for CISB population.
        # These include constants and state variables that need initial sampling.
        self.sample_ranges = {
                    'xr': (-10.0, 0.0), # Initial position before waypoint
                    'vr': (0.0, 0.0),   # Initial velocity is 0
                    'ar': (0.0, 0.0),   # Initial acceleration is 0, set by agent later
                    'xg': (5.0, 15.0),  # Goal position
                    't': (0.0, 0.0),    # Time starts at 0
                    'T': (10.0, 50.0),  # Remaining time
                    'ep': (0.1, 0.5),   # Time limit for control decisions
                    'b': (0.5, 5.0),    # Minimum braking capability
                    'A': (1.0, 10.0),   # Maximum acceleration
                    'GDelta': (1.0, 10.0), # Goal area size
                    'Vmax': (5.0, 30.0), # Maximum robot velocity
                    'minV': (0.0, 10.0), # Derived, so a reasonable range for Z3 to work with
                }

        self._populate_cisb_diverse_z3()

        self._force_acceleration_action = False # This was a debug flag, ensure it's False

        # Call reset at the end of init to set initial state and constants
        self.reset()

    def _build_action_space_from_config(self, action_config_obj: Any) -> spaces.Space:
        """
        Dynamically builds a gymnasium.spaces.Dict from the parsed action_space_config.
        """
        action_dict_spaces = {}
        if isinstance(action_config_obj, SequenceActionConfig):
            for sub_action in action_config_obj.actions:
                if isinstance(sub_action, NondeterministicAssignmentActionConfig):
                    var_name = sub_action.variable
                    lower_bound_expr = sub_action.lower_bound if sub_action.lower_bound is not None else '-inf'
                    upper_bound_expr = sub_action.upper_bound if sub_action.upper_bound is not None else 'inf'

                    # Evaluate bounds in the context of the environment's constants
                    # self.constants must be populated at this point from super().__init__
                    try:
                        low = float(self._evaluate_expression(lower_bound_expr))
                    except Exception as e:
                        if self.debug_mode:
                            print(f"DEBUG_ACTION_SPACE: Could not evaluate lower bound '{lower_bound_expr}': {e}. Using -np.inf.")
                        low = -np.inf
                    try:
                        high = float(self._evaluate_expression(upper_bound_expr))
                    except Exception as e:
                        if self.debug_mode:
                            print(f"DEBUG_ACTION_SPACE: Could not evaluate upper bound '{upper_bound_expr}': {e}. Using np.inf.")
                        high = np.inf

                    action_dict_spaces[f'assign_{var_name}'] = spaces.Box(low=low, high=high, shape=(), dtype=np.float32)
                elif isinstance(sub_action, ChoiceActionConfig):
                    # For a choice action within a sequence, we'd need to create a discrete space for the choice itself
                    # and then potentially nested spaces for parameters of the chosen option.
                    # This is more complex and not directly needed for the 'ar' assignment.
                    if self.debug_mode:
                        print(f"DEBUG_ACTION_SPACE: Warning: Nested ChoiceActionConfig within SequenceActionConfig detected. This is not fully supported for automatic space generation.")
                elif isinstance(sub_action, ActionConstraintConfig):
                    # Action constraints don't define action space components directly
                    pass
                elif isinstance(sub_action, DeterministicAssignmentActionConfig):
                    # Deterministic assignments don't define action space components directly
                    pass
                else:
                    if self.debug_mode:
                        print(f"DEBUG_ACTION_SPACE: Warning: Unhandled action type in SequenceActionConfig for action space generation: {type(sub_action)}")

        elif isinstance(action_config_obj, ChoiceActionConfig):
            # If the top-level is a Choice, the agent needs to select an index.
            # This would typically be a Discrete space for the choice itself.
            action_dict_spaces[f'choice_{action_config_obj.level_name}'] = spaces.Discrete(len(action_config_obj.choices))
            if self.debug_mode:
                print(f"DEBUG_ACTION_SPACE: Warning: Top-level ChoiceActionConfig detected. Only discrete choice index is currently exposed. Nested parameters are not automatically added to action space.")
        else:
            if self.debug_mode:
                print(f"DEBUG_ACTION_SPACE: Warning: Unsupported action_space_config type: {type(action_config_obj)}. Creating empty action space.")
            return spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32) # Empty action space

        if not action_dict_spaces:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32) # Empty action space
        return spaces.Dict(action_dict_spaces)

    def reset(self, seed=None, options=None):
        # Call the base class reset
        observation, info = super().reset(seed=seed, options=options)
        # Re-evaluate minV after constants might have been updated by CISB in super().reset()
        self.constants['minV'] = self._evaluate_expression('A*ep')
        return observation, info

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        # Ensure minV is up-to-date before evaluating loop invariants
        self.constants['minV'] = self._evaluate_expression('A*ep')

        loop_invariant_formula_kx = self._problem_config.loop_execution.loop_invariant_formula
        expanded_loop_inv = self._expand_formula(loop_invariant_formula_kx, self._definitions_config)
        loop_inv_conjuncts = self._split_top_level_and_local(str(expanded_loop_inv)) # Ensure it's a string

        loop_invariant_satisfied_pre_step = True
        for conj_str in loop_inv_conjuncts:
            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', conj_str)
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {}

            if not conjunct_result:
                loop_invariant_satisfied_pre_step = False
                if self.debug_mode:
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (pre-control): '{conj_str}'")
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                    if terminated or truncated:
                        state_record_before_early_return = self.state.copy()
                        state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                        self.state_history.append(state_record_before_early_return)
                        return self._get_obs(), reward, terminated, truncated, info

        if not loop_invariant_satisfied_pre_step:
            if self.debug_mode:
                print(f"DEBUG_STEP: General loop invariant FAILED (pre-control).")
            reward, terminated, truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-500.0', 'on_failure': 'terminate:status=failure'},
                reward, terminated, truncated, is_failure_case=True
            )
            state_record_before_early_return = self.state.copy()
            state_record_before_early_return['episode_time'] = self.cumulative_episode_time
            self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info


        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: State before control action: xr={self.state['xr']:.6f}, vr={self.state['vr']:.6f}, ar={self.state['ar']:.6f}, xg={self.state['xg']:.6f}, T={self.state['T']:.6f}, ep={self.constants['ep']:.6f}")
        control_reward, control_terminated, control_applied = self._apply_control_action(action)
        reward += control_reward
        terminated = terminated or control_terminated
        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: State after _apply_control_action: xr={self.state['xr']:.6f}, vr={self.state['vr']:.6f}, ar={self.state['ar']:.6f}, xg={self.state['xg']:.6f}, T={self.state['T']:.6f}, control_applied={control_applied}")

        if not control_applied:
            if self.debug_mode:
                print(f"DEBUG_STEP: Control action not applied. Terminating early.")
            state_record_before_early_return = self.state.copy()
            state_record_before_early_return['episode_time'] = self.cumulative_episode_time
            self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info

        y0 = np.array([self.state[var_name] for var_name in self._ode_state_map.keys()], dtype=np.float64)

        ep_val = self.constants['ep']
        t_span = (0, ep_val)

        ode_terminated_by_event = False
        solution = None # Initialize solution outside try-except

        if self.debug_mode:
            print(f"DEBUG_ODE: Before solve_ivp: y0={y0}, t_span={t_span}, ep_val={ep_val:.6f}")

        if ep_val <= 0:
            self.state['t'] = 0.0
            if self.debug_mode:
                print(f"DEBUG_ODE: Warning: 'ep' constant is non-positive ({ep_val:.6f}). Skipping ODE integration and penalizing.")
            bounds_meta = self._get_predicate_metadata('bounds')
            reward, terminated, truncated = self._apply_meta_effect(
                bounds_meta or {'type': 'safety', 'penalty': '-200.0', 'on_failure': 'terminate:status=failure'},
                reward, terminated, truncated, is_failure_case=True
            )
            ode_terminated_by_event = True
        else:
            try:
                solution = solve_ivp(
                    self._ode_func_compiled,
                    t_span,
                    y0,
                    events=self._ode_events_compiled,
                    dense_output=True,
                    rtol=1e-6, atol=1e-8
                )

                if self.debug_mode:
                    print(f"DEBUG_ODE: After solve_ivp: solution.status={solution.status}, solution.message={solution.message}")
                    print(f"DEBUG_ODE: solution.t={solution.t}")
                    if solution.y.shape[1] > 0:
                        print(f"DEBUG_ODE: solution.y_initial={solution.y[:,0]}, solution.y_final={solution.y[:,-1]}")
                    if solution.t_events:
                        print(f"DEBUG_ODE: solution.t_events={solution.t_events}")

                if solution.success and solution.sol is not None:
                    actual_ode_duration = solution.t[-1] if solution.t.size > 0 else 0.0

                    if actual_ode_duration > 1e-9:
                        num_ode_points = max(20, int(actual_ode_duration * 100))
                        t_eval = np.linspace(0, actual_ode_duration, num_ode_points)

                        ode_solution_points = solution.sol(t_eval)

                        start_idx = 1 if len(self.state_history) > 0 and \
                                         np.isclose(self.state_history[-1]['episode_time'], self.cumulative_episode_time) and \
                                         np.isclose(t_eval[0], 0.0) else 0

                        for k in range(start_idx, ode_solution_points.shape[1]):
                            current_ode_state = {}
                            for i, var_name in enumerate(self._ode_state_map.keys()):
                                current_ode_state[var_name] = ode_solution_points[i, k]

                            intermediate_episode_time = self.cumulative_episode_time + current_ode_state['t']
                            current_ode_state['episode_time'] = intermediate_episode_time
                            self.state_history.append(current_ode_state)
                            if self.debug_mode and k % (ode_solution_points.shape[1] // 5 + 1) == 0:
                                print(f"DEBUG_ODE_APPEND: Appended intermediate state k={k}: xr={current_ode_state['xr']:.2f}, vr={current_ode_state['vr']:.2f}, ar={current_ode_state['ar']:.2f}, T={current_ode_state['T']:.2f}, ep_time={intermediate_episode_time:.2f}")

                if solution.status == 0: # Integration completed successfully
                    final_y = solution.y[:, -1]
                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]
                elif solution.status == 1: # Integration terminated by an event
                    if solution.y_events and len(solution.y_events) > 0 and solution.y_events[0].size > 0:
                        # Event 0 is event_t_reaches_ep, Event 1 is event_vr_non_negative
                        # Check which event triggered
                        if solution.t_events[0] and solution.t_events[0].size > 0 and np.isclose(solution.t_events[0][0], solution.t[-1]):
                            # event_t_reaches_ep triggered, normal termination
                            final_y = solution.y[:, -1]
                            for i, var_name in enumerate(self._ode_state_map.keys()):
                                self.state[var_name] = final_y[i]
                        elif solution.t_events[1] and solution.t_events[1].size > 0 and np.isclose(solution.t_events[1][0], solution.t[-1]):
                            # event_vr_non_negative triggered (vr became negative)
                            final_y = solution.y_events[1][0] # Use the state at the event
                            for i, var_name in enumerate(self._ode_state_map.keys()):
                                self.state[var_name] = final_y[i]
                            if self.state.get('vr', 0.0) < -self.equality_tolerance:
                                if self.debug_mode:
                                    print(f"DEBUG_ODE: Event: vr became negative ({self.state['vr']:.6f}).")
                                # You might need to define a 'isPhysicallyValid' predicate in your config if you want to use it
                                phys_valid_meta = self._get_predicate_metadata('isPhysicallyValid')
                                reward, terminated, truncated = self._apply_meta_effect(
                                    phys_valid_meta or {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'truncate:status=failure'},
                                    reward, terminated, truncated, is_failure_case=True
                                )
                                ode_terminated_by_event = True
                        else: # Fallback if event type is unclear
                            final_y = solution.y[:, -1]
                            for i, var_name in enumerate(self._ode_state_map.keys()):
                                self.state[var_name] = final_y[i]
                    else: # No event data, but status is 1
                        final_y = solution.y[:, -1]
                        for i, var_name in enumerate(self._ode_state_map.keys()):
                            self.state[var_name] = final_y[i]

                else: # Integration failed
                    if self.debug_mode:
                        print(f"DEBUG_ODE: ODE integration failed with status {solution.status}.")
                    reward, terminated, truncated = self._apply_meta_effect(
                        {'type': 'safety', 'penalty': '-150.0', 'on_failure': 'terminate:status=failure'},
                        reward, terminated, truncated, is_failure_case=True
                    )
                    ode_terminated_by_event = True

            except Exception as e:
                if self.debug_mode:
                    print(f"DEBUG_ODE: Exception during ODE integration: {e}")
                reward, terminated, truncated = self._apply_meta_effect(
                    {'type': 'safety', 'penalty': '-200.0', 'on_failure': 'terminate:status=failure'},
                    reward, terminated, truncated, is_failure_case=True
                )
                ode_terminated_by_event = True

        terminated = terminated or ode_terminated_by_event
        truncated = truncated or ode_terminated_by_event

        # Only update cumulative_episode_time if ODE integration was successful and produced a duration
        if solution and solution.success and solution.t.size > 0:
            self.cumulative_episode_time += solution.t[-1]
        else:
            # If ODE failed or didn't run, 't' in state might be 0, so add 'ep_val' as a fallback for time progression
            # This ensures that time progresses even if ODE fails early, preventing infinite loops in some cases.
            if not terminated and not truncated: # Only if not already terminated/truncated by ODE failure
                self.cumulative_episode_time += ep_val

        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: Updated cumulative_episode_time={self.cumulative_episode_time:.6f}")
            print(f"DEBUG_STEP {self.current_step}: State after ODE integration: xr={self.state['xr']:.6f}, vr={self.state['vr']:.6f}, ar={self.state['ar']:.6f}, xg={self.state['xg']:.6f}, T={self.state['T']:.6f}, t={self.state['t']:.6f}")


        if terminated or truncated:
            if self.debug_mode:
                print(f"DEBUG_STEP: Terminated/Truncated after ODE. Returning early.")
            # Ensure the final state is recorded if it wasn't already by intermediate steps
            if not (solution and solution.success and solution.sol is not None and solution.t.size > 1 and (solution.t[-1] > 1e-9)):
                state_record_before_early_return = self.state.copy()
                state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info

        loop_invariant_satisfied_post_ode = True
        for conj_str in loop_inv_conjuncts:
            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', conj_str)
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {}

            if not conjunct_result:
                loop_invariant_satisfied_post_ode = False
                if self.debug_mode:
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (post-ODE): '{conj_str}'")
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': 'terminate:status=failure'},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                if metadata.get('on_success') and metadata.get('on_success').split(':')[0] != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
                    if terminated or truncated:
                        state_record_before_early_return = self.state.copy()
                        state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                        self.state_history.append(state_record_before_early_return)
                        return self._get_obs(), reward, terminated, truncated, info

        if not loop_invariant_satisfied_post_ode:
            if self.debug_mode:
                print(f"DEBUG_STEP: General loop invariant FAILED (post-ODE).")
            reward, terminated, truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-300.0', 'on_failure': 'terminate:status=failure'},
                reward, terminated, truncated, is_failure_case=True
            )
            if terminated or truncated:
                state_record_before_early_return = self.state.copy()
                state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                self.state_history.append(state_record_before_early_return)
                return self._get_obs(), reward, terminated, truncated, info

        # These predicates (isWellformedDir, isEfficient) are not defined in the provided config.
        # If they were, their metadata would be retrieved and applied.
        # For now, this part will not add reward unless such predicates are added to the config.
        # wellformed_meta = self._get_predicate_metadata('isWellformedDir')
        # if wellformed_meta.get('type') == 'progress' and 'shape' in wellformed_meta:
        #     shape_value = self._evaluate_shape_expression(wellformed_meta['shape'])
        #     reward += shape_value
        #     if self.debug_mode:
        #         print(f"DEBUG_STEP: Applied isWellformedDir shape reward: {shape_value:.6f}")

        # efficient_meta = self._get_predicate_metadata('isEfficient')
        # if efficient_meta.get('type') == 'progress' and 'shape' in efficient_meta:
        #     shape_value = self._evaluate_shape_expression(efficient_meta['shape'])
        #     reward += shape_value
        #     if self.debug_mode:
        #         print(f"DEBUG_STEP: Applied isEfficient shape reward: {shape_value:.6f}")


        postcondition_formula_kx = self._problem_config.postcondition.formula
        postcondition_satisfied = self._evaluate_expression(postcondition_formula_kx, is_constraint=True)
        ensure_meta = self._get_predicate_metadata('ensure')

        if postcondition_satisfied:
            if self.debug_mode:
                print(f"DEBUG_STEP: Postcondition '{postcondition_formula_kx}' SATISFIED.")
            reward, terminated, truncated = self._apply_meta_effect(
                ensure_meta, reward, terminated, truncated, is_failure_case=False
            )
        else:
            if self.debug_mode:
                print(f"DEBUG_STEP: Postcondition '{postcondition_formula_kx}' NOT SATISFIED.")
            reward, terminated, truncated = self._apply_meta_effect(
                ensure_meta, reward, terminated, truncated, is_failure_case=True
            )

            if ensure_meta.get('type') == 'progress' and 'shape' in ensure_meta:
                shape_value = self._evaluate_shape_expression(ensure_meta['shape'])
                reward += shape_value
                if self.debug_mode:
                    print(f"DEBUG_STEP: Applied ensure shape reward: {shape_value:.6f}")

        if self.current_step >= self.max_episode_steps:
            truncated = True
            if self.debug_mode:
                print(f"DEBUG_STEP: Max episode steps ({self.max_episode_steps}) reached. Truncating.")
            if not terminated and not postcondition_satisfied:
                reward -= 10.0

        # Ensure the final state is recorded if it wasn't already by intermediate steps
        if not (solution and solution.success and solution.sol is not None and solution.t.size > 1 and (solution.t[-1] > 1e-9)):
            state_record_after_step = self.state.copy()
            state_record_after_step['episode_time'] = self.cumulative_episode_time
            self.state_history.append(state_record_after_step)
            if self.debug_mode:
                print(f"DEBUG_STEP: Appended final state for step {self.current_step} to history.")

        observation = self._get_obs()
        info = self._get_info()
        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: Step complete. Reward={reward:.6f}, Terminated={terminated}, Truncated={truncated}")
        return observation, reward, terminated, truncated, info