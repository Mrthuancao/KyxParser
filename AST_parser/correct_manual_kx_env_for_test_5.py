import sys
import os
from typing import Optional, Dict, Any, Tuple, List, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import re
import math
import z3

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Adjust project_root to point to 'd:\Reinforcement learning'
# Assuming manual_kx_env_for_test_4.py is in d:\Reinforcement learning\kyx_parser\AST_parser\
# So, two levels up from current_script_dir
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add project_root and current_script_dir to sys.path to ensure all modules are found
# This makes pyast_config discoverable if it's in either of these locations.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

# Import the base environment class (now discoverable via sys.path)
from gym_generator.kx_env_base import _KeYmaeraXEnvBase
# NEW: Import safe_div directly into this module's scope
from gym_generator.kx_env_base import safe_div

# Import necessary config classes (assuming pyast_config is also discoverable)
from pyast_config import (
    GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig,
    ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig,
    HybridProgramConfig, ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig,
    VariableDeclarationConfig, PreconditionConfig, PostconditionConfig,
    LoopExecutionConfig, HybridProgramStepConfig,
    ActionType, ChoiceActionConfig, SequenceActionConfig,
    DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig,
    ActionConstraintConfig, CallHybridProgramConfig
)

from gym_generator.expression_helpers import expand_formula

# this is filed AST_parser/manual_kx_env_for_test_5.py

class GeneratedKXEnv(_KeYmaeraXEnvBase):
    metadata = {"render_modes": ["human", "plot_data", "matplotlib"], "render_fps": 30}

    def __init__(self, render_mode=None, cisb_size: int = 2000, cisb_populate_attempts: int = 4000, seed: Optional[int] = None, debug_mode: bool = False):
        # Store the serialized config string locally. This string is passed to the base class.
        self._config_serialized_str = '''GymEnvironmentConfig(
            env_name='IJRR17_Theorem_15__Cross_intersection_with_deadline',
            definitions=DefinitionsConfig(
                constants=[
                    ConstantDefinitionConfig(name='ep', type='Real', value=None, metadata={'type': 'const', 'bound': '0.01:0.1'}),
                    ConstantDefinitionConfig(name='b', type='Real', value=None, metadata={'type': 'const', 'bound': '0.5:5.0'}),
                    ConstantDefinitionConfig(name='A', type='Real', value=None, metadata={'type': 'const', 'bound': '1.0:10.0'}),
                    ConstantDefinitionConfig(name='Vmin', type='Real', value=None, metadata={'type': 'const', 'bound': '0.1:2.0'}),
                    ConstantDefinitionConfig(name='ixr', type='Real', value='0'), # Fixed value
                    ConstantDefinitionConfig(name='ixo', type='Real', value='0'), # Fixed value
                    ConstantDefinitionConfig(name='D', type='Real', value=None, metadata={'type': 'const', 'bound': '1.0:10.0'})
                ],
                functions=[
                    FunctionDefinitionConfig(name='minV', return_type='Real', parameters=[], body='A*ep'),
                    FunctionDefinitionConfig(name='stopDist', return_type='Real', parameters=[('Real', 'v')], body='v^2/(2*b)'),
                    FunctionDefinitionConfig(name='accComp', return_type='Real', parameters=[('Real', 'v')], body='((A/b + 1)*(A/2*ep^2 + ep*v))'),
                    FunctionDefinitionConfig(name='OAfterX', return_type='Bool', parameters=[('Real', 'xo')], body='xo>ixo'),
                    FunctionDefinitionConfig(name='RAfterX', return_type='Bool', parameters=[('Real', 'xr')], body='xr>ixr')
                ],
                predicates=[
                    PredicateDefinitionConfig(name='bounds', parameters=[], body=("A > 0 & b > 0 & ep > 0 & "
                                                                                   "Vmin > 0 & D >= ep")),
                    PredicateDefinitionConfig(name='initialState', parameters=[('Real', 'xr'), ('Real', 'vr'), ('Real', 'ixr'), ('Real', 'xo'), ('Real', 'vo'), ('Real', 'ixo'), ('Real', 'T')],
                                              body=("vr = 0 & vo>=Vmin & xr < ixr & "
                                                    "xr + A/2*(D - ep)^2 > ixr & " # Corrected from ixr-xr < ... to match formal model
                                                    "T = D")), # FIXED: T = D as per formal model
                    PredicateDefinitionConfig(name='assumptions', parameters=[('Real', 'xr'), ('Real', 'vr'), ('Real', 'ixr'), ('Real', 'xo'), ('Real', 'vo'), ('Real', 'ixo'), ('Real', 'T')],
                                              body='bounds() & initialState(xr, vr, ixr, xo, vo, ixo, T)'),
                    PredicateDefinitionConfig(
                        name='safetyloopinv',
                        parameters=[('Real', 'xr'), ('Real', 'vr'), ('Real', 'ixr'), ('Real', 'xo'), ('Real', 'vo'), ('Real', 'ixo')],
                        body=("0 <= vr & Vmin<=vo & (RAfterX(xr) | OAfterX(xo) | xr + stopDist(vr) < ixr | "
                              "(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/vr)))"),
                        metadata={'type': 'safety', 'penalty': '-1000', 'on_failure': {'action': 'truncate', 'status': 'failure'}, 'on_success': {'action': 'continue', 'status': 'none'}}
                    ),
                    PredicateDefinitionConfig(
                        name='deadlineloopinv',
                        parameters=[('Real', 'xr'), ('Real', 'vr'), ('Real', 'ixr'), ('Real', 'T')], # FIXED: Removed xo, vo, ixo
                        body=("RAfterX(xr + vr * (if (T > 0) then T else 0) + A/2 * ((if (T > 0) then T else 0)^2))"), # FIXED: Simplified to match formal model, using if-then-else for max(0,T)
                        metadata={'type': 'progress', 'reward': '10', 'on_success': {'action': 'continue', 'status': 'none'}, 'on_failure': {'action': 'continue', 'status': 'none'}, 'shape': '-T'}
                    ),
                    PredicateDefinitionConfig(
                        name='finalSafety',
                        parameters=[('Real', 'xr'), ('Real', 'xo'), ('Real', 'T')],
                        body="( (xr=ixr -> xo!=ixo) & (T<=0 -> RAfterX(xr)) )", # FIXED: T<=0 as per formal model
                        metadata={'type': 'safety', 'penalty': '-1000', 'on_failure': {'action': 'terminate', 'status': 'failure'}, 'on_success': {'action': 'continue', 'status': 'none'}}
                    ),
                    PredicateDefinitionConfig(
                        name='ensure',
                        parameters=[('Real', 'xr')],
                        body='RAfterX(xr)',
                        metadata={'type': 'progress', 'reward': '500', 'on_success': {'action': 'terminate', 'status': 'success'}, 'on_failure': {'action': 'terminate', 'status': 'failure'}, 'shape': '(xr - ixr) / 10.0'}
                    )
                ],
                hybrid_programs=[
                    ControlFuncConfig(
                        name='controlFunc',
                        action_space_config=ChoiceActionConfig(
                            level_name='ChoiceLevel_0',
                            choices=[
                                # Choice 0: If obstacle passed or robot passed, robot can go full speed (ar := A)
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(formula='OAfterX(xo) | RAfterX(xr)'),
                                    DeterministicAssignmentActionConfig(variable='ar', value='A')
                                ]),
                                # Choice 1: If intersection is ahead for both AND it's safe to accelerate (ar := A)
                                # Guard: !(OAfterX(xo) | RAfterX(xr)) AND (vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/vr ))
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(
                                        formula=('!(OAfterX(xo) | RAfterX(xr)) & '
                                                 '(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | '
                                                 'ixo < xo + Vmin*(ixr-xr)/vr ))') # Note: Formal model uses /vr here, not /(vr+A*ep)
                                    ),
                                    DeterministicAssignmentActionConfig(variable='ar', value='A') # FIXED: Deterministic ar
                                ]),
                                # Choice 2: If intersection is ahead for both AND not safe to accelerate AND vr = 0 (ar := 0)
                                # Guard: !(OAfterX(xo) | RAfterX(xr)) AND !(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & vr = 0
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(
                                        formula=('!(OAfterX(xo) | RAfterX(xr)) & '
                                                 '!(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                                 'vr = 0')
                                    ),
                                    DeterministicAssignmentActionConfig(variable='ar', value='0') # FIXED: Deterministic ar
                                ]),
                                # Choice 3: If intersection is ahead for both AND not safe to accelerate AND vr != 0 (ar := -b)
                                # Guard: !(OAfterX(xo) | RAfterX(xr)) AND !(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & vr != 0
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(
                                        formula=('!(OAfterX(xo) | RAfterX(xr)) & '
                                                 '!(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                                 'vr != 0')
                                    ),
                                    DeterministicAssignmentActionConfig(variable='ar', value='-b') # FIXED: Deterministic ar
                                ])
                            ]
                        )
                    ),
                    StepFuncConfig(
                        name='stepFunc',
                        odes=[
                            ODEConfig(variable='xr', equation='vr'),
                            ODEConfig(variable='vr', equation='ar'),
                            ODEConfig(variable='ar', equation='0'), # ADDED: ar is a state variable, its derivative is 0 during ODE
                            ODEConfig(variable='xo', equation='vo'),
                            ODEConfig(variable='vo', equation='ao'),
                            ODEConfig(variable='ao', equation='0'), # ADDED: ao is a state variable, its derivative is 0 during ODE
                            ODEConfig(variable='t', equation='1'),
                            ODEConfig(variable='T', equation='-1')
                        ],
                        evolution_domain=EvolutionDomainConfig(constraints=['t <= ep', 'vr >= 0', 'vo>=Vmin']),
                        raw_body_snippet="{xr' = vr, vr' = ar, ar'=0, xo'=vo, vo'=ao, ao'=0, t' = 1, T'=-1 & t <= ep & vr >= 0 & vo>=Vmin }" # Updated snippet
                    )
                ]
            ),
            variables=VariablesConfig(
                state_variables=[
                    VariableDeclarationConfig(name='xr', type='Real', lower_bound='-200', upper_bound='200', description='robot position: x'),
                    VariableDeclarationConfig(name='vr', type='Real', lower_bound='0', upper_bound='30', description='robot translational velocity'),
                    VariableDeclarationConfig(name='ar', type='Real', lower_bound='-10', upper_bound='10', description='robot translational acceleration'),
                    VariableDeclarationConfig(name='xo', type='Real', lower_bound='-200', upper_bound='200', description='obstacle position: x'),
                    VariableDeclarationConfig(name='vo', type='Real', lower_bound='0', upper_bound='30', description='obstacle velocity'),
                    VariableDeclarationConfig(name='ao', type='Real', lower_bound='-10', upper_bound='10', description='obstacle acceleration'),
                    VariableDeclarationConfig(name='t', type='Real', lower_bound='0', upper_bound='1', description='time elapsed in current control step'),
                    VariableDeclarationConfig(name='T', type='Real', lower_bound='-100', upper_bound='100', description='remaining time until deadline')
                ]
            ),
            problem=ProblemConfig(
                # FIXED: Simplified precondition to match formal model's assumptions and initialState
                precondition=PreconditionConfig(formula="A > 0 & b > 0 & ep > 0 & Vmin > 0 & D >= ep & vr = 0 & vo>=Vmin & xr < ixr & xr + A/2*(D - ep)^2 > ixr & T = D"),
                loop_execution=LoopExecutionConfig(
                    iterations='inf',
                    # FIXED: deadlineloopinv parameters adjusted
                    loop_invariant_formula='safetyloopinv(xr, vr, ixr, xo, vo, ixo) & deadlineloopinv(xr, vr, ixr, T)',
                    loop_body=SequenceActionConfig(actions=[
                        CallHybridProgramConfig(hp_name='controlFunc'), # This now represents robot's choice
                        HybridProgramStepConfig(hp_name='stepFunc', invariant_formula='t>=0')
                    ])
                )
                ,
                postconditions=[
                    PostconditionConfig(formula='ensure(xr)'), # Primary goal for agent (liveness)
                    PostconditionConfig(formula='finalSafety(xr, xo, T)') # Secondary safety property
                ]
            )
        )'''

        if debug_mode:
            print(f"DEBUG_GEN_ENV_INIT_PRE_SUPER: _state_keys before super().__init__: {getattr(self, '_state_keys', 'Not set')}, length: {len(getattr(self, '_state_keys', []))}")
            print(f"DEBUG_GEN_ENV_INIT_PRE_SUPER: id(self._state_keys) before super().__init__: {id(getattr(self, '_state_keys', None))}")
            print(f"DEBUG_GEN_ENV_INIT_PRE_SUPER: id(self._variables_config) before super().__init__: {id(getattr(self, '_variables_config', None))}")

        super().__init__(config_serialized_str=self._config_serialized_str, render_mode=render_mode, cisb_size=cisb_size, cisb_populate_attempts=cisb_populate_attempts, seed=seed, debug_mode=debug_mode)

        if self.debug_mode:
            print(f"DEBUG_GEN_ENV_INIT_POST_SUPER: _state_keys from base class (after super().__init__): {self._state_keys}, length: {len(self._state_keys)}")
            print(f"DEBUG_GEN_ENV_INIT_POST_SUPER: id(self._state_keys) after super().__init__: {id(self._state_keys)}")
            print(f"DEBUG_GEN_ENV_INIT_POST_SUPER: id(self._variables_config) after super().__init__: {id(self._variables_config)}")
            print(f"DEBUG_GEN_ENV_INIT_POST_SUPER_CONFIG_VARS: self.config.variables.state_variables names (after super().__init__): {[v.name for v in self.config.variables.state_variables]}")

        # --- CRITICAL FIX: Explicitly set _state_keys and _variables_config to the correct values ---
        # This overrides any potential misparsing by the base class's eval, ensuring consistency.
        correct_state_variables = [
            VariableDeclarationConfig(name='xr', type='Real', lower_bound='-200', upper_bound='200', description='robot position: x'),
            VariableDeclarationConfig(name='vr', type='Real', lower_bound='0', upper_bound='30', description='robot translational velocity'),
            VariableDeclarationConfig(name='ar', type='Real', lower_bound='-10', upper_bound='10', description='robot translational acceleration'),
            VariableDeclarationConfig(name='xo', type='Real', lower_bound='-200', upper_bound='200', description='obstacle position: x'),
            VariableDeclarationConfig(name='vo', type='Real', lower_bound='0', upper_bound='30', description='obstacle velocity'),
            VariableDeclarationConfig(name='ao', type='Real', lower_bound='-10', upper_bound='10', description='obstacle acceleration'),
            VariableDeclarationConfig(name='t', type='Real', lower_bound='0', upper_bound='1', description='time elapsed in current control step'),
            VariableDeclarationConfig(name='T', type='Real', lower_bound='-100', upper_bound='100', description='remaining time until deadline')
        ]
        # Ensure self._variables_config points to the config's variables object.
        # This is important because the base class might have created a new object or assigned it differently.
        if not hasattr(self, '_variables_config') or self._variables_config is None:
            self._variables_config = self.config.variables
        self._variables_config.state_variables = correct_state_variables
        # Re-assign self._state_keys directly based on the correct list
        self._state_keys = [v.name for v in correct_state_variables]

        if self.debug_mode:
            print(f"DEBUG_GEN_ENV_INIT_OVERRIDE_SUCCESS: id(self._variables_config) after override: {id(self._variables_config)}")
            print(f"DEBUG_GEN_ENV_INIT_OVERRIDE_SUCCESS: _variables_config.state_variables names after override: {[v.name for v in self._variables_config.state_variables]}")
            print(f"DEBUG_GEN_ENV_INIT_OVERRIDE_SUCCESS: _state_keys after override: {self._state_keys}, length: {len(self._state_keys)}")
            print(f"DEBUG_GEN_ENV_INIT_OVERRIDE_SUCCESS: id(self._state_keys) after override: {id(self._state_keys)}")
            print(f"DEBUG_GEN_ENV_INIT_BEFORE_CISB: self._variables_config.state_variables names before _populate_cisb_diverse_z3: {[v.name for v in self._variables_config.state_variables]}")
        # --- END CRITICAL FIX ---

        # Set fixed constant values *after* super().__init__ has initialized self.constants
        # These are constants with a fixed 'value' in the definition, not just bounds.
        self.constants['ixr'] = 0.0
        self.constants['ixo'] = 0.0

        # Initialize functions using lambdas. These will now use the fixed values for ixr, ixo.
        self.functions['minV'] = lambda: ((self.constants['A']) * (self.constants['ep']))
        self.functions['stopDist'] = lambda v: ((((v) ** (2))) / ((2) * (self.constants['b'])))

        # Refactored 'accComp' for improved readability
        self.functions['accComp'] = lambda v: (
            (safe_div(self.constants['A'], self.constants['b']) + 1) *
            (safe_div(self.constants['A'], 2) * (self.constants['ep'] ** 2) + (self.constants['ep'] * v))
        )

        self.functions['OAfterX'] = lambda xo: ((xo) > (self.constants['ixo']))
        self.functions['RAfterX'] = lambda xr: ((xr) > (self.constants['ixr']))

        # Initialize predicates using lambdas.
        self.predicates['bounds'] = lambda: (
            (self.constants['A'] > 0) and (self.constants['b'] > 0) and
            (self.constants['ep'] > 0) and (self.constants['Vmin'] > 0) and
            (self.constants['D'] >= self.constants['ep'])
        )

        # Refactored 'initialState' for improved readability
        self.predicates['initialState'] = lambda xr, vr, ixr, xo, vo, ixo, T: (
            self._approx_eq(vr, 0) and (vo >= self.constants['Vmin']) and
            (xr < ixr) and
            (xr + self.constants['A'] / 2 * ((self.constants['D'] - self.constants['ep']) ** 2) > ixr) and # FIXED: Direct translation from formal model
            self._approx_eq(T, self.constants['D']) # FIXED: T = D
        )

        self.predicates['assumptions'] = lambda xr, vr, ixr, xo, vo, ixo, T: (
            self.predicates['bounds']() and self.predicates['initialState'](xr, vr, ixr, xo, vo, ixo, T)
        )

        # Refactored 'safetyloopinv' for improved readability and to resolve Pylance errors
        self.predicates['safetyloopinv'] = lambda xr, vr, ixr, xo, vo, ixo: (
            ((0) <= (vr)) and ((self.constants['Vmin']) <= (vo))
        ) and (
            (self.functions['RAfterX'](xr)) or
            (self.functions['OAfterX'](xo)) or
            (((xr) + (self.functions['stopDist'](vr))) < (ixr)) or
            (
                ((vr) > (0)) and (
                    (
                        ((xo) + ((vo) * (safe_div(ixr - xr, vr))) + ((self.constants['A']) / (2) * (((safe_div(ixr - xr, vr))) ** (2)))) < (ixo)
                    ) or (
                        (ixo) < ((xo) + ((self.constants['Vmin']) * (safe_div(ixr - xr, vr))))
                    )
                )
            )
        )

        # Refactored 'deadlineloopinv' for improved readability and to resolve Pylance errors
        self.predicates['deadlineloopinv'] = lambda xr, vr, ixr, T: ( # FIXED: Parameters adjusted
            self.functions['RAfterX'](xr + vr * np.maximum(0, T) + self.constants['A'] / 2 * (np.maximum(0, T) ** 2)) # FIXED: Simplified to match formal model
        )

        # Refactored 'finalSafety' for improved readability
        self.predicates['finalSafety'] = lambda xr, xo, T: (
            # Corrected implication for finalSafety: (A -> B) is (not A) or B
            ( (not self._approx_eq(xr, self.constants['ixr'])) or (not self._approx_eq(xo, self.constants['ixo'])) ) and
            ( (not (T <= 0)) or self.functions['RAfterX'](xr) ) # FIXED: T <= 0 as per formal model
        )

        self.predicates['ensure'] = lambda xr: (self.functions['RAfterX'](xr))

        # ADDED: Call _populate_cisb_diverse_z3 here, after all constants and functions are set up.
        if self.debug_mode:
            print(f"DEBUG_GEN_ENV_INIT: Calling _populate_cisb_diverse_z3 from GeneratedKXEnv.__init__.")
        self._populate_cisb_diverse_z3()
        if self.debug_mode:
            print(f"DEBUG_GEN_ENV_INIT: _populate_cisb_diverse_z3 completed.")

        # Ensure self.state is initialized with all 8 keys
        # This line is good and ensures all keys exist.
        self.state = {key: 0.0 for key in self._state_keys}

        low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)

        # Action space for controlFunc:
        # 'choice_ChoiceLevel_0' for the 4 robot choices (all deterministic ar)
        self.action_space = spaces.Dict({
                'choice_ChoiceLevel_0': spaces.Discrete(4), # FIXED: 4 choices for robot's ar, all deterministic
                # 'assign_ar' removed as ar is now always deterministic
                })
        self._action_space_keys = ['choice_ChoiceLevel_0'] # FIXED: Removed 'assign_ar'
        self._control_func_action_config = ChoiceActionConfig(
            level_name='ChoiceLevel_0',
            choices=[
                # Choice 0: If obstacle passed or robot passed, robot can go full speed (ar := A)
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(formula='OAfterX(xo) | RAfterX(xr)'),
                    DeterministicAssignmentActionConfig(variable='ar', value='A')
                ]),
                # Choice 1: If intersection is ahead for both AND it's safe to accelerate (ar := A)
                # Guard: !(OAfterX(xo) | RAfterX(xr)) AND (vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/vr ))
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(
                        formula=('!(OAfterX(xo) | RAfterX(xr)) & '
                                 '(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | '
                                 'ixo < xo + Vmin*(ixr-xr)/vr ))')
                    ),
                    DeterministicAssignmentActionConfig(variable='ar', value='A')
                ]),
                # Choice 2: If intersection is ahead for both AND not safe to accelerate AND vr = 0 (ar := 0)
                # Guard: !(OAfterX(xo) | RAfterX(xr)) AND !(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & vr = 0
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(
                        formula=('!(OAfterX(xo) | RAfterX(xr)) & '
                                 '!(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                 'vr = 0')
                    ),
                    DeterministicAssignmentActionConfig(variable='ar', value='0')
                ]),
                # Choice 3: If intersection is ahead for both AND not safe to accelerate AND vr != 0 (ar := -b)
                # Guard: !(OAfterX(xo) | RAfterX(xr)) AND !(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & vr != 0
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(
                        formula=('!(OAfterX(xo) | RAfterX(xr)) & '
                                 '!(vr > 0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                 'vr != 0')
                    ),
                    DeterministicAssignmentActionConfig(variable='ar', value='-b')
                ])
            ]
        )

        self._ode_state_map = {'xr': 0, 'vr': 1, 'ar': 2, 'xo': 3, 'vo': 4, 'ao': 5, 't': 6, 'T': 7}
        self._ode_state_idx_map = {0: 'xr', 1: 'vr', 2: 'ar', 3: 'xo', 4: 'vo', 5: 'ao', 6: 't', 7: 'T'}

        _ode_func_str_local = '''def ode_func(t_val, y_array):
    xr = y_array[0]
    vr = y_array[1]
    ar = y_array[2] # Now ar is part of y_array
    xo = y_array[3]
    vo = y_array[4]
    ao = y_array[5] # Now ao is part of y_array
    t = y_array[6]
    T = y_array[7]

    dydt = [
        vr, # dxr/dt
        ar, # dvr/dt
        0.0, # dar/dt (ar is control input, constant during ODE)
        vo, # dxo/dt
        ao, # dvo/dt (ao is control input, constant during ODE)
        0.0, # dao/dt
        1, # dt/dt
        -1, # dT/dt
    ]
    return dydt'''
        # Updated _ode_events_str_local
        _ode_events_str_local = '''def event_t_max(t_val, y_array):
    return self.constants['ep'] - y_array[6] # t is at index 6
event_t_max.terminal = True
event_t_max.direction = -1

def event_vr_min(t_val, y_array):
    vr = y_array[1] # vr is at index 1
    ar = y_array[2] # ar is at index 2

    # If vr is effectively zero AND ar is non-negative (meaning vr will stay zero or increase),
    # then vr is not crossing from positive to negative.
    # Return a small positive value to effectively "disarm" the event in this specific case.
    if abs(vr) < 1e-9 and ar >= -1e-9: # Use a small epsilon for floating point comparison
        return 1e-12 # Small positive value
    
    # Otherwise, return vr. The direction=-1 will handle actual crossings from positive to negative.
    return vr
event_vr_min.terminal = True
event_vr_min.direction = -1

def event_vo_min(t_val, y_array):
    vo = y_array[4] # vo is at index 4
    ao = y_array[5] # ao is at index 5
    Vmin = self.constants['Vmin']

    # If (vo - Vmin) is effectively zero AND ao is non-negative (meaning vo will stay at or above Vmin),
    # then vo is not crossing below Vmin.
    # Return a small positive value to effectively "disarm" the event in this specific case.
    if abs(vo - Vmin) < 1e-9 and ao >= -1e-9: # Use a small epsilon for floating point comparison
        return 1e-12 # Small positive value
    
    # Otherwise, return vo - Vmin. The direction=-1 will handle actual crossings from positive to negative.
    return vo - Vmin
event_vo_min.terminal = True
event_vo_min.direction = -1

ode_events = [
    event_t_max,
    event_vr_min,
    event_vo_min,
]'''
        self._ode_func_compiled = self._compile_ode_func(_ode_func_str_local)
        self._ode_events_compiled = self._compile_ode_events(_ode_events_str_local)

        # ADDED DEBUG PRINT
        if self.debug_mode:
            print(f"DEBUG_GEN_ENV_INIT_BEFORE_RESET_CALL: _state_keys just before calling self.reset(): {self._state_keys}, len: {len(self._state_keys)}")
            print(f"DEBUG_GEN_ENV_INIT_BEFORE_RESET_CALL: id(self._state_keys) before calling self.reset(): {id(self._state_keys)}")

        # Call reset here to ensure initial state is properly set up
        # and _get_obs is called for the first observation.
        # super().reset() is called inside this, which will call _get_obs.
        if self.debug_mode:
            print(f"DEBUG_GEN_ENV_INIT: Calling self.reset() at end of __init__.")
        self.reset()
        if self.debug_mode:
            print(f"DEBUG_GEN_ENV_INIT: self.reset() completed.")

    def _get_obs(self):
        """
        Returns the current observation of the environment.
        This method is called by reset() and step().
        """
        if self.debug_mode:
            print(f"DEBUG_GEN_GET_OBS_ENTRY: self._state_keys (in _get_obs): {self._state_keys}, length: {len(self._state_keys)}")
            print(f"DEBUG_GEN_GET_OBS_ENTRY: id(self._state_keys) (in _get_obs): {id(self._state_keys)}")
            print(f"DEBUG_GEN_GET_OBS_ENTRY: self.state keys (in _get_obs): {self.state.keys()}, length: {len(self.state.keys())}")

        # Defensive check: if _state_keys is somehow still wrong, correct it as a fallback.
        expected_keys = ['xr', 'vr', 'ar', 'xo', 'vo', 'ao', 't', 'T']
        if len(self._state_keys) != len(expected_keys) or any(key not in self._state_keys for key in expected_keys):
            if self.debug_mode:
                print(f"CRITICAL_ERROR_GET_OBS: _state_keys has {len(self._state_keys)} elements or is missing expected keys. Current _state_keys: {self._state_keys}. Re-asserting.")
            self._state_keys = expected_keys
            # Also ensure self._variables_config.state_variables is aligned
            correct_state_variables = [
                VariableDeclarationConfig(name='xr', type='Real', lower_bound='-200', upper_bound='200', description='robot position: x'),
                VariableDeclarationConfig(name='vr', type='Real', lower_bound='0', upper_bound='30', description='robot translational velocity'),
                VariableDeclarationConfig(name='ar', type='Real', lower_bound='-10', upper_bound='10', description='robot translational acceleration'),
                VariableDeclarationConfig(name='xo', type='Real', lower_bound='-200', upper_bound='200', description='obstacle position: x'),
                VariableDeclarationConfig(name='vo', type='Real', lower_bound='0', upper_bound='30', description='obstacle velocity'),
                VariableDeclarationConfig(name='ao', type='Real', lower_bound='-10', upper_bound='10', description='obstacle acceleration'),
                VariableDeclarationConfig(name='t', type='Real', lower_bound='0', upper_bound='1', description='time elapsed in current control step'),
                VariableDeclarationConfig(name='T', type='Real', lower_bound='-100', upper_bound='100', description='remaining time until deadline')
            ]
            # Ensure self._variables_config is correctly referenced before modification
            if not hasattr(self, '_variables_config') or self._variables_config is None:
                self._variables_config = self.config.variables # Fallback to config's variables
            self._variables_config.state_variables = correct_state_variables
            if self.debug_mode:
                print(f"CRITICAL_ERROR_GET_OBS: _state_keys re-asserted to: {self._state_keys} as fallback.")

        obs_array = np.array([float(self.state.get(key, 0.0)) for key in self._state_keys], dtype=np.float32)
        if self.debug_mode:
            print(f"DEBUG_GEN_GET_OBS_EXIT: _get_obs returning observation of length: {len(obs_array)}")
            print(f"DEBUG_GEN_GET_OBS_EXIT: _get_obs content: {obs_array}")
        return obs_array

    def reset(self, seed=None, options=None):
        if self.debug_mode:
            print(f"DEBUG_GEN_RESET_ENTRY: _state_keys at start of GeneratedKXEnv.reset(): {self._state_keys}, len: {len(self._state_keys)}")
            print(f"DEBUG_GEN_RESET_ENTRY: id(self._state_keys) at start of GeneratedKXEnv.reset(): {id(self._state_keys)}")
            print(f"DEBUG_GEN_RESET_ENTRY: self.state keys at start of GeneratedKXEnv.reset(): {self.state.keys()}, len: {len(self.state.keys())}")
            print(f"DEBUG_GEN_RESET_ENTRY: self._variables_config.state_variables names at start of GeneratedKXEnv.reset(): {[v.name for v in self._variables_config.state_variables]}")

        # --- AGGRESSIVE FIX: Ensure _state_keys and _variables_config are correct BEFORE calling super().reset() ---
        correct_state_variables = [
            VariableDeclarationConfig(name='xr', type='Real', lower_bound='-200', upper_bound='200', description='robot position: x'),
            VariableDeclarationConfig(name='vr', type='Real', lower_bound='0', upper_bound='30', description='robot translational velocity'),
            VariableDeclarationConfig(name='ar', type='Real', lower_bound='-10', upper_bound='10', description='robot translational acceleration'),
            VariableDeclarationConfig(name='xo', type='Real', lower_bound='-200', upper_bound='200', description='obstacle position: x'),
            VariableDeclarationConfig(name='vo', type='Real', lower_bound='0', upper_bound='30', description='obstacle velocity'),
            VariableDeclarationConfig(name='ao', type='Real', lower_bound='-10', upper_bound='10', description='obstacle acceleration'),
            VariableDeclarationConfig(name='t', type='Real', lower_bound='0', upper_bound='1', description='time elapsed in current control step'),
            VariableDeclarationConfig(name='T', type='Real', lower_bound='-100', upper_bound='100', description='remaining time until deadline')
        ]
        # Ensure self._variables_config is correctly referenced before modification
        if not hasattr(self, '_variables_config') or self._variables_config is None:
            self._variables_config = self.config.variables # Fallback to config's variables
        self._variables_config.state_variables = correct_state_variables
        # Re-assign self._state_keys directly based on the correct list
        self._state_keys = [v.name for v in correct_state_variables]

        if self.debug_mode:
            print(f"DEBUG_GEN_RESET_FIXED_PRE_SUPER: _state_keys re-asserted to: {self._state_keys}, len: {len(self._state_keys)}")
            print(f"DEBUG_GEN_RESET_FIXED_PRE_SUPER: id(self._state_keys) after re-assertion: {id(self._state_keys)}")
            print(f"DEBUG_GEN_RESET_FIXED_PRE_SUPER: self._variables_config.state_variables names: {[v.name for v in self._variables_config.state_variables]}")
        # --- END AGGRESSIVE FIX ---

        if self.debug_mode:
            print(f"DEBUG_GEN_RESET_PRE_SUPER: _state_keys before super().reset(): {self._state_keys}, len: {len(self._state_keys)}")
            print(f"DEBUG_GEN_RESET_PRE_SUPER: id(self._state_keys) before super().reset(): {id(self._state_keys)}")

        # Call super().reset() once to get an initial state from Z3 (via CISB)
        observation_from_super, info = super().reset(seed=seed, options=options)

        if self.debug_mode:
            print(f"DEBUG_GEN_RESET_POST_SUPER: super().reset() completed.")
            print(f"DEBUG_GEN_RESET_POST_SUPER: Initial self.state from CISB: {self.state}")
            print(f"DEBUG_GEN_RESET_POST_SUPER: self.constants['Vmin'] from CISB: {self.constants.get('Vmin')}")

        # --- Fix for vo < Vmin if CISB provided an invalid value ---
        current_vo = self.state.get('vo', float('nan'))
        current_Vmin = self.constants.get('Vmin', float('nan'))

        if current_vo < current_Vmin:
            if self.debug_mode:
                print(f"CRITICAL_DEBUG: Initial vo ({current_vo:.6f}) from CISB < Vmin ({current_Vmin:.6f}). Attempting to fix vo.")
            self.state['vo'] = current_Vmin + 1e-6 # Ensure it's strictly greater
            if self.debug_mode:
                print(f"CRITICAL_DEBUG: vo adjusted to {self.state['vo']:.6f}.")
        
        # --- Validate the entire precondition with the (potentially fixed) state ---
        precondition_formula = self.config.problem.precondition.formula
        precondition_satisfied = self._evaluate_expression(precondition_formula, is_constraint=True)

        if not precondition_satisfied:
            # This indicates a fundamental problem with CISB generation or precondition evaluation.
            final_vo = self.state.get('vo', float('nan'))
            final_Vmin = self.constants.get('Vmin', float('nan'))
            raise RuntimeError(f"CISB generated an initial state that violates the precondition even after vo adjustment. Last state: {self.state}. Specific issue: vo={final_vo:.6f}, Vmin={final_Vmin:.6f}. (vo >= Vmin) is {final_vo >= final_Vmin}.")

        # After super().reset(), ensure all state variables are present in self.state
        # The base class's sampling might not populate all variables if not explicitly constrained in precondition.

        # Remove 'xg' if it was incorrectly added by the base class (should not happen with __init__ fix, but defensive)
        if 'xg' in self.state:
            if self.debug_mode:
                print(f"DEBUG_GEN_RESET: Found 'xg' in state, removing it.")
            del self.state['xg']

        # Ensure all expected state variables are present and initialized
        expected_state_keys = ['xr', 'vr', 'ar', 'xo', 'vo', 'ao', 't', 'T']
        for key in expected_state_keys:
            if key not in self.state:
                # Provide a reasonable default if a key is missing
                # For 'vo', use Vmin as per initial state predicate
                if key == 'vo':
                    self.state[key] = self.constants.get('Vmin', 0.1)
                elif key == 'T': # FIXED: Initialize T to D if missing
                    self.state[key] = self.constants.get('D', 1.0)
                else:
                    self.state[key] = 0.0
                if self.debug_mode:
                    print(f"DEBUG_GEN_RESET: '{key}' not in state, setting to {self.state[key]:.6f}")

        # Re-generate observation after ensuring self.state is complete
        # This will call GeneratedKXEnv._get_obs() again, which should now use the fully populated self.state
        if self.debug_mode:
            print(f"DEBUG_GEN_RESET: Calling _get_obs() for final_observation.")
        final_observation = self._get_obs()

        if self.debug_mode:
            print(f"DEBUG_GEN_RESET: State after all initializations: {self.state.keys()}")
            print(f"DEBUG_GEN_RESET: Final observation returned by reset: {final_observation}, length: {len(final_observation)}")
            print(f"DEBUG_GEN_RESET_EXIT: _state_keys before returning final_observation: {self._state_keys}, len: {len(self._state_keys)}")
            print(f"DEBUG_GEN_RESET_EXIT: id(self._state_keys) before returning final_observation: {id(self._state_keys)}")

        return final_observation, info

    def _split_top_level_and_local(self, formula: str) -> List[str]:
        """
        Splits a logical formula string by top-level '&' operators, respecting parentheses.
        Example: "A & (B & C) & D" -> ["A", "(B & C)", "D"]
        This method is added here as a fallback if it's not present in the base class.
        It handles single '&' as the AND operator.
        """
        conjuncts = []
        balance = 0
        current_conjunct_start = 0
        for i, char in enumerate(formula):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            elif char == '&' and balance == 0:
                conjuncts.append(formula[current_conjunct_start:i].strip())
                current_conjunct_start = i + 1
        conjuncts.append(formula[current_conjunct_start:].strip())
        return [c for c in conjuncts if c] # Remove empty strings

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        # --- Pre-control loop invariant check ---
        # Check if loop_execution is None before accessing its attributes
        if self._problem_config.loop_execution is None:
            if self.debug_mode:
                print("DEBUG_STEP: self._problem_config.loop_execution is None. Cannot check loop invariant.")
            # Handle this as an environment setup error, likely terminating the episode.
            reward -= 1000.0 # Severe penalty
            terminated = True
            info['error'] = "Environment configuration error: loop_execution is None."
            return self._get_obs(), reward, terminated, truncated, info

        loop_invariant_formula_kx = self._problem_config.loop_execution.loop_invariant_formula
        expanded_loop_inv = expand_formula(loop_invariant_formula_kx, self._definitions_config)
        loop_inv_conjuncts = self._split_top_level_and_local(expanded_loop_inv)

        loop_invariant_satisfied_pre_step = True
        for conj_str in loop_inv_conjuncts:
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', conj_str)
            # FIX: Change {} to None for pred_name
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {}

            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)

            if not conjunct_result:
                loop_invariant_satisfied_pre_step = False
                if self.debug_mode:
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (pre-control): '{conj_str}'")
                
                # Capture the reward change from this specific failure
                reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                    0.0, False, False, is_failure_case=True
                )
                if self.debug_mode:
                    print(f"DEBUG_STEP: Applied meta effect for '{pred_name if pred_name else 'fallback'}' failure. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                reward += reward_change
                terminated = terminated or new_terminated
                truncated = truncated or new_truncated

                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                # Apply on_success effects for invariants if not 'continue'
                if metadata.get('on_success') and metadata.get('on_success').get('action') != 'continue':
                    reward_change, new_terminated, new_truncated = self._apply_meta_effect(metadata, 0.0, False, False, is_failure_case=False)
                    if self.debug_mode:
                        print(f"DEBUG_STEP: Applied meta effect for '{pred_name if pred_name else 'fallback'}' success. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                    reward += reward_change
                    terminated = terminated or new_terminated
                    truncated = truncated or new_truncated
                    if terminated or truncated:
                        state_record_before_early_return = self.state.copy()
                        state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                        self.state_history.append(state_record_before_early_return)
                        return self._get_obs(), reward, terminated, truncated, info
                # Apply shape reward for progress invariants
                if metadata.get('type') == 'progress' and 'shape' in metadata:
                    shape_value = self._evaluate_shape_expression(metadata['shape'])
                    reward += shape_value
                    if self.debug_mode:
                        print(f"DEBUG_STEP: Applied invariant shape reward for '{pred_name}': {shape_value:.6f}")

        if not loop_invariant_satisfied_pre_step:
            if self.debug_mode:
                print(f"DEBUG_STEP: General loop invariant FAILED (pre-control).")
            reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-500.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                0.0, False, False, is_failure_case=True
            )
            if self.debug_mode:
                print(f"DEBUG_STEP: Applied meta effect for general loop invariant failure. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
            reward += reward_change
            terminated = terminated or new_terminated
            truncated = truncated or new_truncated
            state_record_before_early_return = self.state.copy()
            state_record_before_early_return['episode_time'] = self.cumulative_episode_time
            self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info

        # --- Environmental Nondeterministic Action (Obstacle) ---
        # Sample ao from [-b, A] as per "ao := *; ?-b<=ao & ao<=A;"
        b_val = self.constants['b']
        A_val = self.constants['A']
        self.state['ao'] = self.np_random.uniform(-b_val, A_val)
        if self.debug_mode:
            print(f"DEBUG_STEP: Sampled obstacle acceleration (ao): {self.state['ao']:.6f}")

        # --- Control Action Application (Robot) ---
        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: State before control action: xr={self.state['xr']:.6f}, vr={self.state['vr']:.6f}, ar={self.state['ar']:.6f}, xo={self.state['xo']:.6f}, vo={self.state['vo']:.6f}, ao={self.state['ao']:.6f}, T={self.state['T']:.6f}, ep={self.constants['ep']:.6f}")
        control_reward, control_terminated, control_applied = self._apply_control_action(action)
        reward += control_reward
        terminated = terminated or control_terminated
        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: State after _apply_control_action: xr={self.state['xr']:.6f}, vr={self.state['vr']:.6f}, ar={self.state['ar']:.6f}, xo={self.state['xo']:.6f}, vo={self.state['vo']:.6f}, ao={self.state['ao']:.6f}, T={self.state['T']:.6f}, control_applied={control_applied}")

        if not control_applied:
            if self.debug_mode:
                print(f"DEBUG_STEP: Control action not applied. Terminating early.")
            # Apply a penalty for not being able to apply a control action
            reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                0.0, False, False, is_failure_case=True
            )
            if self.debug_mode:
                print(f"DEBUG_STEP: Applied meta effect for control action not applied. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
            reward += reward_change
            terminated = terminated or new_terminated
            truncated = truncated or new_truncated

            state_record_before_early_return = self.state.copy()
            state_record_before_early_return['episode_time'] = self.cumulative_episode_time
            self.state_history.append(state_record_before_early_return)
            return self._get_obs(), reward, terminated, truncated, info

        # --- ODE Integration ---
        y0 = np.array([self.state[var_name] for var_name in self._ode_state_map.keys()], dtype=np.float64)

        ep_val = self.constants['ep']
        t_span = (0, ep_val)

        ode_terminated_by_event = False

        if self.debug_mode:
            print(f"DEBUG_ODE: Before solve_ivp: y0={y0}, t_span={t_span}, ep_val={ep_val:.6f}")
            print(f"DEBUG_ODE: Initial state for ODE: vr={self.state['vr']:.6f}, ar={self.state['ar']:.6f}, vo={self.state['vo']:.6f}, ao={self.state['ao']:.6f}")
            print(f"DEBUG_ODE: Constants for ODE: Vmin={self.constants['Vmin']:.6f}, ep={self.constants['ep']:.6f}")
            # Evaluate event functions at t=0
            event_vr_min_val = y0[1] # vr
            event_vo_min_val = y0[4] - self.constants['Vmin'] # vo - Vmin
            event_t_max_val = self.constants['ep'] - y0[6] # ep - t
            print(f"DEBUG_ODE: Event function values at t=0: event_vr_min={event_vr_min_val:.6f}, event_vo_min={event_vo_min_val:.6f}, event_t_max={event_t_max_val:.6f}")


        if ep_val <= 0:
            self.state['t'] = 0.0
            if self.debug_mode:
                print(f"DEBUG_ODE: Warning: 'ep' constant is non-positive ({ep_val:.6f}). Skipping ODE integration and penalizing.")
            bounds_meta = self._get_predicate_metadata('bounds')
            reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                bounds_meta or {'type': 'safety', 'penalty': '-200.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                0.0, False, False, is_failure_case=True
            )
            if self.debug_mode:
                print(f"DEBUG_ODE: Applied meta effect for non-positive ep. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
            reward += reward_change
            terminated = terminated or new_terminated
            truncated = truncated or new_truncated
            ode_terminated_by_event = True # Treat as ODE termination

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

                        # Ensure the initial state is only appended once per step, not duplicated if t_eval[0] is 0.
                        # The base class's step might already add the initial state.
                        start_idx = 0
                        if len(self.state_history) > 0 and np.isclose(t_eval[0], 0.0):
                            # Check if the last recorded state matches the current initial state (before ODE)
                            # This is a heuristic to avoid duplicate entries if the base class already added it.
                            last_recorded_state = self.state_history[-1]
                            current_initial_state_values = [self.state[k] for k in self._state_keys]
                            if all(np.isclose(last_recorded_state.get(k, 0.0), current_initial_state_values[i]) for i, k in enumerate(self._state_keys)):
                                start_idx = 1 # Skip the first point if it's a duplicate of the last history entry

                        for k in range(start_idx, ode_solution_points.shape[1]):
                            current_ode_state = {}
                            for i, var_name in enumerate(self._ode_state_map.keys()):
                                current_ode_state[var_name] = ode_solution_points[i, k]

                            intermediate_episode_time = self.cumulative_episode_time + current_ode_state['t']
                            current_ode_state['episode_time'] = intermediate_episode_time
                            self.state_history.append(current_ode_state)
                            if self.debug_mode and k % (ode_solution_points.shape[1] // 5 + 1) == 0:
                                print(f"DEBUG_ODE_APPEND: Appended intermediate state k={k}: xr={current_ode_state['xr']:.2f}, vr={current_ode_state['vr']:.2f}, ar={current_ode_state['ar']:.2f}, xo={current_ode_state['xo']:.2f}, vo={current_ode_state['vo']:.2f}, ao={current_ode_state['ao']:.2f}, T={current_ode_state['T']:.2f}, ep_time={intermediate_episode_time:.2f}")

                if solution.status == 0: # Integration successful, no events
                    final_y = solution.y[:, -1]
                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]
                elif solution.status == 1: # Event occurred
                    # Determine which event triggered and apply its specific metadata
                    triggered_event_indices = []
                    for i, t_event_list in enumerate(solution.t_events):
                        if t_event_list.size > 0:
                            triggered_event_indices.append(i)

                    if triggered_event_indices:
                        # Use the state at the first occurrence of the first triggered terminal event
                        # solution.y_events is a list of arrays, one for each event.
                        # Each array contains the state vector at the time of the event.
                        final_y_at_event = None
                        for event_idx in triggered_event_indices:
                            if event_idx < len(solution.y_events) and solution.y_events[event_idx].size > 0:
                                final_y_at_event = solution.y_events[event_idx][0] # Take the first state for this event
                                break # Use the first event's state

                        if final_y_at_event is None: # Fallback if no specific event state found
                            final_y_at_event = solution.y[:, -1]

                        for i, var_name in enumerate(self._ode_state_map.keys()):
                            self.state[var_name] = final_y_at_event[i]

                        # Apply penalty based on the triggered event
                        # event_t_max is at index 0, event_vr_min is at index 1, event_vo_min is at index 2
                        if 1 in triggered_event_indices: # event_vr_min triggered
                            if self.debug_mode:
                                print(f"DEBUG_ODE: Event: event_vr_min triggered. vr={self.state['vr']:.6f}, ar={self.state['ar']:.6f}")
                            reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                                {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                                0.0, False, False, is_failure_case=True
                            )
                            if self.debug_mode:
                                print(f"DEBUG_ODE: Applied meta effect for event_vr_min. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                            reward += reward_change
                            terminated = terminated or new_terminated
                            truncated = truncated or new_truncated
                            ode_terminated_by_event = True
                        elif 2 in triggered_event_indices: # event_vo_min triggered
                            if self.debug_mode:
                                print(f"DEBUG_ODE: Event: event_vo_min triggered. vo={self.state['vo']:.6f}, Vmin={self.constants['Vmin']:.6f}, ao={self.state['ao']:.6f}")
                            reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                                {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                                0.0, False, False, is_failure_case=True
                            )
                            if self.debug_mode:
                                print(f"DEBUG_ODE: Applied meta effect for event_vo_min. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                            reward += reward_change
                            terminated = terminated or new_terminated
                            truncated = truncated or new_truncated
                            ode_terminated_by_event = True
                        elif 0 in triggered_event_indices: # event_t_max triggered (normal evolution, no penalty)
                            if self.debug_mode:
                                print(f"DEBUG_ODE: Event: event_t_max triggered. t={self.state['t']:.6f}, ep={self.constants['ep']:.6f}")
                            # No specific penalty for t_max unless other conditions are violated
                            pass # Keep existing reward/terminated/truncated unless other events also triggered
                        else: # Unknown terminal event
                            if self.debug_mode:
                                print(f"DEBUG_ODE: Unknown terminal event triggered (index not 0, 1, or 2).")
                            reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                                {'type': 'safety', 'penalty': '-50.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                                0.0, False, False, is_failure_case=True
                            )
                            if self.debug_mode:
                                print(f"DEBUG_ODE: Applied meta effect for unknown terminal event. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                            reward += reward_change
                            terminated = terminated or new_terminated
                            truncated = truncated or new_truncated
                            ode_terminated_by_event = True
                    else: # solution.status == 1 but no specific terminal event identified (should not happen for terminal events)
                        if self.debug_mode:
                            print(f"DEBUG_ODE: Event occurred (status 1) but no specific terminal event identified. This is unexpected.")
                        final_y = solution.y[:, -1]
                        for i, var_name in enumerate(self._ode_state_map.keys()):
                            self.state[var_name] = final_y[i]
                        # Default penalty if event occurred but not handled
                        reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                            {'type': 'safety', 'penalty': '-50.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                            0.0, False, False, is_failure_case=True
                        )
                        if self.debug_mode:
                            print(f"DEBUG_ODE: Applied meta effect for unhandled event. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                        reward += reward_change
                        terminated = terminated or new_terminated
                        truncated = truncated or new_truncated
                        ode_terminated_by_event = True
                else: # Integration failed for other reasons
                    if self.debug_mode:
                        print(f"DEBUG_ODE: ODE integration failed with status {solution.status}.")
                    reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                        {'type': 'safety', 'penalty': '-150.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                        0.0, False, False, is_failure_case=True
                    )
                    if self.debug_mode:
                        print(f"DEBUG_ODE: Applied meta effect for ODE integration failure. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                    reward += reward_change
                    terminated = terminated or new_terminated
                    truncated = truncated or new_truncated
                    ode_terminated_by_event = True

            except Exception as e:
                if self.debug_mode:
                    print(f"DEBUG_ODE: Exception during ODE integration: {e}")
                reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                    {'type': 'safety', 'penalty': '-200.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                    0.0, False, False, is_failure_case=True
                )
                if self.debug_mode:
                    print(f"DEBUG_ODE: Applied meta effect for ODE exception. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                reward += reward_change
                terminated = terminated or new_terminated
                truncated = truncated or new_truncated
                ode_terminated_by_event = True

        terminated = terminated or ode_terminated_by_event
        truncated = truncated or ode_terminated_by_event

        self.cumulative_episode_time += self.state['t']
        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: Updated cumulative_episode_time={self.cumulative_episode_time:.6f}")
            print(f"DEBUG_STEP {self.current_step}: State after ODE integration: xr={self.state['xr']:.6f}, vr={self.state['vr']:.6f}, ar={self.state['ar']:.6f}, xo={self.state['xo']:.6f}, vo={self.state['vo']:.6f}, ao={self.state['ao']:.6f}, t={self.state['t']:.6f}, T={self.state['T']:.6f}")

        if terminated or truncated:
            if self.debug_mode:
                print(f"DEBUG_STEP: Terminated/Truncated after ODE. Returning early.")
            # Ensure the final state after ODE is recorded if not already done by intermediate steps
            if not (solution.success and solution.sol is not None and solution.t.size > 1 and actual_ode_duration > 1e-9):
                state_record_after_step = self.state.copy()
                state_record_after_step['episode_time'] = self.cumulative_episode_time
                self.state_history.append(state_record_after_step)
            return self._get_obs(), reward, terminated, truncated, info

        # --- Post-ODE loop invariant check ---
        loop_invariant_satisfied_post_ode = True
        for conj_str in loop_inv_conjuncts:
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', conj_str)
            # FIX: Change {} to None for pred_name
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {}

            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)

            if not conjunct_result:
                loop_invariant_satisfied_post_ode = False
                if self.debug_mode:
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (post-ODE): '{conj_str}'")
                reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                    0.0, False, False, is_failure_case=True
                )
                if self.debug_mode:
                    print(f"DEBUG_STEP: Applied meta effect for '{pred_name if pred_name else 'fallback'}' failure (post-ODE). Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                reward += reward_change
                terminated = terminated or new_terminated
                truncated = truncated or new_truncated

                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                # Apply on_success effects for invariants if not 'continue'
                if metadata.get('on_success') and metadata.get('on_success').get('action') != 'continue':
                    reward_change, new_terminated, new_truncated = self._apply_meta_effect(metadata, 0.0, False, False, is_failure_case=False)
                    if self.debug_mode:
                        print(f"DEBUG_STEP: Applied meta effect for '{pred_name if pred_name else 'fallback'}' success (post-ODE). Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                    reward += reward_change
                    terminated = terminated or new_terminated
                    truncated = truncated or new_truncated
                    if terminated or truncated:
                        state_record_before_early_return = self.state.copy()
                        state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                        self.state_history.append(state_record_before_early_return)
                        return self._get_obs(), reward, terminated, truncated, info
                # Apply shape reward for progress invariants
                if metadata.get('type') == 'progress' and 'shape' in metadata:
                    shape_value = self._evaluate_shape_expression(metadata['shape'])
                    reward += shape_value
                    if self.debug_mode:
                        print(f"DEBUG_STEP: Applied invariant shape reward for '{pred_name}' (post-ODE): {shape_value:.6f}")

        if not loop_invariant_satisfied_post_ode:
            if self.debug_mode:
                print(f"DEBUG_STEP: General loop invariant FAILED (post-ODE).")
            reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-300.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                0.0, False, False, is_failure_case=True
            )
            if self.debug_mode:
                print(f"DEBUG_STEP: Applied meta effect for general loop invariant failure (post-ODE). Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
            reward += reward_change
            terminated = terminated or new_terminated
            truncated = truncated or new_truncated

            if terminated or truncated:
                state_record_before_early_return = self.state.copy()
                state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                self.state_history.append(state_record_before_early_return)
                return self._get_obs(), reward, terminated, truncated, info

        # --- Postcondition / Goal Check ---
        # The primary goal for the agent is ensure(xr), which is postconditions[0].
        postcondition_formula_kx = self._problem_config.postconditions[0].formula
        postcondition_satisfied = self._evaluate_expression(postcondition_formula_kx, is_constraint=True)
        ensure_meta = self._get_predicate_metadata('ensure')

        deadline_met = self.state.get('T', 0.0) <= 0
        if deadline_met:
            if self.debug_mode:
                print(f"DEBUG_STEP: Deadline T <= 0 met. Checking primary postcondition ('ensure').")
            if not postcondition_satisfied:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Deadline met but primary postcondition '{postcondition_formula_kx}' NOT SATISFIED. Failure.")
                reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                    ensure_meta, 0.0, False, False, is_failure_case=True
                )
                if self.debug_mode:
                    print(f"DEBUG_STEP: Applied meta effect for ensure failure (deadline met). Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                reward += reward_change
                terminated = terminated or new_terminated
                truncated = truncated or new_truncated
            else:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Deadline met and primary postcondition '{postcondition_formula_kx}' SATISFIED. Success.")
                reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                    ensure_meta, 0.0, False, False, is_failure_case=False
                )
                if self.debug_mode:
                    print(f"DEBUG_STEP: Applied meta effect for ensure success (deadline met). Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                reward += reward_change
                terminated = terminated or new_terminated
                truncated = truncated or new_truncated
        else:
            if postcondition_satisfied:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Primary postcondition '{postcondition_formula_kx}' SATISFIED (before deadline).")
                reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                    ensure_meta, 0.0, False, False, is_failure_case=False
                )
                if self.debug_mode:
                    print(f"DEBUG_STEP: Applied meta effect for ensure success (before deadline). Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                reward += reward_change
                terminated = terminated or new_terminated
                truncated = truncated or new_truncated
            else:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Primary postcondition '{postcondition_formula_kx}' NOT SATISFIED (deadline not met).")
                if ensure_meta.get('type') == 'progress' and 'shape' in ensure_meta:
                    # FIX: Use ensure_meta instead of metadata here
                    shape_value = self._evaluate_shape_expression(ensure_meta['shape'])
                    reward += shape_value
                    if self.debug_mode:
                        print(f"DEBUG_STEP: Applied ensure shape reward: {shape_value:.6f}")

        # Check secondary postcondition (finalSafety) if not already terminated/truncated
        if not (terminated or truncated) and len(self._problem_config.postconditions) > 1:
            final_safety_formula_kx = self._problem_config.postconditions[1].formula
            final_safety_satisfied = self._evaluate_expression(final_safety_formula_kx, is_constraint=True)
            final_safety_meta = self._get_predicate_metadata('finalSafety')
            if not final_safety_satisfied:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Secondary postcondition ('finalSafety') FAILED: '{final_safety_formula_kx}'.")
                reward_change, new_terminated, new_truncated = self._apply_meta_effect(
                    final_safety_meta, 0.0, False, False, is_failure_case=True
                )
                if self.debug_mode:
                    print(f"DEBUG_STEP: Applied meta effect for finalSafety failure. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                reward += reward_change
                terminated = terminated or new_terminated
                truncated = truncated or new_truncated
            else:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Secondary postcondition ('finalSafety') SATISFIED: '{final_safety_formula_kx}'.")
                # Apply on_success effects for finalSafety if not 'continue'
                if final_safety_meta.get('on_success') and final_safety_meta.get('on_success').get('action') != 'continue':
                    reward_change, new_terminated, new_truncated = self._apply_meta_effect(final_safety_meta, 0.0, False, False, is_failure_case=False)
                    if self.debug_mode:
                        print(f"DEBUG_STEP: Applied meta effect for finalSafety success. Reward change: {reward_change:.6f}, new_terminated: {new_terminated}, new_truncated: {new_truncated}")
                    reward += reward_change
                    terminated = terminated or new_terminated
                    truncated = truncated or new_truncated

        if self.current_step >= self.max_episode_steps:
            truncated = True
            if self.debug_mode:
                print(f"DEBUG_STEP: Max episode steps ({self.max_episode_steps}) reached. Truncating.")
            if not terminated and not postcondition_satisfied: # If not already terminated by success/failure, apply small penalty
                reward -= 10.0

        # Ensure the final state for the step is recorded if not already done by intermediate ODE points.
        # This covers cases where ODE integration was skipped or only had one point.
        if not (solution.success and solution.sol is not None and solution.t.size > 1 and actual_ode_duration > 1e-9):
            state_record_after_step = self.state.copy()
            state_record_after_step['episode_time'] = self.cumulative_episode_time
            self.state_history.append(state_record_after_step)
            if self.debug_mode:
                print(f"DEBUG_STEP: Appended final state for step {self.current_step} to history (after post-checks).")

        observation = self._get_obs()
        info = self._get_info()
        if self.debug_mode:
            print(f"DEBUG_STEP {self.current_step}: Step complete. Reward={reward:.6f}, Terminated={terminated}, Truncated={truncated}")
        return observation, reward, terminated, truncated, info

    def render(self, mode="human", plot_config: Optional[Union[List[str], Dict[str, str]]] = None):
        """
        Renders the environment.
        - mode="human": Renders to the screen (not implemented for this base class).
        - mode="plot_data": Returns the raw state history for external plotting.
        - mode="matplotlib": Generates and displays matplotlib plots of the state history.
          `plot_config` can be:
          - A list of strings: ['var1', 'var2'] - plots these variables using their DataFrame column names as labels.
          - A dictionary: {'Plot Label 1': 'df_col_name1', 'Plot Label 2': 'df_col_name2'} - plots df_col_name1 with 'Plot Label 1'.
          If `plot_config` is None, all numerical columns in the DataFrame (except 'episode_time') will be plotted.
        """
        if mode == "human":
            pass
        elif mode == "plot_data":
            return self.state_history
        elif mode == "matplotlib":
            self._plot_state_history(plot_config=plot_config, history_data=self.last_episode_history)
            return None
        return None

    def _plot_state_history(self, plot_config: Optional[Union[List[str], Dict[str, str]]] = None, history_data: Optional[List[Dict[str, Any]]] = None):
        """
        Generates and displays a matplotlib plot of the state history.
        `plot_config` can be:
        - A list of strings: ['var1', 'var2'] - plots these variables using their DataFrame column names as labels.
        - A dictionary: {'Plot Label 1': 'df_col_name1', 'Plot Label 2': 'df_col_name2'} - plots df_col_name1 with 'Plot Label 1'.
        If `plot_config` is None, all numerical columns in the DataFrame (except 'episode_time') will be plotted.
        """
        data_to_plot = history_data if history_data is not None else self.state_history

        if not data_to_plot:
            if self.debug_mode:
                print("DEBUG_PLOT: No state history available to plot.")
            return

        df = pd.DataFrame(data_to_plot)
        if self.debug_mode:
            print(f"DEBUG_PLOT: DataFrame shape: {df.shape}")
            print(f"DEBUG_PLOT: DataFrame head:\n{df.head()}")
            print(f"DEBUG_PLOT: DataFrame tail:\n{df.tail()}")
            print(f"DEBUG_PLOT: DataFrame columns: {df.columns.tolist()}")


        x_axis_key = 'episode_time'
        if x_axis_key not in df.columns:
            print(f"DEBUG_PLOT: Error: Time key '{x_axis_key}' not found in state history for plotting. This indicates an issue with state history recording.")
            return

        variables_to_plot_map: Dict[str, str] = {} # {plot_label: df_column_name}

        if plot_config is None:
            # Default behavior: plot all numerical columns except the x_axis_key
            for col_name in df.columns:
                if col_name != x_axis_key and pd.api.types.is_numeric_dtype(df[col_name]):
                    variables_to_plot_map[col_name] = col_name # Use df name as label
            if self.debug_mode and not variables_to_plot_map:
                print("DEBUG_PLOT: No numerical columns found in DataFrame for default plotting.")

        elif isinstance(plot_config, list):
            for var_name in plot_config:
                if var_name in df.columns:
                    variables_to_plot_map[var_name] = var_name
                else:
                    if self.debug_mode:
                        print(f"DEBUG_PLOT: Warning: Variable '{var_name}' requested for plotting but not found in DataFrame columns: {df.columns.tolist()}.")
        elif isinstance(plot_config, dict):
            for plot_label, df_col_name in plot_config.items():
                if df_col_name in df.columns:
                    variables_to_plot_map[plot_label] = df_col_name
                else:
                    if self.debug_mode:
                        print(f"DEBUG_PLOT: Warning: DataFrame column '{df_col_name}' (for plot label '{plot_label}') not found in DataFrame columns: {df.columns.tolist()}.")
        else:
            if self.debug_mode:
                print(f"DEBUG_PLOT: Invalid plot_config type: {type(plot_config)}. Expected list or dict.")
            return

        if not variables_to_plot_map:
            if self.debug_mode:
                print(f"DEBUG_PLOT: No valid variables to plot after processing plot_config.")
            return

        num_vars = len(variables_to_plot_map)
        cols = 2
        rows = (num_vars + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        fig.suptitle('Agent State Evolution Over Time', fontsize=16)

        min_y_range_absolute = 0.05

        # Iterate through the map to plot
        for i, (plot_label, df_col_name) in enumerate(variables_to_plot_map.items()):
            ax = axes[i]
            ax.plot(df[x_axis_key], df[df_col_name], label=plot_label)
            ax.set_xlabel(f"Time ({x_axis_key})")
            ax.set_ylabel(plot_label) # Use the plot_label here
            ax.set_title(f"Evolution of {plot_label}") # Use the plot_label here
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            min_time = df[x_axis_key].min()
            max_time = df[x_axis_key].max()
            if max_time <= min_time + 1e-9:
                ax.set_xlim(0, max_time + 0.01)
            else:
                ax.set_xlim(0, max_time)

            min_val = df[df_col_name].min()
            max_val = df[df_col_name].max()

            data_range = max_val - min_val

            if self.debug_mode:
                print(f"DEBUG_PLOT: Plotting {plot_label} (from DataFrame column '{df_col_name}'): min_val={min_val:.6f}, max_val={max_val:.6f}, data_range={data_range:.6f}")

            if data_range < min_y_range_absolute:
                avg_val = (min_val + max_val) / 2
                lower_bound = avg_val - min_y_range_absolute / 2
                upper_bound = avg_val + min_y_range_absolute / 2
                ax.set_ylim(lower_bound, upper_bound)
                if self.debug_mode:
                    print(f"  DEBUG_PLOT: Applied forced y-lim: [{lower_bound:.6f}, {upper_bound:.6f}] (range {min_y_range_absolute})")
            else:
                buffer = data_range * 0.1
                lower_bound = min_val - buffer
                upper_bound = max_val + buffer
                ax.set_ylim(min_val - buffer, max_val + buffer)
                if self.debug_mode:
                    print(f"  DEBUG_PLOT: Applied buffered y-lim: [{lower_bound:.6f}, {upper_bound:.6f}] (range {upper_bound - lower_bound:.6f})")

        # Hide any unused subplots
        for j in range(0, len(axes)):
            if j >= num_vars:
                fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def close(self):
        plt.close('all')
        pass