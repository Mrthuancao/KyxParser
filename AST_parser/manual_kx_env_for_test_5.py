import sys
import os
from typing import Optional

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__)) # e.g., d:\Reinforcement learning\kyx_parser\AST_parser

# Adjust project_root to point to 'd:\Reinforcement learning'
# Assuming manual_kx_env_for_test_4.py is in d:\Reinforcement learning\kyx_parser\AST_parser\
# So, two levels up from current_script_dir
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..')) # e.forcement learning

# Add project_root and current_script_dir to sys.path to ensure all modules are found
# This makes pyast_config discoverable if it's in either of these locations.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import re
import math
from typing import (Dict, Any, Tuple, List, Optional, Union)

# Import the base environment class (now discoverable via sys.path)
from gym_generator.kx_env_base import _KeYmaeraXEnvBase
# NEW: Import safe_div directly into this module's scope
from gym_generator.kx_env_base import safe_div

import z3

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
                                                    "ixr-xr < A/2*(D-ep)^2 & "
                                                    "T = (if (((xo - ixo) / Vmin) <= 0) then (((xo - ixo) / Vmin) - ep) else (-ep)) & " # MODIFIED: Replaced min with if-then-else for Z3 compatibility
                                                    "RAfterX(xr + A/2*(D - ep)^2)")),
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
                        parameters=[('Real', 'xr'), ('Real', 'vr'), ('Real', 'ixr'), ('Real', 'xo'), ('Real', 'vo'), ('Real', 'ixo'), ('Real', 'T')],
                        body=("0 <= vr & Vmin<=vo & T <= (xo-ixo)/Vmin & "
                              "( (T <= 0 & RAfterX(xr + A/2*(D - ep)^2)) | "
                              "(T > 0 & RAfterX(xr + vr*(max(0,D-T)) + A/2*(max(0,D-T))^2)) )"),
                        metadata={'type': 'progress', 'reward': '10', 'on_success': {'action': 'continue', 'status': 'none'}, 'on_failure': {'action': 'continue', 'status': 'none'}, 'shape': '-T'}
                    ),
                    PredicateDefinitionConfig(
                        name='finalSafety',
                        parameters=[('Real', 'xr'), ('Real', 'xo'), ('Real', 'T')],
                        body="( (xr=ixr -> xo!=ixo) & (T>=D -> RAfterX(xr)) )",
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
                                # Choice 0: ar := A; if OAfterX(xo)
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(formula='OAfterX(xo)'),
                                    DeterministicAssignmentActionConfig(variable='ar', value='A')
                                ]),
                                # Choice 1: ar := *; ?-b<=ar & ar<=A; if !OAfterX(xo) & RAfterX(xr)
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(formula='!OAfterX(xo) & RAfterX(xr)'),
                                    NondeterministicAssignmentActionConfig(variable='ar', bounds_formula='-b<=ar & ar<=A') # Agent provides ar
                                ]),
                                # Choice 2: ar := *; ?0<=ar & ar<=A; if !OAfterX(xo) & !RAfterX(xr) & (vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/vr ))
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(
                                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                                 '(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | '
                                                 'ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) ))')
                                    ),
                                    NondeterministicAssignmentActionConfig(variable='ar', bounds_formula='0<=ar & ar<=A') # Agent provides ar
                                ]),
                                # Choice 3: ar := 0; if !OAfterX(xo) & !RAfterX(xr) & !(...) & (vr>0 & ixo < xo + Vmin*(ixr-xr)/vr)
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(
                                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                                 '!(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                                 '(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr)')
                                    ),
                                    DeterministicAssignmentActionConfig(variable='ar', value='0')
                                ]),
                                # Choice 4: ar := -b; if !OAfterX(xo) & !RAfterX(xr) & !(...) & !(...) & vr != 0 & !(xr + stopDist(vr) + accComp(vr) < ixr)
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(
                                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                                 '!(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                                 '!(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr) & '
                                                 'vr != 0 & !(xr + stopDist(vr) + accComp(vr) < ixr)')
                                    ),
                                    DeterministicAssignmentActionConfig(variable='ar', value='-b')
                                ]),
                                # Choice 5: ar := 0; if !OAfterX(xo) & !RAfterX(xr) & !(...) & !(...) & vr = 0
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(
                                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                                 '!(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                                 '!(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr) & '
                                                 'vr = 0')
                                    ),
                                    DeterministicAssignmentActionConfig(variable='ar', value='0')
                                ]),
                                # Choice 6: ar := A; if !OAfterX(xo) & !RAfterX(xr) & !(...) & !(...) & xr + stopDist(vr) + accComp(vr) < ixr
                                SequenceActionConfig(actions=[
                                    ActionConstraintConfig(
                                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                                 '!(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                                 '!(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr) & '
                                                 'xr + stopDist(vr) + accComp(vr) < ixr')
                                    ),
                                    DeterministicAssignmentActionConfig(variable='ar', value='A')
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
                # MODIFIED: Flattened the top-level conjunction in the precondition formula
                # MODIFIED AGAIN: Replaced if-then-else with equivalent logical disjunction for Z3 compatibility
                precondition=PreconditionConfig(formula="A > 0 & b > 0 & ep > 0 & Vmin > 0 & D >= ep & vr = 0 & vo>=Vmin & xr < ixr & ixr-xr < A/2*(D-ep)^2 & (( ( (xo - ixo) / Vmin ) <= 0 ) & ( T = ( ( (xo - ixo) / Vmin ) - ep ) )) | ( !( ( (xo - ixo) / Vmin ) <= 0 ) & ( T = ( -ep ) ) ) & (xr + A/2*(D - ep)^2 > ixr)"),
                loop_execution=LoopExecutionConfig(
                    iterations='inf',
                    loop_invariant_formula='safetyloopinv(xr, vr, ixr, xo, vo, ixo) & deadlineloopinv(xr, vr, ixr, xo, vo, ixo, T)',
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

        # --- CRITICAL FIX: Explicitly set _state_keys and _variables_config.state_variables to the correct values ---
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
        # Re-assign the entire list of state_variables in the config object
        # Ensure self._variables_config points to the correct object before modifying it
        if self._variables_config is not self.config.variables:
            if self.debug_mode:
                print("DEBUG_GEN_ENV_INIT_WARNING: self._variables_config is not self.config.variables. Re-assigning.")
            self._variables_config = self.config.variables # Ensure it points to the config's variables

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
            ((ixr - xr) < (self.constants['A'] / 2 * ((self.constants['D'] - self.constants['ep']) ** 2))) and
            self._approx_eq(T, (np.minimum(0, safe_div(xo - ixo, self.constants['Vmin'])) - self.constants['ep'])) and # Using np.minimum for Python evaluation
            self.functions['RAfterX'](xr + (self.constants['A'] / 2 * ((self.constants['D'] - self.constants['ep']) ** 2)))
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
        self.predicates['deadlineloopinv'] = lambda xr, vr, ixr, xo, vo, ixo, T: (
            ((0) <= (vr)) and ((self.constants['Vmin']) <= (vo))
        ) and (
            ((T) <= (safe_div(xo - ixo, self.constants['Vmin']))) and
            ( # Corrected logical OR for deadlineloopinv
                ((T <= 0) and self.functions['RAfterX'](
                    ((xr) + ((self.constants['A']) / (2) * (((self.constants['D']) - (self.constants['ep'])) ** (2))))
                )) or
                ((T > 0) and self.functions['RAfterX'](
                    ((xr) + ((vr) * (max(0, (self.constants['D']) - (T)))) + ((self.constants['A']) / (2) * (((max(0, (self.constants['D']) - (T)))) ** (2))))
                ))
            )
        )

        # Refactored 'finalSafety' for improved readability
        self.predicates['finalSafety'] = lambda xr, xo, T: (
            # Corrected implication for finalSafety: (A -> B) is (not A) or B
            ( (not self._approx_eq(xr, self.constants['ixr'])) or (not self._approx_eq(xo, self.constants['ixo'])) ) and
            ( (not (T >= self.constants['D'])) or self.functions['RAfterX'](xr) )
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
        self.state = {key: 0.0 for key in self._state_keys} # This line is good and ensures all keys exist.

        low_bounds = np.array([-np.inf] * len(self._state_keys), dtype=np.float32)
        high_bounds = np.array([np.inf] * len(self._state_keys), dtype=np.float32)
        self.observation_space = spaces.Box(low_bounds, high_bounds, shape=(len(self._state_keys),), dtype=np.float32)

        # Action space for controlFunc:
        # 'choice_ChoiceLevel_0' for the 7 robot choices
        # 'assign_ar' for when ar is nondeterministic (choices 1 and 2)
        self.action_space = spaces.Dict({
                'choice_ChoiceLevel_0': spaces.Discrete(7), # 7 choices for robot's ar
                'assign_ar': spaces.Box(low=-10.0, high=10.0, shape=(), dtype=np.float32) # Max A is 10, max b is 5, so [-5, 10] for ar. Using [-10, 10] for safety.
                })
        self._action_space_keys = ['choice_ChoiceLevel_0', 'assign_ar']
        self._control_func_action_config = ChoiceActionConfig(
            level_name='ChoiceLevel_0',
            choices=[
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(formula='OAfterX(xo)'),
                    DeterministicAssignmentActionConfig(variable='ar', value='A')
                ]),
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(formula='!OAfterX(xo) & RAfterX(xr)'),
                    NondeterministicAssignmentActionConfig(variable='ar', bounds_formula='-b<=ar & ar<=A')
                ]),
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(
                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                 '(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | '
                                 'ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) ))')
                    ),
                    NondeterministicAssignmentActionConfig(variable='ar', bounds_formula='0<=ar & ar<=A')
                ]),
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(
                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                 '!(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                 '(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr)')
                    ),
                    DeterministicAssignmentActionConfig(variable='ar', value='0')
                ]),
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(
                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                 '!(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                 '!(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr) & '
                                 'vr != 0 & !(xr + stopDist(vr) + accComp(vr) < ixr)')
                    ),
                    DeterministicAssignmentActionConfig(variable='ar', value='-b')
                ]),
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(
                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                 '!(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                 '!(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr) & '
                                 'vr = 0')
                    ),
                    DeterministicAssignmentActionConfig(variable='ar', value='0')
                ]),
                SequenceActionConfig(actions=[
                    ActionConstraintConfig(
                        formula=('!OAfterX(xo) & !RAfterX(xr) & '
                                 '!(vr>0 & ( xo+vo*(ixr-xr)/vr+A/2*((ixr-xr)/vr)^2 < ixo | ixo < xo + Vmin*(ixr-xr)/(vr+A*ep) )) & '
                                 '!(vr>0 & ixo < xo + Vmin*(ixr-xr)/vr) & '
                                 'xr + stopDist(vr) + accComp(vr) < ixr')
                    ),
                    DeterministicAssignmentActionConfig(variable='ar', value='A')
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
        _ode_events_str_local = '''def event_t_max(t_val, y_array):
    return self.constants['ep'] - y_array[6] # t is at index 6
event_t_max.terminal = True
event_t_max.direction = -1

def event_vr_min(t_val, y_array):
    vr_val = y_array[1] # vr is at index 1
    ar_val = y_array[2] # ar is at index 2
    if vr_val <= 0 and ar_val <= 0:
        return vr_val
    return 1.0
event_vr_min.terminal = True
event_vr_min.direction = -1

def event_vo_min(t_val, y_array):
    vo_val = y_array[4] # vo is at index 4
    ao_val = y_array[5] # ao is at index 5
    if vo_val <= self.constants['Vmin'] and ao_val <= 0: # Check against Vmin, not 0
        return vo_val - self.constants['Vmin']
    return 1.0
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
            print(f"DEBUG_GEN_ENV_INIT_BEFORE_RESET_CALL: _state_keys just before calling self.reset(): {self._state_keys}, length: {len(self._state_keys)}")
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
        # Re-assign the entire list of state_variables in the config object
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


        # Call super().reset(), which internally calls GeneratedKXEnv._get_obs()
        if self.debug_mode:
            print(f"DEBUG_GEN_RESET_PRE_SUPER: Calling super().reset().")
        observation_from_super, info = super().reset(seed=seed, options=options)
        if self.debug_mode:
            print(f"DEBUG_GEN_RESET_POST_SUPER: super().reset() completed.")
            print(f"DEBUG_GEN_RESET_POST_SUPER: _state_keys after super().reset(): {self._state_keys}, len: {len(self._state_keys)}")
            print(f"DEBUG_GEN_RESET_POST_SUPER: id(self._state_keys) after super().reset(): {id(self._state_keys)}")
            print(f"DEBUG_GEN_RESET_POST_SUPER: self.state after super().reset(): {self.state.keys()}")
            print(f"DEBUG_GEN_RESET_POST_SUPER: Observation from super().reset() (should be from GeneratedKXEnv._get_obs): {observation_from_super}, length: {len(observation_from_super)}")

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

    def _split_top_level_and_and_local(self, formula: str) -> List[str]:
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
        loop_inv_conjuncts = self._split_top_level_and_and_local(expanded_loop_inv)

        loop_invariant_satisfied_pre_step = True
        for conj_str in loop_inv_conjuncts:
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', conj_str)
            pred_name = predicate_name_match.group(1) if predicate_name_match else None
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {}

            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)

            if not conjunct_result:
                loop_invariant_satisfied_pre_step = False
                if self.debug_mode:
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (pre-control): '{conj_str}'")
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                # Apply on_success effects for invariants if not 'continue'
                if metadata.get('on_success') and metadata.get('on_success').get('action') != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
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
            reward, terminated, truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-500.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                reward, terminated, truncated, is_failure_case=True
            )
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

        if ep_val <= 0:
            self.state['t'] = 0.0
            print(f"DEBUG_ODE: Warning: 'ep' constant is non-positive ({ep_val:.6f}). Skipping ODE integration and penalizing.")
            bounds_meta = self._get_predicate_metadata('bounds')
            reward, terminated, truncated = self._apply_meta_effect(
                bounds_meta or {'type': 'safety', 'penalty': '-200.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
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
                                print(f"DEBUG_ODE_APPEND: Appended intermediate state k={k}: xr={current_ode_state['xr']:.2f}, vr={current_ode_state['vr']:.2f}, ar={current_ode_state['ar']:.2f}, xo={current_ode_state['xo']:.2f}, vo={current_ode_state['vo']:.2f}, ao={current_ode_state['ao']:.2f}, T={current_ode_state['T']:.2f}, ep_time={intermediate_episode_time:.2f}")

                if solution.status == 0: # Integration successful, no events
                    final_y = solution.y[:, -1]
                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]
                elif solution.status == 1: # Event occurred
                    if solution.y_events and solution.y_events[0].size > 0:
                        final_y = solution.y_events[0][0]
                    else:
                        final_y = solution.y[:, -1]

                    for i, var_name in enumerate(self._ode_state_map.keys()):
                        self.state[var_name] = final_y[i]

                    # Check which event terminated it
                    if self.state.get('vr', 0.0) < -self.equality_tolerance:
                        if self.debug_mode:
                            print(f"DEBUG_ODE: Event: vr became negative ({self.state['vr']:.6f}).")
                        reward, terminated, truncated = self._apply_meta_effect(
                            {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                            reward, terminated, truncated, is_failure_case=True
                        )
                        ode_terminated_by_event = True
                    # Check for vo < Vmin
                    if self.state.get('vo', self.constants['Vmin']) < self.constants['Vmin'] - self.equality_tolerance:
                         if self.debug_mode:
                             print(f"DEBUG_ODE: Event: vo became less than Vmin ({self.state['vo']:.6f} < {self.constants['Vmin']:.6f}).")
                         reward, terminated, truncated = self._apply_meta_effect(
                             {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                             reward, terminated, truncated, is_failure_case=True
                         )
                         ode_terminated_by_event = True
                else: # Integration failed for other reasons
                    if self.debug_mode:
                        print(f"DEBUG_ODE: ODE integration failed with status {solution.status}.")
                    reward, terminated, truncated = self._apply_meta_effect(
                        {'type': 'safety', 'penalty': '-150.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                        reward, terminated, truncated, is_failure_case=True
                    )
                    ode_terminated_by_event = True

            except Exception as e:
                if self.debug_mode:
                    print(f"DEBUG_ODE: Exception during ODE integration: {e}")
                reward, terminated, truncated = self._apply_meta_effect(
                    {'type': 'safety', 'penalty': '-200.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                    reward, terminated, truncated, is_failure_case=True
                )
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
            if not (solution.success and solution.sol is not None and solution.t.size > 1 and actual_ode_duration > 1e-9):
                state_record_after_step = self.state.copy()
                state_record_after_step['episode_time'] = self.cumulative_episode_time
                self.state_history.append(state_record_after_step)
            return self._get_obs(), reward, terminated, truncated, info

        # --- Post-ODE loop invariant check ---
        loop_invariant_satisfied_post_ode = True
        for conj_str in loop_inv_conjuncts:
            predicate_name_match = re.match(r'^\s*(\w+)\s*\(', conj_str)
            pred_name = predicate_name_match.group(1) if predicate_name_match else {}
            metadata = self._get_predicate_metadata(pred_name) if pred_name else {}

            conjunct_result = self._evaluate_expression(conj_str, is_constraint=True)

            if not conjunct_result:
                loop_invariant_satisfied_post_ode = False
                if self.debug_mode:
                    print(f"DEBUG_STEP: Loop invariant conjunct FAILED (post-ODE): '{conj_str}'")
                reward, terminated, truncated = self._apply_meta_effect(
                    metadata or {'type': 'safety', 'penalty': '-100.0', 'on_failure': {'action': 'truncate', 'status': 'failure'}},
                    reward, terminated, truncated, is_failure_case=True
                )
                if terminated or truncated:
                    state_record_before_early_return = self.state.copy()
                    state_record_before_early_return['episode_time'] = self.cumulative_episode_time
                    self.state_history.append(state_record_before_early_return)
                    return self._get_obs(), reward, terminated, truncated, info
            else:
                # Apply on_success effects for invariants if not 'continue'
                if metadata.get('on_success') and metadata.get('on_success').get('action') != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(metadata, reward, terminated, truncated, is_failure_case=False)
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
            reward, terminated, truncated = self._apply_meta_effect(
                {'type': 'safety', 'penalty': '-300.0', 'on_failure': {'action': 'terminate', 'status': 'failure'}},
                reward, terminated, truncated, is_failure_case=True
            )
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
                reward, terminated, truncated = self._apply_meta_effect(
                    ensure_meta, reward, terminated, truncated, is_failure_case=True
                )
            else:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Deadline met and primary postcondition '{postcondition_formula_kx}' SATISFIED. Success.")
                reward, terminated, truncated = self._apply_meta_effect(
                    ensure_meta, reward, terminated, truncated, is_failure_case=False
                )
        else:
            if postcondition_satisfied:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Primary postcondition '{postcondition_formula_kx}' SATISFIED (before deadline).")
                reward, terminated, truncated = self._apply_meta_effect(
                    ensure_meta, reward, terminated, truncated, is_failure_case=False
                )
            else:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Primary postcondition '{postcondition_formula_kx}' NOT SATISFIED (deadline not met).")
                if ensure_meta.get('type') == 'progress' and 'shape' in ensure_meta:
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
                reward, terminated, truncated = self._apply_meta_effect(
                    final_safety_meta, reward, terminated, truncated, is_failure_case=True
                )
            else:
                if self.debug_mode:
                    print(f"DEBUG_STEP: Secondary postcondition ('finalSafety') SATISFIED: '{final_safety_formula_kx}'.")
                # Apply on_success effects for finalSafety if not 'continue'
                if final_safety_meta.get('on_success') and final_safety_meta.get('on_success').get('action') != 'continue':
                    reward, terminated, truncated = self._apply_meta_effect(final_safety_meta, reward, terminated, truncated, is_failure_case=False)


        if self.current_step >= self.max_episode_steps:
            truncated = True
            if self.debug_mode:
                print(f"DEBUG_STEP: Max episode steps ({self.max_episode_steps}) reached. Truncating.")
            if not terminated and not postcondition_satisfied: # If not already terminated by success/failure, apply small penalty
                reward -= 10.0

        if not (solution.success and solution.sol is not None and solution.t.size > 1 and actual_ode_duration > 1e-9):
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