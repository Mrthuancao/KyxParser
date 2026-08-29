# environment_template_code.py

ENVIRONMENT_TEMPLATE_CODE = """
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from scipy.integrate import solve_ivp
import sympy
from collections import OrderedDict
import copy
import re
import math # Added for eval() context
from typing import Dict, Any

# Configuration for the environment, generated from the KeYmaera model
_ENV_CONFIG_JSON = {embedded_config_json_str}

class KeYmaeraGymEnv(gym.Env):
    metadata = {metadata}

    def __init__(self, render_mode=None, config_override=None): # <--- FIX IS HERE
        super().__init__()
        self.config = self._load_config(_ENV_CONFIG_JSON, config_override)
        
        self.debug_mode = self.config.get('debug_mode', False)
        
        # DEBUG PRINTS START
        self._debug_print(f"Initialized self.config['variables']: {{self.config['variables']}}")
        self._debug_print(f"Initialized self.config['constants']: {{self.config['constants']}}")
        # DEBUG PRINTS END

        self.environment_name = self.config['environment_name']
        self.render_mode = render_mode
        self.current_state = OrderedDict()
        self.time = 0.0 # Time elapsed within current control step (t in dL)
        self.global_time_remaining = 0.0 # T in dL
        self.current_step_count = 0 # Number of control steps taken
        self.max_episode_steps = self.config.get('max_episode_steps', 200)
        
        # Default time_step, will be overridden by 'ep' constant if present
        self.time_step = self.config.get('time_step', 0.01) 
        
        self.ode_solver_method = self.config.get('ode_solver_method', 'dopri5')
        self.ode_solver_step_size = self.config.get('ode_solver_step_size', 0.001)
        self.ode_solver_tolerance = self.config.get('ode_solver_tolerance', 1e-6)

        # SymPy symbols for dynamic evaluation of ODEs
        self.sympy_symbols = {{}}
        # Variables and constants need to be SymPy Symbols for ODE parsing
        for var_config in self.config['variables']:
            self._debug_print(f"DEBUG: Initializing sympy_symbols with variable: {{var_config['name']}}")
            self.sympy_symbols[var_config['name']] = sympy.Symbol(var_config['name'])
        for const_config in self.config['constants']:
            print(f"DEBUG: Initializing sympy_symbols with constant: {{const_config['name']}}")
            self.sympy_symbols[const_config['name']] = sympy.Symbol(const_config['name'])

        # Initialize user_defined_functions before _setup_constants calls _evaluate_expression
        self.user_defined_functions = {{}} 

        self._setup_constants()
        self._setup_functions() 

        ep_value = None
        # Check if 'ep' is a constant
        ep_const_config = next((c for c in self.config['constants'] if c['name'] == 'ep'), None)
        if ep_const_config:
            # If it's a constant, its value is already in self.current_state from _setup_constants
            ep_value = self.current_state['ep'] 
        else: # Check if 'ep' is a function without arguments
            ep_func_config = next((f for f in self.config['function_definitions'] if f['name'] == 'ep' and not f['args']), None)
            if ep_func_config:
                ep_value = self.user_defined_functions['ep']() # Call the generated _ep_impl method

        if ep_value is not None:
            self.time_step = ep_value

        self._setup_spaces()
        self._setup_initial_state_samplers()
        self._setup_hybrid_program()
        
    def _debug_print(self, message):
        if self.debug_mode:
            print(f"DEBUG: {{message}}")

    # User-defined functions from KeYmaera model
{user_defined_methods_code_str}

    def _load_config(self, config_json_str, config_override):
        config = json.loads(config_json_str)
        if config_override:
            for key, override_value in config_override.items():
                if key == 'constants' and isinstance(override_value, list):
                    # Merge constants list by name
                    existing_constants_map = {{c['name']: c for c in config['constants']}}
                    for ov_const in override_value:
                        if ov_const['name'] in existing_constants_map:
                            existing_constants_map[ov_const['name']].update(ov_const)
                        else:
                            existing_constants_map[ov_const['name']] = ov_const # Add new constant
                    config['constants'] = list(existing_constants_map.values())
                elif key == 'variables' and isinstance(override_value, list):
                    # Merge variables list by name
                    existing_variables_map = {{v['name']: v for v in config['variables']}}
                    for ov_var in override_value:
                        if ov_var['name'] in existing_variables_map:
                            existing_variables_map[ov_var['name']].update(ov_var)
                        else:
                            existing_variables_map[ov_var['name']] = ov_var # Add new variable
                    config['variables'] = list(existing_variables_map.values())
                elif key in config and isinstance(config[key], dict) and isinstance(override_value, dict):
                    # For nested dictionaries, update them
                    config[key].update(override_value)
                else:
                    # For other keys, just replace the value
                    config[key] = override_value
        return config

    def _setup_spaces(self):
        action_space_type = self.config.get('action_space_type', 'continuous')
        action_bounds = self.config.get('action_space_bounds', {{}})

        if action_space_type == 'continuous':
            lows = []
            highs = []
            self.action_variable_names = []
            for var_name, bounds in action_bounds.items():
                self.action_variable_names.append(var_name)
                lows.append(self._evaluate_expression(bounds['min'], {{}}))
                highs.append(self._evaluate_expression(bounds['max'], {{}}))
            self.action_space = spaces.Box(low=np.array(lows, dtype=np.float32),
                                           high=np.array(highs, dtype=np.float32),
                                           dtype=np.float32)
        elif action_space_type == 'discrete':
            n = action_bounds.get('n', 1)
            self.action_space = spaces.Discrete(n)
            self.action_variable_names = []
        else:
            raise ValueError(f"Unsupported action space type: {{action_space_type}}")

        observation_space_type = self.config.get('observation_space_type', 'continuous')
        observation_bounds = self.config.get('observation_space_bounds', {{}})

        if observation_space_type == 'continuous':
            lows = []
            highs = []
            self.observation_variable_names = []
            for var_name, bounds in observation_bounds.items():
                self.observation_variable_names.append(var_name)
                lows.append(self._evaluate_expression(bounds['min'], {{}}))
                highs.append(self._evaluate_expression(bounds['max'], {{}}))
            self.observation_space = spaces.Box(low=np.array(lows, dtype=np.float32),
                                                high=np.array(highs, dtype=np.float32),
                                                dtype=np.float32)
        elif observation_space_type == 'discrete':
            raise NotImplementedError("Discrete observation space not yet supported.")
        else:
            raise ValueError(f"Unsupported observation space type: {{observation_space_type}}")

    def _setup_initial_state_samplers(self):
        self.initial_state_samplers = {{}}
        for var_name, sampling_config in self.config['initial_state_sampling_config'].items():
            if 'value' in sampling_config:
                val = self._evaluate_expression(sampling_config['value'], {{}})
                self.initial_state_samplers[var_name] = lambda v=val: v
            elif 'min' in sampling_config and 'max' in sampling_config:
                min_val = self._evaluate_expression(sampling_config['min'], {{}})
                max_val = self._evaluate_expression(sampling_config['max'], {{}})
                self.initial_state_samplers[var_name] = lambda mn=min_val, mx=max_val: self.np_random.uniform(mn, mx)
            else:
                self.initial_state_samplers[var_name] = lambda: 0.0

    def _setup_constants(self):
        for const_config in self.config['constants']:
            self._debug_print(f"DEBUG: _setup_constants evaluating constant: {{const_config['name']}}")
            self.current_state[const_config['name']] = self._evaluate_expression(const_config['value'], {{}})
            # Ensure constants are also SymPy symbols if they are used in ODEs or other symbolic expressions
            if const_config['name'] not in self.sympy_symbols:
                self._debug_print(f"DEBUG: Adding constant to sympy_symbols (late): {{const_config['name']}}")
                self.sympy_symbols[const_config['name']] = sympy.Symbol(const_config['name'])


    def _setup_functions(self):
        self.user_defined_functions = {{}}
{user_defined_function_assignments_str}


    def _setup_hybrid_program(self):
        ode_program_str = self.config['problem']['program']
        self.differential_variables = []
        self.differential_equations = {{}}
        self.differential_domain_constraint = None

        self._debug_print(f"DEBUG: _setup_hybrid_program: ode_program_str = '{{ode_program_str}}'")

        if ode_program_str.startswith('(') and ode_program_str.endswith(')'):
            ode_program_str = ode_program_str[1:-1]

        self._debug_print(f"DEBUG: _setup_hybrid_program: ode_program_str after paren removal = '{{ode_program_str}}'")

        # Split using the new distinct separator
        parts = ode_program_str.split('@@@DOMAIN_CONSTRAINT@@@', 1)

        self._debug_print(f"DEBUG: _setup_hybrid_program: parts = {{parts}}")

        equations_str = parts[0].strip() # Ensure no leading/trailing whitespace
        if len(parts) > 1:
            self.differential_domain_constraint = parts[1].strip()

        self._debug_print(f"DEBUG: _setup_hybrid_program: equations_str = '{{equations_str}}'")
        self._debug_print(f"DEBUG: _setup_hybrid_program: differential_domain_constraint = '{{self.differential_domain_constraint}}'")

        equation_parts = [p.strip() for p in equations_str.split(',') if p.strip()]
        self._debug_print(f"DEBUG: _setup_hybrid_program: equation_parts (filtered) = {{equation_parts}}")

        # Prepare sympy_locals for ODE parsing: symbols + symbolic functions
        ode_sympy_locals = {{k: v for k, v in self.sympy_symbols.items()}}
        for func_name in self.user_defined_functions: # Iterate over names of user-defined functions
            ode_sympy_locals[func_name] = sympy.Function(func_name) # Map to symbolic SymPy function

        for eq_part in equation_parts:
            if "'" in eq_part and '=' in eq_part:
                var_name_prime, expr = eq_part.split('=', 1)
                var_name = var_name_prime.replace("'", "").strip()
                expr = expr.strip()
                # Use ode_sympy_locals for sympify of ODE expressions
                self.differential_equations[var_name] = sympy.sympify(expr, locals=ode_sympy_locals)
                self.differential_variables.append(var_name)
            else:
                raise ValueError(f"Malformed ODE equation in program string: {{eq_part}}")

        for var in self.differential_variables:
            if var not in self.sympy_symbols:
                self.sympy_symbols[var] = sympy.Symbol(var)

    def _evaluate_expression(self, expression_str: str, context: Dict[str, Any]):
        self._debug_print(f"DEBUG: _evaluate_expression called for: {{expression_str}} (type: {{type(expression_str)}})")

        # Prepare the execution environment for eval()
        eval_globals = {{
            '__builtins__': None, # Restrict builtins for safety
            'True': True, 'False': False, 'None': None,
            'self': self, # Allow access to self for calling methods
            'np': np, # Allow numpy functions
            'math': math, # Allow math functions (e.g., for power `**`)
            'float': float, 'int': int, 'bool': bool,
            # Add sympy objects explicitly if any sympy constructs are still generated for eval
            'sympy': sympy,
            'Symbol': sympy.Symbol, 'Eq': sympy.Eq, 'Ne': sympy.Ne,
            'Gt': sympy.Gt, 'Lt': sympy.Lt, 'Ge': sympy.Ge, 'Le': sympy.Le,
            'And': sympy.And, 'Or': sympy.Or, 'Not': sympy.Not,
            'Implies': sympy.Implies, 'Equivalent': sympy.Equivalent,
            'Function': sympy.Function, # Needed if symbolic functions are passed to eval
        }}
        
        # Add current state variables and constants to locals
        eval_locals = {{}}
        eval_locals.update(self.current_state)
        eval_locals.update(context) # Context variables (e.g., function arguments)

        try:
            result = eval(expression_str, eval_globals, eval_locals)
            
            # Convert SymPy boolean results to Python booleans
            if isinstance(result, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
                return bool(result)
            # Convert SymPy numbers to Python floats
            if isinstance(result, sympy.Number):
                return float(result)
            # Ensure Python booleans are returned as bool
            if isinstance(result, (bool, np.bool_)):
                return bool(result)
            # Ensure Python numbers are returned as float
            if isinstance(result, (int, np.integer, np.floating)):
                return float(result)

            return result
        except NameError as ne:
            missing_name = str(ne).split("'")[1]
            raise RuntimeError(f"NameError: '{{missing_name}}' not found when evaluating expression '{{expression_str}}'. "
                               f"Available locals: {{list(eval_locals.keys())}}. Available globals: {{list(eval_globals.keys())}}") from ne
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression '{{expression_str}}': {{e}}")

    def _evaluate_sympy_function_calls(self, sympy_expr_node, current_subs):
        # Helper function to evaluate symbolic function calls within a SymPy expression.
        # This is necessary because user-defined functions are Python callables, not SymPy's built-in functions.
        # Base case 1: If it's a number (sympy.Float or sympy.Integer), return its Python float value.
        # This handles direct numerical constants.
        if isinstance(sympy_expr_node, (sympy.Float, sympy.Integer)):
            return float(sympy_expr_node)

        # Base case 2: If it's a simple sympy.Symbol, substitute its value directly.
        # This must come BEFORE the `hasattr(sympy_expr_node, 'args')` check to prevent the Symbol() TypeError.
        if isinstance(sympy_expr_node, sympy.Symbol):
            # If the symbol has a numerical value in the substitution dictionary, use it.
            if sympy_expr_node in current_subs:
                return current_subs[sympy_expr_node]
            # If not in subs_dict, it means it's a symbol that should remain symbolic.
            # In contexts like ODE evaluation, all variables should ideally be numerical at this point.
            # Returning the symbol itself might lead to further errors if the solver expects only numbers.
            return sympy_expr_node

        # If it's a SymPy Function and its name matches a user-defined function
        if isinstance(sympy_expr_node, sympy.Function) and str(sympy_expr_node.func) in self.user_defined_functions:
            func_name = str(sympy_expr_node.func)
            # Recursively evaluate arguments before calling the Python function
            evaluated_args = [self._evaluate_sympy_function_calls(arg, current_subs) for arg in sympy_expr_node.args]
            # Call the actual Python implementation method (which uses _evaluate_expression)
            # self.user_defined_functions stores the bound methods, so no need to pass 'self' explicitly.
            return self.user_defined_functions[func_name](*evaluated_args) 
        
        # Recursively apply to arguments of other SymPy expressions (e.g., Add, Mul, Pow, etc.)
        # This check should come after handling specific base types like Symbol, Float, Integer.
        if hasattr(sympy_expr_node, 'args'): # Check if it's a SymPy expression with arguments
            # Create a new expression with evaluated arguments
            return sympy_expr_node.func(*[self._evaluate_sympy_function_calls(arg, current_subs) for arg in sympy_expr_node.args])
        
        # Fallback: If it's none of the above (e.g., a Python literal like an int/float that wasn't part of a SymPy expression,
        # or a SymPy object that doesn't have args and wasn't caught by earlier checks like sympy.Number),
        # attempt a final substitution or just return the node.
        # The original `sympy_expr_node.subs(current_subs)` is generally robust for SymPy objects.
        try:
            return sympy_expr_node.subs(current_subs)
        except AttributeError:
            # If it's a pure Python object that doesn't have a .subs method, return it as is.
            return sympy_expr_node

    def _get_ode_system(self, t_ode, y):
        # Map y (from solver) back to named variables
        state_map = {{var: y[i] for i, var in enumerate(self.differential_variables)}}
        
        # Update current_state for evaluation context, but only for ODE variables
        # The full current_state is updated after the ODE step
        temp_state = copy.copy(self.current_state)
        temp_state.update(state_map)
        
        # Add current ODE solver time 't_ode' to context for ODEs that depend on it
        # This 't_ode' corresponds to the dL variable 't' (time elapsed in current control step)
        temp_state['t'] = t_ode 

        # Helper function to evaluate symbolic function calls within a SymPy expression
        # This is necessary because user-defined functions are Python callables, not SymPy's built-in functions.
        def evaluate_sympy_function_calls_for_ode(sympy_expr_node, current_subs):
            # If it's a SymPy Function and its name matches a user-defined function
            if isinstance(sympy_expr_node, sympy.Function) and str(sympy_expr_node.func) in self.user_defined_functions:
                func_name = str(sympy_expr_node.func)
                # Recursively evaluate arguments before calling the Python function
                evaluated_args = [evaluate_sympy_function_calls_for_ode(arg, current_subs) for arg in sympy_expr_node.args]
                # Call the actual Python implementation method (which uses _evaluate_expression)
                # self.user_defined_functions stores the bound methods, so no need to pass 'self' explicitly.
                return self.user_defined_functions[func_name](*evaluated_args) 
            
            # Recursively apply to arguments of other SymPy expressions
            if hasattr(sympy_expr_node, 'args'): # Check if it's a SymPy expression with arguments
                # Create a new expression with evaluated arguments
                return sympy_expr_node.func(*[evaluate_sympy_function_calls_for_ode(arg, current_subs) for arg in sympy_expr_node.args])
            
            # If it's a simple symbol or literal, substitute its value
            return sympy_expr_node.subs(current_subs)

        # Evaluate differential equations
        derivs = []
        for var_name in self.differential_variables:
            expr = self.differential_equations[var_name] # This is already a SymPy expression
            
            # Create a substitution dictionary from temp_state and sympy_symbols
            subs_dict = {{self.sympy_symbols[k]: v for k, v in temp_state.items() if k in self.sympy_symbols}}
            
            # Evaluate the ODE expression, handling symbolic function calls
            evaluated_ode_expr = self._evaluate_sympy_function_calls(expr, subs_dict)
            
            # The result should now be a numerical value or a simple SymPy expression
            # that can be converted to float.
            derivs.append(float(evaluated_ode_expr))
        return derivs

    def _run_continuous_dynamics(self, action):
        # Apply action to relevant variables before ODE integration
        for i, var_name in enumerate(self.action_variable_names):
            self.current_state[var_name] = action[i]

        # Set dL 't' (time elapsed in current control step) to 0 at start of each step
        # This is implicitly handled by the ODE solver's t_span starting from 0.
        # The ODE system itself updates 't' from 0 to self.time_step.
        # self.current_state['t'] will be updated by the ODE solver.
        
        # Update dL 'T' (global time remaining) is handled by ODE `T' = -1`

        # Prepare initial state for ODE solver
        y0 = [self.current_state[var] for var in self.differential_variables]
        
        t_span = (0, self.time_step) # Integrate over the control time step 'ep'

        # Solve ODEs
        sol = solve_ivp(self._get_ode_system, t_span, y0,
                        method=self.ode_solver_method,
                        max_step=self.ode_solver_step_size,
                        rtol=self.ode_solver_tolerance,
                        atol=self.ode_solver_tolerance)

        if sol.status == 0: # Success
            # Update state with the last point from the solver
            for i, var_name in enumerate(self.differential_variables):
                self.current_state[var_name] = sol.y[i, -1]
            
            # Update dL 't' to reflect the end of the control step
            self.current_state['t'] = sol.t[-1] # Should be close to self.time_step

            # Check differential domain constraint
            if self.differential_domain_constraint:
                # The domain constraint is a SymPy expression string, needs SymPy evaluation
                # Prepare locals for SymPy eval
                ode_sympy_locals = {{k: v for k, v in self.sympy_symbols.items()}}
                ode_sympy_locals['sympy'] = sympy 
                ode_sympy_locals['And'] = sympy.And # Explicitly add common sympy functions
                ode_sympy_locals['Le'] = sympy.Le
                ode_sympy_locals['Ge'] = sympy.Ge
                ode_sympy_locals['Float'] = sympy.Float
                for func_name in self.user_defined_functions:
                    ode_sympy_locals[func_name] = sympy.Function(func_name)
                
                # Sympy-fy the constraint string
                constraint_sympy_expr = sympy.sympify(self.differential_domain_constraint, locals=ode_sympy_locals)

                # Create substitution dictionary for concrete values
                concrete_subs = {{self.sympy_symbols[k]: v for k, v in self.current_state.items() if k in self.sympy_symbols}}

                # Helper to evaluate symbolic function calls within the SymPy expression
                def evaluate_sympy_function_calls_for_constraint(sympy_expr_node, current_subs):
                    if isinstance(sympy_expr_node, sympy.Function) and str(sympy_expr_node.func) in self.user_defined_functions:
                        func_name = str(sympy_expr_node.func)
                        evaluated_args = [evaluate_sympy_function_calls_for_constraint(arg, current_subs) for arg in sympy_expr_node.args]
                        # self.user_defined_functions stores the bound methods, so no need to pass 'self' explicitly.
                        return self.user_defined_functions[func_name](*evaluated_args) 
                    if hasattr(sympy_expr_node, 'args'):
                        return sympy_expr_node.func(*[evaluate_sympy_function_calls_for_constraint(arg, current_subs) for arg in sympy_expr_node.args])
                    return sympy_expr_node.subs(current_subs)

                # Evaluate the constraint
                constraint_met = self._evaluate_sympy_function_calls(constraint_sympy_expr, concrete_subs)

                if not bool(constraint_met): # Convert SymPy boolean to Python bool
                    print(f"Differential domain constraint violated: {{self.differential_domain_constraint}}")
                    self._debug_print(f"DEBUG: _run_continuous_dynamics returning True due to domain violation.")
                    return True # Indicate episode should terminate due to violation
            self._debug_print(f"DEBUG: _run_continuous_dynamics returning False (solver success, no domain violation).")
            return False # Indicate episode continues
        else:
            print(f"ODE solver failed: {{sol.message}}")
            self._debug_print(f"DEBUG: _run_continuous_dynamics returning True due to solver failure.")
            return True # Indicate episode should terminate


    def step(self, action):
        terminated = False
        truncated = False

        # Apply action to relevant variables before ODE integration
        # This is already done in _run_continuous_dynamics, but let's ensure action_to_apply is correctly set.
        action_to_apply = action

        # Run continuous dynamics (ODE integration)
        solver_failed = self._run_continuous_dynamics(action_to_apply)
        self._debug_print(f"DEBUG: step method: solver_failed = {{solver_failed}}")
        if solver_failed:
            terminated = True # Terminate if ODE solver fails or domain constraint violated

        # Evaluate reward
        reward_expr = self.config.get('reward_function_body', '0.0')
        self._debug_print(f"DEBUG: step method: reward_expr before evaluation: '{{reward_expr}}' (type: {{type(reward_expr)}})")
        reward = self._evaluate_expression(reward_expr, self.current_state)

        # Evaluate termination condition
        termination_expr = self.config.get('termination_condition_body', 'False')
        self._debug_print(f"DEBUG: step method: termination_expr for evaluation: '{{termination_expr}}'")
        if not terminated: # Only evaluate if not already terminated by solver failure
            evaluated_termination_cond = self._evaluate_expression(termination_expr, self.current_state)
            self._debug_print(f"DEBUG: step method: evaluated_termination_cond = {{evaluated_termination_cond}}")
            terminated = bool(evaluated_termination_cond)
        self._debug_print(f"DEBUG: step method: final terminated status = {{terminated}}")

        # Check for truncation (max steps or global time)
        self.current_step_count += 1
        if self.current_step_count >= self.max_episode_steps:
            truncated = True
        
        # Update global_time_remaining from current_state['T'] after ODE step.
        self.global_time_remaining = self.current_state.get('T', 0.0)

        # Get observation
        observation = self._get_observation()

        # Render (if applicable)
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        self.current_state = OrderedDict()

        # Re-apply constants (already done in __init__, but good to ensure for reset)
        self._setup_constants()

        # Ensure functions are set up for _evaluate_expression calls during reset
        self._setup_functions() # ADDED THIS CALL HERE FOR RESET

        # Initialize variables based on sampling config
        for var_config in self.config['variables']:
            var_name = var_config['name']
            if var_name in self.initial_state_samplers:
                self.current_state[var_name] = self.initial_state_samplers[var_name]()
            else:
                self.current_state[var_name] = 0.0 # Default if no sampler

        # Ensure 'ep' (time_step) is correctly set from constants if available
        ep_value = None
        ep_const_config = next((c for c in self.config['constants'] if c['name'] == 'ep'), None)
        if ep_const_config:
            ep_value = self.current_state['ep'] # Value already evaluated and stored in current_state
        else: # Check if 'ep' is a function without arguments
            ep_func_config = next((f for f in self.config['function_definitions'] if f['name'] == 'ep' and not f['args']), None)
            if ep_func_config:
                ep_value = self.user_defined_functions['ep']() # Call the generated _ep_impl method

        if ep_value is not None:
            self.time_step = ep_value
        
        # Initialize global_time_remaining (T) from current_state
        if 'T' in self.current_state:
            self.global_time_remaining = self.current_state['T']
        else:
            # Fallback if T is not explicitly sampled or set in initial_state_sampling_config
            self.global_time_remaining = 10.0 # Default value if not found

        # Initialize 't' (time elapsed in current control step)
        self.current_state['t'] = 0.0

        # Evaluate initial_condition to ensure initial state is valid
        initial_cond_expr = self.config['problem']['initial_condition']
        initial_cond_met = self._evaluate_expression(initial_cond_expr, self.current_state)
        if not initial_cond_met:
            print(f"Warning: Initial state does not satisfy initial condition: {{initial_cond_expr}}. "
                  f"Current state: {{self.current_state}}")
            # In a real environment, you might want to resample or raise an error here.
            # For now, we proceed, but this might lead to immediate termination or bad behavior.

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_observation(self):
        observation_values = [self.current_state.get(name, 0.0) for name in self.observation_variable_names]
        return np.array(observation_values, dtype=np.float32)

    def _get_info(self):
        return {{
            "time": self.current_state.get('t', 0.0), # Report 't' from current_state
            "global_time_remaining": self.current_state.get('T', 0.0), # Report 'T' from current_state
            "state": dict(self.current_state)
        }}

    def _render_frame(self):
        # This method is called by reset() and step() for human rendering,
        # and by render() for both human and rgb_array.
        if self.render_mode == "human":
            print(f"Step {{self.current_step_count:3d}} | t: {{self.current_state.get('t', 0.0):.2f}} | T: {{self.current_state.get('T', 0.0):.2f}} | "
                  f"xr: {{self.current_state.get('xr', 0.0):.2f}}, vr: {{self.current_state.get('vr', 0.0):.2f}}, ar: {{self.current_state.get('ar', 0.0):.2f}} | "
                  f"xg: {{self.current_state.get('xg', 0.0):.2f}}")
        elif self.render_mode == "rgb_array":
            # For 'rgb_array' mode, return a dummy image for now
            # Actual implementation would involve drawing the environment state
            return np.zeros((100, 100, 3), dtype=np.uint8)
        # If render_mode is None or unsupported, do nothing.

    def render(self):
        # The public render method just calls the internal _render_frame
        # which handles the different modes.
        return self._render_frame()

    def close(self):
        pass
"""