# environment_template_code.py

ENVIRONMENT_TEMPLATE_CODE = """
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from scipy.integrate import solve_ivp
import sympy
import z3
from collections import OrderedDict, deque
import copy
import re
import math
import ast
from typing import Dict, Any, Optional, Union, List

from ast_config import (
    ASTNode, Term, Formula,
    Literal, VarRef, BinaryTermOp, UnaryTermOp, FunctionCall,
    BooleanLiteral, BinaryFormulaOp, UnaryFormulaOp, Comparison, PredicateCall,
    Assignment, NondetAssignment, Test, Sequence, Choice, Iteration,
    ODE, ODEEquation, HPBlock, ModalFormula, Quantifier, C_Formula,
    HybridProgram
)

# Embedded environment config (generator will replace the placeholder)
_ENV_CONFIG_JSON = {embedded_config_json_str}


class KeYmaeraGymEnv(gym.Env):
    metadata = {metadata}

    def __init__(self, render_mode=None, config_override=None):
        super().__init__()
        self.config = self._load_config(_ENV_CONFIG_JSON, config_override)
        
        self.debug_mode = self.config.get('debug_mode', False)
        
        # DEBUG PRINTS START
        self._debug_print(f"Initialized self.config['variables']: {self.config['variables']}")
        self._debug_print(f"Initialized self.config['constants']: {self.config['constants']}")
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
        self.sympy_symbols = {}
        # Variables and constants need to be SymPy Symbols for ODE parsing
        for var_config in self.config['variables']:
            self._debug_print(f"DEBUG: Initializing sympy_symbols with variable: {var_config['name']}")
            self.sympy_symbols[var_config['name']] = sympy.Symbol(var_config['name'])
        for const_config in self.config['constants']:
            self._debug_print(f"DEBUG: Initializing sympy_symbols with constant: {const_config['name']}")
            self.sympy_symbols[const_config['name']] = sympy.Symbol(const_config['name'])

        # Initialize user_defined_functions before _setup_constants calls _evaluate_expression
        self.user_defined_functions = {} 

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

        self.control_flow_program_ast: Optional[HybridProgram] = None
        if self.config.get('control_flow_program_ast_json'):
            self.control_flow_program_ast = self._deserialize_ast_node(self.config['control_flow_program_ast_json'])
            self._debug_print(f"DEBUG: Deserialized control_flow_program_ast: {{self.control_flow_program_ast._repr_indented(0)}}")

        self._setup_spaces()
        self._setup_initial_state_samplers()
        self._setup_hybrid_program()
        
        # Initial State Buffer (ISB) setup
        self.initial_state_buffer_config = self.config.get('initial_state_buffer_config', {})
        self.initial_state_buffer_strategy = self.initial_state_buffer_config.get('strategy', 'random')
        self.initial_state_buffer = deque(maxlen=self.initial_state_buffer_config.get('capacity', 100))
        
        if self.initial_state_buffer_config.get('enabled', False) and self.initial_state_buffer_config.get('populate_on_init', True):
            self._populate_initial_state_buffer()
        
    def _debug_print(self, message):
        if self.debug_mode:
            print(f"DEBUG: {message}")

    # User-defined functions from KeYmaera model
{user_defined_methods_code_str}

    # --- AST Deserialization ---
    def _deserialize_ast_node(self, data: Dict[str, Any]) -> ASTNode:
        if data is None:
            return None
        
        node_type = data["_type"]
        if node_type == "Literal":
            return Literal(data["value"])
        elif node_type == "VarRef":
            return VarRef(data["name"])
        elif node_type == "BinaryTermOp":
            return BinaryTermOp(data["op"], self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "UnaryTermOp":
            return UnaryTermOp(data["op"], self._deserialize_ast_node(data["operand"]))
        elif node_type == "FunctionCall":
            return FunctionCall(data["name"], [self._deserialize_ast_node(arg) for arg in data["args"]])
        elif node_type == "BooleanLiteral":
            return BooleanLiteral(data["value"])
        elif node_type == "BinaryFormulaOp":
            return BinaryFormulaOp(data["op"], self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "UnaryFormulaOp":
            return UnaryFormulaOp(data["op"], self._deserialize_ast_node(data["operand"]))
        elif node_type == "Comparison":
            return Comparison(data["op"], self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "PredicateCall":
            return PredicateCall(data["name"], [self._deserialize_ast_node(arg) for arg in data["args"]])
        elif node_type == "Assignment":
            return Assignment(self._deserialize_ast_node(data["var"]), self._deserialize_ast_node(data["expr"]))
        elif node_type == "NondetAssignment":
            return NondetAssignment(self._deserialize_ast_node(data["var"]))
        elif node_type == "Test":
            return Test(self._deserialize_ast_node(data["condition"]))
        elif node_type == "Sequence":
            return Sequence([self._deserialize_ast_node(p) for p in data["programs"]])
        elif node_type == "Choice":
            return Choice(self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "Iteration":
            return Iteration(self._deserialize_ast_node(data["program"]), self._deserialize_ast_node(data["invariant"]))
        elif node_type == "ODE":
            return ODE([self._deserialize_ast_node(eq) for eq in data["equations"]], self._deserialize_ast_node(data["domain_constraint"]))
        elif node_type == "ODEEquation":
            return ODEEquation(self._deserialize_ast_node(data["var"]), self._deserialize_ast_node(data["expr"]))
        elif node_type == "HPBlock":
            return HPBlock(self._deserialize_ast_node(data["program"]))
        elif node_type == "ModalFormula":
            return ModalFormula(data["modality"], self._deserialize_ast_node(data["program"]), self._deserialize_ast_node(data["formula"]))
        elif node_type == "Quantifier":
            return Quantifier(data["quantifier_type"], [self._deserialize_ast_node(v) for v in data["variables"]], self._deserialize_ast_node(data["body"]))
        elif node_type == "C_Formula":
            return C_Formula(self._deserialize_ast_node(data["formula"]))
        else:
            raise ValueError(f"Unknown AST node type for deserialization: {{node_type}}")

    # --- Python Eval String Conversion for Environment's ASTs ---
    _PYTHON_TERM_OPS = {'+': '+', '-': '-', '*': '*', '/': '/', '^': '**'}
    _PYTHON_FORMULA_OPS = {
        '=': '==', '!=': '!=', '>': '>', '<': '<', '>=': '>=', '<=': '<=',
        '&&': 'and', '&': 'and', '|': 'or'
    }

    def _term_to_python_eval_str_env(self, term: ASTNode) -> str:
        \"\"\"Converts an AST Term node to its string representation for Python eval() within the environment.\"\"\"
        if isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, VarRef):
            # Access variable from current_state
            return f"self.current_state['{term.name}']"
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_python_eval_str_env(term.left)
            right_str = self._term_to_python_eval_str_env(term.right)
            op = self._PYTHON_TERM_OPS.get(term.op, term.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_python_eval_str_env(term.operand)
            if term.op == "'":
                raise ValueError(f"Derivative operator '{term.op}' not supported in Python term evaluation in environment: {term}")
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_python_eval_str_env(arg) for arg in term.args)
            # Call the bound method from user_defined_functions
            return f"self.user_defined_functions['{term.name}']({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for Python eval in environment: {{type(term)}}")

    def _formula_to_python_eval_str_env(self, formula: ASTNode) -> str:
        \"\"\"Converts an AST Formula node to its string representation for Python eval() within the environment.\"\"\"
        if isinstance(formula, BooleanLiteral):
            return str(formula.value)
        elif isinstance(formula, Comparison):
            left_str = self._term_to_python_eval_str_env(formula.left)
            right_str = self._term_to_python_eval_str_env(formula.right)
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_python_eval_str_env(formula.left)
            right_str = self._formula_to_python_eval_str_env(formula.right)
            if formula.op == "->":
                return f"(not ({left_str}) or ({right_str}))"
            elif formula.op == "<->":
                return f"((({left_str}) and ({right_str})) or (not ({left_str}) and not ({right_str})))"
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_python_eval_str_env(formula.operand)
            if formula.op == "!":
                return f"(not ({operand_str}))"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_python_eval_str_env(arg) for arg in formula.args)
            return f"self.user_defined_functions['{formula.name}']({args_str})"
        elif isinstance(formula, (ModalFormula, Quantifier, C_Formula)):
            raise ValueError(f"Formula type {type(formula)} cannot be directly converted to Python for evaluation in environment: {formula}")
        else:
            raise ValueError(f"Unsupported Formula type for Python eval in environment: {type(formula)}")


    def _load_config(self, config_json_str, config_override):
        config = json.loads(config_json_str)
        if config_override:
            for key, override_value in config_override.items():
                if key == 'constants' and isinstance(override_value, list):
                    # Merge constants list by name
                    existing_constants_map = {c['name']: c for c in config.get('constants', [])}
                    for ov_const in override_value:
                        if ov_const['name'] in existing_constants_map:
                            existing_constants_map[ov_const['name']].update(ov_const)
                        else:
                            existing_constants_map[ov_const['name']] = ov_const # Add new constant
                    config['constants'] = list(existing_constants_map.values())
                elif key == 'variables' and isinstance(override_value, list):
                    # Merge variables list by name
                    existing_variables_map = {v['name']: v for v in config.get('variables', [])}
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
        # Determine action space based on controlFunc AST if available
        if self.control_flow_program_ast and isinstance(self.control_flow_program_ast, Choice):
            # Assuming Choice between HPBlocks, each representing a distinct action branch
            # We need to count the number of distinct choices.
            # The example controlFunc is nested: ((A) ++ (B)) ++ ((C) ++ (D)).
            # We need a BFS to find the leaf branches.
            num_choices = 0
            q = deque([self.control_flow_program_ast])
            while q:
                node = q.popleft()
                if isinstance(node, Choice):
                    q.append(node.left)
                    q.append(node.right)
                else: # This is a leaf branch (e.g., HPBlock, Assignment, Test, Sequence)
                    num_choices += 1
            
            if num_choices > 0:
                self.action_space_type = 'discrete'
                self.action_space = spaces.Discrete(num_choices)
                self.action_variable_names = ['control_choice'] # Placeholder name for discrete action
                self._debug_print(f"DEBUG: Action space set to Discrete({{num_choices}}) based on controlFunc.")
            else:
                raise ValueError("controlFunc is a Choice but no valid branches found.")
        elif self.control_flow_program_ast:
            # If controlFunc is present but not a Choice, it's a single, fixed discrete action.
            # Or it could be a NondetAssignment, which is more complex.
            # For now, if it's not a Choice, assume it's a single "action" that executes the HP.
            self.action_space_type = 'discrete'
            self.action_space = spaces.Discrete(1)
            self.action_variable_names = ['control_choice']
            self._debug_print(f"DEBUG: Action space set to Discrete(1) based on controlFunc (not a Choice).")
        else: # Fallback to continuous if no controlFunc or it's not interpretable
            self.action_space_type = self.config.get('action_space_type', 'continuous')
            action_bounds = self.config.get('action_space_bounds', {})

            if self.action_space_type == 'continuous':
                lows = []
                highs = []
                self.action_variable_names = []
                for var_name, bounds in action_bounds.items():
                    self.action_variable_names.append(var_name)
                    lows.append(self._evaluate_expression(bounds['min'], {}))
                    highs.append(self._evaluate_expression(bounds['max'], {}))
                self.action_space = spaces.Box(low=np.array(lows, dtype=np.float32),
                                               high=np.array(highs, dtype=np.float32),
                                               dtype=np.float32)
            elif self.action_space_type == 'discrete':
                n = action_bounds.get('n', 1)
                self.action_space = spaces.Discrete(n)
                self.action_variable_names = [] # No specific variable names for generic discrete
            else:
                raise ValueError(f"Unsupported action space type: {{self.action_space_type}}")

        observation_space_type = self.config.get('observation_space_type', 'continuous')
        observation_bounds = self.config.get('observation_space_bounds', {})

        if observation_space_type == 'continuous':
            lows = []
            highs = []
            self.observation_variable_names = []
            for var_name, bounds in observation_bounds.items():
                self.observation_variable_names.append(var_name)
                lows.append(self._evaluate_expression(bounds['min'], {}))
                highs.append(self._evaluate_expression(bounds['max'], {}))
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

    def _evaluate_expression(self, expression: Union[str, ASTNode], context: Dict[str, Any]):
        self._debug_print(f"DEBUG: _evaluate_expression called for: {{expression}} (type: {{type(expression)}})")

        if isinstance(expression, ASTNode):
            if isinstance(expression, Term):
                expression_str = self._term_to_python_eval_str_env(expression)
            elif isinstance(expression, Formula):
                expression_str = self._formula_to_python_eval_str_env(expression)
            else:
                raise ValueError(f"Unsupported AST node type for evaluation: {{type(expression)}}")
            self._debug_print(f"DEBUG: Converted AST to expression_str: '{{expression_str}}'")
        else: # Assume it's already a string
            expression_str = expression

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
        eval_locals.update(context) # Context variables (e.g., function arguments) - these override self.current_state values

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
            if isinstance(sympy_expr_node, sympy.Function) and str(sympy_expr_node.func) in self.user_defined_functions:
                func_name = str(sympy_expr_node.func)
                evaluated_args = [evaluate_sympy_function_calls_for_ode(arg, current_subs) for arg in sympy_expr_node.args]
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

    def _run_continuous_dynamics(self, action_to_apply):
        # Apply action to relevant variables before ODE integration
        # This block is now only relevant if action_space_type is 'continuous'
        # and controlFunc is NOT being used to set variables.
        # If controlFunc is active, the variables like 'ar' are already set in self.current_state.
        if self.action_space_type == 'continuous' and self.action_variable_names:
            for i, var_name in enumerate(self.action_variable_names):
                self.current_state[var_name] = action_to_apply[i]

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

    def _execute_control_flow(self, action_choice: int) -> Dict[str, Any]:
        \"\"\"
        Executes the discrete control flow logic defined by the controlFunc AST.
        Modifies self.current_state based on assignments and checks conditions.
        Returns a dictionary with execution status (e.g., 'terminated', 'reward_delta').
        \"\"\"
        status = {"terminated": False, "reward_delta": 0.0, "info": ""}

        if not self.control_flow_program_ast:
            self._debug_print("DEBUG: _execute_control_flow called but no control_flow_program_ast defined.")
            status["info"] = "no_control_flow_program"
            return status

        # For a discrete action space, the agent provides an integer representing the choice.
        # We need to map this integer to the correct branch in the Choice AST.
        # The example controlFunc is nested: ((A) ++ (B)) ++ ((C) ++ (D)).
        # We need a BFS to find the Nth leaf branch.
        
        selected_branch = None
        current_choice_index = 0
        q = deque([self.control_flow_program_ast])
        
        while q and selected_branch is None:
            node = q.popleft()
            if isinstance(node, Choice):
                q.append(node.left)
                q.append(node.right)
            else: # This is a leaf branch
                if current_choice_index == action_choice:
                    selected_branch = node
                current_choice_index += 1
        
        if selected_branch is None:
            self._debug_print(f"ERROR: Invalid action_choice {{action_choice}} for controlFunc. Max choices: {{current_choice_index}}.")
            status["terminated"] = True
            status["reward_delta"] = -100.0 # Penalty for invalid choice
            status["info"] = "invalid_control_choice"
            return status

        self._debug_print(f"DEBUG: Executing selected control branch: {{selected_branch._repr_indented(0)}}")
        
        # Now execute the selected_branch
        return self._execute_hp_node(selected_branch)

    def _execute_hp_node(self, hp_node: HybridProgram) -> Dict[str, Any]:
        \"\"\"
        Recursively executes a hybrid program node.
        Returns execution status, potentially modifying self.current_state.
        \"\"\"
        status = {"terminated": False, "reward_delta": 0.0, "info": ""}

        if isinstance(hp_node, HPBlock):
            # Execute the program inside the block
            return self._execute_hp_node(hp_node.program)
        
        elif isinstance(hp_node, Sequence):
            for program in hp_node.programs:
                sub_status = self._execute_hp_node(program)
                status["terminated"] = status["terminated"] or sub_status["terminated"]
                status["reward_delta"] += sub_status["reward_delta"]
                if status["terminated"]: # If any part of sequence terminates, the whole sequence terminates
                    status["info"] = sub_status["info"] or "sequence_terminated"
                    break
            return status

        elif isinstance(hp_node, Choice):
            # This should ideally not be reached if _execute_control_flow already picked a branch.
            # But if it is, it indicates a problem or a complex nested choice not handled by simple action_choice.
            self._debug_print("WARNING: _execute_hp_node encountered a Choice node. This should be handled by _execute_control_flow.")
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "unexpected_choice_node"
            return status
        
        elif isinstance(hp_node, Test):
            condition_met = self._evaluate_expression(hp_node.condition, self.current_state)
            if not condition_met:
                self._debug_print(f"DEBUG: Test condition '{{hp_node.condition._repr_indented(0)}}' failed. Terminating.")
                status["terminated"] = True
                status["reward_delta"] = -100.0 # Penalty for failing a test
                status["info"] = "test_condition_failed"
            return status

        elif isinstance(hp_node, Assignment):
            var_name = hp_node.var.name
            expr_value = self._evaluate_expression(hp_node.expr, self.current_state)
            self.current_state[var_name] = expr_value
            self._debug_print(f"DEBUG: Assignment: {{var_name}} := {{expr_value}}")
            return status

        elif isinstance(hp_node, NondetAssignment):
            # This is more complex. For now, we'll raise an error or apply a default.
            # A proper implementation would sample from the allowed range.
            self._debug_print(f"ERROR: Nondeterministic assignment '{{hp_node.var.name}} := *' not yet fully supported. Defaulting to 0.0.")
            self.current_state[hp_node.var.name] = 0.0 # Default fallback
            status["info"] = "nondet_assignment_defaulted"
            return status

        elif isinstance(hp_node, Iteration):
            self._debug_print("WARNING: Iteration node encountered in controlFunc. Iterations are typically handled by the RL loop, not directly executed here.")
            # If an iteration is part of controlFunc, it implies the agent decides to loop.
            # This is usually a higher-level concept. For now, treat as a single execution.
            return self._execute_hp_node(hp_node.program)

        elif isinstance(hp_node, VarRef):
            # If the HP is just a reference to another HPDef, resolve and execute it.
            # We need to look up the HPDef by name.
            # This requires a map of HPDef names to their bodies.
            # For simplicity, we'll assume the generator only passes the 'controlFunc' AST directly.
            # If other HP references are expected in controlFunc, the environment needs a way to resolve them.
            self._debug_print(f"ERROR: HP VarRef '{{hp_node.name}}' encountered in control flow. Direct HP references within controlFunc are not yet supported for recursive execution.")
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "unsupported_hp_varref_in_control_flow"
            return status
        
        elif isinstance(hp_node, ODE):
            self._debug_print("WARNING: ODE node encountered in controlFunc. ODEs are handled by _run_continuous_dynamics, not discrete control flow.")
            # This indicates a malformed controlFunc if it directly contains an ODE.
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "unexpected_ode_in_control_flow"
            return status

        else:
            self._debug_print(f"ERROR: Unsupported HybridProgram node type in _execute_hp_node: {{type(hp_node)}}")
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "unsupported_hp_node_type"
            return status


    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0 # Initialize reward

        # Execute discrete control flow based on agent's action
        if self.control_flow_program_ast:
            control_flow_status = self._execute_control_flow(action)
            terminated = control_flow_status["terminated"]
            reward += control_flow_status["reward_delta"]
            if terminated:
                self._debug_print(f"DEBUG: step method: Terminated by control flow: {{control_flow_status['info']}}")
                observation = self._get_observation()
                info = self._get_info()
                info["control_flow_info"] = control_flow_status["info"]
                if self.render_mode == "human":
                    self._render_frame()
                return observation, reward, terminated, truncated, info
        else:
            # If no controlFunc, then the action is directly applied to continuous variables
            # This path is for environments without a controlFunc or where it's ignored
            # and action is directly continuous values for ODE variables.
            if self.action_space_type == 'continuous':
                action_to_apply = action
            else:
                # This case should ideally not be reached if action_space_type is discrete
                # but control_flow_program_ast is None.
                # If it is reached, it means a discrete action was given without a controlFunc to interpret it.
                self._debug_print(f"WARNING: Discrete action received but no controlFunc to interpret it. Action: {{action}}")
                action_to_apply = [] # No direct action for ODEs
        
        # Run continuous dynamics (ODE integration)
        # If controlFunc was active, it would have set the control variables in self.current_state.
        # So, _run_continuous_dynamics will pick them up. action_to_apply will be empty in this case.
        solver_failed = self._run_continuous_dynamics(action if self.action_space_type == 'continuous' else [])
        self._debug_print(f"DEBUG: step method: solver_failed = {{solver_failed}}")
        if solver_failed:
            terminated = True # Terminate if ODE solver fails or domain constraint violated
            reward += self.config.get('ode_failure_penalty', -100.0) # Penalty for ODE failure (configurable)

        # Evaluate reward (from problem definition)
        reward_expr = self.config.get('reward_function_body', '0.0')
        self._debug_print(f"DEBUG: step method: reward_expr before evaluation: '{{reward_expr}}' (type: {{type(reward_expr)}})")
        reward += self._evaluate_expression(reward_expr, self.current_state) # Add to existing reward

        # Evaluate termination condition (from problem definition)
        termination_expr = self.config.get('termination_condition_body', 'False')
        self._debug_print(f"DEBUG: step method: termination_expr for evaluation: '{{termination_expr}}'")
        if not terminated: # Only evaluate if not already terminated by solver failure or control flow
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

    def _generate_valid_initial_state(self, max_attempts=None):
        # Generates a single initial state that satisfies the initial_condition.
        # Returns a dictionary of sampled variable values, 't', and 'T'.
        # Raises RuntimeError if a valid state cannot be found within max_attempts.
        if max_attempts is None:
            max_attempts = self.initial_state_buffer_config.get('max_populate_attempts', 1000)

        initial_cond_expr = self.config['problem']['initial_condition']
        
        for attempt in range(max_attempts):
            sampled_vars = {{}}
            # Sample values for variables
            for var_config in self.config['variables']:
                var_name = var_config['name']
                if var_name in self.initial_state_samplers:
                    sampled_vars[var_name] = self.initial_state_samplers[var_name]()
                else:
                    sampled_vars[var_name] = 0.0 # Default if no sampler

            # Prepare a full state dictionary for evaluation (constants + sampled vars + t, T)
            # Start with current constants (which are in self.current_state after _setup_constants)
            temp_eval_state = copy.copy(self.current_state) 
            temp_eval_state.update(sampled_vars) # Overlay sampled variables

            # Set 't' for initial condition evaluation
            temp_eval_state['t'] = 0.0
            # Ensure 'T' is present for evaluation. If it's a variable, its sampled value is used.
            # If it's a constant, its value is already in temp_eval_state.
            # If it's neither, provide a default.
            if 'T' not in temp_eval_state:
                temp_eval_state['T'] = 10.0 # Default global time remaining

            # Evaluate initial_condition using this temporary state
            initial_cond_met = self._evaluate_expression(initial_cond_expr, temp_eval_state)
            
            if initial_cond_met:
                # Return only the parts that need to be set into self.current_state
                # after constants are already loaded.
                result_state_update = {{k: v for k, v in sampled_vars.items()}}
                result_state_update['t'] = 0.0
                result_state_update['T'] = temp_eval_state['T'] # Use the T that was used for evaluation
                return result_state_update

        raise RuntimeError(f"Failed to find a valid initial state after {{max_attempts}} attempts "
                           f"for initial condition: '{{initial_cond_expr}}'")

    def _build_initial_state_sample_ranges(self):
        sample_ranges = {{}
        for var_name, sampling_config in self.config.get('initial_state_sampling_config', {{}}).items():
            if 'value' in sampling_config:
                value = self._evaluate_expression(sampling_config['value'], {{}})
                sample_ranges[var_name] = (float(value), float(value))
            elif 'min' in sampling_config and 'max' in sampling_config:
                low = self._evaluate_expression(sampling_config['min'], {{}})
                high = self._evaluate_expression(sampling_config['max'], {{}})
                sample_ranges[var_name] = (float(low), float(high))
            else:
                sample_ranges[var_name] = (-100.0, 100.0)
        return sample_ranges

    def _expr_to_z3(self, expr: str, z3_symbols: Dict[str, z3.ArithRef], fixed_constants_map: Dict[str, float]):
        expr = expr.strip()
        if not expr:
            return z3.BoolVal(True)

        def find_function_body(function_name: str):
            for func_config in self.config.get('function_definitions', []):
                if func_config.get('name') == function_name:
                    return func_config
            return None

        def is_boolean_function_definition(function_name: str):
            func_def = find_function_body(function_name)
            if func_def is None:
                return False
            body = func_def.get('body')
            if not isinstance(body, str):
                return False
            try:
                parsed = ast.parse(body, mode='eval').body
            except SyntaxError:
                return False
            if isinstance(parsed, ast.BoolOp):
                return True
            if isinstance(parsed, ast.Compare):
                return True
            if isinstance(parsed, ast.UnaryOp) and isinstance(parsed.op, ast.Not):
                return True
            if isinstance(parsed, ast.Call):
                func = parsed.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'self':
                    return is_boolean_function_definition(func.attr.replace('_impl', '').lstrip('_'))
                if isinstance(func, ast.Name):
                    return is_boolean_function_definition(func.id)
            if isinstance(parsed, ast.Name):
                return parsed.id in ('True', 'False')
            return False

        def convert(node, local_bindings=None):
            local_bindings = {} if local_bindings is None else local_bindings

            if isinstance(node, ast.Expression):
                return convert(node.body, local_bindings)
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    return z3.And(*[convert(v, local_bindings) for v in node.values])
                if isinstance(node.op, ast.Or):
                    return z3.Or(*[convert(v, local_bindings) for v in node.values])
            if isinstance(node, ast.UnaryOp):
                operand = convert(node.operand, local_bindings)
                if isinstance(node.op, ast.Not):
                    return z3.Not(operand)
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return operand
            if isinstance(node, ast.BinOp):
                left = convert(node.left, local_bindings)
                right = convert(node.right, local_bindings)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    return left ** right
                if isinstance(node.op, ast.Mod):
                    return left % right
                raise ValueError(f"Unsupported binary operator in initial condition: {{type(node.op).__name__}}")
            if isinstance(node, ast.Compare):
                left = convert(node.left, local_bindings)
                comparisons = []
                for op, comp in zip(node.ops, node.comparators):
                    right = convert(comp, local_bindings)
                    if isinstance(op, ast.Gt):
                        comparisons.append(left > right)
                    elif isinstance(op, ast.GtE):
                        comparisons.append(left >= right)
                    elif isinstance(op, ast.Lt):
                        comparisons.append(left < right)
                    elif isinstance(op, ast.LtE):
                        comparisons.append(left <= right)
                    elif isinstance(op, ast.Eq):
                        comparisons.append(left == right)
                    elif isinstance(op, ast.NotEq):
                        comparisons.append(left != right)
                    else:
                        raise ValueError(f"Unsupported comparison operator in initial condition: {{type(op).__name__}}")
                    left = right
                if not comparisons:
                    return z3.BoolVal(True)
                if len(comparisons) == 1:
                    return comparisons[0]
                return z3.And(*comparisons)
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == 'self':
                    attr_name = node.attr
                    if attr_name.endswith('_impl'):
                        attr_name = attr_name[:-5]
                    attr_name = attr_name.lstrip('_')
                    if attr_name in local_bindings:
                        return local_bindings[attr_name]
                    if attr_name in z3_symbols:
                        return z3_symbols[attr_name]
                    if attr_name in fixed_constants_map:
                        return z3.RealVal(str(fixed_constants_map[attr_name]))
                    if attr_name in ('True', 'False'):
                        return z3.BoolVal(attr_name == 'True')
                    return z3.Real(attr_name)
                raise ValueError(f"Unsupported attribute access in initial condition: {{ast.dump(node)}}")
            if isinstance(node, ast.Name):
                if node.id in local_bindings:
                    return local_bindings[node.id]
                if node.id in z3_symbols:
                    return z3_symbols[node.id]
                if node.id in fixed_constants_map:
                    return z3.RealVal(str(fixed_constants_map[node.id]))
                if node.id in ('True', 'False'):
                    return z3.BoolVal(node.id == 'True')
                return z3.Real(node.id)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool):
                    return z3.BoolVal(node.value)
                if isinstance(node.value, (int, float)):
                    return z3.RealVal(str(node.value))
                return z3.RealVal(str(node.value))
            if isinstance(node, ast.Num):
                return z3.RealVal(str(node.n))
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                    method_name = node.func.attr
                    if method_name.endswith('_impl'):
                        method_name = method_name[:-5]
                    method_name = method_name.lstrip('_')
                    func_def = find_function_body(method_name)
                    if func_def is not None:
                        arg_names = func_def.get('args', [])
                        if not node.args and not arg_names:
                            return convert(ast.parse(func_def['body'], mode='eval').body, local_bindings)
                        new_local_bindings = dict(local_bindings)
                        for arg_name, arg_node in zip(arg_names, node.args):
                            new_local_bindings[arg_name] = convert(arg_node, local_bindings)
                        return convert(ast.parse(func_def['body'], mode='eval').body, new_local_bindings)

                    if not node.args:
                        if method_name in local_bindings:
                            return local_bindings[method_name]
                        if method_name in z3_symbols:
                            return z3_symbols[method_name]
                        if method_name in fixed_constants_map:
                            return z3.RealVal(str(fixed_constants_map[method_name]))
                        if is_boolean_function_definition(method_name):
                            return z3.Bool(method_name)
                        return z3.Real(method_name)

                    if method_name in ('min', 'max'):
                        values = [convert(arg, local_bindings) for arg in node.args]
                        if len(values) == 2:
                            a, b = values
                            return z3.If(a <= b, a, b) if method_name == 'min' else z3.If(a >= b, a, b)
                        return values[0]

                    raise ValueError(f"Unsupported method call in initial condition: {{method_name}}")

                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    func_def = find_function_body(func_name)
                    if func_def is not None:
                        arg_names = func_def.get('args', [])
                        if not node.args and not arg_names:
                            return convert(ast.parse(func_def['body'], mode='eval').body, local_bindings)
                        new_local_bindings = dict(local_bindings)
                        for arg_name, arg_node in zip(arg_names, node.args):
                            new_local_bindings[arg_name] = convert(arg_node, local_bindings)
                        return convert(ast.parse(func_def['body'], mode='eval').body, new_local_bindings)

                    if func_name in ('min', 'max'):
                        values = [convert(arg, local_bindings) for arg in node.args]
                        if len(values) == 2:
                            a, b = values
                            return z3.If(a <= b, a, b) if func_name == 'min' else z3.If(a >= b, a, b)
                        if values:
                            return values[0]
                    if func_name == 'pow' and len(node.args) == 2:
                        base, exp = [convert(arg, local_bindings) for arg in node.args]
                        return base ** exp
                    if func_name in ('float', 'int', 'bool') and len(node.args) == 1:
                        return convert(node.args[0], local_bindings)
                    # Treat unknown function calls as a symbolic placeholder for the function name.
                    if not node.args:
                        if is_boolean_function_definition(func_name):
                            return z3.Bool(func_name)
                        return z3.Real(func_name)
                    return z3.Bool(func_name)

                raise ValueError(f"Unsupported call in initial condition: {{ast.dump(node)}}")
            raise ValueError(f"Unsupported expression node in initial condition: {{type(node).__name__}}")

        try:
            parsed = ast.parse(expr, mode='eval')
            return convert(parsed)
        except Exception as exc:
            raise ValueError(f"Failed to translate initial condition to Z3: {{expr}} -> {{exc}}") from exc

    def _populate_initial_state_buffer_diverse_z3(self):
        # Populate a diverse set of valid initial states using Z3 optimizer.
        capacity = self.initial_state_buffer_config.get('capacity', 100)
        max_populate_attempts_total = self.initial_state_buffer_config.get('max_populate_attempts', 1000)
        self._debug_print(f"DEBUG: Populating diverse Z3 initial state buffer (capacity: {{capacity}}, max_attempts_total: {{max_populate_attempts_total}})")

        sample_ranges = self._build_initial_state_sample_ranges()
        z3_symbols = {}
        fixed_constants_map = {}

        for var_config in self.config.get('variables', []):
            var_name = var_config['name']
            z3_symbols[var_name] = z3.Real(var_name)
        for const_config in self.config.get('constants', []):
            const_name = const_config['name']
            const_value = const_config.get('value')
            if const_value is not None and isinstance(const_value, (int, float)):
                fixed_constants_map[const_name] = float(const_value)
            elif const_value is not None:
                try:
                    numeric_value = float(const_value)
                    fixed_constants_map[const_name] = numeric_value
                except (TypeError, ValueError):
                    z3_symbols[const_name] = z3.Real(const_name)

        opt = z3.Optimize()
        opt.set(timeout=10000)

        initial_condition_expr = self.config.get('problem', {{}}).get('initial_condition', 'True')
        try:
            z3_initial_condition = self._expr_to_z3(initial_condition_expr, z3_symbols, fixed_constants_map)
            opt.add(z3_initial_condition)
        except ValueError as e:
            print(f"CRITICAL ERROR: Failed to convert initial condition to Z3 during CISB population: {{e}}")
            return

        for var_name, z3_var in z3_symbols.items():
            low, high = sample_ranges.get(var_name, (-100.0, 100.0))
            opt.add(z3_var >= low, z3_var <= high)

        attempts_made = 0
        while len(self.initial_state_buffer) < capacity and attempts_made < max_populate_attempts_total:
            attempts_made += 1
            opt.push()

            soft_constraints_added = False
            for var_name, z3_var in z3_symbols.items():
                if var_name in fixed_constants_map:
                    continue
                low, high = sample_ranges.get(var_name, (-100.0, 100.0))
                if low == high:
                    continue
                target = self.np_random.uniform(low, high)
                opt.add_soft(z3_var == target, weight=1)
                soft_constraints_added = True

            check_result = opt.check()
            if check_result == z3.sat:
                model = opt.model()
                solved_values = {{}}
                for k, v in z3_symbols.items():
                    evaluated_val = model.evaluate(v, model_completion=True)
                    if evaluated_val is not None:
                        if hasattr(evaluated_val, 'as_fraction'):
                            solved_values[k] = float(evaluated_val.as_fraction())
                        elif hasattr(evaluated_val, 'as_long'):
                            solved_values[k] = float(evaluated_val.as_long())
                        else:
                            solved_values[k] = float(str(evaluated_val))
                solved_values.update(fixed_constants_map)

                if 't' not in solved_values:
                    solved_values['t'] = 0.0
                if 'T' not in solved_values:
                    solved_values['T'] = 10.0

                full_state_to_store = OrderedDict()
                full_state_to_store.update(self.current_state)
                full_state_to_store.update(solved_values)
                self.initial_state_buffer.append(copy.deepcopy(full_state_to_store))
                self._debug_print(f"DEBUG: Added diverse Z3-generated state to buffer. Buffer size: {{len(self.initial_state_buffer)}}")

                blocking_conditions = []
                for var_name, z3_var in z3_symbols.items():
                    val = solved_values.get(var_name)
                    if val is not None:
                        diversity_epsilon = 0.1
                        blocking_conditions.append(z3.Or(z3_var < val - diversity_epsilon, z3_var > val + diversity_epsilon))

                if blocking_conditions:
                    opt.add(z3.Or(blocking_conditions))
                opt.pop()
            elif check_result == z3.unsat:
                print(f"DEBUG: Z3 UNSAT during diverse buffer population. No more states available.")
                opt.pop()
                break
            else:
                print(f"DEBUG: Z3 UNKNOWN during diverse buffer population. Retrying with new soft constraints.")
                opt.pop()

        if len(self.initial_state_buffer) < capacity:
            print(f"Warning: Diverse initial state buffer could only be filled to {{len(self.initial_state_buffer)}}/{{capacity}} after {{attempts_made}} attempts.")

    def _populate_initial_state_buffer(self):
        # Populates the initial state buffer with valid initial states.
        isb_config = self.initial_state_buffer_config
        strategy = isb_config.get('strategy', 'random')
        if strategy == 'z3_diverse':
            self._populate_initial_state_buffer_diverse_z3()
            return

        capacity = isb_config.get('capacity', 100)
        max_populate_attempts_total = isb_config.get('max_populate_attempts', 1000) # Total attempts for population

        self._debug_print(f"DEBUG: Populating initial state buffer (capacity: {{capacity}}, max_attempts_total: {{max_populate_attempts_total}})")

        attempts_made = 0
        while len(self.initial_state_buffer) < capacity and attempts_made < max_populate_attempts_total:
            try:
                valid_state_update = self._generate_valid_initial_state(max_attempts=100)
                full_state_to_store = OrderedDict()
                full_state_to_store.update(self.current_state)
                full_state_to_store.update(valid_state_update)
                self.initial_state_buffer.append(full_state_to_store)
                self._debug_print(f"DEBUG: Added state to buffer. Buffer size: {{len(self.initial_state_buffer)}}")
            except RuntimeError as e:
                self._debug_print(f"DEBUG: Failed to generate a valid state during population: {{e}}. Continuing attempts.")
            attempts_made += 1
        
        if len(self.initial_state_buffer) < capacity:
            print(f"Warning: Initial state buffer could only be filled to {{len(self.initial_state_buffer)}}/{{capacity}} after {{attempts_made}} attempts.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        self.current_state = OrderedDict()

        # Re-apply constants (already done in __init__, but good to ensure for reset)
        self._setup_constants()

        # Ensure functions are set up for _evaluate_expression calls during reset
        self._setup_functions()

        isb_enabled = self.initial_state_buffer_config.get('enabled', False)
        
        if isb_enabled and len(self.initial_state_buffer) > 0:
            # Use a state from the buffer
            self.current_state = self.initial_state_buffer.popleft()
            self._debug_print(f"DEBUG: Resetting from initial state buffer. Buffer size: {{len(self.initial_state_buffer)}}")
        else:
            # Generate a new initial state
            try:
                # This will return a dict of sampled variables, 't', and 'T'
                new_state_update = self._generate_valid_initial_state()
                self.current_state.update(new_state_update) # Merge sampled vars, t, T into current_state (which has constants)
                self._debug_print(f"DEBUG: Generated new initial state.")

                # If ISB is enabled, add this newly generated state to the buffer
                if isb_enabled:
                    # Store the full current_state
                    self.initial_state_buffer.append(copy.deepcopy(self.current_state))
                    self._debug_print(f"DEBUG: Added newly generated state to buffer. Buffer size: {{len(self.initial_state_buffer)}}")

            except RuntimeError as e:
                print(f"Error during reset: {{e}}. Using default initial state (all zeros for variables).")
                # Fallback: if generation fails, initialize variables to 0.0
                for var_config in self.config['variables']:
                    var_name = var_config['name']
                    if var_name not in self.current_state: # Don't overwrite constants
                        self.current_state[var_name] = 0.0
                self.current_state['t'] = 0.0
                self.current_state['T'] = 10.0 # Default if not set

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
        
        # Ensure global_time_remaining is updated from the state
        self.global_time_remaining = self.current_state.get('T', 0.0)

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