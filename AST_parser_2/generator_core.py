import json
import textwrap
from typing import List, Dict, Any, Optional
import math
import sympy

# AST node imports
from ast_config import (
    ASTNode, KeYmaeraModel, ArchiveEntry, DefinitionsBlock, Definition, FuncDef, PredDef, HPDef,
    ProgramVariablesBlock, ProgramVariableDecl, ProblemBlock, TacticBlock,
    Term, Literal, VarRef, BinaryTermOp, UnaryTermOp, FunctionCall,
    Formula, BooleanLiteral, BinaryFormulaOp, UnaryFormulaOp, Comparison, Quantifier,
    ModalFormula, PredicateCall, C_Formula,
    HybridProgram, Assignment, NondetAssignment, Test, Sequence, Choice, Iteration,
    ODE, ODEEquation, HPBlock
)

# Parser/lexer imports
from keymaera_parser import KeYmaeraParser

# Configuration dataclasses
from pyast_config import (
    GymEnvironmentConfig,
    VariableDeclarationConfig,
    ConstantDefinitionConfig,
    FuncDefConfig,
    HPDefConfig,
    ProblemConfig
)

# Template string
from environment_template_code import ENVIRONMENT_TEMPLATE_CODE


class GymnasiumEnvironmentGenerator:
    """Core generator class (moved out of the CLI wrapper).

    This class contains the logic to convert parsed KeYmaera ASTs into the
    generated Python source for a Gymnasium environment. It mirrors the
    previous implementation but is placed in a dedicated module to keep the
    original CLI file small.
    """
    def __init__(self, environment_name: str):
        self.environment_name = environment_name
        self.config = GymEnvironmentConfig(environment_name=environment_name)
        self.indent_level = 0
        self.indent_str = "    "
        self.hp_definition_map: Dict[str, HPDef] = {}
        self.func_definition_map: Dict[str, FuncDef] = {}
        self.pred_definition_map: Dict[str, PredDef] = {}
        # Reused to parse standalone dL expressions found inside annotation values
        # (e.g. `@progress: reward=xr - waypointStartDist(xg)`).
        self._expr_parser = KeYmaeraParser("")

    def _annotation_value_to_python_expr(self, value: Any) -> str:
        """Converts an `@progress`/`@safety` annotation value into a Python-evaluable expression
        string. Plain numbers (the common case, e.g. `reward=1000`) pass through unchanged; a
        value that failed numeric parsing (e.g. `reward=xr - waypointStartDist(xg)`) is a raw dL
        expression string, so it's parsed as a Term and converted the same way user-defined
        function bodies are, letting the reward/penalty be shaped by the model itself."""
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            try:
                term = self._expr_parser.parse_term(expression=value)
                return self._term_to_string_python(term)
            except Exception:
                return value
        return str(value)

    def _indent(self, text: str) -> str:
        lines = text.splitlines()
        indented_lines = [self.indent_str * self.indent_level + line for line in lines]
        return "\n".join(indented_lines)

    # --- Python evaluation conversions ---
    _PYTHON_TERM_OPS = {'+': '+', '-': '-', '*': '*', '/': '/', '^': '**'}
    _PYTHON_FORMULA_OPS = {
        '=': '==', '!=': '!=', '>': '>', '<': '<', '>=': '>=', '<=': '<=',
        '&&': 'and', '&': 'and', '|': 'or'
    }

    def _term_to_string_python(self, term: Term) -> str:
        if isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, VarRef):
            var_name = term.name
            if var_name in self.func_definition_map:
                func_def = self.func_definition_map[var_name]
                if not func_def.args and not func_def.annotations.get('const'):
                    return f"self._{var_name}_impl()"
            return var_name
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_string_python(term.left)
            right_str = self._term_to_string_python(term.right)
            op = self._PYTHON_TERM_OPS.get(term.op, term.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_string_python(term.operand)
            if term.op == "'":
                raise ValueError(f"Derivative operator '{term.op}' not supported in Python term evaluation: {term}")
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_string_python(arg) for arg in term.args)
            return f"self._{term.name}_impl({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for Python eval: {type(term)}")

    def _formula_to_string_python(self, formula: Formula) -> str:
        if isinstance(formula, BooleanLiteral):
            return str(formula.value)
        elif isinstance(formula, Comparison):
            left_str = self._term_to_string_python(formula.left)
            right_str = self._term_to_string_python(formula.right)
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_string_python(formula.left)
            right_str = self._formula_to_string_python(formula.right)
            if formula.op == "->":
                return f"(not ({left_str}) or ({right_str}))"
            elif formula.op == "<->":
                return f"((({left_str}) and ({right_str})) or (not ({left_str}) and not ({right_str})))"
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_string_python(formula.operand)
            if formula.op == "!":
                return f"(not ({operand_str}))"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_string_python(arg) for arg in formula.args)
            return f"self._{formula.name}_impl({args_str})"
        elif isinstance(formula, (ModalFormula, Quantifier, C_Formula)):
            raise ValueError(f"Formula type {type(formula)} cannot be directly converted to Python for evaluation: {formula}")
        else:
            raise ValueError(f"Unsupported Formula type for Python eval: {type(formula)}")

    # --- SymPy conversions ---
    def _term_to_string_sympy(self, term: Term) -> str:
        if isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, VarRef):
            return term.name
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_string_sympy(term.left)
            right_str = self._term_to_string_sympy(term.right)
            return f"({left_str} {term.op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_string_sympy(term.operand)
            if term.op == "'":
                return f"{operand_str}_prime"
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_string_sympy(arg) for arg in term.args)
            return f"sympy.Function('{term.name}')({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for SymPy: {type(term)}")

    def _formula_to_string_sympy(self, formula: Formula) -> str:
        if isinstance(formula, BooleanLiteral):
            return str(formula.value)
        elif isinstance(formula, Comparison):
            left_str = self._term_to_string_sympy(formula.left)
            right_str = self._term_to_string_sympy(formula.right)
            if formula.op == '=': return f"sympy.Eq({left_str}, {right_str})"
            if formula.op == '!=': return f"sympy.Ne({left_str}, {right_str})"
            if formula.op == '>': return f"sympy.Gt({left_str}, {right_str})"
            if formula.op == '<': return f"sympy.Lt({left_str}, {right_str})"
            if formula.op == '>=': return f"sympy.Ge({left_str}, {right_str})"
            if formula.op == '<=': return f"sympy.Le({left_str}, {right_str})"
            return f"({left_str} {formula.op} {right_str})"
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_string_sympy(formula.left)
            right_str = self._formula_to_string_sympy(formula.right)
            if formula.op == "->":
                return f"sympy.Implies({left_str}, {right_str})"
            elif formula.op == "<->":
                return f"sympy.Equivalent({left_str}, {right_str})"
            elif formula.op in ["&&", "&"]:
                return f"sympy.And({left_str}, {right_str})"
            elif formula.op == "|":
                return f"sympy.Or({left_str}, {right_str})"
            return f"({left_str} {formula.op} {right_str})"
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_string_sympy(formula.operand)
            if formula.op == "!":
                return f"sympy.Not({operand_str})"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_string_sympy(arg) for arg in formula.args)
            return f"sympy.Function('{formula.name}')({args_str})"
        elif isinstance(formula, ModalFormula):
            program_str = self._hp_to_string(formula.program)
            formula_str = self._formula_to_string_sympy(formula.formula)
            return f"[{program_str}]({formula_str})"
        elif isinstance(formula, Quantifier):
            vars_str = ", ".join(v.name for v in formula.variables)
            body_str = self._formula_to_string_sympy(formula.body)
            return f"{formula.quantifier_type}({vars_str}).({body_str})"
        elif isinstance(formula, C_Formula):
            return f"C({self._formula_to_string_sympy(formula.formula)})"
        else:
            raise ValueError(f"Unsupported Formula type for SymPy: {type(formula)}")

    def _hp_to_string(self, hp: HybridProgram) -> str:
        if isinstance(hp, Assignment):
            var_str = self._term_to_string_python(hp.var)
            expr_str = self._term_to_string_python(hp.expr)
            return f"{var_str} := {expr_str}"
        elif isinstance(hp, NondetAssignment):
            var_str = self._term_to_string_python(hp.var)
            return f"{{{var_str} := *}}"
        elif isinstance(hp, Test):
            condition_str = self._formula_to_string_python(hp.condition)
            return f"?({condition_str})"
        elif isinstance(hp, Sequence):
            programs_str = "; ".join(self._hp_to_string(p) for p in hp.programs)
            return f"({programs_str})"
        elif isinstance(hp, Choice):
            left_str = self._hp_to_string(hp.left)
            right_str = self._hp_to_string(hp.right)
            return f"({left_str} ++ {right_str})"
        elif isinstance(hp, Iteration):
            program_str = self._hp_to_string(hp.program)
            invariant_str = f"@invariant({self._formula_to_string_python(hp.invariant)})" if hp.invariant else ""
            return f"({program_str})*{invariant_str}"
        elif isinstance(hp, ODE):
            equations_str = ", ".join(f"{self._term_to_string_sympy(eq.var)}' = {self._term_to_string_sympy(eq.expr)}" for eq in hp.equations)
            if hp.domain_constraint:
                domain_constraint_sympy_str = self._formula_to_string_sympy(hp.domain_constraint)
                domain_str = f"@@@DOMAIN_CONSTRAINT@@@{domain_constraint_sympy_str}"
            else:
                domain_str = ""
            return f"({equations_str}{domain_str})"
        elif isinstance(hp, HPBlock):
            return f"{{{self._hp_to_string(hp.program)}}}"
        elif isinstance(hp, VarRef):
            return hp.name
        else:
            raise ValueError(f"Unsupported HybridProgram type: {type(hp)}")

    def _resolve_hp_to_ode(self, hp_name: str) -> Optional[ODE]:
        if hp_name not in self.hp_definition_map:
            return None
        current_hp_def = self.hp_definition_map[hp_name]
        current_hp_body = current_hp_def.body

        def find_ode_in_hp(hp_node: HybridProgram, path: List[str]) -> Optional[ODE]:
            if isinstance(hp_node, ODE):
                return hp_node
            elif isinstance(hp_node, Sequence):
                for sub_hp in hp_node.programs:
                    ode = find_ode_in_hp(sub_hp, path)
                    if ode:
                        return ode
            elif isinstance(hp_node, Choice):
                ode = find_ode_in_hp(hp_node.left, path)
                if ode:
                    return ode
                ode = find_ode_in_hp(hp_node.right, path)
                if ode:
                    return ode
            elif isinstance(hp_node, Iteration):
                return find_ode_in_hp(hp_node.program, path)
            elif isinstance(hp_node, HPBlock):
                return find_ode_in_hp(hp_node.program, path)
            elif isinstance(hp_node, VarRef):
                return self._resolve_hp_to_ode(hp_node.name)
            return None

        return find_ode_in_hp(current_hp_body, [hp_name])

    def _collect_loop_invariants(self, hp_name: str, _visited: Optional[set] = None) -> List[Formula]:
        """Walks a hybrid program (resolving named HP references) and collects every
        `@invariant(...)` formula attached to an Iteration node."""
        if _visited is None:
            _visited = set()
        if hp_name in _visited or hp_name not in self.hp_definition_map:
            return []
        _visited.add(hp_name)

        invariants: List[Formula] = []

        def walk(node):
            if isinstance(node, Iteration):
                if node.invariant is not None:
                    invariants.append(node.invariant)
                walk(node.program)
            elif isinstance(node, Sequence):
                for p in node.programs:
                    walk(p)
            elif isinstance(node, Choice):
                walk(node.left)
                walk(node.right)
            elif isinstance(node, HPBlock):
                walk(node.program)
            elif isinstance(node, VarRef):
                invariants.extend(self._collect_loop_invariants(node.name, _visited))
            # Assignment / NondetAssignment / Test / ODE have no nested invariants.

        walk(self.hp_definition_map[hp_name].body)
        return invariants

    def _find_safety_penalty(self, *formulas: Formula, default: str = "-100.0") -> str:
        """Searches formulas (loop invariants, safety postcondition, ...) for a PredicateCall
        whose definition carries an `@safety: penalty=X` annotation, returning the first X found
        as a Python-evaluable expression string (X may be a plain number or a dL expression)."""
        def search(node) -> Optional[str]:
            if node is None:
                return None
            if isinstance(node, PredicateCall):
                pred_def = self.pred_definition_map.get(node.name)
                if pred_def:
                    safety_annotation = pred_def.annotations.get('safety')
                    if safety_annotation and 'penalty' in safety_annotation:
                        return self._annotation_value_to_python_expr(safety_annotation['penalty'])
                for arg in node.args:
                    found = search(arg)
                    if found is not None:
                        return found
                return None
            if isinstance(node, (BinaryFormulaOp, BinaryTermOp)):
                return search(node.left) or search(node.right)
            if isinstance(node, (UnaryFormulaOp, UnaryTermOp)):
                return search(node.operand)
            if isinstance(node, Comparison):
                return search(node.left) or search(node.right)
            if isinstance(node, FunctionCall):
                for arg in node.args:
                    found = search(arg)
                    if found is not None:
                        return found
                return None
            if isinstance(node, Quantifier):
                return search(node.body)
            return None

        for formula in formulas:
            found = search(formula)
            if found is not None:
                return found
        return default

    def _extract_reward_components_from_annotation(self, annotation: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Extracts shape, reward, and penalty from @safety, @progress, or @stability annotations.
        According to the RL-Annotation DSL spec, rewards are calculated as:
        R = shape_expr + (reward_val if φ is True, else penalty_val)
        
        Returns a dict with keys 'shape', 'reward', 'penalty', each as a Python-evaluable string.
        Defaults to '0.0' for any missing component.
        """
        components = {
            'shape': '0.0',
            'reward': '0.0',
            'penalty': '0.0'
        }
        
        if annotation:
            if 'shape' in annotation:
                components['shape'] = self._annotation_value_to_python_expr(annotation['shape'])
            if 'reward' in annotation:
                components['reward'] = self._annotation_value_to_python_expr(annotation['reward'])
            if 'penalty' in annotation:
                components['penalty'] = self._annotation_value_to_python_expr(annotation['penalty'])
        
        return components

    def _extract_problem_components(self, problem_formula: Formula) -> Dict[str, Any]:
        # Implementation mirrors previous logic; kept compact for readability
        initial_condition = "True"
        program_ode_str = ""
        goal_formula_str = "True"
        termination_condition_str = "True"
        reward_function_str = "0.0"
        loop_invariants = []

        current_formula = problem_formula

        if isinstance(current_formula, BinaryFormulaOp) and current_formula.op == "->":
            initial_condition = self._formula_to_string_python(current_formula.left)
            current_formula = current_formula.right

        safety_modal_formula = None
        liveness_modal_formula = None

        if isinstance(current_formula, ModalFormula) and current_formula.modality == 'box':
            safety_modal_formula = current_formula
        elif isinstance(current_formula, BinaryFormulaOp) and current_formula.op in ["&&", "&"]:
            if isinstance(current_formula.left, ModalFormula) and current_formula.left.modality == 'box':
                safety_modal_formula = current_formula.left
            if isinstance(current_formula.right, ModalFormula) and current_formula.right.modality == 'diamond':
                liveness_modal_formula = current_formula.right
        elif isinstance(current_formula, ModalFormula) and current_formula.modality == 'box':
            safety_modal_formula = current_formula
            if isinstance(safety_modal_formula.formula, BinaryFormulaOp) and safety_modal_formula.formula.op in ["&&", "&"]:
                if isinstance(safety_modal_formula.formula.right, ModalFormula) and safety_modal_formula.formula.right.modality == 'diamond':
                    liveness_modal_formula = safety_modal_formula.formula.right
                    safety_modal_formula = ModalFormula(modality='box', program=safety_modal_formula.program, formula=safety_modal_formula.formula.left)

        raw_invariant_formulas: List[Formula] = []

        if safety_modal_formula is not None:
            program = safety_modal_formula.program
            if isinstance(program, VarRef):
                hp_name = program.name
                ode_block = self._resolve_hp_to_ode(hp_name)
                if ode_block:
                    program_ode_str = self._hp_to_string(ode_block)
                else:
                    raise ValueError(f"Could not resolve ODE block from hybrid program '{hp_name}'")
                raw_invariant_formulas = self._collect_loop_invariants(hp_name)
            elif hasattr(program, 'programs'):
                # Generated programs can also be explicit hybrid-program blocks that are already in AST form.
                program_ode_str = self._hp_to_string(program)

            try:
                goal_formula_str = self._formula_to_string_python(safety_modal_formula.formula)
            except Exception:
                goal_formula_str = "True"

            termination_condition_str = f"(not ({goal_formula_str}))"

        for inv_formula in raw_invariant_formulas:
            try:
                inv_formula_str = self._formula_to_string_python(inv_formula)
            except Exception:
                continue
            loop_invariants.append({"formula_str": inv_formula_str})
            termination_condition_str = f"({termination_condition_str} or (not ({inv_formula_str})))"

        # `@safety: penalty=X` on a loop-invariant predicate (or the safety postcondition itself)
        # gives the model author's intended penalty for violating that condition; fall back to a
        # generic -100.0 if the model doesn't declare one.
        # NOTE: Per the RL-Annotation DSL spec, safety predicates can also have shape= and reward= fields.
        safety_penalty = self._find_safety_penalty(
            *raw_invariant_formulas,
            safety_modal_formula.formula if safety_modal_formula is not None else None
        )
        not_safe_str = f"(({safety_penalty}) if ({termination_condition_str}) else 0.0)"
        # Collect reward contributions from ALL annotated predicates in the model.
        # Per RL-Annotation DSL spec: R = shape_expr + (reward_val if φ is True, else penalty_val)
        # KEY RULES:
        # 1. @progress and @safety can annotate ANY predicate regardless of name/modality
        # 2. A predicate can have ONLY ONE: either @progress or @safety, not both (mutually exclusive)
        reward_components_list = []
        
        for pred_name, pred_def in self.pred_definition_map.items():
            # Enforce mutual exclusivity: @progress OR @safety, not both
            if 'progress' in pred_def.annotations:
                progress_annotation = pred_def.annotations['progress']
                components = self._extract_reward_components_from_annotation(progress_annotation)
                args_str = ", ".join([arg.name for arg in pred_def.args])
                pred_call_str = f"self._{pred_name}_impl({args_str})" if pred_def.args else f"self._{pred_name}_impl()"
                reward_components_list.append({
                    'predicate_name': pred_name,
                    'predicate_str': pred_call_str,
                    'components': components,
                    'type': 'progress'
                })
            elif 'safety' in pred_def.annotations:
                safety_annotation = pred_def.annotations['safety']
                components = self._extract_reward_components_from_annotation(safety_annotation)
                args_str = ", ".join([arg.name for arg in pred_def.args])
                pred_call_str = f"self._{pred_name}_impl({args_str})" if pred_def.args else f"self._{pred_name}_impl()"
                reward_components_list.append({
                    'predicate_name': pred_name,
                    'predicate_str': pred_call_str,
                    'components': components,
                    'type': 'safety'
                })
        
        # Build the reward function by combining all components.
        # Formula per spec: R = shape + (reward if φ true else penalty)
        # When safe: sum up all contributions; when unsafe: apply safety_penalty override
        if reward_components_list:
            # Build a list of reward expressions, one per annotated predicate
            reward_terms = []
            for component in reward_components_list:
                pred_str = component['predicate_str']
                comps = component['components']
                # Build: shape + (reward if pred else penalty)
                term = f"(({comps['shape']}) + (({comps['reward']}) if ({pred_str}) else ({comps['penalty']})))"
                reward_terms.append(term)
            
            # Sum all components when safe; apply safety_penalty when unsafe
            combined_rewards = " + ".join(reward_terms)
            reward_function_str = f"({combined_rewards}) if not ({termination_condition_str}) else ({safety_penalty})"
        else:
            # Fallback to default behavior if no annotations found
            if liveness_modal_formula and isinstance(liveness_modal_formula.formula, PredicateCall):
                ensure_goal_str = self._formula_to_string_python(liveness_modal_formula.formula)
                reward_function_str = f"1.0 if ({ensure_goal_str}) else {not_safe_str}"
            else:
                reward_function_str = f"1.0 if ({goal_formula_str}) else {not_safe_str}"

        return {
            "initial_condition": initial_condition,
            "program": program_ode_str,
            "goal": goal_formula_str,
            "loop_invariants": [inv.get("formula_str") for inv in loop_invariants],
            "differential_invariants": [],
            "differential_ghosts": [],
            "termination_condition_body": termination_condition_str,
            "reward_function_body": reward_function_str
        }

    def _process_definitions(self, model_ast: KeYmaeraModel):
        if not model_ast.entries:
            return
        archive_entry = model_ast.entries[0]

        for definition in archive_entry.definitions.defs:
            if isinstance(definition, FuncDef):
                if not definition.args and isinstance(definition.body, VarRef) and definition.body.name == definition.name:
                    # Bare declarations such as `Real ep;` are symbolic constants, not zero-arg functions.
                    # Keep them as constants so they do not generate self-recursive `self._ep_impl()` calls.
                    self.config.constants.append(ConstantDefinitionConfig(name=definition.name, value="1.0"))
                    continue
                self.func_definition_map[definition.name] = definition
            elif isinstance(definition, PredDef):
                self.pred_definition_map[definition.name] = definition
            elif isinstance(definition, HPDef):
                self.hp_definition_map[definition.name] = definition

        for pv_decl in archive_entry.program_variables.vars:
            var_name = pv_decl.name
            annotations = pv_decl.annotations
            const_annotation = annotations.get('const')
            if const_annotation:
                value = const_annotation.get('value')
                if value is None and const_annotation.get('bound'):
                    bounds = const_annotation['bound']
                    if isinstance(bounds, list) and len(bounds) == 2:
                        if bounds[0] == bounds[1]:
                            value = str(bounds[0])
                        else:
                            value = str(bounds[0])
                self.config.constants.append(ConstantDefinitionConfig(name=var_name, value=str(value if value is not None else var_name)))
            else:
                lower_bound = None
                upper_bound = None
                initial_value = None
                var_annotation = annotations.get('var')
                if var_annotation:
                    if var_annotation.get('bound'):
                        bounds = var_annotation['bound']
                        if isinstance(bounds, list) and len(bounds) == 2:
                            lower_bound = str(bounds[0])
                            upper_bound = str(bounds[1])
                    if var_annotation.get('initial_value'):
                        initial_value = str(var_annotation['initial_value'])
                self.config.variables.append(VariableDeclarationConfig(name=var_name, type=pv_decl.type, lower_bound=lower_bound, upper_bound=upper_bound, initial_value=initial_value))

        for definition in archive_entry.definitions.defs:
            if isinstance(definition, FuncDef):
                if not definition.args and isinstance(definition.body, VarRef) and definition.body.name == definition.name:
                    # Treat implicit constant declarations such as `Real ep;` as constants with a safe default value.
                    self.config.constants.append(ConstantDefinitionConfig(name=definition.name, value="1.0"))
                    continue

                const_annotation = definition.annotations.get('const')
                if const_annotation:
                    value = const_annotation.get('value')
                    if value is None and const_annotation.get('bound'):
                        bounds = const_annotation['bound']
                        if isinstance(bounds, list) and len(bounds) == 2:
                            value = str(bounds[0])
                    self.config.constants.append(ConstantDefinitionConfig(name=definition.name, value=str(value if value is not None else definition.name)))
                    continue
                body_str = self._term_to_string_python(definition.body)
                self.config.function_definitions.append(FuncDefConfig(name=definition.name, args=[arg.name for arg in definition.args], body=body_str))
            elif isinstance(definition, PredDef):
                body_str = self._formula_to_string_python(definition.body)
                self.config.function_definitions.append(FuncDefConfig(name=definition.name, args=[arg.name for arg in definition.args], body=body_str))
            elif isinstance(definition, HPDef):
                self.config.hp_definitions.append(HPDefConfig(name=definition.name, args=[], body=self._hp_to_string(definition.body)))

        for var_config in self.config.variables:
            if var_config.initial_value:
                self.config.initial_state_sampling_config[var_config.name] = {"value": var_config.initial_value}
            elif var_config.lower_bound is not None and var_config.upper_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {"min": var_config.lower_bound, "max": var_config.upper_bound}
            elif var_config.lower_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {"min": var_config.lower_bound, "max": f"({var_config.lower_bound}) + 1.0"}
            elif var_config.upper_bound is not None:
                self.config.initial_state_sampling_config[var_config.name] = {"min": f"({var_config.upper_bound}) - 1.0", "max": var_config.upper_bound}
            else:
                self.config.initial_state_sampling_config[var_config.name] = {"min": "-1.0", "max": "1.0"}

        for const_config in self.config.constants:
            self.config.initial_state_sampling_config[const_config.name] = {"value": const_config.value}

    def _process_problem(self, model_ast: KeYmaeraModel):
        if not model_ast.entries:
            return
        archive_entry = model_ast.entries[0]
        problem_block_formula = archive_entry.problem.formula
        extracted_components = self._extract_problem_components(problem_block_formula)
        self.config.problem = ProblemConfig(initial_condition=extracted_components["initial_condition"], program=extracted_components["program"], goal=extracted_components["goal"], loop_invariants=extracted_components["loop_invariants"], differential_invariants=extracted_components["differential_invariants"], differential_ghosts=extracted_components["differential_ghosts"])
        self.config.termination_condition_body = extracted_components["termination_condition_body"]
        self.config.reward_function_body = extracted_components["reward_function_body"]

    def _determine_action_and_observation_spaces(self):
        action_variables = []
        observation_variables = []
        if 'ar' in [v.name for v in self.config.variables]:
            action_variables.append('ar')
        for var_config in self.config.variables:
            observation_variables.append(var_config.name)
        if action_variables:
            self.config.action_space_type = "continuous"
            self.config.action_space_bounds = {var.name: {"min": var.lower_bound if var.lower_bound else "-float('inf')", "max": var.upper_bound if var.upper_bound else "float('inf')"} for var in self.config.variables if var.name in action_variables}
        else:
            self.config.action_space_type = "discrete"
            self.config.action_space_bounds = {"n": 1}
        if observation_variables:
            self.config.observation_space_type = "continuous"
            self.config.observation_space_bounds = {var.name: {"min": var.lower_bound if var.lower_bound is not None else "-float('inf')", "max": var.upper_bound if var.upper_bound is not None else "float('inf')"} for var in self.config.variables if var.name in observation_variables}
        else:
            self.config.observation_space_type = "continuous"
            self.config.observation_space_bounds = {"dummy_obs": {"min": -1.0, "max": 1.0}}

    def generate_environment_code(self, model_ast: KeYmaeraModel) -> str:
        code_template = ENVIRONMENT_TEMPLATE_CODE
        self._process_definitions(model_ast)
        self._process_problem(model_ast)
        self._determine_action_and_observation_spaces()
        import dataclasses
        config_dict = dataclasses.asdict(self.config)
        config_json_str = json.dumps(config_dict, indent=4)
        embedded_config_json_str = repr(textwrap.indent(config_json_str, '    '))

        user_defined_methods_code_lines = []
        user_defined_function_assignments_lines = []
        for func_def_config in self.config.function_definitions:
            args_str = ", ".join(func_def_config.args)
            func_name = func_def_config.name
            params = f"self, {args_str}" if args_str else "self"
            body_expr = func_def_config.body
            user_defined_methods_code_lines.append(
                f"    def _{func_name}_impl({params}):\n"
                f"        context = {{}}\n"
                f"        arg_names = {json.dumps(func_def_config.args)}\n"
                f"        for name, val in zip(arg_names, [{args_str}]):\n"
                f"            context[name] = val\n"
                f"        return self._evaluate_expression({json.dumps(body_expr)}, context)\n"
            )
            user_defined_function_assignments_lines.append(f"        self.user_defined_functions['{func_name}'] = self._{func_name}_impl\n")

        user_defined_methods_code_str = "\n".join(user_defined_methods_code_lines)
        user_defined_function_assignments_str = "\n".join(user_defined_function_assignments_lines)

        generated_code = code_template
        generated_code = generated_code.replace('{embedded_config_json_str}', embedded_config_json_str)
        generated_code = generated_code.replace('{metadata}', json.dumps(self.config.metadata, indent=4))
        generated_code = generated_code.replace('{user_defined_methods_code_str}', user_defined_methods_code_str)
        generated_code = generated_code.replace('{user_defined_function_assignments_str}', user_defined_function_assignments_str)
        generated_code = generated_code.replace('{{', '{').replace('}}', '}')

        return generated_code


def generate_gym_environment_file(dl_content: str, environment_name: str, output_filename: str):
    parser = KeYmaeraParser(dl_content)
    model_ast = parser.parse()
    generator = GymnasiumEnvironmentGenerator(environment_name)
    generated_code = generator.generate_environment_code(model_ast)
    if generated_code:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(generated_code)
        return True
    return False
