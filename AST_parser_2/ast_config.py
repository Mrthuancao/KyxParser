#file description: This file contains the AST node classes for representing the structure of KeYmaera models, including terms, formulas, hybrid programs, and model definitions. Each class includes a method for generating an indented string representation of the node and its children.
#name: ast_config.py
from typing import List, Optional, Union, Dict, Any

# Helper for indentation
def _indent(text: str, level: int):
    prefix = "  " * level
    return "\\n".join(prefix + line for line in text.splitlines())

class ASTNode:
    """Base class for all Abstract Syntax Tree nodes."""
    def __repr__(self):
        # Public __repr__ calls the indented version starting at level 0
        return self._repr_indented(0)

    def _repr_indented(self, indent_level: int):
        # Default implementation for simple nodes, can be overridden
        # Fallback to default repr if not specifically handled
        return super().__repr__()

# --- Term Nodes ---
class Term(ASTNode):
    pass

class Literal(Term):
    def __init__(self, value: Union[int, float, bool]):
        self.value = value
    def _repr_indented(self, indent_level: int):
        return f"Literal({self.value})"

class VarRef(Term):
    def __init__(self, name: str):
        self.name = name
    def _repr_indented(self, indent_level: int):
        return f"VarRef('{self.name}')"

class BinaryTermOp(Term):
    def __init__(self, op: str, left: Term, right: Term):
        self.op = op
        self.left = left
        self.right = right
    def _repr_indented(self, indent_level: int):
        left_repr = _indent(self.left._repr_indented(indent_level + 1), 0)
        right_repr = _indent(self.right._repr_indented(indent_level + 1), 0)
        return (f"BinaryTermOp('{self.op}',\\n"
                f"{left_repr},\\n"
                f"{right_repr})")

class UnaryTermOp(Term):
    def __init__(self, op: str, operand: Term):
        self.op = op
        self.operand = operand
    def _repr_indented(self, indent_level: int):
        operand_repr = _indent(self.operand._repr_indented(indent_level + 1), 0)
        return (f"UnaryTermOp('{self.op}',\\n"
                f"{operand_repr})")

class FunctionCall(Term):
    def __init__(self, name: str, args: List[Term]):
        self.name = name
        self.args = args
    def _repr_indented(self, indent_level: int):
        args_repr = ",\\n".join(_indent(arg._repr_indented(indent_level + 1), 0) for arg in self.args)
        return (f"FunctionCall('{{self.name}}', [\\n"
                f"{args_repr}\\n"
                f"{'  ' * indent_level}])")

# --- Formula Nodes ---
class Formula(ASTNode):
    pass

class BooleanLiteral(Formula):
    def __init__(self, value: bool):
        self.value = value
    def _repr_indented(self, indent_level: int):
        return f"BooleanLiteral({self.value})"

class BinaryFormulaOp(Formula):
    def __init__(self, op: str, left: Formula, right: Formula):
        self.op = op
        self.left = left
        self.right = right
    def _repr_indented(self, indent_level: int):
        left_repr = _indent(self.left._repr_indented(indent_level + 1), 0)
        right_repr = _indent(self.right._repr_indented(indent_level + 1), 0)
        return (f"BinaryFormulaOp('{self.op}',\\n"
                f"{left_repr},\\n"
                f"{right_repr})")

class UnaryFormulaOp(Formula):
    def __init__(self, op: str, operand: Formula):
        self.op = op
        self.operand = operand
    def _repr_indented(self, indent_level: int):
        operand_repr = _indent(self.operand._repr_indented(indent_level + 1), 0)
        return (f"UnaryFormulaOp('{self.op}',\\n"
                f"{operand_repr})")

class Comparison(Formula):
    def __init__(self, op: str, left: Term, right: Term):
        self.op = op
        self.left = left
        self.right = right
    def _repr_indented(self, indent_level: int):
        left_repr = _indent(self.left._repr_indented(indent_level + 1), 0)
        right_repr = _indent(self.right._repr_indented(indent_level + 1), 0)
        return (f"Comparison('{{self.op}}',\\n"
                f"{left_repr},\\n"
                f"{right_repr})")

class Quantifier(Formula):
    def __init__(self, quantifier_type: str, variables: List[VarRef], body: Formula):
        self.quantifier_type = quantifier_type
        self.variables = variables
        self.body = body
    def _repr_indented(self, indent_level: int):
        vars_repr = ", ".join(v._repr_indented(0) for v in self.variables)
        body_repr = _indent(self.body._repr_indented(indent_level + 1), 0)
        return (f"Quantifier('{{self.quantifier_type}}', [{{vars_repr}}],\\n"
                f"{{body_repr}})")

class ModalFormula(Formula):
    def __init__(self, modality: str, program: 'HybridProgram', formula: Formula):
        self.modality = modality
        self.program = program
        self.formula = formula
    def _repr_indented(self, indent_level: int):
        program_repr = _indent(self.program._repr_indented(indent_level + 2), 0)
        formula_repr = _indent(self.formula._repr_indented(indent_level + 2), 0)
        return (f"ModalFormula('{{self.modality}}',\\n"
                f"{{'  ' * (indent_level + 1)}}program={{program_repr}},\\n"
                f"{{'  ' * (indent_level + 1)}}formula={{formula_repr}})")

class PredicateCall(Formula):
    def __init__(self, name: str, args: List[Term]):
        self.name = name
        self.args = args
    def _repr_indented(self, indent_level: int):
        if self.args:
            args_repr = ",\\n".join(_indent(arg._repr_indented(indent_level + 1), 0) for arg in self.args)
            return (f"PredicateCall('{{self.name}}', [\\n"
                    f"{{args_repr}}\\n"
                    f"{{'  ' * indent_level}}])")
        return f"PredicateCall('{{self.name}}', [])"

class C_Formula(Formula):
    def __init__(self, formula: Formula):
        self.formula = formula
    def _repr_indented(self, indent_level: int):
        formula_repr = _indent(self.formula._repr_indented(indent_level + 1), 0)
        return (f"C_Formula(\\n"
                f"{formula_repr})")

# --- Hybrid Program Nodes ---
class HybridProgram(ASTNode):
    pass

class Assignment(HybridProgram):
    def __init__(self, var: VarRef, expr: Term):
        self.var = var
        self.expr = expr
    def _repr_indented(self, indent_level: int):
        var_repr = _indent(self.var._repr_indented(indent_level + 1), 0)
        expr_repr = _indent(self.expr._repr_indented(indent_level + 1), 0)
        return (f"Assignment(\\n"
                f"{var_repr},\\n"
                f"{expr_repr})")

class NondetAssignment(HybridProgram):
    def __init__(self, var: VarRef):
        self.var = var
    def _repr_indented(self, indent_level: int):
        var_repr = _indent(self.var._repr_indented(indent_level + 1), 0)
        return f"NondetAssignment({var_repr})"

class Test(HybridProgram):
    def __init__(self, condition: Formula):
        self.condition = condition
    def _repr_indented(self, indent_level: int):
        condition_repr = _indent(self.condition._repr_indented(indent_level + 1), 0)
        return (f"Test(\\n"
                f"{condition_repr})")

class Sequence(HybridProgram):
    def __init__(self, programs: List[HybridProgram]):
        self.programs = programs
    def _repr_indented(self, indent_level: int):
        programs_repr = ",\\n".join(_indent(p._repr_indented(indent_level + 1), 0) for p in self.programs)
        return (f"Sequence([\\n"
                f"{programs_repr}\\n"
                f"{'  ' * indent_level}])")

class Choice(HybridProgram):
    def __init__(self, left: HybridProgram, right: HybridProgram):
        self.left = left
        self.right = right
    def _repr_indented(self, indent_level: int):
        left_repr = _indent(self.left._repr_indented(indent_level + 2), 0)
        right_repr = _indent(self.right._repr_indented(indent_level + 2), 0)
        return (f"Choice(\\n"
                f"{{'  ' * (indent_level + 1)}}left={{left_repr}},\\n"
                f"{{'  ' * (indent_level + 1)}}right={{right_repr}})")

class Iteration(HybridProgram):
    def __init__(self, program: HybridProgram, invariant: Optional[Formula] = None):
        self.program = program
        self.invariant = invariant
    def _repr_indented(self, indent_level: int):
        program_repr = _indent(self.program._repr_indented(indent_level + 2), 0)
        invariant_repr = _indent(self.invariant._repr_indented(indent_level + 2), 0) if self.invariant else "None"
        return (f"Iteration(\\n"
                f"{{'  ' * (indent_level + 1)}}program={{program_repr}},\\n"
                f"{{'  ' * (indent_level + 1)}}invariant={{invariant_repr}})")

class ODE(HybridProgram):
    def __init__(self, equations: List['ODEEquation'], domain_constraint: Optional[Formula] = None):
        self.equations = equations
        self.domain_constraint = domain_constraint
    def _repr_indented(self, indent_level: int):
        equations_repr = ",\\n".join(_indent(eq._repr_indented(indent_level + 3), 0) for eq in self.equations)
        domain_repr = _indent(self.domain_constraint._repr_indented(indent_level + 3), 0) if self.domain_constraint else "None"
        return (f"ODE(\\n"
                f"{{'  ' * (indent_level + 1)}}equations=[\\n"
                f"{{equations_repr}}\\n"
                f"{{'  ' * (indent_level + 1)}}],\\n"
                f"{{'  ' * (indent_level + 1)}}domain_constraint={{domain_repr}})")

class ODEEquation(ASTNode):
    def __init__(self, var: VarRef, expr: Term):
        self.var = var
        self.expr = expr
    def _repr_indented(self, indent_level: int):
        var_repr = self.var._repr_indented(0)
        expr_repr = self.expr._repr_indented(0)
        return f"ODEEquation({{var_repr}}, {{expr_repr}})"

class HPBlock(HybridProgram):
    def __init__(self, program: HybridProgram):
        self.program = program
    def _repr_indented(self, indent_level: int):
        program_repr = _indent(self.program._repr_indented(indent_level + 2), 0)
        return (f"HPBlock(\\n"
                f"{{'  ' * (indent_level + 1)}}program={{program_repr}})")


# --- Model Structure Nodes ---
class Definition(ASTNode):
    def __init__(self, name: str, annotations: Dict[str, Any]):
        self.name = name
        self.annotations = annotations
    def _repr_indented(self, indent_level: int):
        return f"Definition('{{self.name}}', {{self.annotations}})" # Default, subclasses override

class FuncDef(Definition):
    def __init__(self, name: str, type: str, body: Term, args: List[VarRef], annotations: Dict[str, Any]):
        super().__init__(name, annotations)
        self.type = type
        self.body = body
        self.args = args
    def _repr_indented(self, indent_level: int):
        body_repr = _indent(self.body._repr_indented(indent_level + 1), 0)
        args_repr = ", ".join(arg._repr_indented(0) for arg in self.args)
        return (f"FuncDef('{{self.name}}', '{{self.type}}',\\n"
                f"{body_repr},\\n"
                f"{{'  ' * (indent_level + 1)}}[{args_repr}],\\n"
                f"{{'  ' * (indent_level + 1)}}{self.annotations})")

class PredDef(Definition):
    def __init__(self, name: str, type: str, body: Formula, args: List[VarRef], annotations: Dict[str, Any]):
        super().__init__(name, annotations)
        self.type = type
        self.body = body
        self.args = args
    def _repr_indented(self, indent_level: int):
        body_repr = _indent(self.body._repr_indented(indent_level + 1), 0)
        args_repr = ", ".join(arg._repr_indented(0) for arg in self.args)
        return (f"PredDef('{{self.name}}', '{{self.type}}',\\n"
                f"{body_repr},\\n"
                f"{{'  ' * (indent_level + 1)}}[{args_repr}],\\n"
                f"{{'  ' * (indent_level + 1)}}{self.annotations})")

class HPDef(Definition):
    def __init__(self, name: str, body: HybridProgram, annotations: Dict[str, Any]):
        super().__init__(name, annotations)
        self.body = body
    def _repr_indented(self, indent_level: int):
        body_repr = _indent(self.body._repr_indented(indent_level + 1), 0)
        return (f"HPDef('{{self.name}}',\\n"
                f"{body_repr},\\n"
                f"{{'  ' * (indent_level + 1)}}{self.annotations})")

class DefinitionsBlock(ASTNode):
    def __init__(self, defs: List[Definition]):
        self.defs = defs
    def _repr_indented(self, indent_level: int):
        defs_repr = ",\\n".join(_indent(d._repr_indented(indent_level + 1), 0) for d in self.defs)
        return (f"DefinitionsBlock([\\n"
                f"{defs_repr}\\n"
                f"{'  ' * indent_level}])")

class ProgramVariableDecl(ASTNode):
    def __init__(self, type: str, name: str, annotations: Dict[str, Any]):
        self.type = type
        self.name = name
        self.annotations = annotations
    def _repr_indented(self, indent_level: int):
        return f"ProgramVariableDecl('{{self.type}}', '{{self.name}}', {{self.annotations}})"

class ProgramVariablesBlock(ASTNode):
    def __init__(self, vars: List[ProgramVariableDecl]):
        self.vars = vars
    def _repr_indented(self, indent_level: int):
        vars_repr = ",\\n".join(_indent(v._repr_indented(indent_level + 1), 0) for v in self.vars)
        return (f"ProgramVariablesBlock([\\n"
                f"{vars_repr}\\n"
                f"{'  ' * indent_level}])")

class ProblemBlock(ASTNode):
    def __init__(self, formula: Formula):
        self.formula = formula
    def _repr_indented(self, indent_level: int):
        formula_repr = _indent(self.formula._repr_indented(indent_level + 1), 0)
        return (f"ProblemBlock(\\n"
                f"{formula_repr})")

class TacticBlock(ASTNode):
    def __init__(self, name: str, args: List[str]):
        self.name = name
        self.args = args
    def _repr_indented(self, indent_level: int):
        return f"TacticBlock('{{self.name}}', {{self.args}})"

class ArchiveEntry(ASTNode):
    def __init__(self, kind: str, name: str, definitions: DefinitionsBlock, program_variables: ProgramVariablesBlock, problem: ProblemBlock, tactics: List[TacticBlock]):
        self.kind = kind
        self.name = name
        self.definitions = definitions
        self.program_variables = program_variables
        self.problem = problem
        self.tactics = tactics
    def _repr_indented(self, indent_level: int):
        defs_repr = _indent(self.definitions._repr_indented(indent_level + 1), 0)
        pv_repr = _indent(self.program_variables._repr_indented(indent_level + 1), 0)
        problem_repr = _indent(self.problem._repr_indented(indent_level + 1), 0)
        tactics_repr = ",\\n".join(_indent(t._repr_indented(indent_level + 1), 0) for t in self.tactics)
        return (f"ArchiveEntry(kind='{{self.kind}}', name='{{self.name}}',\\n"
                f"{{'  ' * (indent_level + 1)}}definitions={{defs_repr}},\\n"
                f"{{'  ' * (indent_level + 1)}}program_variables={{pv_repr}},\\n"
                f"{{'  ' * (indent_level + 1)}}problem={{problem_repr}},\\n"
                f"{{'  ' * (indent_level + 1)}}tactics=[\\n"
                f"{tactics_repr}\\n"
                f"{{'  ' * (indent_level + 1)}}])")

class KeYmaeraModel(ASTNode):
    def __init__(self, entries: List[ArchiveEntry]):
        self.entries = entries

    def _repr_indented(self, indent_level: int):
        entries_repr = ",\n".join(_indent(e._repr_indented(indent_level + 1), 0) for e in self.entries)
        return (f"KeYmaeraModel([\n"
                f"{entries_repr}\n"
                f"{'  ' * indent_level}])")