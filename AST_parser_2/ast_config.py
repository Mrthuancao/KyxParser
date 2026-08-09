from typing import List, Optional, Union, Dict, Any

class ASTNode:
    """Base class for all Abstract Syntax Tree nodes."""
    pass

# --- Term Nodes ---
class Term(ASTNode):
    pass

class Literal(Term):
    def __init__(self, value: Union[int, float, bool]):
        self.value = value
    def __repr__(self):
        return f"Literal({self.value})"

class VarRef(Term):
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"VarRef('{self.name}')"

class BinaryTermOp(Term):
    def __init__(self, op: str, left: Term, right: Term):
        self.op = op
        self.left = left
        self.right = right
    def __repr__(self):
        return f"BinaryTermOp('{self.op}', {self.left}, {self.right})"

class UnaryTermOp(Term):
    def __init__(self, op: str, operand: Term):
        self.op = op
        self.operand = operand
    def __repr__(self):
        return f"UnaryTermOp('{self.op}', {self.operand})"

class FunctionCall(Term):
    def __init__(self, name: str, args: List[Term]):
        self.name = name
        self.args = args
    def __repr__(self):
        return f"FunctionCall('{self.name}', {self.args})"

# --- Formula Nodes ---
class Formula(ASTNode):
    pass

class BooleanLiteral(Formula):
    def __init__(self, value: bool):
        self.value = value
    def __repr__(self):
        return f"BooleanLiteral({self.value})"

class BinaryFormulaOp(Formula):
    def __init__(self, op: str, left: Formula, right: Formula):
        self.op = op
        self.left = left
        self.right = right
    def __repr__(self):
        return f"BinaryFormulaOp('{self.op}',\n  {self.left},\n  {self.right})"

class UnaryFormulaOp(Formula):
    def __init__(self, op: str, operand: Formula):
        self.op = op
        self.operand = operand
    def __repr__(self):
        return f"UnaryFormulaOp('{self.op}', {self.operand})"

class Comparison(Formula):
    def __init__(self, op: str, left: Term, right: Term):
        self.op = op
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Comparison('{self.op}', {self.left}, {self.right})"

class Quantifier(Formula):
    def __init__(self, quantifier_type: str, variables: List[VarRef], body: Formula):
        self.quantifier_type = quantifier_type
        self.variables = variables
        self.body = body
    def __repr__(self):
        return f"Quantifier('{self.quantifier_type}', {self.variables}, {self.body})"

class ModalFormula(Formula):
    def __init__(self, modality: str, program: 'HybridProgram', formula: Formula):
        self.modality = modality
        self.program = program
        self.formula = formula
    def __repr__(self):
        return f"ModalFormula('{self.modality}',\n    program={self.program},\n    formula={self.formula})"

class PredicateCall(Formula):
    def __init__(self, name: str, args: List[Term]):
        self.name = name
        self.args = args
    def __repr__(self):
        return f"PredicateCall('{self.name}', {self.args})"

class C_Formula(Formula):
    def __init__(self, formula: Formula):
        self.formula = formula
    def __repr__(self):
        return f"C_Formula({self.formula})"

# --- Hybrid Program Nodes ---
class HybridProgram(ASTNode):
    pass

class Assignment(HybridProgram):
    def __init__(self, var: VarRef, expr: Term):
        self.var = var
        self.expr = expr
    def __repr__(self):
        return f"Assignment({self.var}, {self.expr})"

class NondetAssignment(HybridProgram):
    def __init__(self, var: VarRef):
        self.var = var
    def __repr__(self):
        return f"NondetAssignment({self.var})"

class Test(HybridProgram):
    def __init__(self, condition: Formula):
        self.condition = condition
    def __repr__(self):
        return f"Test({self.condition})"

class Sequence(HybridProgram):
    def __init__(self, programs: List[HybridProgram]):
        self.programs = programs
    def __repr__(self):
        return f"Sequence({self.programs})"

class Choice(HybridProgram):
    def __init__(self, left: HybridProgram, right: HybridProgram):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Choice(\n  left={self.left},\n  right={self.right})"

class Iteration(HybridProgram):
    def __init__(self, program: HybridProgram, invariant: Optional[Formula] = None):
        self.program = program
        self.invariant = invariant
    def __repr__(self):
        return f"Iteration(\n    program={self.program},\n    invariant={self.invariant})"

class ODE(HybridProgram):
    def __init__(self, equations: List['ODEEquation'], domain_constraint: Optional[Formula] = None):
        self.equations = equations
        self.domain_constraint = domain_constraint
    def __repr__(self):
        return f"ODE(\n      equations={self.equations},\n      domain_constraint={self.domain_constraint})"

class ODEEquation(ASTNode):
    def __init__(self, var: VarRef, expr: Term):
        self.var = var
        self.expr = expr
    def __repr__(self):
        return f"ODEEquation({self.var}, {self.expr})"

class HPBlock(HybridProgram):
    def __init__(self, program: HybridProgram):
        self.program = program
    def __repr__(self):
        return f"HPBlock(\n    program={self.program})"


# --- Model Structure Nodes ---
class Definition(ASTNode):
    def __init__(self, name: str, annotations: Dict[str, Any]):
        self.name = name
        self.annotations = annotations

class FuncDef(Definition):
    def __init__(self, name: str, type: str, body: Term, args: List[VarRef], annotations: Dict[str, Any]):
        super().__init__(name, annotations)
        self.type = type
        self.body = body
        self.args = args
    def __repr__(self):
        return f"FuncDef('{self.name}', '{self.type}', {self.body}, {self.args}, {self.annotations})"

class PredDef(Definition):
    def __init__(self, name: str, type: str, body: Formula, args: List[VarRef], annotations: Dict[str, Any]):
        super().__init__(name, annotations)
        self.type = type
        self.body = body
        self.args = args
    def __repr__(self):
        return f"PredDef('{self.name}', '{self.type}', {self.body}, {self.args}, {self.annotations})"

class HPDef(Definition):
    def __init__(self, name: str, body: HybridProgram, annotations: Dict[str, Any]):
        super().__init__(name, annotations)
        self.body = body
    def __repr__(self):
        return f"HPDef('{self.name}', {self.body}, {self.annotations})"

class DefinitionsBlock(ASTNode):
    def __init__(self, defs: List[Definition]):
        self.defs = defs
    def __repr__(self):
        return f"DefinitionsBlock({self.defs})"

class ProgramVariableDecl(ASTNode):
    def __init__(self, type: str, name: str, annotations: Dict[str, Any]):
        self.type = type
        self.name = name
        self.annotations = annotations
    def __repr__(self):
        return f"ProgramVariableDecl('{self.type}', '{self.name}', {self.annotations})"

class ProgramVariablesBlock(ASTNode):
    def __init__(self, vars: List[ProgramVariableDecl]):
        self.vars = vars
    def __repr__(self):
        return f"ProgramVariablesBlock({self.vars})"

class ProblemBlock(ASTNode):
    def __init__(self, formula: Formula):
        self.formula = formula
    def __repr__(self):
        return f"ProblemBlock({self.formula})"

class TacticBlock(ASTNode):
    def __init__(self, name: str, args: List[str]):
        self.name = name
        self.args = args
    def __repr__(self):
        return f"TacticBlock('{self.name}', {self.args})"

class ArchiveEntry(ASTNode):
    def __init__(self, kind: str, name: str, definitions: DefinitionsBlock, program_variables: ProgramVariablesBlock, problem: ProblemBlock, tactics: List[TacticBlock]):
        self.kind = kind
        self.name = name
        self.definitions = definitions
        self.program_variables = program_variables
        self.problem = problem
        self.tactics = tactics
    def __repr__(self):
        return (f"ArchiveEntry(kind='{self.kind}', name='{self.name}',\n"
                f"  definitions={self.definitions},\n"
                f"  program_variables={self.program_variables},\n"
                f"  problem={self.problem},\n"
                f"  tactics={self.tactics})")

class KeYmaeraModel(ASTNode):
    def __init__(self, entries: List[ArchiveEntry]):
        self.entries = entries
    def __repr__(self):
        return f"KeYmaeraModel({self.entries})"