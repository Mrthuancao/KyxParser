import abc
from typing import List, Optional, Union, Dict, Any

# --- Helper for hashing annotations ---
def _hash_annotations(annotations: Dict[str, Any]) -> int:
    """
    Converts a dictionary of annotations into a hashable format.
    Handles nested dictionaries and lists by converting them to frozensets and tuples.
    """
    hashable_items = []
    for k, v in annotations.items():
        if isinstance(v, dict):
            # If a value is a dict, convert its items to a frozenset
            hashable_items.append((k, frozenset(v.items())))
        elif isinstance(v, list):
            # If a value is a list, convert it to a tuple
            hashable_items.append((k, tuple(v)))
        else:
            hashable_items.append((k, v))
    return hash(frozenset(hashable_items))

# Base AST Node
class ASTNode(abc.ABC):
    """Abstract base class for all nodes in the Abstract Syntax Tree."""
    # No attributes, so no __eq__ or __hash__ needed at this level.
    # Subclasses will implement their own based on their attributes.
    def __repr__(self, indent=0):
        # Default representation, can be overridden by subclasses
        return f"{self.__class__.__name__}()"

# KeYmaeraModel and ArchiveEntry
class KeYmaeraModel(ASTNode):
    """Represents the top-level KeYmaera model, containing a list of archive entries."""
    def __init__(self, entries: List['ArchiveEntry']):
        self.entries = entries
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.entries == other.entries
    def __hash__(self):
        return hash(tuple(self.entries)) # Hash of tuple of entries
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        next_indent_str = "  " * (indent + 1)
        entries_str = ",\n".join(entry.__repr__(indent + 1) for entry in self.entries)
        return (f"{indent_str}KeYmaeraModel(\n"
                f"{next_indent_str}entries=[\n{entries_str}\n{next_indent_str}]\n"
                f"{indent_str})")

class ArchiveEntry(ASTNode):
    """Represents a single entry in the KeYmaera model archive (e.g., a Theorem or Lemma)."""
    def __init__(self, kind: str, name: str, definitions: 'DefinitionsBlock',
                 program_variables: 'ProgramVariablesBlock', problem: 'ProblemBlock',
                 tactics: List['TacticBlock']):
        self.kind = kind
        self.name = name
        self.definitions = definitions
        self.program_variables = program_variables
        self.problem = problem
        self.tactics = tactics
    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.kind == other.kind and
                self.name == other.name and
                self.definitions == other.definitions and
                self.program_variables == other.program_variables and
                self.problem == other.problem and
                self.tactics == other.tactics)
    def __hash__(self):
        return hash((self.kind, self.name, self.definitions,
                     self.program_variables, self.problem, tuple(self.tactics)))
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        next_indent_str = "  " * (indent + 1)
        tactics_str = ", ".join(tactic.__repr__(indent + 1) for tactic in self.tactics)
        return (f"{indent_str}ArchiveEntry(kind='{self.kind}', name='{self.name}',\n"
                f"{next_indent_str}definitions={self.definitions.__repr__(indent + 1)},\n"
                f"{next_indent_str}program_variables={self.program_variables.__repr__(indent + 1)},\n"
                f"{next_indent_str}problem={self.problem.__repr__(indent + 1)},\n"
                f"{next_indent_str}tactics=[\n{next_indent_str}  {tactics_str}\n{next_indent_str}])")

# Definitions Block
class DefinitionsBlock(ASTNode):
    """Contains a list of definitions (functions, predicates, hybrid programs)."""
    def __init__(self, defs: List['Definition']):
        self.defs = defs
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.defs == other.defs
    def __hash__(self):
        return hash(tuple(self.defs))
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        next_indent_str = "  " * (indent + 1)
        defs_str = ",\n".join(d.__repr__(indent + 1) for d in self.defs)
        return (f"{indent_str}DefinitionsBlock(\n"
                f"{next_indent_str}defs=[\n{defs_str}\n{next_indent_str}])")

class Definition(ASTNode, abc.ABC):
    """Abstract base class for all definition types."""
    def __init__(self, name: str, args: Optional[List['VarRef']] = None,
                 annotations: Optional[Dict[str, Any]] = None):
        self.name = name
        self.args = args if args is not None else []
        self.annotations = annotations if annotations is not None else {}
    def __eq__(self, other):
        # For an abstract base class, ensure exact type match for comparison
        if not isinstance(other, Definition) or self.__class__ is not other.__class__:
            return NotImplemented
        return (self.name == other.name and
                self.args == other.args and
                self.annotations == other.annotations)
    def __hash__(self):
        return hash((self.name, tuple(self.args), _hash_annotations(self.annotations)))
    def __repr__(self, indent=0):
        # This is a base class, concrete subclasses will provide more specific reprs
        return (f"Definition(name='{self.name}', args={self.args.__repr__(indent + 1) if self.args else '[]'}, "
                f"annotations={self.annotations})")

class FuncDef(Definition):
    """Represents a function definition."""
    def __init__(self, name: str, type: str, body: 'Term',
                 args: Optional[List['VarRef']] = None, annotations: Optional[Dict[str, Any]] = None):
        super().__init__(name, args, annotations)
        self.type = type
        self.body = body
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (super().__eq__(other) and # Compare inherited attributes
                self.type == other.type and
                self.body == other.body)
    def __hash__(self):
        return hash((super().__hash__(), self.type, self.body))
    def __repr__(self, indent=0):
        args_str = ", ".join(arg.__repr__(indent + 1) for arg in self.args)
        return (f"FuncDef(name='{self.name}', type='{self.type}', body={self.body.__repr__(indent + 1)}, "
                f"args=[{args_str}], annotations={self.annotations})")

class PredDef(Definition):
    """Represents a predicate definition."""
    def __init__(self, name: str, type: str, body: 'Formula',
                 args: Optional[List['VarRef']] = None, annotations: Optional[Dict[str, Any]] = None):
        super().__init__(name, args, annotations)
        self.type = type
        self.body = body
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (super().__eq__(other) and
                self.type == other.type and
                self.body == other.body)
    def __hash__(self):
        return hash((super().__hash__(), self.type, self.body))
    def __repr__(self, indent=0):
        args_str = ", ".join(arg.__repr__(indent + 1) for arg in self.args)
        return (f"PredDef(name='{self.name}', type='{self.type}', body={self.body.__repr__(indent + 1)}, "
                f"args=[{args_str}], annotations={self.annotations})")

class HPDef(Definition):
    """Represents a hybrid program definition."""
    def __init__(self, name: str, body: 'HybridProgram',
                 args: Optional[List['VarRef']] = None, annotations: Optional[Dict[str, Any]] = None):
        super().__init__(name, args, annotations)
        self.body = body
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (super().__eq__(other) and
                self.body == other.body)
    def __hash__(self):
        return hash((super().__hash__(), self.body))
    def __repr__(self, indent=0):
        args_str = ", ".join(arg.__repr__(indent + 1) for arg in self.args)
        return (f"HPDef(name='{self.name}', body={self.body.__repr__(indent + 1)}, "
                f"args=[{args_str}], annotations={self.annotations})")

# Program Variables Block
class ProgramVariablesBlock(ASTNode):
    """Contains a list of program variable declarations."""
    def __init__(self, vars: List['ProgramVariableDecl']):
        self.vars = vars
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.vars == other.vars
    def __hash__(self):
        return hash(tuple(self.vars))
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        next_indent_str = "  " * (indent + 1)
        vars_str = ",\n".join(v.__repr__(indent + 1) for v in self.vars)
        return (f"{indent_str}ProgramVariablesBlock(\n"
                f"{next_indent_str}vars=[\n{vars_str}\n{next_indent_str}])")

class ProgramVariableDecl(ASTNode):
    """Represents a declaration of a program variable."""
    def __init__(self, type: str, name: str, annotations: Optional[Dict[str, Any]] = None):
        self.type = type
        self.name = name
        self.annotations = annotations if annotations is not None else {}
    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.type == other.type and
                self.name == other.name and
                self.annotations == other.annotations)
    def __hash__(self):
        return hash((self.type, self.name, _hash_annotations(self.annotations)))
    def __repr__(self, indent=0):
        return (f"ProgramVariableDecl(type='{self.type}', name='{self.name}', "
                f"annotations={self.annotations})")

# Problem Block
class ProblemBlock(ASTNode):
    """Contains the main formula representing the problem statement."""
    def __init__(self, formula: 'Formula'):
        self.formula = formula
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.formula == other.formula
    def __hash__(self):
        return hash(self.formula)
    def __repr__(self, indent=0):
        return f"ProblemBlock(formula={self.formula.__repr__(indent + 1)})"

# Tactic Block
class TacticBlock(ASTNode):
    """Represents a block of tactic code."""
    def __init__(self, name: str, tactic_code: str):
        self.name = name
        self.tactic_code = tactic_code
    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.name == other.name and
                self.tactic_code == other.tactic_code)
    def __hash__(self):
        return hash((self.name, self.tactic_code))
    def __repr__(self, indent=0):
        return f"TacticBlock(name='{self.name}', tactic_code='{self.tactic_code}')"

# Terms
class Term(ASTNode, abc.ABC):
    """Abstract base class for all term types."""
    pass

class Literal(Term):
    """Represents a literal value (integer, float, or fraction)."""
    def __init__(self, value: Union[int, float, 'Fraction']): # Assuming Fraction is a custom type or from fractions module
        self.value = value
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.value == other.value
    def __hash__(self):
        return hash(self.value)
    def __repr__(self, indent=0):
        return f"Literal({self.value})"

class VarRef(Term):
    """Represents a reference to a variable."""
    def __init__(self, name: str):
        self.name = name
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name
    def __hash__(self):
        return hash(self.name)
    def __repr__(self, indent=0): # FIX: Added indent=0
        return f"VarRef('{self.name}')"

class BinaryTermOp(Term):
    """Represents a binary operation on terms (e.g., +, -, *)."""
    def __init__(self, operator: str, left: Term, right: Term):
        self.operator = operator
        self.left = left
        self.right = right
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.operator == other.operator and self.left == other.left and self.right == other.right
    def __hash__(self):
        return hash((self.operator, self.left, self.right))
    def __repr__(self, indent=0):
        return (f"BinaryTermOp('{self.operator}', "
                f"{self.left.__repr__(indent + 1)}, "
                f"{self.right.__repr__(indent + 1)})")

class UnaryTermOp(Term):
    """Represents a unary operation on a term (e.g., -, derivative)."""
    def __init__(self, operator: str, operand: Term):
        self.operator = operator
        self.operand = operand
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.operator == other.operator and self.operand == other.operand
    def __hash__(self):
        return hash((self.operator, self.operand))
    def __repr__(self, indent=0):
        return f"UnaryTermOp('{self.operator}', {self.operand.__repr__(indent + 1)})"

class FunctionCall(Term):
    """Represents a function call with arguments."""
    def __init__(self, name: str, args: List[Term]):
        self.name = name
        self.args = args
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name and self.args == other.args
    def __hash__(self):
        return hash((self.name, tuple(self.args)))
    def __repr__(self, indent=0):
        # indent_str = "  " * indent # Not used here, but kept for consistency if needed
        args_str = ", ".join(arg.__repr__(indent + 1) if hasattr(arg, '__repr__') else repr(arg) for arg in self.args)
        return f"FunctionCall('{self.name}', [{args_str}])"

# Formulas
class Formula(ASTNode, abc.ABC):
    """Abstract base class for all formula types."""
    pass

class BooleanLiteral(Formula):
    """Represents a boolean literal (true or false)."""
    def __init__(self, value: bool):
        self.value = value
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.value == other.value
    def __hash__(self):
        return hash(self.value)
    def __repr__(self, indent=0):
        return f"BooleanLiteral({self.value})"

class BinaryFormulaOp(Formula):
    """Represents a binary operation on formulas (e.g., &, |, ->)."""
    def __init__(self, operator: str, left: Formula, right: Formula):
        self.operator = operator
        self.left = left
        self.right = right
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.operator == other.operator and self.left == other.left and self.right == other.right
    def __hash__(self):
        return hash((self.operator, self.left, self.right))
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        return (f"BinaryFormulaOp('{self.operator}',\n{indent_str}  {self.left.__repr__(indent + 1)},\n"
                f"{indent_str}  {self.right.__repr__(indent + 1)})")

class UnaryFormulaOp(Formula):
    """Represents a unary operation on a formula (e.g., !)."""
    def __init__(self, operator: str, operand: Formula):
        self.operator = operator
        self.operand = operand
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.operator == other.operator and self.operand == other.operand
    def __hash__(self):
        return hash((self.operator, self.operand))
    def __repr__(self, indent=0):
        return f"UnaryFormulaOp('{self.operator}', {self.operand.__repr__(indent + 1)})"

class Comparison(Formula):
    """Represents a comparison between two terms (e.g., =, >, <=)."""
    def __init__(self, operator: str, left: Term, right: Term):
        self.operator = operator
        self.left = left
        self.right = right
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.operator == other.operator and self.left == other.left and self.right == other.right
    def __hash__(self):
        return hash((self.operator, self.left, self.right))
    def __repr__(self, indent=0):
        return (f"Comparison('{self.operator}', {self.left.__repr__(indent + 1)}, "
                f"{self.right.__repr__(indent + 1)})")

class Quantifier(Formula):
    """Represents a quantified formula (forall or exists)."""
    def __init__(self, quantifier_type: str, variables: List[VarRef], body: Formula): # Changed 'variable' to 'variables' and type hint to List
        self.quantifier_type = quantifier_type
        self.variables = variables # Changed 'variable' to 'variables'
        self.body = body
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.quantifier_type == other.quantifier_type and self.variables == other.variables and self.body == other.body
    def __hash__(self):
        return hash((self.quantifier_type, tuple(self.variables), self.body)) # Hash tuple of variables
    def __repr__(self, indent=0):
        vars_str = ', '.join(v.__repr__(indent + 1) for v in self.variables)
        return f"Quantifier('{self.quantifier_type}', [{vars_str}], {self.body.__repr__(indent + 1)})"

class ModalFormula(Formula):
    """Represents a modal formula (box or diamond)."""
    def __init__(self, modality: str, program: 'HybridProgram', formula: Formula):
        self.modality = modality
        self.program = program
        self.formula = formula
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.modality == other.modality and self.program == other.program and self.formula == other.formula
    def __hash__(self):
        return hash((self.modality, self.program, self.formula))
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        return (f"ModalFormula('{self.modality}',\n{indent_str}  program={self.program.__repr__(indent + 1)},\n"
                f"{indent_str}  formula={self.formula.__repr__(indent + 1)})")

class PredicateCall(Formula):
    """Represents a predicate call with arguments."""
    def __init__(self, name: str, args: List[Term]):
        self.name = name
        self.args = args
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name and self.args == other.args
    def __hash__(self):
        return hash((self.name, tuple(self.args)))
    def __repr__(self, indent=0):
        # indent_str = "  " * indent # Not used here, but kept for consistency if needed
        args_str = ", ".join(arg.__repr__(indent + 1) if hasattr(arg, '__repr__') else repr(arg) for arg in self.args)
        return f"PredicateCall('{self.name}', [{args_str}])"

class C_Formula(Formula):
    """Represents a C-style formula (e.g., for differential invariants)."""
    def __init__(self, body: Formula):
        self.body = body
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.body == other.body
    def __hash__(self):
        return hash(self.body)
    def __repr__(self, indent=0):
        return f"C_Formula({self.body.__repr__(indent + 1)})"

# Hybrid Programs
class HybridProgram(ASTNode, abc.ABC):
    """Abstract base class for all hybrid program types."""
    pass

class Assignment(HybridProgram):
    """Represents an assignment statement."""
    def __init__(self, variable: VarRef, expression: Term):
        self.variable = variable
        self.expression = expression
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.variable == other.variable and self.expression == other.expression
    def __hash__(self):
        return hash((self.variable, self.expression))
    def __repr__(self, indent=0):
        return (f"Assignment({self.variable.__repr__(indent + 1)}, "
                f"{self.expression.__repr__(indent + 1)})")

class NondetAssignment(HybridProgram):
    """Represents a non-deterministic assignment statement."""
    def __init__(self, variable: VarRef):
        self.variable = variable
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.variable == other.variable
    def __hash__(self):
        return hash(self.variable)
    def __repr__(self, indent=0):
        return f"NondetAssignment({self.variable.__repr__(indent + 1)})"

class Test(HybridProgram):
    """Represents a test statement (condition)."""
    def __init__(self, condition: Formula):
        self.condition = condition
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.condition == other.condition
    def __hash__(self):
        return hash(self.condition)
    def __repr__(self, indent=0):
        return f"Test({self.condition.__repr__(indent + 1)})"

class Sequence(HybridProgram):
    """Represents a sequence of hybrid programs."""
    def __init__(self, programs: List[HybridProgram]):
        self.programs = programs
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.programs == other.programs
    def __hash__(self):
        return hash(tuple(self.programs))
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        programs_str = ",\n".join(f"{indent_str}  {p.__repr__(indent + 1)}" for p in self.programs)
        return f"Sequence([\n{programs_str}\n{indent_str}])"

class Choice(HybridProgram):
    """Represents a non-deterministic choice between two hybrid programs."""
    def __init__(self, left: HybridProgram, right: HybridProgram):
        self.left = left
        self.right = right
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.left == other.left and self.right == other.right
    def __hash__(self):
        return hash((self.left, self.right))
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        return (f"Choice(\n{indent_str}  left={self.left.__repr__(indent + 1)},\n"
                f"{indent_str}  right={self.right.__repr__(indent + 1)})")

class Iteration(HybridProgram):
    """Represents an iteration (loop) of a hybrid program."""
    def __init__(self, program: HybridProgram, invariant: Optional[Formula] = None):
        self.program = program
        self.invariant = invariant
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.program == other.program and self.invariant == other.invariant
    def __hash__(self):
        return hash((self.program, self.invariant))
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        invariant_repr = self.invariant.__repr__(indent + 1) if self.invariant else 'None'
        return (f"Iteration(\n{indent_str}  program={self.program.__repr__(indent + 1)},\n"
                f"{indent_str}  invariant={invariant_repr})")

class ODE(HybridProgram):
    """Represents a differential equation block."""
    def __init__(self, equations: List['ODEEquation'], domain_constraint: Optional[Formula] = None):
        self.equations = equations
        self.domain_constraint = domain_constraint
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.equations == other.equations and self.domain_constraint == other.domain_constraint
    def __hash__(self):
        return hash((tuple(self.equations), self.domain_constraint))
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        equations_str = ",\n".join(f"{indent_str}  {eq.__repr__(indent + 1)}" for eq in self.equations)
        domain_repr = self.domain_constraint.__repr__(indent + 1) if self.domain_constraint else 'None'
        return (f"ODE(\n{indent_str}  equations=[\n{equations_str}\n{indent_str}  ],\n"
                f"{indent_str}  domain_constraint={domain_repr})")

class ODEEquation(ASTNode):
    """Represents a single differential equation within an ODE block."""
    def __init__(self, variable: VarRef, expression: Term):
        self.variable = variable
        self.expression = expression
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.variable == other.variable and self.expression == other.expression
    def __hash__(self):
        return hash((self.variable, self.expression))
    def __repr__(self, indent=0):
        return (f"ODEEquation({self.variable.__repr__(indent + 1)}, "
                f"{self.expression.__repr__(indent + 1)})")

class HPBlock(HybridProgram):
    """Represents a block containing a single hybrid program."""
    def __init__(self, program: HybridProgram):
        self.program = program
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.program == other.program
    def __hash__(self):
        return hash(self.program)
    def __repr__(self, indent=0):
        indent_str = "  " * indent
        return f"HPBlock(\n{indent_str}  program={self.program.__repr__(indent + 1)})"