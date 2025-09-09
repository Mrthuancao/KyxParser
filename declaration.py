from abc import ABC, abstractmethod

class Declaration(ABC):
    """Abstract base class for declarations."""
    @abstractmethod
    def getDescription(self):
        pass

class VariableDeclaration(Declaration):
    def __init__(self, name, type, description):
        self.name = name
        self.type = type
        self.description = description

    def getDescription(self):
        return f"Variable Declaration: {self.type} {self.name}  /* {self.description} */"

class PredicateDeclaration(Declaration):
    def __init__(self, name, parameters, expression):
        self.name = name
        self.parameters = parameters
        self.expression = expression

    def getDescription(self):
        return f"Predicate Declaration: Bool {self.name}({self.parameters}) <-> {self.expression}"

class HybridProgramDeclaration(Declaration):
    def __init__(self, name, program):
        self.name = name
        self.program = program

    def getDescription(self):
        return f"Hybrid Program Declaration: HP {self.name} ::= {self.program}"
