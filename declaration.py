from abc import ABC, abstractmethod

class Declaration(ABC):
    """Abstract base class for declarations."""
    @abstractmethod
    def getDescription(self):
        pass
    
    @abstractmethod
    def toPython(self):
        pass

class VariableDeclaration(Declaration):
    def __init__(self, name, return_type, description = ''):
        self.name = name
        self.return_type = return_type
        self.description = description

    def getDescription(self):
        return f"Variable Declaration: {self.return_type} {self.name}  /* {self.description} */"
    
    def toPython(self):
        result_string = ''
        if self.return_type == 'Bool':
            result_string += 'bool '
        elif self.return_type == 'Real':
            result_string += 'float '
            
        result_string += self.name
        return result_string

class PredicateDeclaration(Declaration):
    def __init__(self, name, return_type, parameters, expression):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.expression = expression

    def getDescription(self):
        return f"Predicate Declaration: {self.return_type} {self.name}({self.parameters}) <-> {self.expression}"
    
    def toPython(self):
        result_string = ''
        if self.return_type == 'Bool':
            result_string += 'bool '
        elif self.return_type == 'Real':
            result_string += 'float '
            
        result_string += f'''{self.name}({self.parameters}): 
            return {self.expression}'''
        return result_string 

class HybridProgramDeclaration(Declaration):
    def __init__(self, name, program):
        self.name = name
        self.program = program

    def getDescription(self):
        return f"Hybrid Program Declaration: HP {self.name} ::= {self.program}"
    
    def toPython(self):
        result_string = f'def {self.name}(): {self.program}'
        
        return result_string
