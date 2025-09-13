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
        self.parameters = PredicateDeclaration.parseParameter(parameters)
        self.expression = expression
        self.parameters_mapping = {
            'Bool': 'bool',
            'Real': 'float'
        }
        
        # print('Python Param: ', self.getParameters(True))
        
    
    @staticmethod    
    def parseParameter(parameter_string):
        try:
            parameters = parameter_string.split(',')  # Split the string into individual parameters
            parsed_parameters = []
            for param in parameters:
                param = param.strip()  # Remove leading/trailing whitespace
                parts = param.split()  # Split into type and name
                if len(parts) != 2:
                    raise ValueError(f"Invalid parameter format: {param}")
                parsed_parameters.append((parts[0], parts[1])) #Append as tuple (type, name)
            return parsed_parameters
        except (ValueError, AttributeError) as e:
            print(f"Error parsing parameter string: {e}")
            return []
        
    def getParameters(self, to_Python = True):
        if to_Python:
            if not self.parameters:
                return ""

            try:
                #Check if all elements are tuples of length 2
                if not all(isinstance(param, tuple) and len(param) == 2 for param in self.parameters):
                    raise ValueError("Invalid input: self.parameters must be a list of (type, name) tuples.")

                parameters = []
                for param_type, param_name in self.parameters:
                    param_type = self.parameters_mapping[param_type]
                    parameters.append(f"{param_type} {param_name}")

                return ", ".join(parameters)

            except (ValueError, TypeError) as e:
                print(f"Error generating parameter string: {e}")
                return ""
            
        return self.parameters


    def getDescription(self):
        return f"Predicate Declaration: {self.return_type} {self.name}({self.parameters}) <-> {self.expression}"
    
    def toPython(self):
        result_string = ''
        if self.return_type == 'Bool':
            result_string += 'bool '
        elif self.return_type == 'Real':
            result_string += 'float '
            
        result_string += f'''{self.name}({self.getParameters()}): 
            return {self.expression}'''
        return result_string 

class HybridProgramDeclaration(Declaration):
    def __init__(self, name, program):
        self.name = name
        self.program = program

    def getDescription(self):
        return f"Hybrid Program Declaration: HP {self.name} ::= {self.program}"
    
    def toPython(self):
        result_string = f'''def {self.name}():
            return {self.program}'''
        
        return result_string
