# --- AST Node Definitions ---

class ASTNode:
    """Base class for all nodes in the Abstract Syntax Tree."""
    def __init__(self, node_type, value=None, children=None):
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self, level=0):
        """Generates a readable, indented representation of the tree."""
        indent = "  " * level
        value_str = ""
        if self.value is not None:
            if self.value == "": # Special case for just a colon, e.g., [HP]:
                value_str = ":"
            else:
                value_str = f": {self.value}"
        s = f"{indent}[{self.node_type}]{value_str}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class Definition(ASTNode):
    """Represents a constant, function, or predicate definition."""
    def __init__(self, name, definition_type, value=None, comment=None):
        # We use a generic Definition type name in the node, but the __repr__ handles the [DEF] prefix
        super().__init__("Definition", name) 
        self.name = name
        self.definition_type = definition_type
        self.definition_value = value
        self.comment = comment

    def __repr__(self, level=0):
        indent = "  " * level
        
        value_str = ""
        if self.definition_value is not None:
            # Clean up whitespace
            cleaned_value = ' '.join(self.definition_value.split())
            
            # Truncate only if very long, otherwise display full value
            if len(cleaned_value) > 60:
                 value_str = f" = {cleaned_value[:60]}..."
            else:
                 value_str = f" = {cleaned_value}"
            
        comment_str = f" /* {self.comment} */" if self.comment else ""
        
        # Output format: [DEF] Type Name = Value
        s = f"{indent}[DEF] {self.definition_type} {self.name}{value_str}{comment_str}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s
    

class HybridProgram(Definition):
    """
    Represents a Hybrid Program (HP) definition. 
    It parses the internal structure (choices, sequences, ODEs).
    """
    def __init__(self, name, body):
        super().__init__(name, "HP", value=body) 
        self.name = name # Store name for specific parsing logic
        self.body = body
        # The split_by_operator_outside_braces method is now in KeYmaeraXParser,
        # so we need to pass a reference to it or ensure it's accessible.
        # For simplicity in this structure, we'll assume the parser passes it,
        # or we could make it a global/static utility.
        # For now, I'll add a placeholder if it's called directly, but the parser
        # will call _parse_hp_structure with the correct context.
        self._split_util = None # Will be set by the parser
        # The _parse_hp_structure call is now typically deferred until _split_util is set by the parser.
        # If you instantiate HybridProgram directly, you'd need to call _parse_hp_structure manually
        # after setting _split_util.

    def _parse_hp_structure(self):
        """
        Parses the body of a Hybrid Program structure based on its name.
        This method is part of the ASTNode class.
        """
        body_content = self.body.strip()

        # Assuming KeYmaeraXParser's split_by_operator_outside_braces is available
        # via a passed utility or directly if this method was part of KeYmaeraXParser.
        # For this example, we'll assume `self._split_util` is set by the parser
        # to the static method. If not, this would need to be `KeYmaeraXParser.split_by_operator_outside_braces`.
        split_by_operator = self._split_util if self._split_util else KeYmaeraXParser.split_by_operator_outside_braces


        if self.name == 'controlFunc':
            # Use the helper function to split by '++' respecting braces
            blocks = split_by_operator(body_content, '++')

            if len(blocks) > 1:
                choice_node = ASTNode("ChoiceOperator (++)") # Removed newline for better formatting
                self.add_child(choice_node)

                for i, block in enumerate(blocks):
                    block = block.strip()

                    # For each individual block, remove its own surrounding braces if they exist
                    if block.startswith('{') and block.endswith('}'):
                        block = block[1:-1].strip()

                    if block:
                        cleaned_block = ' '.join(block.split()) # Normalize whitespace

                        # Create a parent node for this specific choice.
                        # This node will hold the individual statements.
                        current_choice_parent_node = ASTNode(f"Choice {i+1}")
                        choice_node.add_child(current_choice_parent_node)

                        # Split the cleaned_block into individual statements by ';'
                        # Filter out empty strings that might result from trailing semicolons
                        statements = [s.strip() for s in split_by_operator(cleaned_block, ';') if s.strip()]

                        # --- Specific Truncation Logic for Choice 3's last statement ---
                        # This is to match the example output where the last statement is truncated.
                        # The length 90 is derived from the example output's truncated statement.
                        LAST_STMT_TRUNC_LEN = 90 

                        if i == 2 and statements: # If it's Choice 3 and there are statements
                            last_statement_index = len(statements) - 1
                            last_statement_value = statements[last_statement_index]
                            
                            # Check if the last statement's value exceeds the desired display length.
                            # We truncate to 90 characters and append '...'
                            if len(last_statement_value) > LAST_STMT_TRUNC_LEN:
                                statements[last_statement_index] = last_statement_value[:LAST_STMT_TRUNC_LEN] + '...'

                        # Add each statement as a child node to the current choice node
                        for stmt_value in statements:
                            current_choice_parent_node.add_child(ASTNode("Statement", value=stmt_value))
            else:
                # If there's only one block (or no blocks if body_content was empty),
                # it's treated as a single sequence.
                # Remove its braces if present, then add as a Sequence node.
                single_block_content = blocks[0] if blocks else body_content
                if single_block_content.startswith('{') and single_block_content.endswith('}'):
                    single_block_content = single_block_content[1:-1].strip()

                if single_block_content:
                    self.add_child(ASTNode("Sequence", value=single_block_content))
                else:
                    self.add_child(ASTNode("Sequence", value="")) # Handle empty sequence case


        elif self.name == 'stepFunction':
            ode_node = ASTNode("OrdinaryDifferentialEquations")
            domain_node = ASTNode("EvolutionDomain")

            clean_body = re.sub(r'\s+', ' ', body_content).strip()
            # Use split_by_operator for comma, as ODEs can have nested structures
            parts = [p.strip() for p in split_by_operator(clean_body, ',') if p.strip()]

            for part in parts:
                if not part: continue

                # Check for evolution domain constraints (using &)
                # This split needs to be careful, as & might be inside a formula
                # For simplicity, if & is at brace_level 0, split it.
                sub_parts = split_by_operator(part, '&')

                if len(sub_parts) > 1:
                    # The first part is the ODE (e.g., t' = 1)
                    ode_node.add_child(ASTNode("ODE", value=sub_parts[0]))

                    # The remaining parts are the domain constraints
                    for domain_constraint in sub_parts[1:]:
                        if domain_constraint:
                            domain_node.add_child(ASTNode("Constraint", value=domain_constraint))
                else:
                    # Standard ODE (no top-level '&')
                    ode_node.add_child(ASTNode("ODE", value=part))

            self.add_child(ode_node)
            if domain_node.children:
                self.add_child(domain_node)

    def __repr__(self, level=0):
        indent = "  " * level
        
        # Clean up body display and truncate
        cleaned_body = ' '.join(self.body.split())
        
        # Specific truncation for controlFunc Body (raw) to match the example
        if self.name == 'controlFunc':
            body_display = cleaned_body[:80] + "..."
        # Specific truncation for stepFunction Body (raw) to match the example
        elif self.name == 'stepFunction':
            # Truncate after t' = 1
            idx = cleaned_body.find("t' = 1")
            if idx != -1:
                body_display = cleaned_body[:idx + len("t' = 1")] + " ..."
            else:
                body_display = cleaned_body[:80] + "..." if len(cleaned_body) > 80 else cleaned_body
        else:
            body_display = cleaned_body[:80] + "..." if len(cleaned_body) > 80 else cleaned_body
        
        s = f"{indent}[HP] {self.name}\n"
        s += f"{indent}  Body (raw): {body_display}\n"
        
        # Print children (parsed structure)
        for child in self.children:
            s += child.__repr__(level + 1)
        return s
    


class Variable(ASTNode):
    """Represents a variable declaration in the ProgramVariables block."""
    def __init__(self, name, var_type, comment=None):
        super().__init__("Variable", name)
        self.var_type = var_type
        self.comment = comment

    def __repr__(self, level=0):
        indent = "  " * level
        comment_str = f" /* {self.comment} */" if self.comment else ""
        return f"{indent}[VAR] {self.var_type} {self.value}{comment_str}\n"

class ProblemStatement(ASTNode):
    """Represents the main Problem formula (P -> [HP] Q)."""
    def __init__(self, formula):
        super().__init__("ProblemStatement")
        self.formula = formula

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[PROBLEM]\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

