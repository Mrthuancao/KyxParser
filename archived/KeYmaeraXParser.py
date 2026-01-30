# --- Parser Implementation ---

class KeYmaeraXParser:
    def __init__(self, source_code):
        self.source_code = source_code
        self.ast_root = ASTNode("KeYmaeraXSpecification")
        self._cleanup_source()

    def _cleanup_source(self):
        # 1. Remove multi-line comments (/* ... */)
        self.source_code = re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m.group(0)), self.source_code, flags=re.DOTALL)
        # 2. Remove single-line comments (//)
        self.source_code = re.sub(r'//.*', '', self.source_code)
        # 3. Normalize line endings and remove excessive blank lines
        self.source_code = re.sub(r'\n\s*\n', '\n', self.source_code)
        self.source_code = self.source_code.strip()

    # Helper method to split by an operator, respecting brace nesting
    @staticmethod
    def split_by_operator_outside_braces(text, operator):
        """
        Splits a string by a given operator, but only if the operator is not
        enclosed within curly braces {}.
        """
        parts = []
        brace_level = 0
        current_part_start = 0

        i = 0
        while i < len(text):
            if text[i] == '{':
                brace_level += 1
            elif text[i] == '}':
                brace_level -= 1

            # Check for operator match, ensuring we don't go out of bounds
            if (i + len(operator) <= len(text) and
                brace_level == 0 and
                text[i:i+len(operator)] == operator):

                part = text[current_part_start:i].strip()
                if part:
                    parts.append(part)
                current_part_start = i + len(operator)
                i += len(operator) - 1 # Adjust i to continue after the operator in the next loop iteration
            i += 1

        # Add the last part after the loop finishes
        last_part = text[current_part_start:].strip()
        if last_part:
            parts.append(last_part)
        return parts
    
    
    def _extract_top_level_brace_blocks(self, text: str) -> list[str | list]:
        """
        Parses a string with nested braces and returns a list representing
        the nested structure. Each element in the list can be:
        - a string (content outside of braces, stripped of leading/trailing whitespace)
        - a list (representing the content of a nested brace block, recursively parsed)

        This function is now recursive to achieve the "nested array" output.
        """
        result = []
        balance = 0
        current_text_segment_start = 0
        block_start_index = -1 # Marks the start of content *inside* a '{'

        for i, char in enumerate(text):
            if char == '{':
                if balance == 0:
                    # If we encounter an opening brace at balance 0,
                    # it means any text collected so far is outside this new block.
                    if i > current_text_segment_start:
                        segment = text[current_text_segment_start:i].strip()
                        if segment: # Only add if not empty after stripping
                            result.append(segment)
                    block_start_index = i + 1 # Content of this new block starts after '{'
                balance += 1
            elif char == '}':
                balance -= 1
                if balance == 0:
                    # If we encounter a closing brace and balance returns to 0,
                    # it means a top-level block (for the current recursion level) has ended.
                    if block_start_index != -1:
                        block_content = text[block_start_index:i]
                        # Recursively parse the content of this block
                        result.append(self._extract_top_level_brace_blocks(block_content))
                        block_start_index = -1 # Reset block start
                    current_text_segment_start = i + 1 # Start collecting text after this block
                elif balance < 0:
                    # Malformed input: closing brace without a matching opening brace.
                    print(f"Warning: Malformed brace structure (unmatched '}}') at index {i} in '{text}'")
                    balance = 0 # Attempt to recover by resetting balance
                    current_text_segment_start = i + 1 # Continue as if this '}' was just text

        # Add any remaining text segment after the last block or if no blocks were found
        if current_text_segment_start < len(text):
            segment = text[current_text_segment_start:len(text)].strip()
            if segment:
                result.append(segment)

        # Check for unclosed braces at the end of the string
        if balance > 0:
            print(f"Warning: Unclosed brace block detected in text: '{text}'")
            # If there's an unclosed block, its content from block_start_index to the end
            # should be treated as a single text segment for this level, or handled as an error.
            if block_start_index != -1 and block_start_index < len(text):
                remaining_content = text[block_start_index:].strip()
                if remaining_content:
                    result.append(remaining_content)

        return result

    def parse(self):
        self._parse_theorem_name()
        self._parse_definitions()
        self._parse_program_variables()
        self._parse_problem()
        return self.ast_root

    def _parse_theorem_name(self):
        match = re.search(r'Theorem\s+"(.*?)"', self.source_code, re.DOTALL)
        if match:
            self.ast_root.add_child(ASTNode("TheoremName", match.group(1)))

    def _extract_block(self, keyword):
        # Extracts content between 'Keyword' and 'End.' or the next major keyword block
        pattern = rf'{keyword}\s*(.*?)(?=\n\s*End\.|\n\s*Theorem|\n\s*Definitions|\n\s*ProgramVariables|\n\s*Problem|$)'
        match = re.search(pattern, self.source_code, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content.endswith('End.'):
                 content = content[:-4].strip()
            return content
        return None
    
    def _parse_definitions(self):
        definitions_block = self._extract_block("Definitions")
        if not definitions_block:
            return

        defs_node = ASTNode("DefinitionsBlock")
        self.ast_root.add_child(defs_node)

        remaining_text = definitions_block
        
        # --- 1. Extract Hybrid Programs (HP) first ---
        
        HP_PATTERN = re.compile(
            r'(HP\s+(\w+)\s*::=\s*\{(.*?)\}\s*;;?)', 
            re.DOTALL
        )
        
        parsed_hps = []
        
        for match in HP_PATTERN.finditer(remaining_text):
            name = match.group(2).strip()
            body = match.group(3).strip()
            
            hp_node = HybridProgram(name, body)
            # Pass the split utility to the HP node for its internal parsing
            hp_node._split_util = self.split_by_operator_outside_braces 
            hp_node._parse_hp_structure() # Re-parse with the utility set
            defs_node.add_child(hp_node)
            
            parsed_hps.append(match.span())

        # Remove the extracted HP definitions from the block text
        parsed_hps.sort(key=lambda x: x[0], reverse=True)
        
        mutable_text = list(remaining_text)
        for start, end in parsed_hps:
            mutable_text[start:end] = [' '] * (end - start)
            
        remaining_text = "".join(mutable_text)
        
        # --- 2. Parse simple Definitions (DEF Real/Bool) ---

        # Use the new split utility for semicolon-separated definitions
        parts = self.split_by_operator_outside_braces(remaining_text, ';')
        
        for part in parts:
            line = part.strip()
            if not line:
                continue

            # Parse Real/Bool Definitions with assignment or equivalence
            # Note: The pattern needs to handle the case where the definition value might contain parentheses (like function calls)
            def_match = re.search(r'(Real|Bool)\s+(\w+)(?:\s*\((.*?)\))?\s*(?:<->|=)\s*(.*?)$', line, re.DOTALL)
            
            if def_match:
                def_type = def_match.group(1)
                name = def_match.group(2)
                params = def_match.group(3)
                value = def_match.group(4).strip()
                
                # Check for outer parentheses removal (standard KeYmaera X practice)
                if value.startswith('(') and value.endswith(')'):
                    value = value[1:-1].strip()
                
                def_node = Definition(name, def_type, value=value)
                
                if params:
                    # KeYmaera X often includes the type in the parameter list (e.g., Real v)
                    param_list = [p.strip() for p in self.split_by_operator_outside_braces(params, ',')]
                    param_node = ASTNode("Parameters", ", ".join(param_list))
                    def_node.add_child(param_node)
                
                defs_node.add_child(def_node)
                continue
            
            # Parse simple constant declarations (e.g., Real ep) - no assignment
            const_match = re.search(r'(Real|Bool)\s+(\w+)\s*$', line, re.DOTALL)
            if const_match:
                def_type = const_match.group(1)
                name = const_match.group(2)
                defs_node.add_child(Definition(name, def_type))


    def _parse_program_variables(self):
        vars_block = self._extract_block("ProgramVariables")
        if not vars_block:
            return

        vars_node = ASTNode("ProgramVariablesBlock")
        self.ast_root.add_child(vars_node)

        # Use the new split utility for semicolon-separated variable declarations
        for line in self.split_by_operator_outside_braces(vars_block, ';'):
            line = line.strip()
            if not line:
                continue
            
            var_match = re.search(r'(Real|Bool)\s+(\w+)\s*$', line, re.DOTALL)
            if var_match:
                var_type = var_match.group(1)
                name = var_match.group(2)
                vars_node.add_child(Variable(name, var_type))

    def _parse_problem(self):
        problem_block = self._extract_block("Problem")
        if not problem_block:
            return

        problem_node = ProblemStatement(problem_block)
        self.ast_root.add_child(problem_node)
        
        # P -> [HP] Q structure
        # Use a non-greedy match for the program block ([...])
        implication_match = re.search(
            r'([\s\S]+?)\s*->\s*\[\s*([\s\S]+?)\s*\]\s*([\s\S]+)', 
            problem_block, 
            re.DOTALL
        )
        
            
        if implication_match:
            precondition_raw = implication_match.group(1).strip()
            program_raw = implication_match.group(2).strip()
            postcondition_raw = implication_match.group(3).strip()
            
            problem_node.add_child(ASTNode("Precondition (P)", precondition_raw))
            
            # Parse the loop structure: { ... } { ... } *@invariant(loopinv)
            # Updated regex to correctly capture the two loop body blocks and the loop invariant
            loop_match = re.search(
                r"(\{\s*(\{\s*.*?\s*\})?\s*(\{\s*.*?\s*\})\s*\})\*@invariant\(([\s\S]*?\))\)", 
                program_raw, 
                re.DOTALL
            )
            
            if loop_match:
                # loop_body_raw will be "{ controlFunc; t := 0; } { stepFunction;@invariant(t>=0, isWellformedDir(dx, dy)) }"
                loop_body_raw = loop_match.group(1).strip()
                # loop_invariant_raw will be "loopinv(x, y, v, dx, dy, xo, yo)"
                loop_invariant_raw = loop_match.group(4).strip() 
                
                program_node = ASTNode("LoopExecution")
                
                # Create the outer [HP]: node
                outer_hp_node = ASTNode("HP", "") # Use empty string for value to get ":" in repr
                
                # Using _extract_top_level_brace_blocks for robust parsing of nested braces
                blocks = self._extract_top_level_brace_blocks(loop_body_raw)[0]
                
                if len(blocks) == 2:
                    control_block_full = blocks[0][0] # e.g., "{ controlFunc; t := 0; }"
                    dynamics_block_full = blocks[1][0] # e.g., "{ stepFunction;@invariant(t>=0, isWellformedDir(dx, dy)) }"

                    # --- Process Control Block (Discrete) ---
                    # Desired: [HP]: -> [Statement]: controlFunc, [Statement]: t:= 0
                    control_hp_node = ASTNode("HP", "") # Inner [HP]: node
                    
                    # Remove outer braces for parsing content
                    # control_content = control_block_full[1:-1].strip() # "controlFunc; t := 0;"
                    
                    # Split by semicolon to get individual statements
                    control_statements = [s.strip() for s in self.split_by_operator_outside_braces(control_block_full, ';') if s.strip()]
                    
                    for stmt_value in control_statements:
                        control_hp_node.add_child(ASTNode("Statement", stmt_value))
                    
                    outer_hp_node.add_child(control_hp_node)

                    # --- Process Dynamics Block (Continuous) ---
                    # Desired: [HP]: -> [Statement]: stepFunction, [Invariant]: t>=0, isWellformedDir(...)
                    dynamics_hp_node = ASTNode("HP", "") # Inner [HP]: node
                    
                    # Remove outer braces for parsing content
                    # dynamics_content = dynamics_block_full[1:-1].strip() # "stepFunction;@invariant(t>=0, isWellformedDir(dx, dy))"
                    
                    # Extract statement and its invariant
                    stmt_inv_match = re.search(r'^(.*?);@invariant\((.*?)\)$', dynamics_block_full, re.DOTALL)
                    if stmt_inv_match:
                        statement_name = stmt_inv_match.group(1).strip() # "stepFunction"
                        invariant_content = stmt_inv_match.group(2).strip() # "t>=0, isWellformedDir(dx, dy)"
                        dynamics_hp_node.add_child(ASTNode("Statement", statement_name))
                        dynamics_hp_node.add_child(ASTNode("Invariant", invariant_content))
                    else:
                        # Fallback if the expected pattern doesn't match
                        dynamics_hp_node.add_child(ASTNode("Statement", dynamics_block_full)) 
                         
                    outer_hp_node.add_child(dynamics_hp_node)
                    
                    # Add the outer HP node to the program node
                    program_node.add_child(outer_hp_node)
                    
                    # Add the loop invariant as a direct child of LoopExecution, after the HP blocks
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                else:
                    # Fallback if the loop body structure is not as expected (e.g., not two blocks)
                    program_node.add_child(ASTNode("Raw Loop Body Content", loop_body_raw))
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw)) # Still add the invariant if found
                
                problem_node.add_child(program_node)
            else:
                problem_node.add_child(ASTNode("RawProgram", program_raw))

            problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
        else:
            problem_node.add_child(ASTNode("Error", "Could not parse P -> [HP] Q structure."))

