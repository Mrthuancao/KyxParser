from __future__ import annotations
import re
from ast_config import * # Import all necessary dataclasses

# --- AST Node Definitions (from your parser's internal AST, not the intermediate PyAST) ---

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
        node_type_str = self.node_type

        if node_type_str == "KeYmaeraXSpecification":
            # Root node has no colon and no value part
            s = f"{indent}[{node_type_str}]\n"
        else:
            # All other nodes get a colon.
            # If they have a value, it's appended after the colon.
            value_part = f" {self.value}" if self.value is not None else ""
            s = f"{indent}[{node_type_str}]:{value_part}\n"
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
        if self.definition_value:
            cleaned_value = ' '.join(self.definition_value.split())
            value_str = f" = {cleaned_value}"

        comment_str = f" /* {self.comment} */" if self.comment else ""

        # Output format: [DEF]: Type Name = Value
        s = f"{indent}[DEF]: {self.definition_type} {self.name}{value_str}{comment_str}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s


class HybridProgram(Definition):
    """
    Represents a Hybrid Program (HP) definition.
    It parses the internal structure (choices, sequences, ODEs).
    """
    def __init__(self, name, body, parser_ref=None):
        super().__init__(name, "HP", value=body)
        self.name = name
        self.body = body
        self._parser_ref = parser_ref

    def _parse_hp_structure(self):
        """
        Parses the body of a Hybrid Program structure based on its name.
        This method is part of the ASTNode class.
        """
        body_content = self.body.strip()

        if self.name == 'controlFunc':
            if self._parser_ref:
                self._parser_ref._parse_program_fragment(body_content, self)
            else:
                print(f"Warning: No parser reference available for HybridProgram '{self.name}'. Cannot parse program fragment.")

        elif self.name == 'stepFunc': # Corrected typo from 'stepFunction' to 'stepFunc'
            ode_node = ASTNode("OrdinaryDifferentialEquations")
            domain_node = ASTNode("EvolutionDomain")

            # body_content should now NOT have outer braces from the definition due to previous fix.
            # So, this conditional removal is no longer strictly needed but harmless if the content is just a block.
            clean_body_no_braces = body_content
            if clean_body_no_braces.startswith('{') and clean_body_no_braces.endswith('}'):
                clean_body_no_braces = clean_body_no_braces[1:-1].strip()
            clean_body = re.sub(r'\s+', ' ', clean_body_no_braces).strip()

            split_by_operator = KeYmaeraXParser.split_by_operator_outside_braces
            parts = [p.strip() for p in split_by_operator(clean_body, ',') if p.strip()]

            for part in parts:
                if not part: continue

                sub_parts = split_by_operator(part, '&')

                if not sub_parts:
                    continue

                ode_or_formula = sub_parts[0]
                if "'" in ode_or_formula: # Simple heuristic to identify an ODE (contains derivative)
                    ode_node.add_child(ASTNode("ODE", value=ode_or_formula))
                else: # Treat as a constraint if it doesn't look like an ODE
                    domain_node.add_child(ASTNode("Constraint", value=ode_or_formula))

                for domain_constraint in sub_parts[1:]:
                    if domain_constraint:
                        domain_node.add_child(ASTNode("Constraint", value=domain_constraint))

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
        # Specific truncation for stepFunc Body (raw) to match the example
        elif self.name == 'stepFunc':
            # Truncate after t' = 1
            idx = cleaned_body.find("t' = 1")
            if idx != -1:
                body_display = cleaned_body[:idx + len("t' = 1")] + " ..."
            else:
                body_display = cleaned_body[:80] + "..." if len(cleaned_body) > 80 else cleaned_body
        else:
            body_display = cleaned_body[:80] + "..." if len(cleaned_body) > 80 else cleaned_body

        # Use ASTNode's __repr__ for the HP node itself, then add Body (raw)
        s = super().__repr__(level) # This will print [DEF]: HP name)
        s_lines = s.splitlines()
        s = s_lines[0] + '\n' # Keep the first line ([DEF]: HP name)
        s += f"{indent}  Body (raw): {body_display}\n" # Add Body (raw)

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
        # Ensure var_type is included as per expected output
        return f"{indent}[VAR]: {self.var_type} {self.value}{comment_str}\n"

class ProblemStatement(ASTNode):
    """Represents the main Problem formula (P -> [HP] Q)."""
    def __init__(self, formula):
        super().__init__("ProblemStatement")
        self.formula = formula

    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}[PROBLEM]:\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s


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

    def _find_matching_brace_span(self, text: str, start_brace_idx: int) -> int | None:
        """
        Finds the index of the closing brace '}' that matches the opening brace
        at start_brace_idx.
        Returns the index of the matching '}' or None if no match is found (unbalanced).
        """
        if text[start_brace_idx] != '{':
            raise ValueError("start_brace_idx must point to an opening brace '{'")

        balance = 1
        for i in range(start_brace_idx + 1, len(text)):
            if text[i] == '{':
                balance += 1
            elif text[i] == '}':
                balance -= 1

            if balance == 0:
                return i

        return None # Unbalanced braces


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

    def _extract_block(self, keyword):
        # Extracts content between 'Keyword' and 'End.' or the next major keyword block
        # Updated regex to handle "Theorem" keyword specifically at the beginning of the file
        if keyword == "Theorem":
            pattern = rf'{keyword}\s+(.*?)(?=\n\s*Definitions|\n\s*ProgramVariables|\n\s*Problem|\n\s*End\.|$)'
        else:
            pattern = rf'{keyword}\s*(.*?)(?=\n\s*End\.|\n\s*Theorem|\n\s*Definitions|\n\s*ProgramVariables|\n\s*Problem|$)'

        match = re.search(pattern, self.source_code, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content.endswith('End.'):
                 content = content[:-4].strip()
            return content
        return None

    def _parse_theorem_name(self):
        """
        Parses the theorem name from the source code and adds it to the AST.
        """
        theorem_name_raw = self._extract_block("Theorem")
        if theorem_name_raw:
            # The theorem name might be enclosed in quotes or just be a string.
            # Remove leading/trailing quotes if present.
            theorem_name = theorem_name_raw.strip().strip('"')
            theorem_node = ASTNode("TheoremName", theorem_name)
            self.ast_root.add_child(theorem_node)
        else:
            # If no Theorem block is found, add a placeholder or log a warning
            theorem_node = ASTNode("TheoremName", "UnnamedTheorem")
            self.ast_root.add_child(theorem_node)
            print("Warning: No 'Theorem' block found in the KeYmaera X input.")


    def _parse_definitions(self):
        definitions_block = self._extract_block("Definitions")
        if not definitions_block:
            return

        defs_node = ASTNode("DefinitionsBlock")
        self.ast_root.add_child(defs_node)

        # Use a mutable list of characters for efficient replacement of processed blocks.
        mutable_definitions_block = list(definitions_block)

        # --- 1. Extract Hybrid Programs (HP) first ---

        # Pattern to find the HP header up to the opening brace '{'
        HP_HEADER_PATTERN = re.compile(
            r'HP\s+(\w+)\s*::=\s*\{',
            re.DOTALL
        )

        current_search_pos = 0
        while current_search_pos < len(mutable_definitions_block):
            # Search for the next HP header from the current position
            search_string = "".join(mutable_definitions_block[current_search_pos:])
            match = HP_HEADER_PATTERN.search(search_string)

            if match:
                hp_header_start_idx = current_search_pos + match.start()
                open_brace_idx = current_search_pos + match.end() - 1

                hp_name = match.group(1).strip()

                full_current_block_string = "".join(mutable_definitions_block)

                closing_brace_idx = self._find_matching_brace_span(
                    full_current_block_string,
                    open_brace_idx
                )

                if closing_brace_idx is not None:
                    # The body content is between the opening '{' and closing '}'
                    # Include the outer braces so _parse_program_fragment can strip them
                    hp_body_start_idx = open_brace_idx
                    hp_body_end_idx = closing_brace_idx + 1

                    hp_body_raw = "".join(mutable_definitions_block[hp_body_start_idx : hp_body_end_idx]).strip()

                    # Determine the end of the full HP block, including optional ';;'
                    full_hp_block_end_idx = closing_brace_idx + 1 # Start after the '}'

                    # Check for optional ';;'
                    if full_hp_block_end_idx + 1 < len(mutable_definitions_block) and \
                       "".join(mutable_definitions_block[full_hp_block_end_idx : full_hp_block_end_idx + 2]).strip() == ';;':
                        full_hp_block_end_idx += 2
                    # Check for optional ';'
                    elif full_hp_block_end_idx < len(mutable_definitions_block) and \
                         mutable_definitions_block[full_hp_block_end_idx] == ';':
                        full_hp_block_end_idx += 1

                    # Create and process the HybridProgram node, passing parser_ref=self
                    hp_node = HybridProgram(hp_name, hp_body_raw, parser_ref=self)
                    hp_node._parse_hp_structure()
                    defs_node.add_child(hp_node)

                    # Replace the extracted HP definition with spaces in the mutable_definitions_block
                    for i in range(hp_header_start_idx, full_hp_block_end_idx):
                        if i < len(mutable_definitions_block):
                            mutable_definitions_block[i] = ' '

                    # Advance the search position past the extracted block
                    current_search_pos = full_hp_block_end_idx
                else:
                    print(f"Warning: Unbalanced braces for HP '{hp_name}' starting at index {open_brace_idx}. Skipping this block.")
                    current_search_pos = open_brace_idx + 1
            else:
                break

        # Convert the modified list back to a string for parsing simple definitions
        remaining_text_after_hps = "".join(mutable_definitions_block)

        # --- 2. Parse simple Definitions (DEF Real/Bool) ---

        # Use the new split utility for semicolon-separated definitions
        parts = self.split_by_operator_outside_braces(remaining_text_after_hps, ';')

        for part in parts:
            line = part.strip()
            if not line:
                continue

            # Parse Real/Bool Definitions with assignment or equivalence
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

    def _parse_program_fragment(self, program_fragment_string: str, parent_node: ASTNode):
        """
        Parses a string representing a program fragment (which can be a sequence, choice,
        assignment, or test) and adds the resulting AST nodes as children to the parent_node.
        This function handles stripping outer braces and then delegates to sequence/choice parsing.
        """
        fragment = program_fragment_string.strip()
        if not fragment:
            return

        # 1. Try to split by '++' (choice operator) at the current top level
        choice_parts = self.split_by_operator_outside_braces(fragment, '++')
        if len(choice_parts) > 1:
            choice_node = ASTNode("ChoiceOperator (++)")
            parent_node.add_child(choice_node)
            for i, part in enumerate(choice_parts):
                child_choice_node = ASTNode(f"Choice {i+1}")
                choice_node.add_child(child_choice_node)
                self._parse_program_fragment(part, child_choice_node) # Recurse on each choice part
            return

        # 2. If not a choice, try to split by ';' (sequence operator) at the current top level
        statements = self.split_by_operator_outside_braces(fragment, ';')
        # If there's more than one statement, or it's a single statement that had a trailing semicolon,
        # then process them as a sequence.
        if len(statements) > 1: # Definitely a sequence
            for stmt_value in statements:
                if stmt_value:
                    self._parse_program_fragment(stmt_value, parent_node) # Recurse on each statement
            return
        elif len(statements) == 1 and statements[0] != fragment: # Single statement that had a trailing semicolon (e.g., "A;" -> ["A"])
            self._parse_program_fragment(statements[0], parent_node)
            return

        # 3. If it's not a top-level choice or sequence, check if it's a braced block { ... }
        if fragment.startswith('{') and fragment.endswith('}'):
            # If it's a block, strip the outer braces and parse its content.
            # The content of this block becomes children of the current parent.
            self._parse_program_fragment(fragment[1:-1].strip(), parent_node)
            return

        # 4. Base case: If it wasn't any of the above, it's a single, simple statement.
        if fragment: # Ensure it's not empty after all processing
            if fragment.startswith('?'):
                parent_node.add_child(ASTNode("Test", value=fragment))
            elif '(' in fragment and ')' in fragment and ':=' not in fragment: # Heuristic for predicate calls as tests
                parent_node.add_child(ASTNode("Test", value=fragment))
            else:
                parent_node.add_child(ASTNode("Statement", value=fragment))
            return


    def _parse_problem(self):
        problem_block = self._extract_block("Problem")
        if not problem_block:
            return

        problem_node = ProblemStatement(problem_block)
        self.ast_root.add_child(problem_node)

        # P -> [HP] Q structure
        # FIX: Corrected regex to escape literal square brackets
        implication_match = re.search(
            r'([\s\S]+?)\s*->\s*\[([\s\S]+?)\]\s*([\s\S]+)', # Corrected this line
            problem_block,
            re.DOTALL
        )

        if implication_match:
            precondition_raw = implication_match.group(1).strip()
            program_raw = implication_match.group(2).strip()
            postcondition_raw = implication_match.group(3).strip()

            problem_node.add_child(ASTNode("Precondition (P)", precondition_raw))

            # FIX: Corrected regex for loop invariant (removed extra \) )
            loop_match = re.search(
                r"(\{[\s\S]*?\})\*@invariant\(([\s\S]*?)\)",
                program_raw,
                re.DOTALL
            )

            if loop_match:
                loop_body_raw = loop_match.group(1).strip()
                loop_invariant_raw = loop_match.group(2).strip()

                program_node = ASTNode("LoopExecution")
                outer_hp_node = ASTNode("HP", "") # This represents the outer { } of the loop body

                # Get the content *inside* the outermost braces of the loop body
                loop_body_inner_content = loop_body_raw[1:-1].strip()

                # Find the first block (discrete part)
                discrete_start_idx = loop_body_inner_content.find('{')
                if discrete_start_idx == -1:
                    print(f"Warning: Could not find discrete block in loop body: '{loop_body_inner_content}'")
                    program_node.add_child(ASTNode("Raw Loop Body Content", loop_body_raw))
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                    problem_node.add_child(program_node)
                    problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
                    return

                discrete_end_idx = self._find_matching_brace_span(loop_body_inner_content, discrete_start_idx)
                if discrete_end_idx is None:
                    print(f"Warning: Unmatched braces in discrete block of loop body: '{loop_body_inner_content}'")
                    program_node.add_child(ASTNode("Raw Loop Body Content", loop_body_raw))
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                    problem_node.add_child(program_node)
                    problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
                    return

                discrete_block_full_str = loop_body_inner_content[discrete_start_idx : discrete_end_idx + 1].strip()

                # Find the second block (continuous part)
                continuous_search_start_idx = discrete_end_idx + 1
                continuous_start_idx = loop_body_inner_content.find('{', continuous_search_start_idx)

                if continuous_start_idx == -1:
                    print(f"Warning: Could not find continuous block in loop body: '{loop_body_inner_content}'")
                    program_node.add_child(ASTNode("Raw Loop Body Content", loop_body_raw))
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                    problem_node.add_child(program_node)
                    problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
                    return

                continuous_end_idx = self._find_matching_brace_span(loop_body_inner_content, continuous_start_idx)
                if continuous_end_idx == None: # CORRECTED: Changed '===' to '=='
                    print(f"Warning: Unmatched braces in continuous block of loop body: '{loop_body_inner_content}'")
                    program_node.add_child(ASTNode("Raw Loop Body Content", loop_body_raw))
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                    problem_node.add_child(program_node)
                    problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
                    return

                continuous_block_full_str = loop_body_inner_content[continuous_start_idx : continuous_end_idx + 1].strip()

                # --- Process Control Block (Discrete) ---
                control_hp_node = ASTNode("HP", "")
                discrete_fragment_content = discrete_block_full_str[1:-1].strip()
                self._parse_program_fragment(discrete_fragment_content, control_hp_node)
                outer_hp_node.add_child(control_hp_node)

                # --- Process Dynamics Block (Continuous) ---
                dynamics_hp_node = ASTNode("HP", "")
                dynamics_fragment_content = continuous_block_full_str[1:-1].strip()

                stmt_inv_match = re.search(r'^(.*?);@invariant\((.*?)\)$', dynamics_fragment_content, re.DOTALL)
                if stmt_inv_match:
                    statement_name = stmt_inv_match.group(1).strip()
                    invariant_content = stmt_inv_match.group(2).strip()
                    dynamics_hp_node.add_child(ASTNode("Statement", statement_name))
                    dynamics_hp_node.add_child(ASTNode("Invariant", invariant_content))
                else:
                    dynamics_hp_node.add_child(ASTNode("Statement", dynamics_fragment_content))
                outer_hp_node.add_child(dynamics_hp_node)

                program_node.add_child(outer_hp_node)
                program_node.add_child(ASTNode("Invariant", loop_invariant_raw))

                problem_node.add_child(program_node)
            else:
                program_node = ASTNode("RawProgram", program_raw)
                problem_node.add_child(program_node)

            problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
        else:
            problem_node.add_child(ASTNode("Error", "Could not parse P -> [HP] Q structure."))

# --- Helper to convert KeYmaeraX AST to Intermediate PyAST ---
def convert_keymaerax_ast_to_intermediate_pyast(kx_ast_root: ASTNode) -> GymEnvironmentConfig:
    """
    Converts the KeYmaera X AST into the intermediate Python AST representation.
    """
    if not kx_ast_root or kx_ast_root.node_type != "KeYmaeraXSpecification":
        raise ValueError("Invalid KeYmaera X AST root node provided.")

    env_name = "KeYmaeraX_RL_Environment" # Default name
    theorem_name_node = next((c for c in kx_ast_root.children if c.node_type == "TheoremName"), None)
    if theorem_name_node and theorem_name_node.value:
        # Clean up the theorem name for use as an environment name
        env_name = theorem_name_node.value.replace(' ', '_').replace(':', '_').replace('"', '').replace('/', '_').replace('.', '_')

    # Initialize top-level PyAST config with placeholder ProblemConfig
    definitions_config_instance = DefinitionsConfig()
    variables_config_instance = VariablesConfig()
    problem_config_instance = ProblemConfig(
        precondition=PreconditionConfig(formula=""),
        loop_execution=LoopExecutionConfig(),
        postcondition=PostconditionConfig(formula="")
    )

    gym_config = GymEnvironmentConfig(
        env_name=env_name,
        definitions=definitions_config_instance,
        variables=variables_config_instance,
        problem=problem_config_instance
    )

    # Process Definitions Block
    defs_block_kx = next((c for c in kx_ast_root.children if c.node_type == "DefinitionsBlock"), None)
    if defs_block_kx:
        defs_config = gym_config.definitions

        for def_node_kx in defs_block_kx.children:
            if def_node_kx.node_type == "Definition" and hasattr(def_node_kx, 'definition_type'): # General definition
                parameters = []
                params_node = next((c for c in def_node_kx.children if c.node_type == "Parameters"), None)
                if params_node and params_node.value: # Check if value is not empty
                    # Parameters value is a comma-separated string like "Real v, Real dx"
                    param_strings = [p.strip() for p in params_node.value.split(',')]
                    for p_str in param_strings:
                        parts = p_str.split(' ')
                        if len(parts) == 2: # e.g., "Real v"
                            parameters.append((parts[0], parts[1]))
                        else: # Fallback if type is missing, assume Real
                            parameters.append(('Real', p_str))

                if def_node_kx.definition_type == "HP":
                    if def_node_kx.name == "controlFunc":
                        control_func_config = ControlFuncConfig(name=def_node_kx.name, body_raw=def_node_kx.body)
                        # The controlFunc HP node's children directly represent its action structure
                        # We expect a ChoiceOperator (++) as the top-level child for this example
                        control_func_root_action = next((c for c in def_node_kx.children if c.node_type == "ChoiceOperator (++)"), None)
                        if control_func_root_action:
                            control_func_config.action_structure = _parse_kx_hp_program_fragment_to_action_config(
                                control_func_root_action, 0
                            )
                        defs_config.hybrid_programs.append(control_func_config)
                    elif def_node_kx.name == "stepFunc":
                        step_func_config = StepFuncConfig(name=def_node_kx.name, body_raw=def_node_kx.body)
                        ode_block = next((c for c in def_node_kx.children if c.node_type == "OrdinaryDifferentialEquations"), None)
                        if ode_block:
                            for ode_child in ode_block.children:
                                if ode_child.node_type == "ODE" and ode_child.value:
                                    # ODEs are like "x' = v * dx", extract variable and equation
                                    parts = ode_child.value.split("=", 1)
                                    if len(parts) == 2:
                                        var = parts[0].strip().replace("'", "") # Remove ' from variable
                                        equation = parts[1].strip()
                                        step_func_config.odes.append(ODEConfig(variable=var, equation=equation))
                                    else:
                                        print(f"Warning: Could not parse ODE: {ode_child.value}")

                        domain_block = next((c for c in def_node_kx.children if c.node_type == "EvolutionDomain"), None)
                        if domain_block:
                            evolution_domain_config = EvolutionDomainConfig()
                            for domain_child in domain_block.children:
                                if domain_child.node_type == "Constraint" and domain_child.value:
                                    evolution_domain_config.constraints.append(domain_child.value)
                            step_func_config.evolution_domain = evolution_domain_config
                        defs_config.hybrid_programs.append(step_func_config)
                elif def_node_kx.definition_type == "Real":
                    if parameters: # Has parameters, so it's a function
                        defs_config.functions.append(FunctionDefinitionConfig(
                            name=def_node_kx.name,
                            return_type=def_node_kx.definition_type,
                            parameters=parameters,
                            body=def_node_kx.definition_value
                        ))
                    else: # No parameters, so it's a constant
                        defs_config.constants.append(ConstantDefinitionConfig(
                            name=def_node_kx.name,
                            type=def_node_kx.definition_type,
                            value=def_node_kx.definition_value
                        ))
                elif def_node_kx.definition_type == "Bool":
                    defs_config.predicates.append(PredicateDefinitionConfig(
                        name=def_node_kx.name,
                        parameters=parameters,
                        body=def_node_kx.definition_value
                    ))

    # Process Program Variables Block
    vars_block_kx = next((c for c in kx_ast_root.children if c.node_type == "ProgramVariablesBlock"), None)
    if vars_block_kx:
        vars_config = gym_config.variables
        for var_node_kx in vars_block_kx.children:
            if var_node_kx.node_type == "Variable":
                vars_config.state_variables.append(VariableConfig(name=var_node_kx.value, type=var_node_kx.var_type))

    # Process Problem Block
    problem_block_kx = next((c for c in kx_ast_root.children if c.node_type == "ProblemStatement"), None)
    if problem_block_kx:
        problem_config = gym_config.problem

        precondition_kx = next((c for c in problem_block_kx.children if c.node_type == "Precondition (P)"), None)
        if precondition_kx and precondition_kx.value:
            problem_config.precondition.formula = precondition_kx.value

            # NEW LOGIC: Handle implicit 'assumptions' predicate
            # If 'assumptions' is used in the precondition but not explicitly defined,
            # we add a placeholder predicate definition.
            if "assumptions" in precondition_kx.value and not any(p.name == "assumptions" for p in definitions_config_instance.predicates):
                assumptions_call_match = re.search(r'assumptions\((.*?)\)', precondition_kx.value)
                parameters = []
                if assumptions_call_match:
                    param_str = assumptions_call_match.group(1)
                    param_names = [p.strip() for p in param_str.split(',') if p.strip()]
                    param_names = [name for name in param_names if name] # Filter out empty strings
                    parameters = [('Real', name) for name in param_names] # Default to Real type for now

                definitions_config_instance.predicates.append(PredicateDefinitionConfig(
                    name="assumptions",
                    parameters=parameters,
                    body="True" # Default body for undefined assumptions: always true
                ))


        loop_execution_kx = next((c for c in problem_block_kx.children if c.node_type == "LoopExecution"), None)
        if loop_execution_kx:
            loop_inv_kx = next((c for c in loop_execution_kx.children if c.node_type == "Invariant"), None)
            problem_config.loop_execution.loop_invariant_formula = loop_inv_kx.value if loop_inv_kx and loop_inv_kx.value else ""
            problem_config.loop_execution.iterations = "inf" # Default, or parse if available

            # The LoopExecution node in KeYmaera X AST contains an HP node for the loop body
            loop_body_hp_kx = next((c for c in loop_execution_kx.children if c.node_type == "HP"), None)
            if loop_body_hp_kx and len(loop_body_hp_kx.children) == 2:
                # Discrete control part: { { controlFunc; }; t := 0; }
                discrete_hp_kx = loop_body_hp_kx.children[0] # This is an HP node
                if discrete_hp_kx.node_type == "HP":
                    # The children of this discrete_hp_kx are the actual statements/calls
                    for stmt_kx in discrete_hp_kx.children:
                        if stmt_kx.node_type == "Statement":
                            if stmt_kx.value == "controlFunc":
                                # controlFunc is the agent's action, not part of the loop body actions
                                # So, we don't add it to the loop_body.actions directly.
                                pass
                            elif ":=" in stmt_kx.value:
                                var, val = stmt_kx.value.split(':=', 1)
                                problem_config.loop_execution.loop_body.actions.append(
                                    DeterministicAssignmentActionConfig(variable=var.strip(), value=val.strip())
                                )
                            else:
                                print(f"Warning: Unhandled statement type in discrete loop body: {stmt_kx.value}")
                        elif stmt_kx.node_type == "Test": # Handle constraints like isWellformedDir, if present here
                            problem_config.loop_execution.loop_body.actions.append(
                                ActionConstraintConfig(formula=stmt_kx.value)
                            )
                        elif stmt_kx.node_type == "HP": # This is the '{ controlFunc; }' block
                            # We already know controlFunc is handled by _apply_agent_action, so we skip it here.
                            pass
                        else:
                            print(f"Warning: Unhandled node type in discrete loop body: {stmt_kx.node_type}")

                # Continuous dynamics part: { stepFunc;@invariant(...) }
                continuous_hp_kx = loop_body_hp_kx.children[1] # This is an HP node
                if continuous_hp_kx.node_type == "HP":
                    step_func_call_kx = next((c for c in continuous_hp_kx.children if c.node_type == "Statement" and c.value == "stepFunc"), None)
                    step_inv_kx = next((c for c in continuous_hp_kx.children if c.node_type == "Invariant"), None)
                    if step_func_call_kx:
                        problem_config.loop_execution.loop_body.actions.append(HybridProgramStepConfig(
                            action=CallHybridProgramConfig(hp_name="stepFunc"),
                            invariant=ActionConstraintConfig(formula=step_inv_kx.value) if step_inv_kx and step_inv_kx.value else None
                        ))
                    else:
                        print(f"Warning: Could not find stepFunc call in continuous loop body.")
                else:
                    print(f"Warning: Expected HP node for continuous part, got {continuous_hp_kx.node_type}")

                # Add the global constraint 't>=0, isWellformedDir(dx, dy)' if it's a fixed part of the loop body
                # This is based on the example in the prompt's `_problem_loop_body_actions`
                # Note: The original example had `stepFunction;@invariant(t>=0, isWellformedDir(dx, dy))`
                # The invariant for the *stepFunc* itself should be separate from the *loop invariant*.
                # The problem statement shows `stepFunction;@invariant(t>=0, isWellformedDir(dx, dy))`.
                # This means `t>=0` and `isWellformedDir(dx, dy)` are invariants for the *continuous evolution step*.
                # The `HybridProgramStepConfig` already handles this.
                # The line below was adding it again as a separate action, which is redundant if correctly
                # parsed into HybridProgramStepConfig's `invariant` field.
                # I'm commenting it out as the `HybridProgramStepConfig` should now correctly capture it.
                # problem_config.loop_execution.loop_body.actions.append(
                #     ActionConstraintConfig(formula='t>=0 and isWellformedDir(dx, dy)') # Combine with 'and'
                # )

            else:
                print(f"Warning: Loop body HP node structure unexpected: {loop_body_hp_kx}")


        postcondition_kx = next((c for c in problem_block_kx.children if c.node_type == "Postcondition (Q)"), None)
        if postcondition_kx and postcondition_kx.value:
            problem_config.postcondition.formula = postcondition_kx.value

    return gym_config

def _parse_kx_hp_program_fragment_to_action_config(kx_node: ASTNode, choice_level: int) -> ActionType:
    """
    Recursively parses a KeYmaera X program fragment (like controlFunc body)
    into an ActionBlockConfig structure (ChoiceActionConfig or SequenceActionConfig).
    This function now expects to receive the *already parsed* ASTNodes from KeYmaeraXParser.
    """
    if not kx_node:
        return SequenceActionConfig() # Return an empty sequence if node is empty

    # If the current node is a ChoiceOperator (++)
    if kx_node.node_type == "ChoiceOperator (++)":
        choice_block = ChoiceActionConfig(level_name=f"ChoiceLevel_{choice_level}")
        for choice_kx_branch in kx_node.children:
            # Each child of ChoiceOperator is a "Choice N" node
            if choice_kx_branch.node_type.startswith("Choice"):
                # The content of this specific choice branch is its children, forming a sequence
                choice_content_sequence = SequenceActionConfig()
                for sub_node_kx in choice_kx_branch.children:
                    # Recursively parse the sub-node. It could be another ChoiceOperator, Statement, or Test.
                    parsed_sub_block = _parse_kx_hp_program_fragment_to_action_config(sub_node_kx, choice_level + 1)
                    # If the recursive call returns a sequence, extend the current sequence
                    if isinstance(parsed_sub_block, SequenceActionConfig):
                        choice_content_sequence.actions.extend(parsed_sub_block.actions)
                    else:
                        # Otherwise, add the single action/block returned
                        choice_content_sequence.actions.append(parsed_sub_block)
                choice_block.choices.append(choice_content_sequence) # Each choice branch is a sequence of actions
        return choice_block

    # If the current node is a single Statement or Test
    elif kx_node.node_type == "Statement":
        if kx_node.value.endswith(":= *"):
            var_name = kx_node.value.split(':=')[0].strip()
            return NondeterministicAssignmentActionConfig(variable=var_name)
        elif ":=" in kx_node.value:
            var_name, value = kx_node.value.split(':=', 1)
            return DeterministicAssignmentActionConfig(variable=var_name.strip(), value=value.strip())
        else:
            print(f"Warning: Unhandled statement type in controlFunc parsing: {kx_node.value}")
            return SequenceActionConfig()

    elif kx_node.node_type == "Test":
        return ActionConstraintConfig(formula=kx_node.value)

    # If the current node is not a ChoiceOperator, Statement, or Test,
    # but it has children, it implies an implicit sequence of its children.
    # This handles cases where the KeYmaera X parser might group multiple statements
    # under a generic ASTNode (e.g., the root of controlFunc's body if it's not a direct choice).
    elif kx_node.children:
        sequence_block = SequenceActionConfig()
        for child_kx in kx_node.children:
            parsed_child_block = _parse_kx_hp_program_fragment_to_action_config(child_kx, choice_level)
            if isinstance(parsed_child_block, SequenceActionConfig):
                sequence_block.actions.extend(parsed_child_block.actions)
            else:
                sequence_block.actions.append(parsed_child_block)
        return sequence_block

    return SequenceActionConfig() # Default for unhandled or empty nodes