#keymaerax_parser.py
#this file contain the building block for AST and also the parser for parsing KeymaeraX file into AST
from __future__ import annotations
import re
import logging # Import the logging module
from ast_config import *

# Configure logging (you might want to customize this further in a real application)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # Changed to INFO for less verbose default

# --- AST Node Definitions ---
# Tested: please dont touch
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
        value_str = ":"
        if self.value is not None:
            value_str += f" {self.value}"
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
        s = f"{indent}[DEF]: {self.definition_type} {self.name}{value_str}{comment_str}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s


class HybridProgram(Definition):
    """
    Represents a Hybrid Program (HP) definition.
    It parses the internal structure (choices, sequences, ODEs).
    """
    def __init__(self, name, body, parser_ref=None): # Added parser_ref
        super().__init__(name, "HP", value=body)
        self.name = name # Store name for specific parsing logic
        self.body = body
        # self._split_util = None # No longer needed here as _parse_program_fragment is moved
        self._parser_ref = parser_ref # Store reference to the parser instance

    def _parse_hp_structure(self):
        """
        Parses the body of a Hybrid Program structure based on its name.
        This method is part of the ASTNode class.
        """
        logging.debug(f"HybridProgram '{self.name}' calling _parse_hp_structure.")
        body_content = self.body.strip()

        if self.name == 'controlFunc':
            if self._parser_ref: # Call the parser's method
                self._parser_ref._parse_program_fragment(body_content, self)
            else:
                logging.warning(f"No parser reference available for HybridProgram '{self.name}'. Cannot parse program fragment.")

        elif self.name == 'stepFunc':
            logging.debug(f"Parsing stepFunc structure for '{self.name}'. Raw body: {body_content[:100]}...")
            ode_node = ASTNode("OrdinaryDifferentialEquations")
            domain_node = ASTNode("EvolutionDomain")

            # --- FIX: Remove outer braces from body_content before splitting ---
            clean_body_no_braces = body_content
            if clean_body_no_braces.startswith('{') and clean_body_no_braces.endswith('}'):
                clean_body_no_braces = clean_body_no_braces[1:-1].strip()
            clean_body = re.sub(r'\s+', ' ', clean_body_no_braces).strip()
            logging.debug(f"stepFunc body after cleaning: {clean_body[:100]}...")
            # --- END FIX ---

            # Use split_by_operator for comma, as ODEs can have nested structures
            split_by_operator = KeYmaeraXParser.split_by_operator_outside_braces # Direct call to static method
            parts = [p.strip() for p in split_by_operator(clean_body, ',') if p.strip()]
            logging.debug(f"stepFunc parts after splitting by comma: {parts}")

            for part in parts:
                if not part: continue

                # Check for evolution domain constraints (using &)
                # This split needs to be careful, as & might be inside a formula
                # For simplicity, if & is at brace_level 0, split it.
                sub_parts = split_by_operator(part, '&')
                logging.debug(f"stepFunc sub-parts for '{part}' after splitting by '&': {sub_parts}")


                if not sub_parts: # Should not happen if part is not empty, but for safety
                    continue

                # The first part is an ODE or a formula that might be an ODE/constraint
                # All subsequent parts (if any) are domain constraints
                ode_or_formula = sub_parts[0]
                if "'" in ode_or_formula: # Simple heuristic to identify an ODE (contains derivative)
                    ode_node.add_child(ASTNode("ODE", value=ode_or_formula))
                    logging.debug(f"Added ODE: {ode_or_formula}")
                else: # Treat as a constraint if it doesn't look like an ODE
                    domain_node.add_child(ASTNode("Constraint", value=ode_or_formula))
                    logging.debug(f"Added Constraint (from ODE/formula part): {ode_or_formula}")

                for domain_constraint in sub_parts[1:]:
                    if domain_constraint:
                        domain_node.add_child(ASTNode("Constraint", value=domain_constraint))
                        logging.debug(f"Added Constraint (from domain part): {domain_constraint}")

            self.add_child(ode_node)
            if domain_node.children:
                self.add_child(domain_node)
            logging.debug(f"Finished parsing stepFunc structure for '{self.name}'. ODE nodes: {len(ode_node.children)}, Domain nodes: {len(domain_node.children)}")


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

        if self.name:
            s = f"{indent}[HP]: {self.name}\n"
        else:
            s = f"{indent}[HP]:\n"
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
        logging.debug("KeYmaeraXParser initialized. Source code cleaned.")

    def _cleanup_source(self):
        # 1. Remove multi-line comments (/* ... */)
        self.source_code = re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m.group(0)), self.source_code, flags=re.DOTALL)
        # 2. Remove single-line comments (//)
        self.source_code = re.sub(r'//.*', '', self.source_code)
        # 3. Normalize line endings and remove excessive blank lines
        self.source_code = re.sub(r'\n\s*\n', '\n', self.source_code)
        self.source_code = self.source_code.strip()
        logging.debug("Source code cleanup complete.")

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
                    logging.warning(f"Malformed brace structure (unmatched '}}') at index {i} in '{text}'")
                    balance = 0 # Attempt to recover by resetting balance
                    current_text_segment_start = i + 1 # Continue as if this '}' was just text

        # Add any remaining text segment after the last block or if no blocks were found
        if current_text_segment_start < len(text):
            segment = text[current_text_segment_start:len(text)].strip()
            if segment:
                result.append(segment)

        # Check for unclosed braces at the end of the string
        if balance > 0:
            logging.warning(f"Unclosed brace block detected in text: '{text}'")
            # If there's an unclosed block, its content from block_start_index to the end
            # should be treated as a single text segment for this level, or handled as an error.
            if block_start_index != -1 and block_start_index < len(text):
                remaining_content = text[block_start_index:].strip()
                if remaining_content:
                    result.append(remaining_content)

        return result

    def parse(self):
        logging.debug("Starting parsing process.")
        self._parse_theorem_name()
        self._parse_definitions()
        self._parse_program_variables()
        self._parse_problem()
        logging.debug("Parsing process finished.")
        return self.ast_root

    def _parse_theorem_name(self):
        match = re.search(r'Theorem\s+"(.*?)"', self.source_code, re.DOTALL)
        if match:
            self.ast_root.add_child(ASTNode("TheoremName", match.group(1)))
            logging.debug(f"Theorem name found: '{match.group(1)}'")

    def _extract_block(self, keyword):
        # Extracts content between 'Keyword' and 'End.' or the next major keyword block
        pattern = rf'{keyword}\s*(.*?)(?=\n\s*End\.|\n\s*Theorem|\n\s*Definitions|\n\s*ProgramVariables|\n\s*Problem|$)'
        match = re.search(pattern, self.source_code, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content.endswith('End.'):
                 content = content[:-4].strip()
            logging.debug(f"Extracted block for '{keyword}'. Content length: {len(content)}. Snippet: '{content[:100]}...'")
            return content
        logging.debug(f"No block found for keyword '{keyword}'.")
        return None

    def _parse_definitions(self):
        logging.debug("Starting _parse_definitions.")
        definitions_block = self._extract_block("Definitions")
        if not definitions_block:
            logging.debug("No Definitions block found. Skipping definition parsing.")
            return

        defs_node = ASTNode("DefinitionsBlock")
        self.ast_root.add_child(defs_node)
        logging.debug(f"Created DefinitionsBlock ASTNode.")

        # Use a mutable list of characters for efficient replacement of processed blocks.
        mutable_definitions_block = list(definitions_block)

        # --- 1. Extract Hybrid Programs (HP) first ---

        # Pattern to find the HP header up to the opening brace '{'
        HP_HEADER_PATTERN = re.compile(
            r'HP\s+(\w+)\s*::=\s*\{',
            re.DOTALL
        )
        logging.debug("Starting HP extraction.")
        current_search_pos = 0
        while current_search_pos < len(mutable_definitions_block):
            # Search for the next HP header from the current position
            search_string = "".join(mutable_definitions_block[current_search_pos:])
            match = HP_HEADER_PATTERN.search(search_string)

            if match:
                hp_header_start_idx = current_search_pos + match.start()
                open_brace_idx = current_search_pos + match.end() - 1
                hp_name = match.group(1).strip()
                logging.debug(f"Found HP header for '{hp_name}' at global index {hp_header_start_idx}.")

                full_current_block_string = "".join(mutable_definitions_block)

                closing_brace_idx = self._find_matching_brace_span(
                    full_current_block_string,
                    open_brace_idx
                )

                if closing_brace_idx is not None:
                    hp_body_raw = "".join(mutable_definitions_block[open_brace_idx : closing_brace_idx + 1]).strip()
                    logging.debug(f"Extracted HP '{hp_name}' body (raw): '{hp_body_raw[:100]}...'")

                    full_hp_block_end_idx = closing_brace_idx + 1

                    if full_hp_block_end_idx + 1 < len(mutable_definitions_block) and \
                       "".join(mutable_definitions_block[full_hp_block_end_idx : full_hp_block_end_idx + 2]).strip() == ';;':
                        full_hp_block_end_idx += 2
                    elif full_hp_block_end_idx < len(mutable_definitions_block) and \
                         mutable_definitions_block[full_hp_block_end_idx] == ';':
                        full_hp_block_end_idx += 1

                    hp_node = HybridProgram(hp_name, hp_body_raw, parser_ref=self)
                    hp_node._parse_hp_structure()
                    defs_node.add_child(hp_node)
                    logging.debug(f"Added HybridProgram ASTNode for '{hp_name}' to DefinitionsBlock.")

                    for i in range(hp_header_start_idx, full_hp_block_end_idx):
                        if i < len(mutable_definitions_block):
                            mutable_definitions_block[i] = ' '

                    current_search_pos = full_hp_block_end_idx
                else:
                    logging.warning(f"Unbalanced braces for HP '{hp_name}' starting at index {open_brace_idx}. Skipping this block.")
                    current_search_pos = open_brace_idx + 1
            else:
                logging.debug("No more HP definitions found in current segment. Breaking HP extraction loop.")
                break

        remaining_text_after_hps = "".join(mutable_definitions_block)
        logging.debug(f"Finished HP extraction. Remaining text for simple definitions:\n'{remaining_text_after_hps.strip()[:200]}...'")

        # --- 2. Parse simple Definitions (DEF Real/Bool) ---
        parts = self.split_by_operator_outside_braces(remaining_text_after_hps, ';')
        logging.debug(f"Simple definitions parts after splitting by ';': {parts}")

        for part in parts:
            line = part.strip()
            if not line:
                continue

            def_match = re.search(r'(Real|Bool)\s+(\w+)(?:\s*\((.*?)\))?\s*(?:<->|=)\s*(.*?)$', line, re.DOTALL)

            if def_match:
                def_type = def_match.group(1)
                name = def_match.group(2)
                params = def_match.group(3)
                value = def_match.group(4).strip()
                logging.debug(f"Found Real/Bool definition with value: Type='{def_type}', Name='{name}', Value='{value}'")

                if value.startswith('(') and value.endswith(')'):
                    value = value[1:-1].strip()

                def_node = Definition(name, def_type, value=value)

                if params:
                    param_list = [p.strip() for p in self.split_by_operator_outside_braces(params, ',')]
                    param_node = ASTNode("Parameters", ", ".join(param_list))
                    def_node.add_child(param_node)

                defs_node.add_child(def_node)
                continue

            const_match = re.search(r'(Real|Bool)\s+(\w+)\s*$', line, re.DOTALL)
            if const_match:
                def_type = const_match.group(1)
                name = const_match.group(2)
                defs_node.add_child(Definition(name, def_type))
                logging.debug(f"Found simple constant definition: Type='{def_type}', Name='{name}'")
        logging.debug("Finished _parse_definitions.")


    def _parse_program_variables(self):
        logging.debug("Starting _parse_program_variables.")
        vars_block = self._extract_block("ProgramVariables")
        if not vars_block:
            logging.debug("No ProgramVariables block found. Skipping variable parsing.")
            return

        vars_node = ASTNode("ProgramVariablesBlock")
        self.ast_root.add_child(vars_node)
        logging.debug("Created ProgramVariablesBlock ASTNode.")

        for line in self.split_by_operator_outside_braces(vars_block, ';'):
            line = line.strip()
            if not line:
                continue

            var_match = re.search(r'(Real|Bool)\s+(\w+)\s*$', line, re.DOTALL)
            if var_match:
                var_type = var_match.group(1)
                name = var_match.group(2)
                vars_node.add_child(Variable(name, var_type))
                logging.debug(f"Added Variable: Type='{var_type}', Name='{name}'")
        logging.debug("Finished _parse_program_variables.")


    def _parse_program_fragment(self, program_fragment_string: str, parent_node: ASTNode):
        logging.debug(f"_parse_program_fragment called for: '{program_fragment_string[:80]}...'")
        fragment = program_fragment_string.strip()
        if not fragment:
            logging.debug("Empty fragment, returning from _parse_program_fragment.")
            return

        split_by_operator = self.split_by_operator_outside_braces

        if fragment.startswith('{') and fragment.endswith('}'):
            fragment_content = fragment[1:-1].strip()
            logging.debug(f"Stripped outer braces. Content: '{fragment_content[:80]}...'")
        else:
            fragment_content = fragment

        choice_parts = split_by_operator(fragment_content, '++')
        if len(choice_parts) > 1:
            logging.debug(f"Detected ChoiceOperator (++) with {len(choice_parts)} parts.")
            choice_node = ASTNode("ChoiceOperator (++)")
            parent_node.add_child(choice_node)
            for i, part in enumerate(choice_parts):
                child_choice_node = ASTNode(f"Choice {i+1}")
                choice_node.add_child(child_choice_node)
                self._parse_program_fragment(part, child_choice_node)
            return

        statements = [s.strip() for s in split_by_operator(fragment_content, ';') if s.strip()]
        if len(statements) > 1 or (len(statements) == 1 and ';' in fragment_content):
            logging.debug(f"Detected Sequence (;) with {len(statements)} statements.")
            for stmt_value in statements:
                if stmt_value:
                    self._parse_program_fragment(stmt_value, parent_node)
            return

        if fragment_content:
            if fragment_content.startswith('?'):
                parent_node.add_child(ASTNode("Test", value=fragment_content))
                logging.debug(f"Added Test statement: '{fragment_content}'")
            else:
                parent_node.add_child(ASTNode("Statement", value=fragment_content))
                logging.debug(f"Added Statement: '{fragment_content}'")
        logging.debug("Finished _parse_program_fragment for this level.")


    def _parse_problem(self):
        logging.debug("Starting _parse_problem.")
        problem_block = self._extract_block("Problem")
        if not problem_block:
            logging.debug("No Problem block found. Skipping problem parsing.")
            return

        problem_node = ProblemStatement(problem_block)
        self.ast_root.add_child(problem_node)
        logging.debug("Created ProblemStatement ASTNode.")

        implication_match = re.search(
            r'([\s\S]+?)\s*->\s*\[([\s\S]+?)\]\s*([\s\S]+)', # Corrected regex for [HP]
            problem_block,
            re.DOTALL
        )

        if implication_match:
            precondition_raw = implication_match.group(1).strip()
            program_raw = implication_match.group(2).strip()
            postcondition_raw = implication_match.group(3).strip()
            logging.debug(f"Problem structure P -> [HP] Q matched.")
            logging.debug(f"Precondition: '{precondition_raw[:50]}...'")
            logging.debug(f"Program: '{program_raw[:50]}...'")
            logging.debug(f"Postcondition: '{postcondition_raw[:50]}...'")


            problem_node.add_child(ASTNode("Precondition (P)", precondition_raw))

            loop_match = re.search(
                r"(\{[\s\S]*?\})\*@invariant\(([\s\S]*?)\)", # Corrected regex for invariant closing paren
                program_raw,
                re.DOTALL
            )

            if loop_match:
                loop_body_raw = loop_match.group(1).strip()
                loop_invariant_raw = loop_match.group(2).strip()
                logging.debug(f"Loop structure matched. Loop body: '{loop_body_raw[:50]}...', Invariant: '{loop_invariant_raw[:50]}...'")

                program_node = ASTNode("LoopExecution")
                outer_hp_node = ASTNode("HP", "")

                loop_body_inner_content = loop_body_raw[1:-1].strip()

                discrete_start_idx = loop_body_inner_content.find('{')
                if discrete_start_idx == -1:
                    logging.warning(f"Could not find discrete block in loop body: '{loop_body_inner_content}'")
                    program_node.add_child(ASTNode("Raw Loop Body Content", loop_body_raw))
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                    problem_node.add_child(program_node)
                    problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
                    return

                discrete_end_idx = self._find_matching_brace_span(loop_body_inner_content, discrete_start_idx)
                if discrete_end_idx is None:
                    logging.warning(f"Unmatched braces in discrete block of loop body: '{loop_body_inner_content}'")
                    program_node.add_child(ASTNode("Raw Loop Body Content", loop_body_raw))
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                    problem_node.add_child(program_node)
                    problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
                    return

                discrete_block_full_str = loop_body_inner_content[discrete_start_idx : discrete_end_idx + 1].strip()
                logging.debug(f"Discrete block extracted: '{discrete_block_full_str[:50]}...'")


                continuous_search_start_idx = discrete_end_idx + 1
                continuous_start_idx = loop_body_inner_content.find('{', continuous_search_start_idx)

                if continuous_start_idx == -1:
                    logging.warning(f"Could not find continuous block in loop body: '{loop_body_inner_content}'")
                    program_node.add_child(ASTNode("Raw Loop Body Content", loop_body_raw))
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                    problem_node.add_child(program_node)
                    problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
                    return

                continuous_end_idx = self._find_matching_brace_span(loop_body_inner_content, continuous_start_idx)
                if continuous_end_idx == None: # Changed from -1 to None
                    logging.warning(f"Unmatched braces in continuous block of loop body: '{loop_body_inner_content}'")
                    program_node.add_child(ASTNode("Raw Loop Body Content", loop_body_raw))
                    program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                    problem_node.add_child(program_node)
                    problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
                    return

                continuous_block_full_str = loop_body_inner_content[continuous_start_idx : continuous_end_idx + 1].strip()
                logging.debug(f"Continuous block extracted: '{continuous_block_full_str[:50]}...'")

                control_hp_node = ASTNode("HP", "")
                discrete_fragment_content = discrete_block_full_str[1:-1].strip()
                self._parse_program_fragment(discrete_fragment_content, control_hp_node)
                outer_hp_node.add_child(control_hp_node)
                logging.debug("Processed discrete control block.")

                dynamics_hp_node = ASTNode("HP", "")
                dynamics_fragment_content = continuous_block_full_str[1:-1].strip()

                stmt_inv_match = re.search(r'^(.*?);@invariant\((.*?)\)$', dynamics_fragment_content, re.DOTALL)
                if stmt_inv_match:
                    statement_name = stmt_inv_match.group(1).strip()
                    invariant_content = stmt_inv_match.group(2).strip()
                    dynamics_hp_node.add_child(ASTNode("Statement", statement_name))
                    dynamics_hp_node.add_child(ASTNode("Invariant", invariant_content))
                    logging.debug(f"Processed dynamics block with invariant: Statement='{statement_name}', Invariant='{invariant_content}'")
                else:
                    dynamics_hp_node.add_child(ASTNode("Statement", dynamics_fragment_content))
                    logging.debug(f"Processed dynamics block without explicit invariant: Statement='{dynamics_fragment_content}'")
                outer_hp_node.add_child(dynamics_hp_node)

                program_node.add_child(outer_hp_node)
                program_node.add_child(ASTNode("Invariant", loop_invariant_raw))
                problem_node.add_child(program_node)
                logging.debug("Added LoopExecution to ProblemStatement.")
            else:
                program_node = ASTNode("RawProgram", program_raw)
                problem_node.add_child(program_node)
                logging.debug(f"No loop structure found in program, added as RawProgram.")

            problem_node.add_child(ASTNode("Postcondition (Q)", postcondition_raw))
        else:
            problem_node.add_child(ASTNode("Error", "Could not parse P -> [HP] Q structure."))
            logging.debug("Failed to parse P -> [HP] Q structure.")
        logging.debug("Finished _parse_problem.")

# Not tested from this point on
# --- Helper to convert KeYmaeraX AST to Intermediate PyAST ---
def convert_keymaerax_ast_to_intermediate_pyast(kx_ast_root: ASTNode) -> GymEnvironmentConfig:
    """
    Converts the KeYmaera X AST into the intermediate Python AST representation.
    """
    logging.debug("Starting convert_keymaerax_ast_to_intermediate_pyast.")
    if not kx_ast_root or kx_ast_root.node_type != "KeYmaeraXSpecification":
        raise ValueError("Invalid KeYmaera X AST root node provided.")

    env_name = "KeYmaeraX_RL_Environment" # Default name
    theorem_name_node = next((c for c in kx_ast_root.children if c.node_type == "TheoremName"), None)
    if theorem_name_node and theorem_name_node.value:
        env_name = theorem_name_node.value.replace(' ', '_').replace(':', '_').replace('"', '').replace('/', '_').replace('.', '_')
        logging.debug(f"Set environment name to '{env_name}' from theorem name.")

    # Initialize sub-configs first
    definitions_config = DefinitionsConfig()
    variables_config = VariablesConfig()
    problem_config = ProblemConfig(
        precondition=PreconditionConfig(formula=""),
        loop_execution=LoopExecutionConfig(),
        postcondition=PostconditionConfig(formula="")
    )

    # Pass them to GymEnvironmentConfig constructor
    gym_config = GymEnvironmentConfig(
        env_name=env_name,
        definitions=definitions_config,
        variables=variables_config,
        problem=problem_config
    )
    logging.debug("Initialized GymEnvironmentConfig.")

    defs_block_kx = next((c for c in kx_ast_root.children if c.node_type == "DefinitionsBlock"), None)
    if defs_block_kx:
        logging.debug("Found DefinitionsBlock in KeYmaeraX AST for conversion.")
        
        for def_node_kx in defs_block_kx.children:
            logging.debug(f"Converting definition AST node: {def_node_kx.node_type} (name: {getattr(def_node_kx, 'name', 'N/A')}, type: {getattr(def_node_kx, 'definition_type', 'N/A')})")
            if def_node_kx.node_type == "Definition" and hasattr(def_node_kx, 'definition_type'):
                parameters = []
                params_node = next((c for c in def_node_kx.children if c.node_type == "Parameters"), None)
                if params_node and params_node.value:
                    param_strings = [p.strip() for p in params_node.value.split(',')]
                    for p_str in param_strings:
                        parts = p_str.split(' ')
                        if len(parts) == 2:
                            parameters.append((parts[0], parts[1]))
                        else:
                            parameters.append(('Real', p_str)) # Default to Real if type not specified
                    logging.debug(f"Extracted parameters for '{def_node_kx.name}': {parameters}")

                if def_node_kx.definition_type == "HP":
                    logging.debug(f"Processing HP definition: {def_node_kx.name}")
                    if def_node_kx.name == "controlFunc":
                        control_func_root_action = next((c for c in def_node_kx.children if c.node_type == "ChoiceOperator (++)"), None)
                        action_space_config = None
                        if control_func_root_action:
                            action_space_config = _parse_kx_hp_program_fragment_to_action_config(
                                control_func_root_action, 0
                            )
                        # Pass action_space_config directly to constructor
                        control_func_config = ControlFuncConfig(name=def_node_kx.name, action_space_config=action_space_config)
                        definitions_config.hybrid_programs.append(control_func_config)
                        logging.debug("Appended ControlFuncConfig to hybrid_programs.")
                    elif def_node_kx.name == "stepFunc":
                        odes = []
                        evolution_domain_constraints = []
                        duration_expr: Optional[str] = None # Will need to be set from somewhere, perhaps a constant 'ep'

                        # The HybridProgram AST node (def_node_kx) itself has children for ODEs and EvolutionDomain
                        ode_block = next((c for c in def_node_kx.children if c.node_type == "OrdinaryDifferentialEquations"), None)
                        if ode_block:
                            logging.debug(f"Found ODE block for stepFunc with {len(ode_block.children)} children.")
                            for ode_child in ode_block.children:
                                if ode_child.node_type == "ODE" and ode_child.value:
                                    parts = ode_child.value.split("=", 1)
                                    if len(parts) == 2:
                                        var = parts[0].strip().replace("'", "")
                                        equation = parts[1].strip()
                                        odes.append(ODEConfig(variable=var, equation=equation))
                                        logging.debug(f"  Parsed ODE: {var}' = {equation}")
                                    else:
                                        logging.warning(f"Could not parse ODE: {ode_child.value}")
                                else:
                                    logging.warning(f"Unexpected node type in ODE block: {ode_child.node_type}")

                        domain_block = next((c for c in def_node_kx.children if c.node_type == "EvolutionDomain"), None)
                        if domain_block:
                            logging.debug(f"Found EvolutionDomain block for stepFunc with {len(domain_block.children)} children.")
                            for domain_child in domain_block.children:
                                if domain_child.node_type == "Constraint" and domain_child.value:
                                    evolution_domain_constraints.append(domain_child.value)
                                    logging.debug(f"  Parsed Domain Constraint: {domain_child.value}")
                                else:
                                    logging.warning(f"Unexpected node type in EvolutionDomain block: {domain_child.node_type}")
                        
                        # The duration for stepFunc is often a constant like 'ep'.
                        # The parser doesn't explicitly extract it into the HybridProgram AST node.
                        # It's typically found in the Problem block's loop execution or as a global constant.
                        # For now, we'll keep it as None and let the generator use a default.
                        
                        evolution_domain = EvolutionDomainConfig(constraints=evolution_domain_constraints) if evolution_domain_constraints else None
                        step_func_config = StepFuncConfig(name=def_node_kx.name, odes=odes, evolution_domain=evolution_domain, duration=duration_expr)
                        definitions_config.hybrid_programs.append(step_func_config)
                        logging.debug("Appended StepFuncConfig to hybrid_programs.")
                    else:
                        logging.debug(f"Unrecognized HP name '{def_node_kx.name}'. Skipping conversion to specific HPConfig.")
                elif def_node_kx.definition_type == "Real":
                    if parameters:
                        definitions_config.functions.append(FunctionDefinitionConfig(
                            name=def_node_kx.name,
                            return_type=def_node_kx.definition_type,
                            parameters=parameters,
                            body=def_node_kx.definition_value
                        ))
                        logging.debug(f"Appended FunctionDefinitionConfig: {def_node_kx.name}")
                    else:
                        definitions_config.constants.append(ConstantDefinitionConfig(
                            name=def_node_kx.name,
                            type=def_node_kx.definition_type, # Using 'type' now
                            value=def_node_kx.definition_value
                        ))
                        logging.debug(f"Appended ConstantDefinitionConfig: {def_node_kx.name}")
                elif def_node_kx.definition_type == "Bool":
                    definitions_config.predicates.append(PredicateDefinitionConfig(
                        name=def_node_kx.name,
                        parameters=parameters,
                        body=def_node_kx.definition_value
                    ))
                    logging.debug(f"Appended PredicateDefinitionConfig: {def_node_kx.name}")
            else:
                logging.debug(f"Skipping non-Definition AST node: {def_node_kx.node_type}")
    else:
        logging.debug("No DefinitionsBlock found in KeYmaeraX AST for conversion.")

    vars_block_kx = next((c for c in kx_ast_root.children if c.node_type == "ProgramVariablesBlock"), None)
    if vars_block_kx:
        logging.debug("Found ProgramVariablesBlock in KeYmaeraX AST for conversion.")
        
        for var_node_kx in vars_block_kx.children:
            if var_node_kx.node_type == "Variable":
                # Changed VariableConfig to VariableDeclarationConfig
                variables_config.state_variables.append(VariableDeclarationConfig(name=var_node_kx.value, type=var_node_kx.var_type))
                logging.debug(f"Added VariableDeclarationConfig: {var_node_kx.value}")

    problem_block_kx = next((c for c in kx_ast_root.children if c.node_type == "ProblemStatement"), None)
    if problem_block_kx:
        logging.debug("Found ProblemStatement in KeYmaeraX AST for conversion.")
        

        precondition_kx = next((c for c in problem_block_kx.children if c.node_type == "Precondition (P)"), None)
        if precondition_kx and precondition_kx.value:
            problem_config.precondition = PreconditionConfig(formula=precondition_kx.value)
            logging.debug(f"Set precondition: '{precondition_kx.value[:50]}...'")

        loop_execution_kx = next((c for c in problem_block_kx.children if c.node_type == "LoopExecution"), None)
        if loop_execution_kx:
            logging.debug("Found LoopExecution in KeYmaeraX AST for conversion.")
            loop_inv_kx = next((c for c in loop_execution_kx.children if c.node_type == "Invariant"), None)
            loop_execution_config = LoopExecutionConfig(
                loop_invariant_formula=loop_inv_kx.value if loop_inv_kx and loop_inv_kx.value else "",
                iterations="inf"
            )
            problem_config.loop_execution = loop_execution_config
            logging.debug(f"Set loop invariant: '{loop_execution_config.loop_invariant_formula[:50]}...'")

            loop_body_hp_kx = next((c for c in loop_execution_kx.children if c.node_type == "HP"), None)
            if loop_body_hp_kx and len(loop_body_hp_kx.children) == 2:
                logging.debug("Found loop body HP structure (discrete and continuous parts).")
                discrete_hp_kx = loop_body_hp_kx.children[0]
                if discrete_hp_kx.node_type == "HP":
                    discrete_sequence_config = SequenceActionConfig()
                    for stmt_kx in discrete_hp_kx.children:
                        if stmt_kx.node_type == "Statement":
                            if stmt_kx.value == "controlFunc":
                                discrete_sequence_config.actions.append(CallHybridProgramConfig(hp_name="controlFunc"))
                                logging.debug("Added CallHybridProgramConfig for 'controlFunc' to loop body.")
                            elif ":=" in stmt_kx.value:
                                var, val = stmt_kx.value.split(':=', 1)
                                discrete_sequence_config.actions.append(DeterministicAssignmentActionConfig(variable=var.strip(), value=val.strip()))
                                logging.debug(f"Added DeterministicAssignmentActionConfig: {var.strip()} := {val.strip()}")
                            else:
                                logging.warning(f"Unhandled statement in discrete loop body: {stmt_kx.value}")
                        else:
                            logging.warning(f"Unhandled node type in discrete loop body: {stmt_kx.node_type}")
                    loop_execution_config.loop_body.actions.append(discrete_sequence_config)

                continuous_hp_kx = loop_body_hp_kx.children[1]
                if continuous_hp_kx.node_type == "HP":
                    step_func_call_kx = next((c for c in continuous_hp_kx.children if c.node_type == "Statement" and c.value == "stepFunc"), None)
                    step_inv_kx = next((c for c in continuous_hp_kx.children if c.node_type == "Invariant"), None)
                    if step_func_call_kx:
                        loop_execution_config.loop_body.actions.append(HybridProgramStepConfig(
                            hp_name="stepFunc",
                            invariant_formula=step_inv_kx.value if step_inv_kx and step_inv_kx.value else None
                        ))
                        logging.debug("Added HybridProgramStepConfig for 'stepFunc' to loop body.")
                    else:
                        logging.warning(f"Could not find stepFunc call in continuous loop body.")
                else:
                    logging.warning(f"Expected HP node for continuous part, got {continuous_hp_kx.node_type}")
            else:
                logging.warning(f"Loop body HP node structure unexpected: {loop_body_hp_kx}")


        postcondition_kx = next((c for c in problem_block_kx.children if c.node_type == "Postcondition (Q)"), None)
        if postcondition_kx and postcondition_kx.value:
            problem_config.postcondition = PostconditionConfig(formula=postcondition_kx.value)
            logging.debug(f"Set postcondition: '{postcondition_kx.value[:50]}...'")

    logging.debug(f"Final GymEnvironmentConfig.definitions.hybrid_programs contains: {[hp.name for hp in gym_config.definitions.hybrid_programs]}")
    logging.debug("Finished convert_keymaerax_ast_to_intermediate_pyast.")
    return gym_config

def _parse_kx_hp_program_fragment_to_action_config(kx_node: ASTNode, choice_level: int) -> ActionType:
    """
    Recursively parses a KeYmaera X program fragment (like controlFunc body)
    into an ActionBlockConfig structure (ChoiceActionConfig or SequenceActionConfig).
    This function now expects to receive the *already parsed* ASTNodes from KeYmaeraXParser.
    """
    logging.debug(f"_parse_kx_hp_program_fragment_to_action_config called for node type: {kx_node.node_type}, level: {choice_level}")
    if not kx_node:
        logging.debug("Empty kx_node, returning empty SequenceActionConfig.")
        return SequenceActionConfig()

    if kx_node.node_type == "ChoiceOperator (++)":
        choice_block = ChoiceActionConfig(level_name=f"ChoiceLevel_{choice_level}", choices=[])
        logging.debug(f"Creating ChoiceActionConfig for ChoiceOperator (++) at level {choice_level}.")
        for choice_kx_branch in kx_node.children:
            if choice_kx_branch.node_type.startswith("Choice"):
                choice_content_sequence = SequenceActionConfig()
                logging.debug(f"Processing choice branch: {choice_kx_branch.node_type}")
                for sub_node_kx in choice_kx_branch.children:
                    parsed_sub_block = _parse_kx_hp_program_fragment_to_action_config(sub_node_kx, choice_level + 1)
                    if isinstance(parsed_sub_block, SequenceActionConfig):
                        choice_content_sequence.actions.extend(parsed_sub_block.actions)
                    else:
                        choice_content_sequence.actions.append(parsed_sub_block)
                choice_block.choices.append(choice_content_sequence)
        return choice_block

    elif kx_node.node_type == "Statement":
        if kx_node.value.endswith(":= *"):
            var_name = kx_node.value.split(':=')[0].strip()
            logging.debug(f"Creating NondeterministicAssignmentActionConfig for '{var_name}'.")
            return NondeterministicAssignmentActionConfig(variable=var_name)
        elif ":=" in kx_node.value:
            var_name, value = kx_node.value.split(':=', 1)
            logging.debug(f"Creating DeterministicAssignmentActionConfig for '{var_name}' := '{value}'.")
            return DeterministicAssignmentActionConfig(variable=var_name.strip(), value=value.strip())
        else:
            logging.warning(f"Unhandled statement type in controlFunc parsing: {kx_node.value}. Returning empty SequenceActionConfig.")
            return SequenceActionConfig()

    elif kx_node.node_type == "Test":
        logging.debug(f"Creating ActionConstraintConfig for Test: '{kx_node.value}'.")
        return ActionConstraintConfig(formula=kx_node.value)

    elif kx_node.children:
        logging.debug(f"Node '{kx_node.node_type}' has children, treating as implicit sequence.")
        sequence_block = SequenceActionConfig()
        for child_kx in kx_node.children:
            parsed_child_block = _parse_kx_hp_program_fragment_to_action_config(child_kx, choice_level)
            # The original code had 'parsed_sub_block' here, which was likely a copy-paste error.
            # It should be 'parsed_child_block' to refer to the result of the recursive call.
            if isinstance(parsed_child_block, SequenceActionConfig):
                sequence_block.actions.extend(parsed_child_block.actions)
            else:
                sequence_block.actions.append(parsed_child_block)
        return sequence_block

    logging.debug(f"Unhandled or empty kx_node type '{kx_node.node_type}'. Returning empty SequenceActionConfig.")
    return SequenceActionConfig()