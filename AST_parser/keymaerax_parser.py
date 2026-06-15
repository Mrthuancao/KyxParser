# keymaerax_parser.py
from __future__ import annotations
import re
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
import sys # Import the sys module

# Increase the recursion limit to handle deeply nested program structures.
# The default is usually 1000 or 3000. 5000 should be sufficient for most cases.
# Adjust if necessary based on the complexity of your KeYmaera X programs.
sys.setrecursionlimit(5000)

# Import AST node definitions from ast_config.py
from ast_config import (
    ASTNode, Definition, HybridProgram, Variable, ProblemStatement,
    IfElseStatement, ChoiceStatement, SequenceStatement, TestStatement,
    AssignmentStatement, HPCall, SafetyPostcondition, LivenessPostcondition
)

# Import ast_config and its helper functions
# Assuming pyast_config.py contains the GymEnvironmentConfig and related classes
from pyast_config import (
    GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig,
    ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig,
    HybridProgramConfig, ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig,
    VariableDeclarationConfig, PreconditionConfig, PostconditionConfig,
    LoopExecutionConfig, HybridProgramStepConfig,
    ActionType, ChoiceActionConfig, SequenceActionConfig,
    DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig,
    ActionConstraintConfig, CallHybridProgramConfig
)


# Configure logging (you might want to customize this further in a real application)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Parser Implementation ---
class KeYmaeraXParser:
    def __init__(self, source_code):
        self.source_code = source_code
        # ASTNode is now imported from ast_config, which has metadata support
        self.ast_root = ASTNode("KeYmaeraXSpecification")
        self._cleanup_source()
        logging.debug("KeYmaeraXParser initialized. Source code cleaned.")

    def _cleanup_source(self):
        # 1. Remove multi-line comments (/* ... */)
        self.source_code = re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m.group(0)), self.source_code, flags=re.DOTALL)
        # 2. Remove single-line comments (//)
        # BUT keep metadata comments (//@)
        # This regex looks for // that is NOT followed by @
        self.source_code = re.sub(r'//(?![ \t]*@).*', '', self.source_code)
        # 3. Normalize line endings and remove excessive blank lines
        self.source_code = re.sub(r'\n\s*\n', '\n', self.source_code)
        self.source_code = self.source_code.strip()

        # print("Source code cleanup complete. Cleaned source:", self.source_code)
        logging.debug("Source code cleanup complete.")

    # Helper method to split by an operator, respecting brace nesting
    @staticmethod
    def split_by_operator_outside_braces(text, operator):
        """
        Splits a string by a given operator, but only if the operator is not
        enclosed within any type of braces (curly {}, parentheses (), or square []).
        """
        parts = []
        brace_level = 0
        paren_level = 0
        square_level = 0
        current_part_start = 0

        i = 0
        while i < len(text):
            char = text[i]

            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
            elif char == '(':
                paren_level += 1
            elif char == ')':
                paren_level -= 1
            elif char == '[':
                square_level += 1
            elif char == ']':
                square_level -= 1

            # Check for operator match, ensuring we don't go out of bounds
            # and that all levels are zero (i.e., outside all delimiters)
            if (i + len(operator) <= len(text) and
                brace_level == 0 and paren_level == 0 and square_level == 0 and
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
        # First, try to match the 'ArchiveEntry' format
        match = re.search(r'ArchiveEntry\s+"(.*?)"', self.source_code, re.DOTALL)
        if match:
            theorem_name = match.group(1)
            self.ast_root.add_child(ASTNode("TheoremName", theorem_name))
            logging.debug(f"Theorem name found from ArchiveEntry: '{theorem_name}'")
            return

        # If ArchiveEntry not found, try the 'Theorem' format (for backward compatibility or other files)
        match = re.search(r'Theorem\s+"(.*?)"', self.source_code, re.DOTALL)
        if match:
            theorem_name = match.group(1)
            self.ast_root.add_child(ASTNode("TheoremName", theorem_name))
            logging.debug(f"Theorem name found from Theorem: '{theorem_name}')")
            return

        logging.debug("No Theorem name (neither ArchiveEntry nor Theorem) found in the source code.")

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
        # This allows us to "remove" parsed content by replacing it with spaces,
        # which regexes will then naturally skip over.
        mutable_definitions_block_chars = list(definitions_block)

        # Pre-compile patterns for efficiency
        # HP_START_PATTERN: Matches the header of an HP definition, including metadata.
        # Uses `[^\n]*` for single-line comments and `*` for multiple metadata lines.
        HP_START_PATTERN = re.compile(
            r'(?P<metadata>(?://\s*@[^\n]*\n)*)?' # Capture all leading comment lines for HP metadata
            r'\s*HP\s+(?P<name>\w+)\s*::=\s*\{',
            re.DOTALL # Allows '.' to match newlines for the overall pattern, though metadata is line-limited
        )

        # DEF_FULL_PATTERN: Matches a complete Real or Bool definition, including metadata,
        # parameters, value, and the terminating semicolon.
        # Uses `[^\n]*` for single-line comments.
        DEF_FULL_PATTERN = re.compile(
            r'(?P<raw_comments>(?:^[ \t]*//[^\n]*\n)*)' # Capture all leading comment lines (strictly single-line)
            r'[ \t]*(?P<type>Real|Bool)\s+(?P<name>\w+)' # Type and Name
            r'(?:\s*\((?P<params>.*?)\))?' # Optional parameters (can span multiple lines)
            r'(?:\s*(?:<->|=)\s*(?P<value>.*?))?' # Optional value/body (can span multiple lines)
            r';', # Terminating semicolon
            re.DOTALL | re.MULTILINE # DOTALL for multi-line params/value, MULTILINE for '^' to match start of line
        )

        # current_search_global_offset tracks the position in the original `definitions_block`
        # (and thus in `mutable_definitions_block_chars`) where the next search should begin.
        current_search_global_offset = 0

        # Loop until no more definitions can be found
        while True:
            # Recreate the string from the mutable list for each search.
            # Regex `search` will naturally skip over the ' ' characters from previous replacements.
            current_definitions_text_for_search = "".join(mutable_definitions_block_chars)

            # Search for the next HP header and Real/Bool definition from the current offset
            hp_match = HP_START_PATTERN.search(current_definitions_text_for_search, current_search_global_offset)
            def_match = DEF_FULL_PATTERN.search(current_definitions_text_for_search, current_search_global_offset)

            # Determine which definition type appears next (or if none are left)
            next_hp_idx = hp_match.start() if hp_match else float('inf')
            next_def_idx = def_match.start() if def_match else float('inf')

            if next_hp_idx == float('inf') and next_def_idx == float('inf'):
                # No more definitions of either type found in the remaining text
                logging.debug("No more definitions found. Exiting parsing loop.")
                break # Exit the loop

            if next_hp_idx < next_def_idx:
                # Process Hybrid Program (HP) because its header appeared first
                hp_header_start_idx = hp_match.start()
                open_brace_idx = hp_match.end() - 1 # The '{' is at the end of the HP header match
                hp_name = hp_match.group('name')
                raw_meta = hp_match.group('metadata')
                metadata_dict = self._parse_metadata_string(raw_meta) if raw_meta else {}
                logging.debug(f"Found HP header for '{hp_name}' at global index {hp_header_start_idx}.")

                # Use the full current string for brace matching
                closing_brace_idx = self._find_matching_brace_span(
                    current_definitions_text_for_search,
                    open_brace_idx
                )

                if closing_brace_idx is not None:
                    # hp_body_raw_within_braces includes the outer { }
                    hp_body_raw_within_braces = current_definitions_text_for_search[open_brace_idx : closing_brace_idx + 1].strip()
                    logging.debug(f"Extracted HP '{hp_name}' body (raw): '{hp_body_raw_within_braces[:100]}...'")

                    full_hp_block_end_idx = closing_brace_idx + 1

                    # Check for optional HP terminators (;; or ;) immediately after the closing brace
                    temp_str_after_brace = current_definitions_text_for_search[full_hp_block_end_idx:].lstrip()
                    if temp_str_after_brace.startswith(';;'):
                        terminator_idx = current_definitions_text_for_search.find(';;', full_hp_block_end_idx)
                        if terminator_idx != -1:
                            full_hp_block_end_idx = terminator_idx + 2
                    elif temp_str_after_brace.startswith(';'):
                        terminator_idx = current_definitions_text_for_search.find(';', full_hp_block_end_idx)
                        if terminator_idx != -1:
                            full_hp_block_end_idx = terminator_idx + 1

                    # --- NEW LOGIC FOR LOOP HPs LIKE dwwpdl ---
                    loop_match = re.search(
                        r'\{\s*(.*?)\s*\}\s*\*@invariant\s*\((.*?)\)',
                        hp_body_raw_within_braces, # Search within the content *including* the outer braces
                        re.DOTALL
                    )

                    if loop_match:
                        loop_body_inner_raw = loop_match.group(1).strip()
                        loop_invariant_formula = loop_match.group(2).strip()

                        hp_node = HybridProgram(hp_name, hp_body_raw_within_braces, metadata=metadata_dict)
                        hp_node.add_child(ASTNode("LoopBodyRaw", value=loop_body_inner_raw))
                        hp_node.add_child(ASTNode("Invariant", value=loop_invariant_formula))

                        # Recursively parse the *inner* loop body using the refactored _parse_program_fragment
                        parsed_loop_body = self._parse_program_fragment(loop_body_inner_raw)
                        hp_node.add_child(ASTNode("ParsedLoopBody", children=[parsed_loop_body]))

                        defs_node.add_child(hp_node)
                        logging.debug(f"Added Loop HybridProgram ASTNode for '{hp_name}' to DefinitionsBlock.")
                    else:
                        # Original logic for non-loop HPs
                        hp_node = HybridProgram(hp_name, hp_body_raw_within_braces, metadata=metadata_dict)
                        defs_node.add_child(hp_node)
                        logging.debug(f"Added HybridProgram ASTNode for '{hp_name}' to DefinitionsBlock.")

                        # Parse HP body structure if specific functions are defined
                        body_content_without_outer_braces = hp_body_raw_within_braces[1:-1].strip() # Remove outer braces for parsing
                        if hp_name == 'controlFunc':
                            # Use the refactored _parse_program_fragment
                            parsed_fragment_root = self._parse_program_fragment(body_content_without_outer_braces)
                            hp_node.add_child(ASTNode("ParsedControlFunc", children=[parsed_fragment_root]))
                            logging.debug(f"Parsed controlFunc structure for '{hp_name}'.")
                        elif hp_name == 'stepFunc':
                            self._parse_step_func_structure(hp_body_raw_within_braces, hp_node) # _parse_step_func_structure expects raw body with braces
                        # Other HPs like 'dwwpdl' without loop syntax would also fall here if not caught above.
                        # For 'dwwpdl' in the example, it *does* have loop syntax, so it's handled above.

                    # Replace the processed HP block with spaces in the mutable character list
                    for i in range(hp_header_start_idx, full_hp_block_end_idx):
                        if i < len(mutable_definitions_block_chars):
                            mutable_definitions_block_chars[i] = ' '

                    # Advance search position past the *original* end of the processed block
                    current_search_global_offset = full_hp_block_end_idx
                else:
                    logging.warning(f"Unbalanced braces for HP '{hp_name}' starting at index {open_brace_idx}. Skipping this block.")
                    # Advance search past the problematic open brace to avoid an infinite loop
                    current_search_global_offset = open_brace_idx + 1
            else:
                # Process Real/Bool Definition because its header appeared first
                def_start_idx = def_match.start()
                d_type = def_match.group('type')
                d_name = def_match.group('name')
                d_params = def_match.group('params')
                d_val = (def_match.group('value') or "").strip()
                raw_comments = def_match.group('raw_comments') or ""

                m_data = self._parse_metadata_string(raw_comments)

                if d_type == "Bool" and d_val.startswith('(') and d_val.endswith(')'):
                    d_val = d_val[1:-1].strip()

                # Pass metadata to the Definition AST node constructor
                def_node = Definition(d_name, d_type, value=d_val, metadata=m_data)

                if d_params:
                    param_list = [p.strip() for p in self.split_by_operator_outside_braces(d_params, ',')]
                    def_node.add_child(ASTNode("Parameters", ", ".join(param_list)))

                defs_node.add_child(def_node)
                logging.debug(f"Added Definition ASTNode for '{d_name}' (Type: {d_type}) to DefinitionsBlock.")

                # Replace the processed definition block with spaces
                def_end_idx = def_match.end()
                for i in range(def_start_idx, def_end_idx):
                    if i < len(mutable_definitions_block_chars):
                        mutable_definitions_block_chars[i] = ' '

                # Advance search position past the end of the processed block
                current_search_global_offset = def_end_idx

        # After the loop, check for any unparsed content (non-space characters)
        remaining_text_after_loop = "".join(mutable_definitions_block_chars).strip()
        if remaining_text_after_loop:
            logging.warning(f"Unparsed content remaining in Definitions block: '{remaining_text_after_loop[:200]}...'")


    def _parse_metadata_string(self, raw_metadata: Optional[str]) -> dict:
        """
        Converts RL-Annotation lines into a dictionary.
        Supports: // @category: key1=val1, key2={complex_expr}, key3=[bounds]
        """
        metadata = {}
        if not raw_metadata:
            return metadata

        # 1. Extract the category (e.g., @const, @var, @progress, @safety)
        # This looks for the word immediately following the '@' and before the ':'
        category_match = re.search(r'@(\w+):', raw_metadata)
        if category_match:
            metadata['type'] = category_match.group(1).strip()

        # 2. Extract key-value pairs
        attr_pairs_raw = re.findall(
            r'(\w+)\s*=\s*'
            r'(?:(\{.*?\})|'   # Group 1: curly braces
            r'([.*?])|'   # Group 2: square brackets (escaped)
            r'(\".*?\")|'   # Group 3: quoted string
            r'([^,]+))',    # Group 4: anything else (non-comma)
            raw_metadata,
            re.DOTALL
        )

        for match_tuple in attr_pairs_raw:
            key = match_tuple[0] # The actual key
            # Find the first non-empty value from the capturing groups (index 1, 2, 3, or 4)
            value_raw = next((v for v in match_tuple[1:] if v), '').strip()

            # Remove surrounding delimiters for the final dict value
            if (value_raw.startswith('{') and value_raw.endswith('}')) or \
               (value_raw.startswith('[') and value_raw.endswith(']')) or \
               (value_raw.startswith('"') and value_raw.endswith('"')):
                clean_val = value_raw[1:-1].strip()
            else:
                clean_val = value_raw

            # Handle on_success/on_failure values with optional status
            if key in ['on_success', 'on_failure']:
                parts = clean_val.split(':', 1) # Split only on the first colon
                action = parts[0].strip()
                status = 'none' # Default status if not specified
                if len(parts) > 1:
                    status_part = parts[1].strip()
                    if status_part.startswith('status='):
                        status = status_part[len('status='):].strip()
                metadata[key] = {'action': action, 'status': status}
            else:
                metadata[key] = clean_val

        return metadata


    def _parse_step_func_structure(self, body_content: str, hp_node: ASTNode):
        """Extracted stepFunc logic into a clean helper"""
        ode_node = ASTNode("OrdinaryDifferentialEquations")
        domain_node = ASTNode("EvolutionDomain")

        clean_body = body_content.strip()
        if clean_body.startswith('{') and clean_body.endswith('}'):
            clean_body = clean_body[1:-1].strip()

        # Split by comma first to separate ODEs and domain constraints
        parts = self.split_by_operator_outside_braces(clean_body, ',')
        for part in parts:
            part = part.strip()
            if not part: continue

            # Further split by '&' for multiple domain constraints or ODEs
            sub_parts = self.split_by_operator_outside_braces(part, '&')
            if not sub_parts: continue

            # The first sub_part might be an ODE or a constraint
            ode_or_formula = sub_parts[0].strip()
            if "'" in ode_or_formula: # Heuristic for ODEs (contains a prime)
                ode_node.add_child(ASTNode("ODE", value=ode_or_formula))
            else:
                domain_node.add_child(ASTNode("Constraint", value=ode_or_formula))

            # Remaining sub_parts are always domain constraints
            for domain_constraint in sub_parts[1:]:
                if domain_constraint.strip():
                    domain_node.add_child(ASTNode("Constraint", value=domain_constraint.strip()))

        hp_node.add_child(ode_node)
        if domain_node.children:
            hp_node.add_child(domain_node)


    def _parse_program_variables(self):
        logging.debug("Starting _parse_program_variables.")
        vars_block = self._extract_block("ProgramVariables")
        if not vars_block:
            logging.debug("No ProgramVariables block found. Skipping variable parsing.")
            return

        vars_node = ASTNode("ProgramVariablesBlock")
        self.ast_root.add_child(vars_node)
        logging.debug("Created ProgramVariablesBlock ASTNode.")

        # Regex to capture optional leading comments (metadata) and the variable declaration itself
        VAR_LINE_PATTERN = re.compile(
            r'(?P<raw_comments>(?:^[ \t]*//[^\n]*\n)*)?' # Optional leading comment lines
            r'[ \t]*(?P<type>Real|Bool)\s+(?P<name>\w+)\s*$', # Type and Name
            re.MULTILINE # Important for '^' to match start of line for comments
        )

        for line_segment in self.split_by_operator_outside_braces(vars_block, ';'):
            line_segment = line_segment.strip()
            if not line_segment:
                continue

            var_match = VAR_LINE_PATTERN.search(line_segment)
            if var_match:
                var_type = var_match.group('type')
                name = var_match.group('name')
                raw_comments = var_match.group('raw_comments') or ""
                metadata = self._parse_metadata_string(raw_comments)

                vars_node.add_child(Variable(name, var_type, metadata=metadata))
                logging.debug(f"Added Variable: Type='{var_type}', Name='{name}', Metadata={metadata}")
            else:
                logging.warning(f"Could not parse variable declaration line: '{line_segment}'")
        logging.debug("Finished _parse_program_variables.")


    def _parse_atomic_statement(self, statement_string: str) -> ASTNode:
        """
        Parses a single, non-compound statement (assignment, test, HP call, or raw).
        This is the base case for program fragment parsing.
        Assumes outer braces have already been stripped by _parse_program_fragment if present.
        """
        fragment = statement_string.strip()
        if not fragment:
            return ASTNode("EmptyFragment")

        if fragment.startswith('?'):
            return TestStatement(fragment[1:].strip())
        elif ":=" in fragment:
            return AssignmentStatement(fragment)
        elif re.match(r'^\w+$', fragment): # Simple identifier, likely an HP call
            return HPCall(fragment)
        else:
            logging.warning(f"Unrecognized program fragment (atomic base case): '{fragment}'. Treating as RawStatement.")
            return ASTNode("RawStatement", value=fragment)


    def _parse_program_fragment(self, program_fragment_string: str) -> ASTNode:
        """
        Parses a program fragment into an ASTNode representing its structure.
        Handles if/else, choice (++) and sequence (;) operators by precedence.
        This function now handles sequence parsing iteratively at the top level
        to prevent RecursionError for long sequences of atomic statements.
        """
        fragment = program_fragment_string.strip()
        if not fragment:
            return ASTNode("EmptyFragment")

        # 1. Handle top-level braces: If the entire fragment is enclosed in braces,
        # strip them and recursively parse the inner content.
        if fragment.startswith('{'):
            matching_brace_idx = self._find_matching_brace_span(fragment, 0)
            if matching_brace_idx == len(fragment) - 1:
                inner_content = fragment[1:-1].strip()
                return self._parse_program_fragment(inner_content)
            # If braces are not balanced or don't enclose the whole thing, proceed.

        # 2. Handle If-Else statements (highest precedence for compound statements)
        if fragment.startswith("if"):
            cond_match = re.match(r'if\s*\((.*?)\)\s*\{', fragment, re.DOTALL)
            if cond_match:
                condition = cond_match.group(1).strip()
                then_block_start = cond_match.end() - 1
                then_block_end = self._find_matching_brace_span(fragment, then_block_start)
                if then_block_end is None:
                    logging.warning(f"Unbalanced braces in 'then' block of if statement: '{fragment}'")
                    return ASTNode("ErrorFragment", value=fragment)

                then_content_raw = fragment[then_block_start + 1 : then_block_end].strip()
                then_node = self._parse_program_fragment(then_content_raw) # Recursive call for then block

                else_node = None
                remaining_after_then = fragment[then_block_end + 1:].strip()
                if remaining_after_then.startswith("else"):
                    else_keyword_end_match = re.match(r'else\s*(\{.*\}|.*)', remaining_after_then, re.DOTALL)
                    if else_keyword_end_match:
                        else_content_full = else_keyword_end_match.group(1).strip()
                        if else_content_full.startswith('{'):
                            else_block_start = remaining_after_then.find('{')
                            else_block_end = self._find_matching_brace_span(remaining_after_then, else_block_start)
                            if else_block_end is None:
                                logging.warning(f"Unbalanced braces in 'else' block of if statement: '{fragment}'")
                                return ASTNode("ErrorFragment", value=fragment)
                            else_content_raw = remaining_after_then[else_block_start + 1 : else_block_end].strip()
                            else_node = self._parse_program_fragment(else_content_raw) # Recursive call for else block
                        else: # 'else' followed by a single statement (or a sequence without explicit braces)
                            else_node = self._parse_program_fragment(else_content_full)

                if_else_ast_node = IfElseStatement(condition)
                if_else_ast_node.add_child(then_node)
                if else_node:
                    if_else_ast_node.add_child(else_node)
                return if_else_ast_node

        # 3. Check for Choice Operator (++) (next precedence)
        choice_parts = self.split_by_operator_outside_braces(fragment, '++')
        if len(choice_parts) > 1:
            choice_ast_node = ChoiceStatement()
            for part in choice_parts:
                if part.strip():
                    choice_ast_node.add_child(self._parse_program_fragment(part)) # Recursive call for each choice branch
            return choice_ast_node

        # 4. Handle Sequence or single Atomic statement
        # If it's not an if/else, a top-level choice, or a fully enclosed braced block,
        # it must be a sequence of statements (separated by ';') or a single atomic statement.
        statements = self.split_by_operator_outside_braces(fragment, ';')
        non_empty_statements = [s.strip() for s in statements if s.strip()]

        if not non_empty_statements:
            return ASTNode("EmptyFragment")

        # If there's only one non-empty statement, it means the fragment itself was a single unit.
        # Since top-level if/else, choice, and fully enclosed braced blocks are handled above,
        # if we reach here with a single statement, it must be atomic.
        if len(non_empty_statements) == 1:
            return self._parse_atomic_statement(non_empty_statements[0])
        
        # If there are multiple non-empty statements, it's a true sequence.
        sequence_ast_node = SequenceStatement()
        for stmt_value in non_empty_statements:
            if stmt_value.strip():
                temp_stmt_value = stmt_value.strip()
                
                # Determine if the current statement within the sequence is itself a compound structure.
                is_compound_nested = False
                
                # Check for nested if-else
                if temp_stmt_value.startswith('if '):
                    is_compound_nested = True
                # Check for nested choice (using split_by_operator_outside_braces for robustness)
                elif len(self.split_by_operator_outside_braces(temp_stmt_value, '++')) > 1:
                    is_compound_nested = True
                # Check for nested braced block (e.g., `{x:=1; y:=2}` within a sequence `a:=0; {x:=1; y:=2}; b:=3;`)
                elif temp_stmt_value.startswith('{') and self._find_matching_brace_span(temp_stmt_value, 0) == len(temp_stmt_value) - 1:
                    is_compound_nested = True
                
                if is_compound_nested:
                    # If it's a nested compound statement, recurse with _parse_program_fragment.
                    sequence_ast_node.add_child(self._parse_program_fragment(stmt_value))
                else:
                    # Otherwise, it's an atomic statement, parse it directly.
                    sequence_ast_node.add_child(self._parse_atomic_statement(stmt_value))
        return sequence_ast_node


    def _parse_problem(self):
        logging.debug("Starting _parse_problem.")
        problem_block = self._extract_block("Problem")
        if not problem_block:
            logging.debug("No Problem block found. Skipping problem parsing.")
            return

        problem_node = ProblemStatement(problem_block)
        self.ast_root.add_child(problem_node)
        logging.debug("Created ProblemStatement ASTNode.")

        # Split the problem_block into precondition, safety, and liveness parts
        # The pattern is: P -> [HP] Q_safety & <HP> Q_liveness

        # First, find P -> ProgramAndPostconditions
        implication_match = re.search(
            r'^(.*?)\s*->\s*(.*)$', # P -> ProgramAndPostconditions
            problem_block,
            re.DOTALL
        )

        if implication_match:
            precondition_raw = implication_match.group(1).strip()
            program_and_postconditions_raw = implication_match.group(2).strip()

            problem_node.add_child(ASTNode("Precondition (P)", precondition_raw))
            logging.debug(f"Precondition: '{precondition_raw[:50]}...'")

            # Now parse the ProgramAndPostconditions_raw: [dwwpdl;](...) & <dwwpdl;>(...)
            # We need to find the '&' that separates the safety and liveness parts.
            # This '&' should be at the top level, outside any parenthesis for the postconditions.
            safety_and_liveness_parts = self.split_by_operator_outside_braces(program_and_postconditions_raw, '&')

            if len(safety_and_liveness_parts) == 2:
                safety_part_raw = safety_and_liveness_parts[0].strip()
                liveness_part_raw = safety_and_liveness_parts[1].strip()

                # Parse Safety Postcondition: [HP;](Q_safety)
                safety_match = re.match(r'\[\s*(\w+)\s*;\s*\]\s*\((.*)\)', safety_part_raw, re.DOTALL)
                if safety_match:
                    hp_name_safety = safety_match.group(1).strip()
                    safety_formula = safety_match.group(2).strip()
                    safety_ast_node = SafetyPostcondition()
                    safety_ast_node.add_child(HPCall(hp_name_safety))
                    safety_ast_node.add_child(ASTNode("Formula", value=safety_formula))
                    problem_node.add_child(safety_ast_node)
                    logging.debug(f"Safety Postcondition: HP='{hp_name_safety}', Formula='{safety_formula[:50]}...'")
                else:
                    problem_node.add_child(ASTNode("Error", f"Could not parse safety part: '{safety_part_raw}'"))
                    logging.warning(f"Could not parse safety part: '{safety_part_raw}'")

                # Parse Liveness Postcondition: <HP;> Q_liveness
                # Note: The liveness formula might not be enclosed in parentheses.
                liveness_match = re.match(r'<\s*(\w+)\s*;\s*>\s*(.*)', liveness_part_raw, re.DOTALL)
                if liveness_match:
                    hp_name_liveness = liveness_match.group(1).strip()
                    liveness_formula = liveness_match.group(2).strip()
                    liveness_ast_node = LivenessPostcondition()
                    liveness_ast_node.add_child(HPCall(hp_name_liveness))
                    liveness_ast_node.add_child(ASTNode("Formula", value=liveness_formula))
                    problem_node.add_child(liveness_ast_node)
                    logging.debug(f"Liveness Postcondition: HP='{hp_name_liveness}', Formula='{liveness_formula[:50]}...'")
                else:
                    problem_node.add_child(ASTNode("Error", f"Could not parse liveness part: '{liveness_part_raw}'"))
                    logging.warning(f"Could not parse liveness part: '{liveness_part_raw}'")
            else:
                problem_node.add_child(ASTNode("Error", f"Could not split program and postconditions by '&': '{program_and_postconditions_raw}'"))
                logging.warning(f"Could not split program and postconditions by '&': '{program_and_postconditions_raw}'")
        else:
            problem_node.add_child(ASTNode("Error", "Could not parse P -> [HP] Q structure."))
            logging.debug("Failed to parse P -> [HP] Q structure.")
        logging.debug("Finished _parse_problem.")

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

    # --- MODIFICATION START ---
    # Initialize ProblemConfig without safety_postcondition and liveness_postcondition
    problem_config = ProblemConfig(
        precondition=PreconditionConfig(formula=""),
        loop_execution=LoopExecutionConfig()
    )
    # These attributes will be set later in the function if found in the AST
    problem_config.safety_postcondition = None
    problem_config.liveness_postcondition = None
    # --- MODIFICATION END ---

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
            # Access metadata directly from the ASTNode instance
            metadata = def_node_kx.metadata

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
                    hp_name = def_node_kx.name
                    hp_body = def_node_kx.body # This is the raw body string

                    # Check if it's a loop HP (like dwwpdl)
                    loop_body_raw_node = next((c for c in def_node_kx.children if c.node_type == "LoopBodyRaw"), None)
                    invariant_node = next((c for c in def_node_kx.children if c.node_type == "Invariant"), None)
                    parsed_loop_body_node = next((c for c in def_node_kx.children if c.node_type == "ParsedLoopBody"), None)

                    if loop_body_raw_node and invariant_node and parsed_loop_body_node:
                        # This is a loop HP (like dwwpdl). Its structure is stored in the HP definition.
                        # The Problem block will reference this.
                        logging.debug(f"Detected and parsed Loop HP '{hp_name}' with invariant: {invariant_node.value}")
                        # We don't add a specific config for it in definitions_config.hybrid_programs
                        # because its primary role is in the problem statement's loop execution.
                        # Its parsed body is available via parsed_loop_body_node.children[0]
                        # for when the Problem block needs to build the LoopExecutionConfig.

                    elif hp_name == "controlFunc":
                        # The children of the HP node should contain the parsed program fragment
                        parsed_control_func_root = next((c for c in def_node_kx.children if c.node_type == "ParsedControlFunc"), None)
                        action_space_config = None
                        if parsed_control_func_root and parsed_control_func_root.children:
                            # The actual root of the parsed control flow is the first child of ParsedControlFunc
                            action_space_config = _parse_kx_hp_program_fragment_to_action_config(
                                parsed_control_func_root.children[0], 0
                            )
                        control_func_config = ControlFuncConfig(name=hp_name, action_space_config=action_space_config, metadata=metadata)
                        definitions_config.hybrid_programs.append(control_func_config)
                        logging.debug("Appended ControlFuncConfig to hybrid_programs.")
                    elif hp_name == "stepFunc":
                        odes = []
                        evolution_domain_constraints = []
                        duration_expr: Optional[str] = None

                        ode_block = next((c for c in def_node_kx.children if c.node_type == "OrdinaryDifferentialEquations"), None)
                        if ode_block:
                            for ode_child in ode_block.children:
                                if ode_child.node_type == "ODE" and ode_child.value:
                                    parts = ode_child.value.split("=", 1)
                                    if len(parts) == 2:
                                        var = parts[0].strip().replace("'", "")
                                        equation = parts[1].strip()
                                        odes.append(ODEConfig(variable=var, equation=equation))

                        domain_block = next((c for c in def_node_kx.children if c.node_type == "EvolutionDomain"), None)
                        if domain_block:
                            for domain_child in domain_block.children:
                                if domain_child.node_type == "Constraint" and domain_child.value:
                                    evolution_domain_constraints.append(domain_child.value)

                        evolution_domain = EvolutionDomainConfig(constraints=evolution_domain_constraints) if evolution_domain_constraints else None
                        step_func_config = StepFuncConfig(name=hp_name, odes=odes, evolution_domain=evolution_domain, duration=duration_expr, raw_body_snippet=hp_body, metadata=metadata)
                        definitions_config.hybrid_programs.append(step_func_config)
                        logging.debug("Appended StepFuncConfig to hybrid_programs.")
                    else:
                        logging.debug(f"Unrecognized HP name '{hp_name}'. Skipping conversion to specific HPConfig.")
                elif def_node_kx.definition_type == "Real":
                    if parameters:
                        definitions_config.functions.append(FunctionDefinitionConfig(
                            name=def_node_kx.name,
                            return_type=def_node_kx.definition_type,
                            parameters=parameters,
                            body=def_node_kx.definition_value,
                            metadata=metadata
                        ))
                        logging.debug(f"Appended FunctionDefinitionConfig: {def_node_kx.name}")
                    else:
                        # Set value to None if it's an empty string for constants
                        constant_value = def_node_kx.definition_value if def_node_kx.definition_value else None
                        definitions_config.constants.append(ConstantDefinitionConfig(
                            name=def_node_kx.name,
                            type=def_node_kx.definition_type,
                            value=constant_value,
                            metadata=metadata
                        ))
                        logging.debug(f"Appended ConstantDefinitionConfig: {def_node_kx.name}")
                elif def_node_kx.definition_type == "Bool":
                    definitions_config.predicates.append(PredicateDefinitionConfig(
                        name=def_node_kx.name,
                        parameters=parameters,
                        body=def_node_kx.definition_value,
                        metadata=metadata
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
                # Get metadata from the Kx AST node
                var_metadata = var_node_kx.metadata # Access metadata directly from the ASTNode instance

                lower_bound = None
                upper_bound = None
                if 'bound' in var_metadata:
                    bound_str = var_metadata['bound']
                    # Use a more robust regex for parsing bounds like [0:100]
                    bound_match = re.match(r'(-?\d+(\.\d+)?):(-?\d+(\.\d+)?)', bound_str)
                    if bound_match:
                        lower_bound = bound_match.group(1)
                        upper_bound = bound_match.group(3)
                    else:
                        logging.warning(f"Could not parse bound string '{bound_str}' for variable '{var_node_kx.value}'.")

                variables_config.state_variables.append(VariableDeclarationConfig(
                    name=var_node_kx.value,
                    type=var_node_kx.var_type, # This should now be correctly present
                    lower_bound=lower_bound,
                    upper_bound=upper_bound
                ))
                # Note: If 'def' metadata (description) for variables is needed in the final
                # GymEnvironmentConfig, the VariableDeclarationConfig in pyast_config needs
                # to be extended with a 'description' field, and metadata['def'] would be
                # passed to it. Currently, it's parsed but not stored in the PyAST.
                logging.debug(f"Added VariableDeclarationConfig: {var_node_kx.value} with metadata: {var_metadata}")

    problem_block_kx = next((c for c in kx_ast_root.children if c.node_type == "ProblemStatement"), None)
    if problem_block_kx:
        logging.debug("Found ProblemStatement in KeYmaeraX AST for conversion.")

        precondition_kx = next((c for c in problem_block_kx.children if c.node_type == "Precondition (P)"), None)
        if precondition_kx and precondition_kx.value:
            problem_config.precondition = PreconditionConfig(formula=precondition_kx.value)
            logging.debug(f"Set precondition: '{precondition_kx.value[:50]}...'")

        # --- NEW LOGIC FOR SAFETY AND LIVENESS POSTCONDITIONS ---
        safety_postcondition_kx = next((c for c in problem_block_kx.children if c.node_type == "SafetyPostcondition"), None)
        if safety_postcondition_kx:
            hp_call_node = next((c for c in safety_postcondition_kx.children if c.node_type == "HPCall"), None)
            formula_node = next((c for c in safety_postcondition_kx.children if c.node_type == "Formula"), None)
            if hp_call_node and formula_node:
                problem_config.safety_postcondition = PostconditionConfig(
                    formula=formula_node.value,
                    hp_name=hp_call_node.value # Store the HP name for reference
                )
                logging.debug(f"Set safety postcondition: HP='{hp_call_node.value}', Formula='{formula_node.value[:50]}...'")

        liveness_postcondition_kx = next((c for c in problem_block_kx.children if c.node_type == "LivenessPostcondition"), None)
        if liveness_postcondition_kx:
            hp_call_node = next((c for c in liveness_postcondition_kx.children if c.node_type == "HPCall"), None)
            formula_node = next((c for c in liveness_postcondition_kx.children if c.node_type == "Formula"), None)
            if hp_call_node and formula_node:
                problem_config.liveness_postcondition = PostconditionConfig(
                    formula=formula_node.value,
                    hp_name=hp_call_node.value # Store the HP name for reference
                )
                logging.debug(f"Set liveness postcondition: HP='{hp_call_node.value}', Formula='{liveness_formula[:50]}...'")

        # The LoopExecutionConfig for the problem should now refer to the dwwpdl HP definition.
        # Find the dwwpdl HP definition in the KeYmaeraX AST to get its parsed loop body and invariant.
        dwwpdl_hp_def = None
        if defs_block_kx:
            for hp_def in defs_block_kx.children: # Iterate through raw Kx AST nodes
                if hp_def.node_type == "HP" and hp_def.name == "dwwpdl":
                    dwwpdl_hp_def = hp_def
                    break

        if dwwpdl_hp_def:
            loop_invariant_node = next((c for c in dwwpdl_hp_def.children if c.node_type == "Invariant"), None)
            parsed_loop_body_node = next((c for c in dwwpdl_hp_def.children if c.node_type == "ParsedLoopBody"), None)

            if loop_invariant_node and parsed_loop_body_node and parsed_loop_body_node.children:
                # The parsed_loop_body_node's first child is the root of the parsed program fragment
                loop_body_root_kx_node = parsed_loop_body_node.children[0]

                # Convert this Kx AST program fragment into PyAST ActionType
                loop_execution_actions_config = _parse_kx_hp_program_fragment_to_action_config(loop_body_root_kx_node, 0)

                # The loop body is expected to be a SequenceActionConfig at the top level
                if not isinstance(loop_execution_actions_config, SequenceActionConfig):
                    # If it's not a sequence, wrap it in one for consistency
                    loop_execution_actions_config = SequenceActionConfig(actions=[loop_execution_actions_config])

                problem_config.loop_execution = LoopExecutionConfig(
                    loop_invariant_formula=loop_invariant_node.value,
                    iterations="inf", # Assumed for general loops
                    loop_body=loop_execution_actions_config
                )
                logging.debug("Populated LoopExecutionConfig from dwwpdl HP definition.")
            else:
                logging.warning("Could not find invariant or parsed loop body for dwwpdl HP.")
        else:
            logging.warning("Could not find dwwpdl HP definition to populate LoopExecutionConfig.")

    logging.debug("Finished convert_keymaerax_ast_to_intermediate_pyast.")
    return gym_config

def _parse_kx_hp_program_fragment_to_action_config(kx_node: ASTNode, choice_level: int) -> ActionType:
    """
    Recursively parses a KeYmaera X program fragment AST node into an ActionType representation.
    """
    logging.debug(f"_parse_kx_hp_program_fragment_to_action_config called for node type: {kx_node.node_type}, value: {kx_node.value}, level: {choice_level}")
    if not kx_node or kx_node.node_type == "EmptyFragment":
        return SequenceActionConfig(actions=[]) # Return an empty sequence for empty fragments

    if kx_node.node_type == "ChoiceStatement":
        choice_block = ChoiceActionConfig(level_name=f"ChoiceLevel_{choice_level}", choices=[])
        for child_kx in kx_node.children:
            # Each child of a ChoiceStatement is a branch (could be Sequence, IfElse, etc.)
            parsed_child_action = _parse_kx_hp_program_fragment_to_action_config(child_kx, choice_level + 1)
            choice_block.choices.append(parsed_child_action)
        return choice_block

    elif kx_node.node_type == "SequenceStatement":
        sequence_block = SequenceActionConfig()
        for child_kx in kx_node.children:
            parsed_child_action = _parse_kx_hp_program_fragment_to_action_config(child_kx, choice_level)
            # If a child is also a sequence, extend the current sequence
            if isinstance(parsed_child_action, SequenceActionConfig):
                sequence_block.actions.extend(parsed_child_action.actions)
            else:
                sequence_block.actions.append(parsed_child_action)
        return sequence_block

    elif kx_node.node_type == "IfElseStatement":
        condition = kx_node.value # The condition is stored in the value of IfElseStatement
        then_block_node = kx_node.children[0] if kx_node.children else None
        else_block_node = kx_node.children[1] if len(kx_node.children) > 1 else None

        if not condition or not then_block_node:
            logging.warning(f"Malformed IfElseStatement: {kx_node.value}")
            return SequenceActionConfig(actions=[])

        # Recursively parse then and else blocks
        then_actions_config = _parse_kx_hp_program_fragment_to_action_config(then_block_node, choice_level + 1)
        else_actions_config = _parse_kx_hp_program_fragment_to_action_config(else_block_node, choice_level + 1) if else_block_node else SequenceActionConfig(actions=[])

        # Ensure then_actions_config and else_actions_config are SequenceActionConfig
        if not isinstance(then_actions_config, SequenceActionConfig):
            then_actions_config = SequenceActionConfig(actions=[then_actions_config])
        if not isinstance(else_actions_config, SequenceActionConfig):
            else_actions_config = SequenceActionConfig(actions=[else_actions_config])

        # Represent if/else as a choice between two guarded sequences
        return ChoiceActionConfig(
            level_name=f"IfElseChoice_{choice_level}",
            choices=[
                SequenceActionConfig(actions=[ActionConstraintConfig(formula=condition)] + then_actions_config.actions),
                SequenceActionConfig(actions=[ActionConstraintConfig(formula=f"!({condition})")] + else_actions_config.actions)
            ]
        )

    elif kx_node.node_type == "AssignmentStatement":
        # Handle both deterministic and non-deterministic assignments
        if kx_node.value.endswith(":= *"):
            var_name = kx_node.value.split(':=')[0].strip()
            return NondeterministicAssignmentActionConfig(variable=var_name)
        else:
            var_name, value = kx_node.value.split(':=', 1)
            return DeterministicAssignmentActionConfig(variable=var_name.strip(), value=value.strip())

    elif kx_node.node_type == "TestStatement":
        return ActionConstraintConfig(formula=kx_node.value)

    elif kx_node.node_type == "HPCall":
        return CallHybridProgramConfig(hp_name=kx_node.value)

    elif kx_node.node_type == "RawStatement":
        logging.warning(f"Converting RawStatement '{kx_node.value}' to a generic sequence action. This might indicate incomplete parsing.")
        # This is a fallback, ideally all statements should be parsed into specific types.
        # For a raw statement, we'll represent it as an assignment to a placeholder for now.
        return SequenceActionConfig(actions=[DeterministicAssignmentActionConfig(variable="raw_statement_placeholder", value=f"'{kx_node.value}'")])

    logging.warning(f"Unhandled or empty kx_node type '{kx_node.node_type}' in _parse_kx_hp_program_fragment_to_action_config. Returning empty SequenceActionConfig.")
    return SequenceActionConfig(actions=[]) # Ensure a return value even for unhandled cases