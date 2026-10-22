# keymaerax_parser.py
from __future__ import annotations
import re
import logging
from typing import Optional, Dict, Any, List, Tuple, Union

# Import AST node definitions from ast_config.py
from ast_config import ASTNode, Definition, HybridProgram, Variable, ProblemStatement

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

    @staticmethod
    def _find_matching_delimiter_span(text: str, start_idx: int, open_delim: str, close_delim: str) -> int | None:
        """
        Finds the index of the closing delimiter that matches the opening delimiter
        at start_idx.
        Returns the index of the matching closing delimiter or None if no match is found (unbalanced).
        """
        if text[start_idx] != open_delim:
            raise ValueError(f"start_idx must point to an opening delimiter '{open_delim}'")

        balance = 1
        for i in range(start_idx + 1, len(text)):
            if text[i] == open_delim:
                balance += 1
            elif text[i] == close_delim:
                balance -= 1

            if balance == 0:
                return i

        return None # Unbalanced delimiters

    def _find_top_level_operator(self, text: str, operator: str) -> Optional[int]:
        """
        Finds the index of the first occurrence of `operator` that is not
        enclosed within any `{}`, `[]`, or `()` braces/brackets.
        """
        balance_curly = 0
        balance_square = 0
        balance_round = 0
        op_len = len(operator)

        i = 0
        while i < len(text):
            if text[i] == '{':
                balance_curly += 1
            elif text[i] == '}':
                balance_curly -= 1
            elif text[i] == '[':
                balance_square += 1
            elif text[i] == ']':
                balance_square -= 1
            elif text[i] == '(':
                balance_round += 1
            elif text[i] == ')':
                balance_round -= 1

            # Check for operator match only if all balances are zero
            if (balance_curly == 0 and balance_square == 0 and balance_round == 0 and
                i + op_len <= len(text) and
                text[i:i+op_len] == operator):
                return i
            i += 1
        return None


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
            logging.debug(f"Theorem name found from Theorem: '{theorem_name}'")
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

                # Find the matching closing brace for the HP's main body
                closing_brace_of_hp_body_idx = self._find_matching_delimiter_span(
                    current_definitions_text_for_search,
                    open_brace_idx, '{', '}'
                )

                if closing_brace_of_hp_body_idx is not None:
                    # The content *inside* the outermost braces of the HP definition
                    full_hp_content_with_potential_invariant = current_definitions_text_for_search[open_brace_idx + 1 : closing_brace_of_hp_body_idx].strip()
                    
                    hp_program_body_content = full_hp_content_with_potential_invariant
                    loop_invariant_formula = None

                    # Check for '*@invariant(...)' pattern within the HP content (at the end)
                    invariant_pattern_in_body = re.compile(r'(?P<program_part>.*?)\s*\*\s*@invariant\((?P<invariant_formula>.*?)\)\s*$', re.DOTALL)
                    invariant_match_in_body = invariant_pattern_in_body.match(full_hp_content_with_potential_invariant)

                    if invariant_match_in_body:
                        hp_program_body_content = invariant_match_in_body.group('program_part').strip()
                        loop_invariant_formula = invariant_match_in_body.group('invariant_formula').strip()
                        logging.debug(f"Extracted program part '{hp_program_body_content[:50]}...' and invariant '{loop_invariant_formula[:50]}...' from HP '{hp_name}' body.")
                    
                    # Reconstruct the raw body string including its outermost braces
                    hp_body_raw = "{" + full_hp_content_with_potential_invariant + "}" 
                    
                    # The full HP block now extends to the closing brace of the HP body
                    full_hp_block_end_idx = closing_brace_of_hp_body_idx + 1
                    
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
                    
                    # Pass metadata to the HybridProgram AST node constructor
                    hp_node = HybridProgram(hp_name, hp_body_raw, metadata=metadata_dict) 
                    defs_node.add_child(hp_node)
                    logging.debug(f"Added HybridProgram ASTNode for '{hp_name}' to DefinitionsBlock.")

                    if loop_invariant_formula:
                        hp_node.add_child(ASTNode("Invariant", value=loop_invariant_formula))
                        # FIX: Store hp_program_body_content directly, it already has the outer braces if it's a loop body
                        # The value for LoopBodyRaw should be the program part itself, without extra braces added here.
                        hp_node.add_child(ASTNode("LoopBodyRaw", value=hp_program_body_content)) 
                        logging.debug(f"Added Invariant and LoopBodyRaw children to HP '{hp_name}'.")

                    # Parse HP body structure if specific functions are defined
                    # Use hp_program_body_content for parsing, not the raw one that might contain invariant
                    body_content_for_parsing = "{" + hp_program_body_content + "}" # Wrap in braces for consistency with _parse_program_fragment
                    if hp_name == 'controlFunc':
                        self._parse_program_fragment(body_content_for_parsing, hp_node)
                        logging.debug(f"Parsed controlFunc structure for '{hp_name}'.")
                    elif hp_name == 'stepFunc':
                        self._parse_step_func_structure(body_content_for_parsing, hp_node)
                    
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
        # Corrected regex: (\w+) for key, and four optional capturing groups for values:
        # (?:(\{.*?\}))? -> curly braces
        # (?:([.*?]))? -> square brackets (FIXED: escaped square brackets)
        # (?:(\".*?\"))? -> quoted string
        # (?:([^,]+))? -> anything not a comma (for single-line values)
        
        attr_pairs_raw = re.findall(
            r'(\w+)\s*=\s*'
            r'(?:(\{.*?\})|'   # Group 1: curly braces
            r'([.*?])|'   # Group 2: square brackets (FIXED: escaped square brackets)
            r'(\".*?\")|'   # Group 3: quoted string
            r'([^,]+))',    # Group 4: anything else (non-comma)
            raw_metadata,
            re.DOTALL # Keep DOTALL to allow multiline expressions if they ever occur within {} or ""
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

        parts = [p.strip() for p in self.split_by_operator_outside_braces(clean_body, ',') if p.strip()]
        for part in parts:
            sub_parts = self.split_by_operator_outside_braces(part, '&')
            if not sub_parts: continue

            ode_or_formula = sub_parts[0]
            if "'" in ode_or_formula:
                ode_node.add_child(ASTNode("ODE", value=ode_or_formula))
            else:
                domain_node.add_child(ASTNode("Constraint", value=ode_or_formula))

            for domain_constraint in sub_parts[1:]:
                domain_node.add_child(ASTNode("Constraint", value=domain_constraint))

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


    def _parse_program_fragment(self, program_fragment_string: str, parent_node: ASTNode):
        fragment = program_fragment_string.strip()
        if not fragment:
            return

        # 1. Peel off ALL outer braces until we find the actual content.
        # This prevents {{...}} from confusing the splitter.
        while fragment.startswith('{') and fragment.endswith('}'):
            # Check if these braces actually match each other
            if self._find_matching_delimiter_span(fragment, 0, '{', '}') == len(fragment) - 1:
                fragment = fragment[1:-1].strip()
            else:
                break # Braces are like {a};{b}, don't peel.

        if not fragment:
            return

        # 2. Check for Choice Operator (++)
        # Use the helper to split, ensuring we only split at the TOP level of this fragment
        choice_parts = self.split_by_operator_outside_braces(fragment, '++')
        
        if len(choice_parts) > 1:
            choice_node = ASTNode("ChoiceOperator (++)")
            parent_node.add_child(choice_node)
            for i, part in enumerate(choice_parts):
                child_choice_node = ASTNode(f"Choice {i+1}")
                choice_node.add_child(child_choice_node)
                # Recurse into the part
                self._parse_program_fragment(part, child_choice_node)
            return

        # 3. Check for Sequence Operator (;)
        statements = self.split_by_operator_outside_braces(fragment, ';')
        
        # If we have multiple statements OR a single statement that ends in a semicolon
        if len(statements) > 1 or (len(statements) == 1 and fragment.endswith(';')):
            for stmt_value in statements:
                if stmt_value.strip():
                    self._parse_program_fragment(stmt_value, parent_node)
            return

        # 4. Base Case: It's a single Action/Test
        if fragment.startswith('?'):
            parent_node.add_child(ASTNode("Test", value=fragment[1:].strip()))
        else:
            # Check for assignment or HP call
            parent_node.add_child(ASTNode("Statement", value=fragment))


    def _parse_problem(self):
        logging.debug("Starting _parse_problem.")
        problem_block = self._extract_block("Problem")
        if not problem_block:
            logging.debug("No Problem block found. Skipping problem parsing.")
            return

        problem_node = ProblemStatement(problem_block) 
        self.ast_root.add_child(problem_node)
        logging.debug("Created ProblemStatement ASTNode.")

        # Find the top-level '->'
        arrow_idx = self._find_top_level_operator(problem_block, '->')

        if arrow_idx is None:
            problem_node.add_child(ASTNode("Error", "Could not find top-level '->' in Problem block."))
            logging.error("Failed to find top-level '->' in Problem block.")
            return

        precondition_raw = problem_block[:arrow_idx].strip()
        rhs_raw = problem_block[arrow_idx + 2:].strip() # +2 for '->'

        problem_node.add_child(ASTNode("Precondition (P)", precondition_raw))
        logging.debug(f"Precondition: '{precondition_raw[:50]}...'")

        # Now parse the RHS which can contain multiple modal formulas connected by '&'
        clauses = []
        current_rhs_segment = rhs_raw
        while True:
            ampersand_idx = self._find_top_level_operator(current_rhs_segment, '&')
            # FIX: Changed '===' to 'is None'
            if ampersand_idx is None: 
                if current_rhs_segment.strip():
                    clauses.append(current_rhs_segment.strip())
                break
            
            clause_part = current_rhs_segment[:ampersand_idx].strip()
            if clause_part:
                clauses.append(clause_part)
            current_rhs_segment = current_rhs_segment[ampersand_idx + 1:].strip()
        
        rhs_clauses = clauses


        for clause in rhs_clauses:
            clause = clause.strip()
            if not clause:
                continue

            # Check for Box Modality: [HP]Q
            box_match = re.match(r'\[\s*(?P<hp_name>.*?)\s*\]\s*(?P<formula>.*)', clause, re.DOTALL)
            # Check for Diamond Modality: <HP>Q
            diamond_match = re.match(r'<\s*(?P<hp_name>.*?)\s*>\s*(?P<formula>.*)', clause, re.DOTALL)

            if box_match:
                hp_name = box_match.group('hp_name').strip()
                formula = box_match.group('formula').strip()
                # Remove trailing semicolon if present in HP name
                if hp_name.endswith(';'):
                    hp_name = hp_name[:-1].strip()
                # Remove outer parentheses if present in formula
                if formula.startswith('(') and formula.endswith(')'):
                    # Ensure these are actual matching outer parentheses
                    if self._find_matching_delimiter_span(formula, 0, '(', ')') == len(formula) - 1:
                        formula = formula[1:-1].strip()
                problem_node.add_child(ASTNode("BoxModality", metadata={'hp': hp_name, 'formula': formula}))
                logging.debug(f"Parsed Box Modality: HP='{hp_name}', Formula='{formula[:50]}...'")
            elif diamond_match:
                hp_name = diamond_match.group('hp_name').strip()
                formula = diamond_match.group('formula').strip()
                # Remove trailing semicolon if present in HP name
                if hp_name.endswith(';'):
                    hp_name = hp_name[:-1].strip()
                # Remove outer parentheses if present in formula
                if formula.startswith('(') and formula.endswith(')'):
                    if self._find_matching_delimiter_span(formula, 0, '(', ')') == len(formula) - 1:
                        formula = formula[1:-1].strip()
                problem_node.add_child(ASTNode("DiamondModality", metadata={'hp': hp_name, 'formula': formula}))
                logging.debug(f"Parsed Diamond Modality: HP='{hp_name}', Formula='{formula[:50]}...'")
            else:
                # If it's not a modal formula, treat it as a simple postcondition
                if clause.startswith('(') and clause.endswith(')'):
                    if self._find_matching_delimiter_span(clause, 0, '(', ')') == len(clause) - 1:
                        clause = clause[1:-1].strip()
                problem_node.add_child(ASTNode("Postcondition (Q)", clause))
                logging.debug(f"Parsed simple Postcondition: '{clause[:50]}...'")
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
        # The replacement logic here correctly handles the string from ArchiveEntry
        env_name = theorem_name_node.value.replace(' ', '_').replace(':', '_').replace('"', '').replace('/', '_').replace('.', '_')
        logging.debug(f"Set environment name to '{env_name}' from theorem name.")

    # Initialize sub-configs first
    definitions_config = DefinitionsConfig()
    variables_config = VariablesConfig()
    # ProblemConfig now takes a list of postconditions
    problem_config = ProblemConfig(
        precondition=PreconditionConfig(formula=""),
        postconditions=[] # Initialize as empty list
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
            # Access metadata directly from the ASTNode instance
            metadata = def_node_kx.metadata 
            
            if def_node_kx.node_type == "Definition" and hasattr(def_node_kx, 'definition_type'):
                parameters = []
                params_node = next((c for c in def_node_kx.children if c.node_type == "Parameters"), None)
                if params_node and params_node.value:
                    param_strings = [p.strip() for p in KeYmaeraXParser.split_by_operator_outside_braces(params_node.value, ',')]
                    for p_str in param_strings:
                        parts = p_str.split(' ')
                        if len(parts) == 2:
                            parameters.append((parts[0], parts[1]))
                        else:
                            parameters.append(('Real', p_str)) # Default to Real if type not specified
                    logging.debug(f"Extracted parameters for '{def_node_kx.name}': {parameters}")

                if def_node_kx.definition_type == "HP":
                    logging.debug(f"Processing HP definition: {def_node_kx.name}")
                    hp_name = def_node_kx.name 
                    hp_body = def_node_kx.body # This is the raw body string { ... }
                    
                    if hp_name == "controlFunc":
                        control_func_root_action = next((c for c in def_node_kx.children if c.node_type == "ChoiceOperator (++)"), None)
                        action_space_config = None
                        if control_func_root_action:
                            action_space_config = _parse_kx_hp_program_fragment_to_action_config(
                                control_func_root_action, 0
                            )
                        control_func_config = ControlFuncConfig(name=hp_name, action_space_config=action_space_config, metadata=metadata, raw_body_snippet=hp_body)
                        definitions_config.hybrid_programs.append(control_func_config)
                        logging.debug("Appended ControlFuncConfig to hybrid_programs.")
                    elif hp_name == "stepFunc":
                        odes = []
                        evolution_domain_constraints = []
                        duration_expr: Optional[str] = None 

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
                        
                        evolution_domain = EvolutionDomainConfig(constraints=evolution_domain_constraints) if evolution_domain_constraints else None
                        step_func_config = StepFuncConfig(name=hp_name, odes=odes, evolution_domain=evolution_domain, duration=duration_expr, raw_body_snippet=hp_body, metadata=metadata)
                        definitions_config.hybrid_programs.append(step_func_config)
                        logging.debug("Appended StepFuncConfig to hybrid_programs.")
                    elif hp_name == "dwwpdl": # Handle the dwwpdl HP with loop structure
                        loop_invariant_node = next((c for c in def_node_kx.children if c.node_type == "Invariant"), None)
                        loop_body_raw_node = next((c for c in def_node_kx.children if c.node_type == "LoopBodyRaw"), None)
                        logging.debug(f"For dwwpdl: loop_invariant_node={loop_invariant_node is not None}, loop_body_raw_node={loop_body_raw_node is not None}")

                        if loop_invariant_node and loop_body_raw_node:
                            loop_execution_config = LoopExecutionConfig(
                                loop_invariant_formula=loop_invariant_node.value,
                                iterations="inf" # Assuming infinite iterations for now
                            )
                            
                            # The raw body for dwwpdl is {{ controlFunc; t := 0; } { stepFunc; }}
                            full_loop_body_content = loop_body_raw_node.value
                            # Remove outermost braces if present
                            if full_loop_body_content.startswith('{') and full_loop_body_content.endswith('}'):
                                # Use the generalized delimiter span to ensure correct peeling
                                if KeYmaeraXParser._find_matching_delimiter_span(full_loop_body_content, 0, '{', '}') == len(full_loop_body_content) - 1:
                                    full_loop_body_content = full_loop_body_content[1:-1].strip()

                            # Parse the discrete and continuous parts
                            # They are separated by a space at the top level, but enclosed in their own braces.
                            # We need to find the first '{' and its matching '}', then the next '{' and its matching '}'
                            
                            discrete_part_start_idx = full_loop_body_content.find('{')
                            if discrete_part_start_idx != -1:
                                discrete_part_end_idx = KeYmaeraXParser._find_matching_delimiter_span(full_loop_body_content, discrete_part_start_idx, '{', '}')
                                if discrete_part_end_idx is not None:
                                    discrete_block_content = full_loop_body_content[discrete_part_start_idx + 1 : discrete_part_end_idx].strip()
                                    
                                    # Parse discrete actions
                                    discrete_sequence_config = SequenceActionConfig()
                                    discrete_statements = [s.strip() for s in KeYmaeraXParser.split_by_operator_outside_braces(discrete_block_content, ';') if s.strip()]
                                    for stmt in discrete_statements:
                                        if stmt == "controlFunc":
                                            discrete_sequence_config.actions.append(CallHybridProgramConfig(hp_name="controlFunc"))
                                        elif ":=" in stmt:
                                            var, val = stmt.split(':=', 1)
                                            discrete_sequence_config.actions.append(DeterministicAssignmentActionConfig(variable=var.strip(), value=val.strip()))
                                        else:
                                            logging.warning(f"Unhandled statement in dwwpdl discrete loop body: '{stmt}'")
                                    loop_execution_config.loop_body.actions.append(discrete_sequence_config)

                                    # Find and parse continuous part
                                    continuous_part_search_start = discrete_part_end_idx + 1
                                    continuous_part_start_idx = full_loop_body_content.find('{', continuous_part_search_start)
                                    if continuous_part_start_idx != -1:
                                        continuous_part_end_idx = KeYmaeraXParser._find_matching_delimiter_span(full_loop_body_content, continuous_part_start_idx, '{', '}')
                                        if continuous_part_end_idx is not None:
                                            continuous_block_content = full_loop_body_content[continuous_part_start_idx + 1 : continuous_part_end_idx].strip()
                                            
                                            # Assuming continuous part is just "stepFunc;"
                                            if continuous_block_content.startswith("stepFunc"): # Allow for optional trailing ';'
                                                # The stepFunc in the problem example doesn't have an invariant, but if it did,
                                                # it would be parsed here. For now, just call stepFunc.
                                                loop_execution_config.loop_body.actions.append(HybridProgramStepConfig(hp_name="stepFunc"))
                                            else:
                                                logging.warning(f"Unhandled continuous loop body content in dwwpdl: '{continuous_block_content}'")
                                        else:
                                            logging.warning(f"Unmatched braces in continuous part of dwwpdl loop body: '{full_loop_body_content}'")
                                    else:
                                        logging.warning(f"Could not find continuous part in dwwpdl loop body: '{full_loop_body_content}'")
                                else:
                                    logging.warning(f"Unmatched braces in discrete part of dwwpdl loop body: '{full_loop_body_content}'")
                            else:
                                logging.warning(f"Could not find discrete part in dwwpdl loop body: '{full_loop_body_content}'")

                            dwwpdl_hp_config = HybridProgramConfig(name=hp_name, raw_body_snippet=hp_body, metadata=metadata, loop_execution_config=loop_execution_config)
                            definitions_config.hybrid_programs.append(dwwpdl_hp_config)
                            logging.debug("Appended dwwpdl HybridProgramConfig with LoopExecutionConfig.")
                        else:
                            logging.warning(f"dwwpdl HP found but missing Invariant or LoopBodyRaw children. Appending without LoopExecutionConfig.")
                            definitions_config.hybrid_programs.append(HybridProgramConfig(name=hp_name, raw_body_snippet=hp_body, metadata=metadata))

                    else: # Generic HP
                        definitions_config.hybrid_programs.append(HybridProgramConfig(name=hp_name, raw_body_snippet=hp_body, metadata=metadata))
                        logging.debug(f"Appended generic HybridProgramConfig for '{hp_name}'.")

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
                        lower_bound = float(bound_match.group(1)) # Convert to float
                        upper_bound = float(bound_match.group(3)) # Convert to float
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

        # Iterate through all children of ProblemStatement to find postconditions
        for child_kx in problem_block_kx.children:
            if child_kx.node_type == "BoxModality":
                hp_name = child_kx.metadata.get('hp')
                formula = child_kx.metadata.get('formula')
                if hp_name and formula:
                    problem_config.postconditions.append(PostconditionConfig(
                        formula=formula,
                        modality="Box",
                        hp_name=hp_name
                    ))
                    logging.debug(f"Added Box Postcondition: HP='{hp_name}', Formula='{formula[:50]}...'")
            elif child_kx.node_type == "DiamondModality":
                hp_name = child_kx.metadata.get('hp')
                formula = child_kx.metadata.get('formula')
                if hp_name and formula:
                    problem_config.postconditions.append(PostconditionConfig(
                        formula=formula,
                        modality="Diamond",
                        hp_name=hp_name
                    ))
                    logging.debug(f"Added Diamond Postcondition: HP='{hp_name}', Formula='{formula[:50]}...'")
            elif child_kx.node_type == "Postcondition (Q)":
                if child_kx.value:
                    problem_config.postconditions.append(PostconditionConfig(
                        formula=child_kx.value,
                        modality=None, # No modality for simple postcondition
                        hp_name=None
                    ))
                    logging.debug(f"Added simple Postcondition: '{child_kx.value[:50]}...'")
    else:
        logging.debug("No ProblemStatement found in KeYmaeraX AST for conversion.")

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
                # Handle nested ChoiceOperator directly without wrapping in SequenceAction
                if len(choice_kx_branch.children) == 1 and choice_kx_branch.children[0].node_type == "ChoiceOperator (++)":
                    parsed_nested_choice = _parse_kx_hp_program_fragment_to_action_config(
                        choice_kx_branch.children[0], choice_level + 1
                    )
                    choice_block.choices.append(parsed_nested_choice)
                else:
                    choice_content_sequence = SequenceActionConfig()
                    logging.debug(f"Processing choice branch: {choice_kx_branch.node_type}")
                    for sub_node_kx in choice_kx_branch.children:
                        parsed_sub_block = _parse_kx_hp_program_fragment_to_action_config(sub_node_kx, choice_level + 1)
                        # Corrected: If parsed_sub_block is a SequenceActionConfig, extend the current sequence
                        # Otherwise, append it as a single action.
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
            if isinstance(parsed_child_block, SequenceActionConfig):
                sequence_block.actions.extend(parsed_child_block.actions)
            else:
                sequence_block.actions.append(parsed_child_block)
        return sequence_block

    logging.debug(f"Unhandled or empty kx_node type '{kx_node.node_type}'. Returning empty SequenceActionConfig.")
    return SequenceActionConfig() # Ensure a return value even for unhandled cases