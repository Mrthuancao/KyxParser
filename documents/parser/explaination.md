You've provided a comprehensive set of Python code for parsing KeYmaera X specifications and converting them into an intermediate Python AST (PyAST) representation, specifically designed for configuring OpenAI Gym environments.

Let's break down the two functions you asked about, `convert_keymaerax_ast_to_intermediate_pyast` and `_parse_kx_hp_program_fragment_to_action_config`, within the context of the full parsing pipeline.

### Overall Pipeline

The process can be visualized as follows:

1.  **KeYmaera X Source Code:** You write your hybrid system specification in KeYmaera X syntax.
2.  **`KeYmaeraXParser`:** This class (which you also provided) takes the raw KeYmaera X source code and parses it into a structured **KeYmaera X AST (Abstract Syntax Tree)**, represented by `ASTNode` objects. This AST is a direct, hierarchical representation of the source code's structure.
3.  **`convert_keymaerax_ast_to_intermediate_pyast`:** This function takes the KeYmaera X AST (the output of `KeYmaeraXParser`) and transforms it into a more abstract, Python-native data structure called `GymEnvironmentConfig`. This `GymEnvironmentConfig` is the **Intermediate PyAST**, tailored for generating an RL environment.
4.  **`_parse_kx_hp_program_fragment_to_action_config`:** This is a helper function, primarily called by `convert_keymaerax_ast_to_intermediate_pyast`, to recursively process complex hybrid program fragments (like `controlFunc` bodies) into specific `ActionType` objects within the `GymEnvironmentConfig`.

---

### Prerequisites: Data Structures

Before explaining the functions, let's ensure we have the necessary data structure definitions (from your `ast_config.py` and `keymaerax_parser.py`) for a complete understanding.

```python
import re
import logging # Import the logging module
from typing import List, Optional, Union, Tuple, Any

# Configure logging for better output visibility
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s') # Set to DEBUG for detailed output

# --- KeYmaera X AST Node (Input to conversion functions) ---
class ASTNode:
    """Base class for all nodes in the Abstract Syntax Tree."""
    def __init__(self, node_type, value=None, children=None, **kwargs):
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []
        for k, v in kwargs.items(): # For dynamic attributes like 'name', 'definition_type', 'var_type'
            setattr(self, k, v)

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self, level=0):
        """Generates a readable, indented representation of the tree."""
        indent = "  " * level
        value_str = ""
        if self.value is not None:
            value_str = f" '{self.value}'"
        
        # Add specific attributes for better debugging/representation
        attrs = []
        if hasattr(self, 'name'): attrs.append(f"name='{self.name}'")
        if hasattr(self, 'definition_type'): attrs.append(f"def_type='{self.definition_type}'")
        if hasattr(self, 'var_type'): attrs.append(f"var_type='{self.var_type}'")
        attr_str = ", ".join(attrs)
        
        if attr_str:
            s = f"{indent}[{self.node_type}]{value_str} ({attr_str})\n"
        else:
            s = f"{indent}[{self.node_type}]{value_str}\n"

        for child in self.children:
            s += child.__repr__(level + 1)
        return s

class Definition(ASTNode):
    """Represents a constant, function, or predicate definition."""
    def __init__(self, name, definition_type, value=None, comment=None):
        super().__init__("Definition", name) # node_type "Definition", value is the name
        self.name = name
        self.definition_type = definition_type
        self.definition_value = value # The actual formula/value
        self.comment = comment

    def __repr__(self, level=0):
        indent = "  " * level
        value_str = ""
        if self.definition_value is not None:
            cleaned_value = ' '.join(self.definition_value.split())
            value_str = f" = {cleaned_value[:60]}..." if len(cleaned_value) > 60 else f" = {cleaned_value}"
        comment_str = f" /* {self.comment} */" if self.comment else ""
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
        super().__init__(name, "HP", value=body) # node_type "Definition", name, def_type "HP", value is raw body
        self.name = name # Store name for specific parsing logic
        self.body = body
        self._parser_ref = parser_ref # Store reference to the parser instance

    # This method is called by the parser to populate children
    def _parse_hp_structure(self):
        """
        Parses the body of a Hybrid Program structure based on its name.
        This method is part of the ASTNode class.
        """
        logging.debug(f"HybridProgram '{self.name}' calling _parse_hp_structure.")
        body_content = self.body.strip()

        if self.name == 'controlFunc':
            if self._parser_ref: # Call the parser's method
                # This is where _parse_program_fragment is called to populate children
                self._parser_ref._parse_program_fragment(body_content, self)
            else:
                logging.warning(f"No parser reference available for HybridProgram '{self.name}'. Cannot parse program fragment.")

        elif self.name == 'stepFunc':
            logging.debug(f"Parsing stepFunc structure for '{self.name}'. Raw body: {body_content[:100]}...")
            ode_node = ASTNode("OrdinaryDifferentialEquations")
            domain_node = ASTNode("EvolutionDomain")

            clean_body_no_braces = body_content
            if clean_body_no_braces.startswith('{') and clean_body_no_braces.endswith('}'):
                clean_body_no_braces = clean_body_no_braces[1:-1].strip()
            clean_body = re.sub(r'\s+', ' ', clean_body_no_braces).strip()
            logging.debug(f"stepFunc body after cleaning: {clean_body[:100]}...")

            split_by_operator = KeYmaeraXParser.split_by_operator_outside_braces # Direct call to static method
            parts = [p.strip() for p in split_by_operator(clean_body, ',') if p.strip()]
            logging.debug(f"stepFunc parts after splitting by comma: {parts}")

            for part in parts:
                if not part: continue

                sub_parts = split_by_operator(part, '&')
                logging.debug(f"stepFunc sub-parts for '{part}' after splitting by '&': {sub_parts}")

                if not sub_parts:
                    continue

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
        cleaned_body = ' '.join(self.body.split())
        if self.name == 'controlFunc':
            body_display = cleaned_body[:80] + "..."
        elif self.name == 'stepFunc':
            idx = cleaned_body.find("t' = 1")
            if idx != -1:
                body_display = cleaned_body[:idx + len("t' = 1")] + " ..."
            else:
                body_display = cleaned_body[:80] + "..." if len(cleaned_body) > 80 else cleaned_body
        else:
            body_display = cleaned_body[:80] + "..." if len(cleaned_body) > 80 else cleaned_body

        s = f"{indent}[HP]: {self.name}\n"
        s += f"{indent}  Body (raw): {body_display}\n"
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

# --- Intermediate Python AST (Output of conversion functions) ---
# Base class for actions within hybrid programs
class ActionType: pass

class NondeterministicAssignmentActionConfig(ActionType):
    def __init__(self, variable: str): self.variable = variable
    def __repr__(self): return f"NondetAssign(var='{self.variable}')"

class DeterministicAssignmentActionConfig(ActionType):
    def __init__(self, variable: str, value: str): self.variable = variable; self.value = value
    def __repr__(self): return f"DetAssign(var='{self.variable}', val='{self.value}')"

class ActionConstraintConfig(ActionType):
    def __init__(self, formula: str): self.formula = formula
    def __repr__(self): return f"Constraint(formula='{self.formula}')"

class SequenceActionConfig(ActionType):
    def __init__(self, actions: Optional[List[ActionType]] = None): self.actions = actions if actions is not None else []
    def __repr__(self): return f"Sequence({self.actions})"

class ChoiceActionConfig(ActionType):
    def __init__(self, level_name: str, choices: Optional[List[SequenceActionConfig]] = None): self.level_name = level_name; self.choices = choices if choices is not None else []
    def __repr__(self): return f"Choice(name='{self.level_name}', choices={self.choices})"

class ControlFuncConfig:
    def __init__(self, name: str, action_space_config: Optional[ActionType] = None): self.name = name; self.action_space_config = action_space_config
    def __repr__(self): return f"ControlFunc(name='{self.name}', actions={self.action_space_config})"

class ODEConfig:
    def __init__(self, variable: str, equation: str): self.variable = variable; self.equation = equation
    def __repr__(self): return f"ODE(var='{self.variable}', eq='{self.equation}')"

class EvolutionDomainConfig:
    def __init__(self, constraints: List[str]): self.constraints = constraints
    def __repr__(self): return f"EvolutionDomain(constraints={self.constraints})"

class StepFuncConfig:
    def __init__(self, name: str, odes: List[ODEConfig], evolution_domain: Optional[EvolutionDomainConfig], duration: Optional[str]):
        self.name = name; self.odes = odes; self.evolution_domain = evolution_domain; self.duration = duration
    def __repr__(self): return f"StepFunc(name='{self.name}', odes={len(self.odes)}, domain={self.evolution_domain}, duration={self.duration})"

class FunctionDefinitionConfig:
    def __init__(self, name: str, return_type: str, parameters: List[Tuple[str, str]], body: str):
        self.name = name; self.return_type = return_type; self.parameters = parameters; self.body = body
    def __repr__(self): return f"FuncDef(name='{self.name}', type='{self.return_type}', params={self.parameters}, body='{self.body}')"

class ConstantDefinitionConfig:
    def __init__(self, name: str, type: str, value: str): self.name = name; self.type = type; self.value = value
    def __repr__(self): return f"ConstDef(name='{self.name}', type='{self.type}', value='{self.value}')"

class PredicateDefinitionConfig:
    def __init__(self, name: str, parameters: List[Tuple[str, str]], body: str):
        self.name = name; self.parameters = parameters; self.body = body
    def __repr__(self): return f"PredDef(name='{self.name}', params={self.parameters}, body='{self.body}')"

class DefinitionsConfig:
    def __init__(self):
        self.hybrid_programs: List[Union[ControlFuncConfig, StepFuncConfig]] = []
        self.functions: List[FunctionDefinitionConfig] = []
        self.constants: List[ConstantDefinitionConfig] = []
        self.predicates: List[PredicateDefinitionConfig] = []
    def __repr__(self): return f"Defs(hps={self.hybrid_programs}, funcs={self.functions}, consts={self.constants}, preds={self.predicates})"

class VariableDeclarationConfig:
    def __init__(self, name: str, type: str): self.name = name; self.type = type
    def __repr__(self): return f"VarDecl(name='{self.name}', type='{self.type}')"

class VariablesConfig:
    def __init__(self): self.state_variables: List[VariableDeclarationConfig] = []
    def __repr__(self): return f"Vars(state_vars={self.state_variables})"

class PreconditionConfig:
    def __init__(self, formula: str): self.formula = formula
    def __repr__(self): return f"Precond(formula='{self.formula}')"

class CallHybridProgramConfig(ActionType):
    def __init__(self, hp_name: str): self.hp_name = hp_name
    def __repr__(self): return f"CallHP(name='{self.hp_name}')"

class HybridProgramStepConfig(ActionType):
    def __init__(self, hp_name: str, invariant_formula: Optional[str]): self.hp_name = hp_name; self.invariant_formula = invariant_formula
    def __repr__(self): return f"HPStep(name='{self.hp_name}', inv='{self.invariant_formula}')"

class LoopBodyConfig:
    def __init__(self, actions: Optional[List[ActionType]] = None): self.actions = actions if actions is not None else []
    def __repr__(self): return f"LoopBody(actions={self.actions})"

class LoopExecutionConfig:
    def __init__(self, loop_invariant_formula: str = "", iterations: str = "inf"):
        self.loop_invariant_formula = loop_invariant_formula
        self.iterations = iterations
        self.loop_body: LoopBodyConfig = LoopBodyConfig()
    def __repr__(self): return f"LoopExec(inv='{self.loop_invariant_formula}', iter='{self.iterations}', body={self.loop_body})"

class PostconditionConfig:
    def __init__(self, formula: str): self.formula = formula
    def __repr__(self): return f"Postcond(formula='{self.formula}')"

class ProblemConfig:
    def __init__(self, precondition: PreconditionConfig, loop_execution: LoopExecutionConfig, postcondition: PostconditionConfig):
        self.precondition = precondition; self.loop_execution = loop_execution; self.postcondition = postcondition
    def __repr__(self): return f"Problem(pre={self.precondition}, loop={self.loop_execution}, post={self.postcondition})"

class GymEnvironmentConfig:
    def __init__(self, env_name: str, definitions: DefinitionsConfig, variables: VariablesConfig, problem: ProblemConfig):
        self.env_name = env_name; self.definitions = definitions; self.variables = variables; self.problem = problem
    def __repr__(self): return f"GymEnv(name='{self.env_name}', defs={self.definitions}, vars={self.variables}, prob={self.problem})"

# --- KeYmaeraXParser (for generating the ASTNode input) ---
class KeYmaeraXParser:
    def __init__(self, source_code):
        self.source_code = source_code
        self.ast_root = ASTNode("KeYmaeraXSpecification")
        self._cleanup_source()
        logging.debug("KeYmaeraXParser initialized. Source code cleaned.")

    def _cleanup_source(self):
        self.source_code = re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m.group(0)), self.source_code, flags=re.DOTALL)
        self.source_code = re.sub(r'//.*', '', self.source_code)
        self.source_code = re.sub(r'\n\s*\n', '\n', self.source_code)
        self.source_code = self.source_code.strip()
        logging.debug("Source code cleanup complete.")

    @staticmethod
    def split_by_operator_outside_braces(text, operator):
        parts = []
        brace_level = 0
        current_part_start = 0
        i = 0
        while i < len(text):
            if text[i] == '{':
                brace_level += 1
            elif text[i] == '}':
                brace_level -= 1
            if (i + len(operator) <= len(text) and
                brace_level == 0 and
                text[i:i+len(operator)] == operator):
                part = text[current_part_start:i].strip()
                if part: parts.append(part)
                current_part_start = i + len(operator)
                i += len(operator) - 1
            i += 1
        last_part = text[current_part_start:].strip()
        if last_part: parts.append(last_part)
        return parts

    def _find_matching_brace_span(self, text: str, start_brace_idx: int) -> int | None:
        if text[start_brace_idx] != '{':
            raise ValueError("start_brace_idx must point to an opening brace '{'")
        balance = 1
        for i in range(start_brace_idx + 1, len(text)):
            if text[i] == '{': balance += 1
            elif text[i] == '}': balance -= 1
            if balance == 0: return i
        return None

    def _extract_top_level_brace_blocks(self, text: str) -> list[str | list]:
        result = []
        balance = 0
        current_text_segment_start = 0
        block_start_index = -1
        for i, char in enumerate(text):
            if char == '{':
                if balance == 0:
                    if i > current_text_segment_start:
                        segment = text[current_text_segment_start:i].strip()
                        if segment: result.append(segment)
                    block_start_index = i + 1
                balance += 1
            elif char == '}':
                balance -= 1
                if balance == 0:
                    if block_start_index != -1:
                        block_content = text[block_start_index:i]
                        result.append(self._extract_top_level_brace_blocks(block_content))
                        block_start_index = -1
                    current_text_segment_start = i + 1
                elif balance < 0:
                    logging.warning(f"Malformed brace structure (unmatched '}}') at index {i} in '{text}'")
                    balance = 0
                    current_text_segment_start = i + 1
        if current_text_segment_start < len(text):
            segment = text[current_text_segment_start:len(text)].strip()
            if segment: result.append(segment)
        if balance > 0:
            logging.warning(f"Unclosed brace block detected in text: '{text}'")
            if block_start_index != -1 and block_start_index < len(text):
                remaining_content = text[block_start_index:].strip()
                if remaining_content: result.append(remaining_content)
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

        mutable_definitions_block = list(definitions_block)

        HP_HEADER_PATTERN = re.compile(
            r'HP\s+(\w+)\s*::=\s*\{',
            re.DOTALL
        )
        logging.debug("Starting HP extraction.")
        current_search_pos = 0
        while current_search_pos < len(mutable_definitions_block):
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
                    hp_node._parse_hp_structure() # This populates the HP's children
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

        parts = self.split_by_operator_outside_braces(remaining_text_after_hps, ';')
        logging.debug(f"Simple definitions parts after splitting by ';': {parts}")

        for part in parts:
            line = part.strip()
            if not line: continue

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
            if not line: continue

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
            r'([\s\S]+?)\s*->\s*\[([\s\S]+?)\]\s*([\s\S]+)',
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
                r"(\{[\s\S]*?\})\*@invariant\(([\s\S]*?)\)",
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
                if continuous_end_idx == None:
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

```

---

### 1. `convert_keymaerax_ast_to_intermediate_pyast`

**Purpose:**
This is the primary function responsible for translating the KeYmaera X AST (generated by `KeYmaeraXParser`) into a `GymEnvironmentConfig` object. This `GymEnvironmentConfig` is a structured Python object that encapsulates all the necessary information to define an OpenAI Gym-compatible reinforcement learning environment, abstracting away the KeYmaera X specific syntax.

**Input:**
*   `kx_ast_root: ASTNode`: The root node of the KeYmaera X AST. This node must be of type `"KeYmaeraXSpecification"`. This is the output of `KeYmaeraXParser.parse()`.

**Output:**
*   `GymEnvironmentConfig`: A comprehensive configuration object that describes the hybrid system's definitions (hybrid programs, functions, constants, predicates), state variables, and the problem statement (precondition, loop execution, postcondition).

**Step-by-step Explanation:**

1.  **Validation and Initialization:**
    *   It first validates that `kx_ast_root` is not `None` and its `node_type` is `"KeYmaeraXSpecification"`. If not, it raises a `ValueError`.
    *   It extracts the environment name (`env_name`) from the `TheoremName` child node of the root AST, if available. It cleans up the name (replaces spaces, colons, etc., with underscores). If no `TheoremName` is found, it defaults to `"KeYmaeraX_RL_Environment"`.
    *   It initializes empty `DefinitionsConfig`, `VariablesConfig`, and `ProblemConfig` objects. These objects will be populated as the function traverses the KeYmaera X AST.
    *   Finally, it creates the main `GymEnvironmentConfig` object, passing these initialized sub-configurations and the `env_name`.

2.  **Processing `DefinitionsBlock`:**
    *   It searches for a `DefinitionsBlock` child within `kx_ast_root`.
    *   If found, it iterates through each child of this `DefinitionsBlock`, which represents individual definitions (Hybrid Programs, Real constants/functions, Bool predicates).
    *   **Hybrid Programs (`HP`):**
        *   If `def_node_kx.name` is `"controlFunc"`: This is crucial for defining the agent's actions. It looks for a `ChoiceOperator (++)` node within `controlFunc`'s children. It then calls the helper function `_parse_kx_hp_program_fragment_to_action_config` (explained next) to recursively convert this complex program fragment into an `ActionType` object (typically a `ChoiceActionConfig`). This result is wrapped in a `ControlFuncConfig` and added to `definitions_config.hybrid_programs`.
        *   If `def_node_kx.name` is `"stepFunc"`: This defines the continuous dynamics. It parses:
            *   `OrdinaryDifferentialEquations`: Extracts each `ODE` child node, splitting its `value` (e.g., `"x' = v * dx"`) into a variable (`x`) and an equation (`v * dx`). These are stored as `ODEConfig` objects.
            *   `EvolutionDomain`: Extracts each `Constraint` child node (e.g., `"t <= ep"`) into a list of strings. These form an `EvolutionDomainConfig`.
            *   A `StepFuncConfig` is created with these ODEs and domain constraints, and added to `definitions_config.hybrid_programs`. The `duration` is left as `None` as it's often determined contextually (e.g., from a global constant like `ep` or the loop execution).
        *   Other HPs are currently logged as unrecognized and skipped.
    *   **Real Definitions (`Real`):**
        *   It checks if the definition node has `Parameters` (e.g., `stopDist(v)`). If so, it's treated as a `FunctionDefinitionConfig`.
        *   Otherwise (e.g., `Real ep`), it's a `ConstantDefinitionConfig`.
    *   **Bool Definitions (`Bool`):** These represent predicates (e.g., `isWellformedDir(dx, dy)`). They are parsed into `PredicateDefinitionConfig` objects.

3.  **Processing `ProgramVariablesBlock`:**
    *   It searches for a `ProgramVariablesBlock` child.
    *   It iterates through each `Variable` child node, extracting its `name` and `var_type` (e.g., `Real x`) into a `VariableDeclarationConfig`, and adds it to `variables_config.state_variables`.

4.  **Processing `ProblemStatement`:**
    *   It searches for a `ProblemStatement` child.
    *   **Precondition (`Precondition (P)`):** Extracts the `formula` from the `Precondition (P)` child node and sets `problem_config.precondition`.
    *   **Loop Execution (`LoopExecution`):**
        *   It extracts the `loop_invariant_formula` from the `Invariant` child of `LoopExecution`.
        *   It then processes the `loop_body`, which is expected to be an `HP` node containing two further `HP` children (one for the discrete part, one for the continuous part).
            *   **Discrete Part:** It iterates through the `Statement` nodes within the discrete `HP`. If a statement is `"controlFunc"`, it adds a `CallHybridProgramConfig`. If it's a deterministic assignment (`:=`), it adds a `DeterministicAssignmentActionConfig`. These are collected into a `SequenceActionConfig`.
            *   **Continuous Part:** It looks for a `Statement` node with value `"stepFunc"` and an associated `Invariant` node. It creates a `HybridProgramStepConfig` for `stepFunc` with its invariant.
        *   These discrete and continuous action configurations are added to `problem_config.loop_execution.loop_body.actions`.
    *   **Postcondition (`Postcondition (Q)`):** Extracts the `formula` from the `Postcondition (Q)` child node and sets `problem_config.postcondition`.

5.  **Return:** The fully populated `gym_config` object.

---

### 2. `_parse_kx_hp_program_fragment_to_action_config`

**Purpose:**
This is a recursive helper function dedicated to translating fragments of KeYmaera X hybrid programs (specifically the bodies of `controlFunc`s) into a hierarchical structure of `ActionType` objects. This structure directly models the choices and sequences of actions an agent can take in the RL environment.

**Inputs:**
*   `kx_node: ASTNode`: A node from the KeYmaera X AST representing a program fragment (e.g., a `ChoiceOperator (++)` node, a `Statement` node, or a `Test` node).
*   `choice_level: int`: An integer that tracks the current nesting depth of `ChoiceOperator (++)` nodes. This is used to generate unique `level_name`s for `ChoiceActionConfig` objects, which can be useful for debugging or specific RL environment implementations.

**Output:**
*   `ActionType`: An object representing the parsed action structure. This can be a `ChoiceActionConfig`, `SequenceActionConfig`, `NondeterministicAssignmentActionConfig`, `DeterministicAssignmentActionConfig`, or `ActionConstraintConfig`.

**Step-by-step Explanation (Recursive Logic):**

1.  **Base Case / Empty Node:** If `kx_node` is `None` (or effectively empty), it returns an empty `SequenceActionConfig`.

2.  **`ChoiceOperator (++)`:**
    *   If `kx_node.node_type` is `"ChoiceOperator (++)"`, it indicates a point where the agent can choose between multiple branches.
    *   It creates a `ChoiceActionConfig` with a `level_name` like `"ChoiceLevel_0"`, `"ChoiceLevel_1"`, etc., based on the `choice_level`.
    *   It then iterates through each child of the `ChoiceOperator` (e.g., `Choice 1`, `Choice 2`). Each child represents a distinct choice branch.
    *   For each branch, it creates a `SequenceActionConfig` to hold the actions within that specific choice.
    *   It *recursively calls itself* (`_parse_kx_hp_program_fragment_to_action_config`) for each sub-node within the current choice branch. The `choice_level` is incremented for these recursive calls to reflect deeper nesting.
    *   The results of these recursive calls (which are `ActionType` objects) are appended to the `actions` list of the `SequenceActionConfig` for the current branch.
    *   Finally, this `SequenceActionConfig` (representing one complete choice) is added to the `choices` list of the main `ChoiceActionConfig`.

3.  **`Statement`:**
    *   If `kx_node.node_type` is `"Statement"`:
        *   If its `value` ends with `":= *"`, it's a nondeterministic assignment (e.g., `w := *`). It creates a `NondeterministicAssignmentActionConfig`.
        *   If its `value` contains `":="` (and is not a nondeterministic assignment), it's a deterministic assignment (e.g., `a := -b`). It creates a `DeterministicAssignmentActionConfig`.
        *   Otherwise, it's an unhandled statement type, logs a warning, and returns an empty `SequenceActionConfig`.

4.  **`Test`:**
    *   If `kx_node.node_type` is `"Test"`, it represents a logical condition that must hold (e.g., `?v = 0`). It creates an `ActionConstraintConfig` with the test's formula.

5.  **Implicit Sequence (Node with Children):**
    *   If `kx_node` is not one of the specific types above but *does* have children, it's treated as an implicit sequence of actions. This handles cases where a block of statements or tests is implicitly sequential in the KeYmaera X AST (e.g., `a := 0; w := 0;` within a `Choice` block).
    *   It creates a `SequenceActionConfig`.
    *   It *recursively calls itself* for each child node, and the resulting `ActionType` objects are added to the `actions` list of the `SequenceActionConfig`.

6.  **Unhandled Node:** If none of the above conditions match, a warning is logged, and an empty `SequenceActionConfig` is returned.

---

### Example Input and Output Walkthrough

Let's use the KeYmaera X source code you provided as the input to the `KeYmaeraXParser`, and then trace how `convert_keymaerax_ast_to_intermediate_pyast` processes the resulting AST.

**1. KeYmaera X Source Code (Input to `KeYmaeraXParser`):**

```keymaerax
Theorem "IJRR17/Theorem 1: Static safety"
Definitions
  HP controlFunc ::= { a := -b; } ++ { ?v = 0; a := 0; w := 0; } ++ { a := A; w := *; ?-W<=w & w<=W; r := *; xo := *; yo := *; ?r!=0 & r*w = v; ? abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v); } ;;
  HP stepFunc ::= { x' = v * dx, y' = v * dy, v' = a, dx' = -w * dy, dy' = w * dx, w' = a/r, t' = 1 & t <= ep & v >= 0 } ;;
  Real ep;
  Real b;
  Real A;
  Real W;
  Real stopDist(v) = v^2 / (2*b);
  Real accelComp(v) = (A/b + 1) * (A/2 * ep^2 + ep*v);
  Real admissibleSeparation(v) = stopDist(v) + accelComp(v);
  Bool isWellformedDir(dx, dy) <-> dx^2 + dy^2 = 1;
  Bool bounds <-> A >= 0 & b > 0 & ep > 0;
End.

ProgramVariables
  Real x;
  Real y;
  Real v;
  Real a;
  Real dx;
  Real dy;
  Real w;
  Real r;
  Real xo;
  Real yo;
  Real t;
End.

Problem
  assumptions(x, y, v, dx, dy, xo, yo) -> [{ {controlFunc; t := 0;} {stepFunc;@invariant(t>=0, isWellformedDir(dx, dy));} }*@invariant(loopinv(x, y, v, dx, dy, xo, yo))] (x - xo)^2 + (y - yo)^2 > 0
End.
```

**2. KeYmaera X AST (Output of `KeYmaeraXParser.parse()` and Input to `convert_keymaerax_ast_to_intermediate_pyast`):**

This is a textual representation of the `ASTNode` tree.

```
[KeYmaeraXSpecification]
  [TheoremName] 'IJRR17/Theorem 1: Static safety'
  [DefinitionsBlock]
    [HP]: controlFunc
      Body (raw): { a := -b; } ++ { ?v = 0; a := 0; w := 0; } ++ { a := A; w := *; ?-W<=w & w<=W; r := *; xo := *; yo := *; ?r!=0 & r*w = v; ? abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v); } ...
      [ChoiceOperator (++)]
        [Choice 1]
          [Statement] 'a := -b'
        [Choice 2]
          [Test] '?v = 0'
          [Statement] 'a := 0'
          [Statement] 'w := 0'
        [Choice 3]
          [Statement] 'a := A'
          [Statement] 'w := *'
          [Test] '?-W<=w & w<=W'
          [Statement] 'r := *'
          [Statement] 'xo := *'
          [Statement] 'yo := *'
          [Test] '?r!=0 & r*w = v'
          [Test] '? abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v)'
    [HP]: stepFunc
      Body (raw): { x' = v * dx, y' = v * dy, v' = a, dx' = -w * dy, dy' = w * dx, w' = a/r, t' = 1 & t <= ep & v >= 0 } ...
      [OrdinaryDifferentialEquations]
        [ODE] 'x\' = v * dx'
        [ODE] 'y\' = v * dy'
        [ODE] 'v\' = a'
        [ODE] 'dx\' = -w * dy'
        [ODE] 'dy\' = w * dx'
        [ODE] 'w\' = a/r'
        [ODE] 't\' = 1'
      [EvolutionDomain]
        [Constraint] 't <= ep'
        [Constraint] 'v >= 0'
    [DEF]: Real ep
    [DEF]: Real b
    [DEF]: Real A
    [DEF]: Real W
    [DEF]: Real stopDist = v^2 / (2*b)
      [Parameters] 'Real v'
    [DEF]: Real accelComp = (A/b + 1) * (A/2 * ep^2 + ep*v)
      [Parameters] 'Real v'
    [DEF]: Real admissibleSeparation = stopDist(v) + accelComp(v)
      [Parameters] 'Real v'
    [DEF]: Bool isWellformedDir = dx^2 + dy^2 = 1
      [Parameters] 'Real dx, Real dy'
    [DEF]: Bool bounds = A >= 0 & b > 0 & ep > 0
  [ProgramVariablesBlock]
    [VAR]: Real x
    [VAR]: Real y
    [VAR]: Real v
    [VAR]: Real a
    [VAR]: Real dx
    [VAR]: Real dy
    [VAR]: Real w
    [VAR]: Real r
    [VAR]: Real xo
    [VAR]: Real yo
    [VAR]: Real t
  [ProblemStatement]
    [Precondition (P)] 'assumptions(x, y, v, dx, dy, xo, yo)'
    [LoopExecution]
      [HP]
        [HP]
          [Statement] 'controlFunc'
          [Statement] 't := 0'
        [HP]
          [Statement] 'stepFunc'
          [Invariant] 't>=0, isWellformedDir(dx, dy)'
      [Invariant] 'loopinv(x, y, v, dx, dy, xo, yo)'
    [Postcondition (Q)] '(x - xo)^2 + (y - yo)^2 > 0'
```

**3. `GymEnvironmentConfig` (Output of `convert_keymaerax_ast_to_intermediate_pyast`):**

This is a textual representation of the `GymEnvironmentConfig` object.

```python
GymEnv(name='IJRR17_Theorem_1__Static_safety',
  defs=Defs(
    hps=[
      ControlFunc(name='controlFunc',
        actions=Choice(name='ChoiceLevel_0',
          choices=[
            Sequence([DetAssign(var='a', val='-b')]),
            Sequence([Constraint(formula='?v = 0'), DetAssign(var='a', val='0'), DetAssign(var='w', val='0')]),
            Sequence([
              DetAssign(var='a', val='A'),
              NondetAssign(var='w'),
              Constraint(formula='?-W<=w & w<=W'),
              NondetAssign(var='r'),
              NondetAssign(var='xo'),
              NondetAssign(var='yo'),
              Constraint(formula='?r!=0 & r*w = v'),
              Constraint(formula='? abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v)')
            ])
          ]
        )
      ),
      StepFunc(name='stepFunc',
        odes=[
          ODE(var='x', eq='v * dx'),
          ODE(var='y', eq='v * dy'),
          ODE(var='v', eq='a'),
          ODE(var='dx', eq='-w * dy'),
          ODE(var='dy', eq='w * dx'),
          ODE(var='w', eq='a/r'),
          ODE(var='t', eq='1')
        ],
        domain=EvolutionDomain(constraints=['t <= ep', 'v >= 0']),
        duration=None
      )
    ],
    funcs=[
      FuncDef(name='stopDist', type='Real', params=[('Real', 'v')], body='v^2 / (2*b)'),
      FuncDef(name='accelComp', type='Real', params=[('Real', 'v')], body='(A/b + 1) * (A/2 * ep^2 + ep*v)'),
      FuncDef(name='admissibleSeparation', type='Real', params=[('Real', 'v')], body='stopDist(v) + accelComp(v)')
    ],
    consts=[
      ConstDef(name='ep', type='Real', value='ep'),
      ConstDef(name='b', type='Real', value='b'),
      ConstDef(name='A', type='Real', value='A'),
      ConstDef(name='W', type='Real', value='W')
    ],
    preds=[
      PredDef(name='isWellformedDir', params=[('Real', 'dx'), ('Real', 'dy')], body='dx^2 + dy^2 = 1'),
      PredDef(name='bounds', params=[], body='A >= 0 & b > 0 & ep > 0')
    ]
  ),
  vars=Vars(
    state_vars=[
      VarDecl(name='x', type='Real'),
      VarDecl(name='y', type='Real'),
      VarDecl(name='v', type='Real'),
      VarDecl(name='a', type='Real'),
      VarDecl(name='dx', type='Real'),
      VarDecl(name='dy', type='Real'),
      VarDecl(name='w', type='Real'),
      VarDecl(name='r', type='Real'),
      VarDecl(name='xo', type='Real'),
      VarDecl(name='yo', type='Real'),
      VarDecl(name='t', type='Real')
    ]
  ),
  prob=Problem(
    pre=Precond(formula='assumptions(x, y, v, dx, dy, xo, yo)'),
    loop=LoopExec(
      inv='loopinv(x, y, v, dx, dy, xo, yo)',
      iter='inf',
      body=LoopBody(
        actions=[
          Sequence([
            CallHP(name='controlFunc'),
            DetAssign(var='t', val='0')
          ]),
          HPStep(name='stepFunc', inv='t>=0, isWellformedDir(dx, dy)')
        ]
      )
    ),
    post=Postcond(formula='(x - xo)^2 + (y - yo)^2 > 0')
  )
)
```