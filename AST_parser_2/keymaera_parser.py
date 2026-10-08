#file description: This file contains the KeYmaeraLexer class, which performs lexical analysis for KeYmaera model text, converting it into a stream of tokens. It also includes a helper function to parse annotations from comments.
#name : keymaera_parser.py
import re
from typing import List, Optional, Union, Dict, Any, Tuple

# Import AST nodes
from ast_config import (
    ASTNode, KeYmaeraModel, ArchiveEntry, DefinitionsBlock, Definition, FuncDef, PredDef, HPDef,
    ProgramVariablesBlock, ProgramVariableDecl, ProblemBlock, TacticBlock,
    Term, Literal, VarRef, BinaryTermOp, UnaryTermOp, FunctionCall,
    Formula, BooleanLiteral, BinaryFormulaOp, UnaryFormulaOp, Comparison, Quantifier,
    ModalFormula, PredicateCall, C_Formula,
    HybridProgram, Assignment, NondetAssignment, Test, Sequence, Choice, Iteration,
    ODE, ODEEquation, HPBlock
)

# Import the lexer and annotation parser
from keymaera_lexer import KeYmaeraLexer, parse_annotations


class KeYmaeraParser:
    """
    A parser for KeYmaera models, converting their textual representation
    into a structured Abstract Syntax Tree (AST).

    It uses a dedicated `KeYmaeraLexer` for tokenization and implements
    recursive descent parsing with operator precedence (Pratt parser)
    for handling terms, formulas, and hybrid programs.
    """
    def __init__(self, text: str):
        """
        Initializes the KeYmaeraParser with the raw input text.
        """
        self.lexer = KeYmaeraLexer(text) # Use the dedicated lexer

    # --- Token Stream Accessors (delegating to lexer) ---
    def _peek_token(self) -> Optional[str]:
        return self.lexer.peek_token()

    def _consume_token(self) -> str:
        return self.lexer.consume_token()

    def _expect_token(self, expected_token: str) -> None:
        self.lexer.expect_token(expected_token)

    # --- Combined Operator Precedence for Terms and Formulas ---
    # Lower number = lower precedence
    # Operators are grouped by precedence level.
    # Note: '->' and '<->' are right-associative. '^' is also right-associative.
    operator_precedence = {
        '<->': 1,
        '->': 2,
        '|': 3,
        '&&': 4, '&': 4, # Logical AND
        '=': 5, '>': 5, '<': 5, '>=': 5, '<=': 5, '!=': 5, # Comparison operators
        '+': 6, '-': 6, # Term addition/subtraction
        '*': 7, '/': 7, # Term multiplication/division
        '^': 8, # Term exponentiation
        "'": 9 # Postfix derivative
    }
    right_associative_ops = {'^', '->', '<->'}

    # --- Primary Expression Parsing (lowest level atom) ---
    def _parse_primary_expression(self) -> Union[Formula, Term]:
        token = self._peek_token()
        if token is None:
            raise ValueError("Unexpected end of expression: expected a primary expression.")

        if token in ['true', 'false']:
            self._consume_token()
            return BooleanLiteral(token == 'true')
        elif token == '[': # Box modality: [hp]F
            self._consume_token()
            program = self._parse_hp_choice_expression()
            self._expect_token(']')
            # [hp]F binds tighter than &,|,->,<->, so F must not swallow trailing connectives
            # (e.g. `[a]P & Q` must parse as `([a]P) & Q`, not `[a](P & Q)`). A parenthesized F
            # still parses fully since '(' resets to min_precedence=0 internally.
            formula_part = self._parse_formula_expression(self.operator_precedence['&'] + 1)
            if not isinstance(formula_part, Formula):
                formula_part = self._convert_term_to_predicate_call(formula_part, "Modal operator expects a formula, but got a term: ")
            return ModalFormula('box', program, formula_part)
        elif token == '<': # Diamond modality: <hp>F
            self._consume_token()
            program = self._parse_hp_choice_expression()
            self._expect_token('>')
            # See box modality above: bind tighter than the logical connectives.
            formula_part = self._parse_formula_expression(self.operator_precedence['&'] + 1)
            if not isinstance(formula_part, Formula):
                formula_part = self._convert_term_to_predicate_call(formula_part, "Modal operator expects a formula, but got a term: ")
            return ModalFormula('diamond', program, formula_part)
        elif token == '(': # Parenthesized expression
            self._consume_token() # Consume '('
            # Recursively parse the inner expression as a general formula/term
            inner_expr = self._parse_formula_expression()
            self._expect_token(')') # Expect and consume ')'
            return inner_expr
        elif re.match(r'^\d+\.\d+$', token) or re.match(r'^\d+$', token): # Number literal
            self._consume_token()
            return Literal(float(token))
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token): # Identifier (VarRef or FunctionCall)
            name = self._consume_token()
            if self._peek_token() == '(': # Function call
                self._consume_token() # Consume '('
                args = []
                if self._peek_token() != ')':
                    while True:
                        # Function arguments should be terms. Use _parse_formula_expression
                        # and then assert it's a Term.
                        arg_node = self._parse_formula_expression()
                        if not isinstance(arg_node, Term):
                            raise ValueError(f"Function argument expects a term, but got a formula: {arg_node}")
                        args.append(arg_node)
                        if self._peek_token() == ',':
                            self._consume_token()
                        else:
                            break
                self._expect_token(')')
                return FunctionCall(name, args)
            else: # Variable reference
                return VarRef(name)
        else:
            raise ValueError(f"Unexpected token in primary expression: '{token}' (at index {self.lexer.current_token_idx}). "
                             f"Full token list: {self.lexer.tokens}. "
                             f"Context: {self.lexer.tokens[max(0, self.lexer.current_token_idx-3):self.lexer.current_token_idx+3]}")

    # --- Unary Expression Parsing (e.g., '-', '!') ---
    def _parse_unary_expression(self) -> Union[Formula, Term]:
        token = self._peek_token()
        if token == '-': # Unary minus
            self._consume_token()
            # Unary minus should bind tighter than binary ops, so give it a high precedence
            # Use the same precedence as exponentiation to ensure it binds before binary arithmetic
            operand = self._parse_formula_expression(self.operator_precedence['^'])
            if not isinstance(operand, Term):
                raise ValueError(f"Unary '-' operator expects a term, but got a formula: {operand}")
            return UnaryTermOp('-', operand)
        elif token == '!': # Unary NOT
            self._consume_token()
            operand = self._parse_formula_expression(self.operator_precedence['&']) # Precedence of logical AND
            if not isinstance(operand, Formula):
                operand = self._convert_term_to_predicate_call(operand, "Unary '!' operator expects a formula, but got a term: ")
            return UnaryFormulaOp('!', operand)
        else:
            return self._parse_primary_expression()

    # --- Main Formula/Term Expression Parsing (Pratt Parser) ---
    def _parse_formula_expression(self, min_precedence=0) -> Union[Formula, Term]:
        left_node: Union[Formula, Term]

        token = self._peek_token()
        if token in ['forall', 'exists']: # Quantifiers are prefix operators handled at this level
            quantifier_type = self._consume_token()
            self._expect_token('(')
            variables: List[VarRef] = []
            while self._peek_token() != ')':
                var_name = self._consume_token()
                variables.append(VarRef(var_name))
                if self._peek_token() == ',':
                    self._consume_token()
                else:
                    break
            self._expect_token(')')
            self._expect_token('.')
            body_formula = self._parse_formula_expression() # Body is a formula
            if not isinstance(body_formula, Formula):
                 body_formula = self._convert_term_to_predicate_call(body_formula, "Quantifier body expects a formula, but got a term: ")
            left_node = Quantifier(quantifier_type, variables, body_formula)
        else:
            left_node = self._parse_unary_expression() # Start with a unary expression (or primary)

        while self._peek_token() in self.operator_precedence and \
              self.operator_precedence[self._peek_token()] >= min_precedence:
            op = self._peek_token()

            # Handle postfix derivative operator
            if op == "'":
                if not isinstance(left_node, Term):
                    raise ValueError(f"Postfix derivative operator '{op}' expects a term, but got a formula: {left_node}")
                self._consume_token()
                left_node = UnaryTermOp(op, left_node)
                continue # Continue the loop to check for more operators

            # Handle comparison operators (Term op Term -> Formula)
            if op in ['=', '>', '<', '>=', '<='
, '!=']:
                if not isinstance(left_node, Term):
                    raise ValueError(f"Comparison operator '{op}' expects a term on its left, but found a formula: {left_node}")
                self._consume_token()
                # Right side of comparison is always a term, parsed with precedence of current operator + 1
                right_term = self._parse_formula_expression(self.operator_precedence[op] + 1)
                if not isinstance(right_term, Term):
                    raise ValueError(f"Comparison operator '{op}' expects a term on its right, but got a formula: {right_term}")
                left_node = Comparison(op, left_node, right_term)
                continue # Continue the loop with the new 'left_node' (which is now a Formula)

            # All other operators are binary (Term op Term -> Term or Formula op Formula -> Formula)
            self._consume_token() # Consume the operator
            precedence = self.operator_precedence[op]

            # Determine precedence for recursive call
            if op in self.right_associative_ops:
                right_node = self._parse_formula_expression(precedence)
            else:
                right_node = self._parse_formula_expression(precedence + 1)

            # Type checking and construction based on operator type
            if op in ['+', '-', '*', '/', '^']: # Term operators
                if not isinstance(left_node, Term) or not isinstance(right_node, Term):
                    raise ValueError(f"Binary term operator '{op}' expects two terms, but got formula(s): {left_node}, {right_node}")
                left_node = BinaryTermOp(op, left_node, right_node)
            elif op in ['&&', '&', '|', '->', '<->']: # Logical operators
                if not isinstance(left_node, Formula):
                    left_node = self._convert_term_to_predicate_call(left_node, f"Binary logical operator '{op}' expects a formula on its left, but found a term: ")
                if not isinstance(right_node, Formula):
                    right_node = self._convert_term_to_predicate_call(right_node, f"Binary logical operator '{op}' expects a formula on its right, but found a term: ")
                left_node = BinaryFormulaOp(op, left_node, right_node)
            else:
                # This case should ideally not be reached if all operators are covered
                raise ValueError(f"Unhandled binary operator: '{op}'")

        return left_node

    def parse_term(self, expression: str = "", tokens: Optional[List[str]] = None) -> Term:
        """
        Public method to parse a standalone term expression string or a list of tokens.
        If 'tokens' is provided, it operates on those tokens.
        """
        is_top_level_call = (tokens is None)
        original_lexer_state = None

        if is_top_level_call:
            original_lexer_state = self.lexer.get_state()
            temp_lexer = KeYmaeraLexer(expression)
            self.lexer.set_tokens(temp_lexer.get_all_tokens())
            self.lexer.current_token_idx = 0
        else:
            # For recursive calls with tokens, just set the tokens and reset index.
            # The caller (e.g., _parse_hp_atom) is responsible for saving/restoring the original state
            # and advancing the main lexer's index.
            self.lexer.set_tokens(tokens)
            self.lexer.current_token_idx = 0
            
        if not self.lexer.tokens and (expression or tokens):
            raise ValueError(f"Term content is empty or contains only untokenizable characters: '{expression}' or provided tokens.")

        parsed_node = self._parse_formula_expression() # This can return a Formula or Term

        if not isinstance(parsed_node, Term):
            raise ValueError(f"Expected a term, but parsed a formula: {parsed_node}")
        
        if self.lexer.current_token_idx != len(self.lexer.tokens):
            raise ValueError(f"Unconsumed tokens after parsing term: {self.lexer.get_remaining_tokens()}")

        if is_top_level_call:
            self.lexer.restore_state(original_lexer_state)
        return parsed_node

    def parse_formula(self, expression: str = "", tokens: Optional[List[str]] = None) -> Formula:
        """
        Public method to parse a standalone formula expression string or a list of tokens.
        If 'tokens' is provided, it operates on those tokens and consumes them.
        If 'expression' is provided, it tokenizes it, parses, and restores the lexer state.
        """
        # Flag to indicate if this is a top-level call that should manage lexer state restoration
        is_top_level_call = (tokens is None)
        original_lexer_state = None

        if is_top_level_call:
            original_lexer_state = self.lexer.get_state()
            temp_lexer = KeYmaeraLexer(expression)
            self.lexer.set_tokens(temp_lexer.get_all_tokens())
            self.lexer.current_token_idx = 0
        else:
            # For recursive calls with tokens, just set the tokens and reset index.
            # The caller (e.g., _parse_hp_atom) is responsible for saving/restoring the original state
            # and advancing the main lexer's index.
            self.lexer.set_tokens(tokens)
            self.lexer.current_token_idx = 0

        if not self.lexer.tokens and (expression or tokens):
            raise ValueError(f"Formula content is empty or contains only untokenizable characters: '{expression}' or provided tokens.")

        # --- Handle implicit [HP]true for Problem section (only for top-level calls) ---
        if is_top_level_call and self._peek_token() == '{':
            hp_attempt_lexer_state = self.lexer.get_state() # Save state for potential rollback
            try:
                hp_ast = self._parse_hp_choice_expression()
                if self.lexer.current_token_idx == len(self.lexer.tokens):
                    # Successfully parsed as an HP block, implying [HP]true
                    self.lexer.restore_state(original_lexer_state) # Restore to before this parse_formula call
                    return ModalFormula('box', hp_ast, BooleanLiteral(True))
            except ValueError:
                self.lexer.restore_state(hp_attempt_lexer_state) # Rollback if HP parsing failed

        # --- Handle comma-separated formulas ---
        # Scan for top-level commas within the current set of tokens.
        # This needs to be done on the *current* lexer's tokens, not a copy of remaining tokens
        # to correctly advance the lexer's position.
        
        # We need to find if there's a top-level comma *within the current scope of tokens*
        # and if so, parse them individually.
        
        # This logic is complex because parse_formula needs to consume tokens.
        # If we find a comma, we need to split the *current* tokens into sub-lists
        # and call parse_formula recursively for each.
        
        # To avoid problems with modifying self.lexer.tokens during iteration,
        # we'll take a snapshot of the current tokens for splitting logic,
        # but ensure the main lexer's state is advanced correctly.

        # Store the current lexer state before attempting comma splitting
        current_parse_formula_lexer_state = self.lexer.get_state()
        tokens_for_splitting = list(self.lexer.tokens[self.lexer.current_token_idx:])
        
        comma_found = False
        balance = 0
        for i, token_to_check in enumerate(tokens_for_splitting):
            if token_to_check in ['(', '[', '{']:
                balance += 1
            elif token_to_check in [')', ']', '}']:
                balance -= 1
            elif token_to_check == ',' and balance == 0:
                comma_found = True
                break
        
        if comma_found:
            parsed_formulas: List[Formula] = []
            current_formula_tokens: List[str] = []
            balance = 0
            
            total_tokens_consumed_by_splitting = 0
            
            for i, token_to_process in enumerate(tokens_for_splitting):
                if token_to_process in ['(', '[', '{']:
                    balance += 1
                elif token_to_process in [')', ']', '}']:
                    balance -= 1

                if token_to_process == ',' and balance == 0:
                    if current_formula_tokens:
                        # Recursively parse the segment using a temporary lexer state
                        # This call will also manage its own lexer state internally.
                        parsed_formulas.append(self.parse_formula(tokens=current_formula_tokens))
                    current_formula_tokens = []
                else:
                    current_formula_tokens.append(token_to_process)
                total_tokens_consumed_by_splitting += 1
            
            if current_formula_tokens:
                parsed_formulas.append(self.parse_formula(tokens=current_formula_tokens))

            if not parsed_formulas:
                raise ValueError(f"No formulas parsed from tokens: {tokens_for_splitting}")
            
            combined_formula = parsed_formulas[0]
            for i in range(1, len(parsed_formulas)):
                combined_formula = BinaryFormulaOp('&&', combined_formula, parsed_formulas[i])
            
            # Restore the lexer state to what it was before this comma-splitting logic,
            # then advance its index by the total tokens processed by the splitting.
            self.lexer.restore_state(current_parse_formula_lexer_state)
            self.lexer.current_token_idx += total_tokens_consumed_by_splitting

            if is_top_level_call:
                self.lexer.restore_state(original_lexer_state)
            return combined_formula
        
        # --- End of comma-separated formula handling ---

        # If no commas were found at the top level, proceed with normal formula expression parsing
        parsed_node = self._parse_formula_expression()

        if self.lexer.current_token_idx != len(self.lexer.tokens):
            raise ValueError(f"Unconsumed tokens after parsing formula: {self.lexer.get_remaining_tokens()}")

        if is_top_level_call:
            self.lexer.restore_state(original_lexer_state)

        # Final conversion: If the top-level expression resolved to a Term,
        # convert it to a PredicateCall or Comparison.
        if isinstance(parsed_node, Term):
            return self._convert_term_to_predicate_call(parsed_node, "Top-level formula expression resolved to a non-predicate term: ")
        return parsed_node

    # --- Helper to convert Term to PredicateCall/Comparison if appropriate ---
    def _convert_term_to_predicate_call(self, term_node: Term, error_prefix: str) -> Formula:
        if isinstance(term_node, VarRef):
            return PredicateCall(term_node.name, [])
        elif isinstance(term_node, FunctionCall):
            return PredicateCall(term_node.name, term_node.args)
        elif isinstance(term_node, (Literal, BinaryTermOp, UnaryTermOp)):
            # Heuristic: If a complex term (like 'x - xo') is found where a formula is expected,
            # assume it implicitly means 'term = 0'.
            return Comparison('=', term_node, Literal(0.0))
        else:
            raise ValueError(f"{error_prefix}{term_node}. Expected a predicate (VarRef or FunctionCall) or a formula, "
                             f"or a term that can implicitly be compared to zero.")

    # --- Hybrid Program Parsing ---
    def _parse_ode_block_content(self, ode_tokens: List[str]) -> ODE:
        equations: List[ODEEquation] = []
        domain_constraint: Optional[Formula] = None

        # Temporarily set the lexer's state to parse these ODE tokens
        original_ode_lexer_state = self.lexer.get_state()
        self.lexer.set_tokens(ode_tokens)
        self.lexer.current_token_idx = 0

        current_part_tokens = []
        parts_to_parse_token_lists = []

        balance = 0
        # Split tokens by ',' or '&' at the top level for ODE equations and domain constraints
        while self.lexer.peek_token() is not None:
            token = self.lexer.peek_token()
            if token in ['(', '[', '{']:
                balance += 1
            elif token in [')', ']', '}']:
                balance -= 1

            # Split only by ',' or '&' that are not inside nested structures
            if (token in [',', '&']) and balance == 0:
                if current_part_tokens:
                    parts_to_parse_token_lists.append(current_part_tokens)
                    current_part_tokens = []
                self._consume_token() # Consume the separator
            else:
                current_part_tokens.append(self._consume_token())
        if current_part_tokens:
            parts_to_parse_token_lists.append(current_part_tokens)

        for part_token_list in parts_to_parse_token_lists:
            if not part_token_list: # Check if the list is empty
                continue

            # Check for ODE equation (e.g., x' = v)
            # Use the tokens directly for checking and parsing
            if len(part_token_list) >= 3 and part_token_list[1] == "'" and part_token_list[2] == '=':
                var_name = part_token_list[0]
                expr_tokens = part_token_list[3:]
                equations.append(ODEEquation(VarRef(var_name), self.parse_term(tokens=expr_tokens)))
            else:
                # If it's not an ODE equation, assume it's a domain constraint formula
                new_domain_formula = self.parse_formula(tokens=part_token_list)
                if domain_constraint is not None:
                    domain_constraint = BinaryFormulaOp('&', domain_constraint, new_domain_formula)
                else:
                    domain_constraint = new_domain_formula

        if not equations and domain_constraint is None:
             raise ValueError(f"Empty or unparseable ODE block content: '{' '.join(ode_tokens)}'")

        # Ensure all tokens were consumed by this ODE parser
        if self.lexer.current_token_idx != len(self.lexer.tokens):
            raise ValueError(f"Unconsumed tokens after parsing ODE block content: {self.lexer.get_remaining_tokens()}")

        self.lexer.restore_state(original_ode_lexer_state) # Restore original lexer state
        return ODE(equations, domain_constraint)


    def _parse_hp_atom(self) -> HybridProgram:
        token = self._peek_token()
        if token is None:
            raise ValueError("Unexpected end of hybrid program: expected an atom.")

        if token == '{': # ODE or HP Block: `{...}`
            self._consume_token() # Consume the opening '{'

            # Find the matching '}' for the current HP block
            temp_idx_scan = self.lexer.current_token_idx
            brace_balance_scan = 1
            inner_block_end_token_idx = -1

            while temp_idx_scan < len(self.lexer.tokens):
                t_scan = self.lexer.tokens[temp_idx_scan]
                if t_scan == '{':
                    brace_balance_scan += 1
                elif t_scan == '}':
                    brace_balance_scan -= 1

                if brace_balance_scan == 0:
                    inner_block_end_token_idx = temp_idx_scan
                    break
                temp_idx_scan += 1

            if inner_block_end_token_idx == -1:
                raise ValueError("Unexpected end of expression: Unmatched '{' in HP block.")

            inner_content_tokens = self.lexer.tokens[self.lexer.current_token_idx:inner_block_end_token_idx]
            
            # Save the current lexer state before parsing inner content
            original_outer_lexer_state = self.lexer.get_state()
            
            # Temporarily set the lexer's state to parse these inner tokens
            self.lexer.set_tokens(inner_content_tokens)
            self.lexer.current_token_idx = 0 # Reset for this sub-parse

            parsed_inner_hp_content: Optional[HybridProgram] = None
            ode_parsed = False
            
            # Heuristic for ODE: Check for ODE characteristics *within the current level of tokens*
            is_likely_ode_at_this_level = False
            for i in range(len(inner_content_tokens)):
                if inner_content_tokens[i] == '=':
                    if i > 0 and (re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', inner_content_tokens[i-1]) or \
                                  (i > 1 and inner_content_tokens[i-1] == "'" and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', inner_content_tokens[i-2]))):
                        is_likely_ode_at_this_level = True
                        break
            
            # Also, if there are no assignments (:=) or tests (?) at this level, it's more likely an ODE.
            # And crucially, if the inner_content_tokens *do not* start with another '{' or '?',
            # then it's probably an ODE.
            starts_with_nested_block = inner_content_tokens and (inner_content_tokens[0] == '{' or inner_content_tokens[0] == '?')

            if is_likely_ode_at_this_level and not any(t in [':=', '?'] for t in inner_content_tokens) and not starts_with_nested_block:
                try:
                    parsed_inner_hp_content = self._parse_ode_block_content(inner_content_tokens)
                    ode_parsed = True
                except ValueError:
                    # If ODE parsing fails, it might be a general HP, so proceed to try that.
                    pass
            
            if not ode_parsed:
                try:
                    # If not an ODE, or ODE parsing failed, treat as a general HPBlock.
                    # The lexer state is already set to inner_content_tokens for this attempt.
                    parsed_inner_hp_content = self._parse_hp_choice_expression()
                    if self.lexer.current_token_idx != len(self.lexer.tokens):
                        # If not all tokens were consumed by the inner HP parser, it's an error.
                        raise ValueError(f"HPBlock content not fully consumed by HP parser. Remaining tokens: {self.lexer.get_remaining_tokens()}")
                except ValueError as e:
                    # Restore state on error and re-raise.
                    self.lexer.restore_state(original_outer_lexer_state)
                    raise ValueError(f"Unparseable content inside HPBlock: '{' '.join(inner_content_tokens)}'. Specific error: {e}")

            # Restore the original lexer state and advance its index past the entire block
            self.lexer.restore_state(original_outer_lexer_state)
            self.lexer.current_token_idx = inner_block_end_token_idx + 1 # This consumes the closing '}'

            if parsed_inner_hp_content is None:
                raise ValueError(f"Empty or unparseable content inside HPBlock: '{' '.join(inner_content_tokens)}'")

            return HPBlock(parsed_inner_hp_content)


        elif token == '?': # Test statement: `?(F)` or `? F`
            self._consume_token() # Consume '?'

            original_lexer_state_for_test = self.lexer.get_state() # Save main lexer state

            condition_tokens_for_subparse: List[str] = []
            tokens_consumed_by_subparse = 0

            if self._peek_token() == '(':
                self._consume_token() # Consume '('
                # Extract tokens for the condition within parentheses
                paren_balance_cond = 1
                cond_start_idx_in_main_lexer = self.lexer.current_token_idx
                temp_idx_cond_scan = self.lexer.current_token_idx
                cond_end_token_idx_in_main_lexer = -1

                while temp_idx_cond_scan < len(self.lexer.tokens):
                    t_scan = self.lexer.tokens[temp_idx_cond_scan]
                    if t_scan == '(':
                        paren_balance_cond += 1
                    elif t_scan == ')':
                        paren_balance_cond -= 1

                    if paren_balance_cond == 0:
                        cond_end_token_idx_in_main_lexer = temp_idx_cond_scan
                        break
                    temp_idx_cond_scan += 1

                if cond_end_token_idx_in_main_lexer == -1:
                    self.lexer.restore_state(original_lexer_state_for_test) # Restore on error
                    raise ValueError("Unexpected end of expression: Unmatched '(' in Test condition.")

                condition_tokens_for_subparse = self.lexer.tokens[cond_start_idx_in_main_lexer:cond_end_token_idx_in_main_lexer]
                tokens_consumed_by_subparse = (cond_end_token_idx_in_main_lexer - cond_start_idx_in_main_lexer) + 2 # +2 for '(' and ')'
                
                # Temporarily set lexer to parse condition_tokens_for_subparse
                self.lexer.set_tokens(condition_tokens_for_subparse)
                self.lexer.current_token_idx = 0
                condition = self.parse_formula(tokens=condition_tokens_for_subparse)
                
                # After parsing, restore the main lexer state and advance its index
                self.lexer.restore_state(original_lexer_state_for_test)
                self.lexer.current_token_idx = cond_end_token_idx_in_main_lexer + 1 # Advance past the ')'
                return Test(condition)
            else:
                # If no '(', try to parse the next part as a formula directly.
                temp_idx_formula_scan = self.lexer.current_token_idx # Start from current main lexer position
                balance = 0
                
                while temp_idx_formula_scan < len(self.lexer.tokens):
                    token_candidate = self.lexer.tokens[temp_idx_formula_scan]
                    if token_candidate in ['(', '[', '{']:
                        balance += 1
                    elif token_candidate in [')', ']', '}']:
                        balance -= 1

                    if balance < 0: # Malformed, e.g., unmatched ')'
                        break
                    
                    if balance == 0 and token_candidate in [';', '++', '}', '@invariant']:
                        break
                    
                    condition_tokens_for_subparse.append(token_candidate)
                    temp_idx_formula_scan += 1

                if not condition_tokens_for_subparse:
                    self.lexer.restore_state(original_lexer_state_for_test) # Restore on error
                    raise ValueError(f"Expected a formula after '?', but found nothing or unexpected token: {self._peek_token()}")

                tokens_consumed_by_subparse = temp_idx_formula_scan - self.lexer.current_token_idx
                
                # Temporarily set lexer to parse condition_tokens_for_subparse
                self.lexer.set_tokens(condition_tokens_for_subparse)
                self.lexer.current_token_idx = 0
                condition = self.parse_formula(tokens=condition_tokens_for_subparse)
                
                # After parsing, restore the main lexer state and advance its index
                self.lexer.restore_state(original_lexer_state_for_test)
                self.lexer.current_token_idx += tokens_consumed_by_subparse # Advance past the collected tokens
                return Test(condition)


        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token): # Assignment or HP reference
            var_name = self._consume_token()
            if self._peek_token() == ':=':
                self._consume_token()
                # Handle Nondeterministic Assignment: var := *
                if self._peek_token() == '*':
                    self._consume_token() # Consume '*'
                    return NondetAssignment(VarRef(var_name))
                
                original_lexer_state_for_assignment = self.lexer.get_state() # Save main lexer state

                expr_tokens_for_subparse = []
                tokens_consumed_by_subparse = 0
                balance = 0
                temp_idx_expr_scan = self.lexer.current_token_idx # Start from current main lexer position
                
                while temp_idx_expr_scan < len(self.lexer.tokens):
                    t = self.lexer.tokens[temp_idx_expr_scan]
                    if t in ['(', '[', '{']:
                        balance += 1
                    elif t in [')', ']', '}']:
                        balance -= 1

                    if balance < 0: # Malformed, e.g., unmatched ')'
                        break
                    
                    if balance == 0 and t in [';', '++', '}', '@invariant']:
                        break

                    expr_tokens_for_subparse.append(t)
                    temp_idx_expr_scan += 1

                if not expr_tokens_for_subparse:
                    self.lexer.restore_state(original_lexer_state_for_assignment) # Restore on error
                    raise ValueError(f"Expected expression after ':=' for assignment to '{var_name}'.")

                tokens_consumed_by_subparse = temp_idx_expr_scan - self.lexer.current_token_idx
                
                # Temporarily set lexer to parse expr_tokens_for_subparse
                self.lexer.set_tokens(expr_tokens_for_subparse)
                self.lexer.current_token_idx = 0
                assigned_term = self.parse_term(tokens=expr_tokens_for_subparse)
                
                # After parsing, restore the main lexer state and advance its index
                self.lexer.restore_state(original_lexer_state_for_assignment)
                self.lexer.current_token_idx += tokens_consumed_by_subparse # Advance past the collected tokens
                return Assignment(VarRef(var_name), assigned_term)
            else:
                # If not an assignment, it's a reference to a defined HP
                return VarRef(var_name)

        elif token == '@invariant': # Interpret @invariant(F) as Test(F) when found as an HP atom
            self._consume_token() # Consume '@invariant'
            self._expect_token('(')

            original_lexer_state_for_invariant = self.lexer.get_state() # Save main lexer state

            invariant_tokens_for_subparse: List[str] = []
            tokens_consumed_by_subparse = 0

            # Extract tokens for the invariant within parentheses
            paren_balance_inv = 1
            inv_start_idx_in_main_lexer = self.lexer.current_token_idx
            temp_idx_inv_scan = self.lexer.current_token_idx
            inv_end_token_idx_in_main_lexer = -1

            while temp_idx_inv_scan < len(self.lexer.tokens):
                t_scan = self.lexer.tokens[temp_idx_inv_scan]
                if t_scan == '(':
                    paren_balance_inv += 1
                elif t_scan == ')':
                    paren_balance_inv -= 1

                if paren_balance_inv == 0:
                    inv_end_token_idx_in_main_lexer = temp_idx_inv_scan
                    break
                temp_idx_inv_scan += 1

            if inv_end_token_idx_in_main_lexer == -1:
                self.lexer.restore_state(original_lexer_state_for_invariant) # Restore on error
                raise ValueError("Unexpected end of expression: Unmatched '(' in @invariant.")

            invariant_tokens_for_subparse = self.lexer.tokens[inv_start_idx_in_main_lexer:inv_end_token_idx_in_main_lexer]
            tokens_consumed_by_subparse = (inv_end_token_idx_in_main_lexer - inv_start_idx_in_main_lexer) + 2 # +2 for '(' and ')'
            
            # Temporarily set lexer to parse invariant_tokens_for_subparse
            self.lexer.set_tokens(invariant_tokens_for_subparse)
            self.lexer.current_token_idx = 0
            invariant_formula = self.parse_formula(tokens=invariant_tokens_for_subparse)
            
            # After parsing, restore the main lexer state and advance its index
            self.lexer.restore_state(original_lexer_state_for_invariant)
            self.lexer.current_token_idx = inv_end_token_idx_in_main_lexer + 1 # Consume ')'
            return Test(invariant_formula)

        else:
            raise ValueError(f"Unexpected token in HP atom: '{token}'")

    def _parse_hp_factor(self) -> HybridProgram:
        """
        Parses a single HP atom and then applies any suffix operators like * or *@invariant.
        """
        hp_atom = self._parse_hp_atom() # This parses the fundamental HP unit

        while self._peek_token() == '*':
            self._consume_token() # Consume '*'
            invariant = None
            if self._peek_token() == '@invariant':
                self._consume_token() # Consume '@invariant'
                self._expect_token('(')

                original_lexer_state_for_invariant_suffix = self.lexer.get_state() # Save main lexer state

                # Extract tokens for the invariant within parentheses
                paren_balance_inv = 1
                inv_start_idx_in_main_lexer = self.lexer.current_token_idx
                temp_idx_inv_scan = self.lexer.current_token_idx
                inv_end_token_idx_in_main_lexer = -1

                while temp_idx_inv_scan < len(self.lexer.tokens):
                    t_scan = self.lexer.tokens[temp_idx_inv_scan]
                    if t_scan == '(':
                        paren_balance_inv += 1
                    elif t_scan == ')':
                        paren_balance_inv -= 1

                    if paren_balance_inv == 0:
                        inv_end_token_idx_in_main_lexer = temp_idx_inv_scan
                        break
                    temp_idx_inv_scan += 1

                if inv_end_token_idx_in_main_lexer == -1:
                    self.lexer.restore_state(original_lexer_state_for_invariant_suffix) # Restore on error
                    raise ValueError("Unexpected end of expression: Unmatched '(' in @invariant.")

                invariant_tokens_for_subparse = self.lexer.tokens[inv_start_idx_in_main_lexer:inv_end_token_idx_in_main_lexer]
                
                # Temporarily set lexer to parse invariant_tokens_for_subparse
                self.lexer.set_tokens(invariant_tokens_for_subparse)
                self.lexer.current_token_idx = 0
                invariant = self.parse_formula(tokens=invariant_tokens_for_subparse)
                
                # After parsing, restore the main lexer state and advance its index
                self.lexer.restore_state(original_lexer_state_for_invariant_suffix)
                self.lexer.current_token_idx = inv_end_token_idx_in_main_lexer + 1 # Consume ')'
            
            # If hp_atom is already an HPBlock, we directly use its inner_hp.
            # Otherwise, we wrap it in an HPBlock.
            # The Iteration AST node expects an HPBlock for its program.
            if isinstance(hp_atom, HPBlock):
                hp_atom = Iteration(hp_atom, invariant)
            else:
                hp_atom = Iteration(HPBlock(hp_atom), invariant) # Wrap the atom in HPBlock for iteration

        return hp_atom


    def _parse_hp_sequence_expression(self) -> HybridProgram:
        programs: List[HybridProgram] = []

        programs.append(self._parse_hp_factor()) # Call _parse_hp_factor here

        while True:
            next_token = self._peek_token()
            # Stop if end of HP expression, or if it's a choice operator, or end of modal operator
            if next_token is None or next_token in ['++', ']', '>']:
                break

            if next_token == ';':
                self._consume_token()
                # Allow trailing semicolon, so if next is end-of-HP, break
                if self._peek_token() is None or self._peek_token() in ['++', ']', '>']:
                    break
                # If not trailing, parse the next program in sequence
                programs.append(self._parse_hp_factor()) # Call _parse_hp_factor here
            else:
                # Implicit sequencing (HP HP): if the next token is the start of an HP atom,
                # parse it as another program in the sequence.
                # An HP atom can start with '{', '?', or an identifier.
                if next_token in ['{', '?'] or re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', next_token) or next_token == '@invariant': # Added @invariant here
                    programs.append(self._parse_hp_factor()) # Call _parse_hp_factor here
                else:
                    # If it's not a separator and not the start of an HP atom,
                    # then we should stop parsing the sequence.
                    break

        if len(programs) == 1:
            return programs[0]
        else:
            return Sequence(programs)

    def _parse_hp_choice_expression(self) -> HybridProgram:
        left_hp = self._parse_hp_sequence_expression()

        if self._peek_token() == '++':
            self._consume_token()
            right_hp = self._parse_hp_choice_expression()
            return Choice(left_hp, right_hp)

        return left_hp

    def parse_hybrid_program(self, expression: str = "", tokens: Optional[List[str]] = None) -> HybridProgram:
        """
        Public method to parse a standalone hybrid program string or a list of tokens.
        This method uses a temporary lexer state to parse the sub-expression.
        """
        original_lexer_state = self.lexer.get_state()

        if tokens is not None:
            self.lexer.set_tokens(tokens)
            self.lexer.current_token_idx = 0
        else:
            temp_lexer = KeYmaeraLexer(expression)
            self.lexer.set_tokens(temp_lexer.get_all_tokens())
            self.lexer.current_token_idx = 0

        if not self.lexer.tokens and (expression or tokens):
            raise ValueError(f"Hybrid program content is empty or contains only untokenizable characters: '{expression}' or provided tokens.")

        hp = self._parse_hp_choice_expression()

        if self.lexer.current_token_idx != len(self.lexer.tokens):
            source_info = f"from tokens: {tokens}" if tokens is not None else f"from '{expression}'"
            raise ValueError(f"Unconsumed tokens after parsing hybrid program: {self.lexer.get_remaining_tokens()} {source_info}")

        self.lexer.restore_state(original_lexer_state)
        return hp

    def parse(self) -> KeYmaeraModel:
        """
        Parses the entire KeYmaera model text into a KeYmaeraModel AST.
        This is the top-level entry point for parsing a complete model file.
        """
        entries = []

        # The lexer has already tokenized the entire input text, including comments.
        # Now we need to re-access the original text for regex matching of sections,
        # as the lexer only provides tokens.

        # 1. Extract Theorem Metadata
        theorem_match = re.search(r'Theorem\s+"([^"]+)"', self.lexer.text)
        if not theorem_match:
            raise ValueError("Theorem name not found in the model file.")

        full_theorem_string_in_quotes = theorem_match.group(1)
        kind = "Theorem"
        name = full_theorem_string_in_quotes.strip()

        kind_prefix_match = re.match(r'(\w+)/(.+)', full_theorem_string_in_quotes)
        if kind_prefix_match:
            kind = kind_prefix_match.group(1)
            name = kind_prefix_match.group(2).strip()

        # 2. Locate and Extract Raw Text for Main Sections
        # NOTE: search only the text after the Theorem header so entry-specific sections
        # (Definitions/ProgramVariables/Problem) aren't confused with a preceding SharedDefinitions block.
        text_after_theorem = self.lexer.text[theorem_match.end():]

        # SharedDefinitions (if present) is shared across archive entries and lives before the Theorem header.
        shared_definitions_section_match = re.search(r'SharedDefinitions\s*(.*?)\s*End\.', self.lexer.text, re.DOTALL)
        definitions_section_match = re.search(r'Definitions\s*(.*?)\s*End\.', text_after_theorem, re.DOTALL)
        program_variables_section_match = re.search(r'ProgramVariables\s*(.*?)\s*End\.', text_after_theorem, re.DOTALL)
        # FIX: Adjusted regex to be more robust for End. followed by any whitespace or EOF
        problem_section_match = re.search(r'Problem\s*(.*?)\s*End\.\s*', text_after_theorem, re.DOTALL)

        if not (shared_definitions_section_match or definitions_section_match) or not (program_variables_section_match and problem_section_match):
            missing_sections = []
            if not (shared_definitions_section_match or definitions_section_match): missing_sections.append("Definitions")
            if not program_variables_section_match: missing_sections.append("ProgramVariables")
            if not problem_section_match: missing_sections.append("Problem")
            raise ValueError(f"Missing sections in the model file: {', '.join(missing_sections)}.")

        # Merge SharedDefinitions (if any) with the entry-specific Definitions block.
        raw_definitions_text = "\n".join(
            m.group(1) for m in (shared_definitions_section_match, definitions_section_match) if m
        )
        raw_program_variables_text = program_variables_section_match.group(1)
        raw_problem_text_with_comments = problem_section_match.group(1)

        # 3. Parse Definitions Section
        definitions_list: List[Definition] = []
        current_annotations: Dict[str, Any] = {}

        def_lines = raw_definitions_text.splitlines()
        i = 0
        while i < len(def_lines):
            line = def_lines[i]

            comment_match = re.match(r'//\s*@(\w+):.*', line.strip())
            if comment_match:
                current_annotations = parse_annotations(line)
                i += 1
                continue

            line_without_comments = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
            line_without_comments = re.sub(r'//.*', '', line_without_comments).strip()

            if not line_without_comments:
                i += 1
                continue

            full_definition_str_builder = [line_without_comments]

            paren_balance = line_without_comments.count('(') - line_without_comments.count(')')
            brace_balance = line_without_comments.count('{') - line_without_comments.count('}')

            is_terminated = line_without_comments.endswith(';') and paren_balance == 0 and brace_balance == 0

            current_line_idx_for_def = i + 1
            while not is_terminated and current_line_idx_for_def < len(def_lines):
                next_line = def_lines[current_line_idx_for_def]

                next_line_without_comments = re.sub(r'/\*.*?\*/', '', next_line, flags=re.DOTALL)
                next_line_without_comments = re.sub(r'//.*', '', next_line_without_comments).strip()

                if not next_line_without_comments:
                    current_line_idx_for_def += 1
                    continue

                full_definition_str_builder.append(next_line_without_comments)

                paren_balance += next_line_without_comments.count('(') - next_line_without_comments.count(')')
                brace_balance += next_line_without_comments.count('{') - next_line_without_comments.count('}')

                is_terminated = next_line_without_comments.endswith(';') and paren_balance == 0 and brace_balance == 0
                current_line_idx_for_def += 1

            if not is_terminated:
                raise ValueError(f"Unterminated definition starting at line {i+1}: '{line}'")

            full_definition_str = ' '.join(full_definition_str_builder).strip()

            if var_decl_match := re.match(r'(Real|Bool)\s+(\w+)\s*;', full_definition_str):
                var_type = var_decl_match.group(1)
                var_name = var_decl_match.group(2)
                annotations = {
                    key: value.copy() if isinstance(value, dict) else value
                    for key, value in current_annotations.items()
                }
                if var_type == "Real":
                    # Bare declarations like `Real ep;` are constant symbols, not zero-arg functions.
                    # Use a positive default rather than 0.0 so common constraints like `b > 0` remain satisfiable
                    # when no explicit value or fixed bound is present in the model.
                    const_annotation = annotations.setdefault('const', {})
                    if 'value' not in const_annotation and 'bound' not in const_annotation:
                        const_annotation['value'] = '1.0'
                    definitions_list.append(FuncDef(var_name, "Real", Literal(1.0), [], annotations))
                else:
                    const_annotation = annotations.setdefault('const', {})
                    const_annotation.setdefault('value', 'True')
                    definitions_list.append(PredDef(var_name, "Bool", BooleanLiteral(True), [], annotations))
            # Refined regex for FuncDef: use non-capturing group for optional arguments
            elif func_def_match := re.match(r'Real\s+(\w+)\s*(?:\(([^)]*)\))?\s*=\s*(.*);', full_definition_str):
                def_name = func_def_match.group(1)
                args_str = func_def_match.group(2) # Now group 2 for args
                body_str = func_def_match.group(3).strip() # Now group 3 for body

                args = []
                if args_str:
                    individual_args_with_types = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                    for arg_with_type in individual_args_with_types:
                        arg_name_match = re.match(r'(?:Real|Bool)?\s*(\w+)', arg_with_type)
                        if arg_name_match:
                            args.append(VarRef(arg_name_match.group(1)))
                        else:
                            raise ValueError(f"Could not parse argument '{arg_with_type}' in function definition '{def_name}'.")

                body_term = self.parse_term(expression=body_str) # Use expression=body_str for definitions
                definitions_list.append(FuncDef(def_name, "Real", body_term, args, current_annotations.copy()))
            # Refined regex for PredDef: use non-capturing group for optional arguments
            elif pred_def_match := re.match(r'Bool\s+(\w+)\s*(?:\(([^)]*)\))?\s*<->\s*(.*);', full_definition_str):
                def_name = pred_def_match.group(1)
                args_str = pred_def_match.group(2) # Now group 2 for args
                body_str = pred_def_match.group(3).strip() # Now group 3 for body

                args = []
                if args_str:
                    individual_args_with_types = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                    for arg_with_type in individual_args_with_types:
                        arg_name_match = re.match(r'(?:Real|Bool)?\s*(\w+)', arg_with_type)
                        if arg_name_match:
                            args.append(VarRef(arg_name_match.group(1)))
                        else:
                            raise ValueError(f"Could not parse argument '{arg_with_type}' in predicate definition '{def_name}'.")

                body_formula = self.parse_formula(expression=body_str) # Use expression=body_str for definitions
                definitions_list.append(PredDef(def_name, "Bool", body_formula, args, current_annotations.copy()))
            elif hp_def_match := re.match(r'HP\s+(\w+)\s*::=\s*(.*);', full_definition_str):
                def_name = hp_def_match.group(1)
                body_str = hp_def_match.group(2).strip()

                body_hp = self.parse_hybrid_program(expression=body_str) # Use expression=body_str for definitions
                definitions_list.append(HPDef(def_name, body_hp, annotations=current_annotations.copy()))
            else:
                raise ValueError(f"Unrecognized definition pattern for: '{full_definition_str}' (starting line {i+1})")

            current_annotations = {}
            i = current_line_idx_for_def

        definitions_block = DefinitionsBlock(definitions_list)

        # 4. Parse ProgramVariables Section
        program_vars_list: List[ProgramVariableDecl] = []
        current_annotations = {}
        pv_lines = raw_program_variables_text.splitlines()
        for line in pv_lines:
            line = line.strip()
            comment_match = re.match(r'//\s*@(\w+):.*', line)
            if comment_match:
                current_annotations = parse_annotations(line)
                continue

            line_for_parsing = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
            line_for_parsing = re.sub(r'//.*', '', line_for_parsing).strip()

            if not line_for_parsing:
                continue

            var_decl_match = re.match(r'(Real|Bool)\s+(\w+)\s*;', line_for_parsing)
            if var_decl_match:
                var_type = var_decl_match.group(1)
                var_name = var_decl_match.group(2)
                program_vars_list.append(ProgramVariableDecl(var_type, var_name, current_annotations.copy()))
                current_annotations = {}
            elif line_for_parsing:
                raise ValueError(f"Unrecognized program variable declaration: '{line}'")

        program_variables_block = ProgramVariablesBlock(program_vars_list)

        # 5. Parse Problem Section
        problem_text_without_c_comments = re.sub(r'/\*.*?\*/', '', raw_problem_text_with_comments, flags=re.DOTALL)
        problem_text = re.sub(r'//.*', '', problem_text_without_c_comments).strip()
        problem_formula = self.parse_formula(expression=problem_text) # Use expression=problem_text for the problem
        problem_block = ProblemBlock(problem_formula)

        # 6. Tactics (Currently not parsed, remains an empty list)
        tactics_list: List[TacticBlock] = []

        # 7. Construct the final KeYmaeraModel AST
        archive_entry = ArchiveEntry(
            kind=kind,
            name=name,
            definitions=definitions_block,
            program_variables=program_variables_block,
            problem=problem_block,
            tactics=tactics_list
        )

        return KeYmaeraModel([archive_entry])