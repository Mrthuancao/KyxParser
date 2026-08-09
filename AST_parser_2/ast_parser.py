"""
File name: ast_parser.py

This module provides a parser for KeYmaera models, converting their textual
representation into a structured Abstract Syntax Tree (AST).

KeYmaera is a hybrid systems theorem prover that uses a specialized syntax
to define hybrid programs, differential equations, and logical formulas.
This parser aims to facilitate programmatic analysis, manipulation, or
translation of KeYmaera models by representing them as a hierarchical
data structure.

The parsing process involves:
1.  **Lexical Analysis (Tokenization):** Breaking down the input text into a
    stream of meaningful tokens (e.g., keywords, identifiers, operators, literals).
2.  **Syntactic Analysis (Parsing):** Building the AST from the token stream
    according to the grammar rules of KeYmaera. This includes handling:
    *   **Terms:** Mathematical expressions involving variables, literals,
        function calls, and arithmetic operators (e.g., `x + y * z`).
    *   **Formulas:** Logical expressions involving predicates, comparisons,
        quantifiers, and logical operators (e.g., `forall(x). x > 0 -> P(x)`).
    *   **Hybrid Programs (HPs):** Dynamic systems combining discrete assignments,
        differential equations, non-deterministic choices, sequences, tests,
        and iterations (e.g., `{x'=v, v'=-g & t<=5} ++ (x:=x+1; ?(x<10))`).
    *   **Model Structure:** Identifying and parsing top-level sections like
        `Definitions`, `ProgramVariables`, and `Problem`.

The parser employs a recursive descent strategy enhanced with operator
precedence parsing (Pratt parser) for terms and formulas to correctly handle
operator associativity and precedence. It also includes helper functions
for parsing annotations embedded in comments.

The AST nodes are defined in `ast_config.py`, which is assumed to be
available in the same environment.
"""
import abc
import re
from typing import List, Optional, Union, Dict, Any

# Assuming ast_config.py is in the same directory or importable
from ast_config import (
    ASTNode, KeYmaeraModel, ArchiveEntry, DefinitionsBlock, Definition, FuncDef, PredDef, HPDef,
    ProgramVariablesBlock, ProgramVariableDecl, ProblemBlock, TacticBlock,
    Term, Literal, VarRef, BinaryTermOp, UnaryTermOp, FunctionCall,
    Formula, BooleanLiteral, BinaryFormulaOp, UnaryFormulaOp, Comparison, Quantifier,
    ModalFormula, PredicateCall, C_Formula,
    HybridProgram, Assignment, NondetAssignment, Test, Sequence, Choice, Iteration,
    ODE, ODEEquation, HPBlock
)

# --- Helper functions for parsing ---

def parse_annotations(comment_line: str) -> Dict[str, Any]:
    """Parses annotations from a comment string.

    Annotations in KeYmaera models often appear as special comments, e.g.,
    '// @const: bound=[0.1:0.5]; def="time limit"'. This function extracts
    structured key-value pairs from such lines.

    Args:
        comment_line: The string line containing the annotation comment.

    Returns:
        A dictionary where the top-level key is the annotation type (e.g., 'const')
        and its value is another dictionary of parsed key-value properties.
        Returns an empty dictionary if no valid annotation is found.

    Example:
        Input: `comment_line = '// @const: bound=[0.1:0.5]; def="time limit"'`
        Output: `{'const': {'bound': [0.1, 0.5], 'def': 'time limit'}}`
    """
    annotations = {}
    # Regex to find '// @<type>:<content>'
    match = re.search(r'//\s*@(\w+):(.*)', comment_line)
    if match:
        annotation_type = match.group(1)
        annotation_str = match.group(2).strip()
        annotations[annotation_type] = {}
        
        # Heuristic: Try splitting by semicolon first for distinct properties.
        # If no semicolons are present, try splitting by comma.
        # This handles cases where different annotation types might use different
        # primary delimiters for their key-value pairs.
        if ';' in annotation_str:
            individual_properties = [p.strip() for p in annotation_str.split(';')]
        else:
            individual_properties = [p.strip() for p in annotation_str.split(',')]

        for prop_pair_str in individual_properties:
            if '=' in prop_pair_str:
                key, value_str = prop_pair_str.split('=', 1) 
                key = key.strip()
                value_str = value_str.strip()

                # Attempt to convert value to appropriate Python types
                if value_str.startswith('[') and value_str.endswith(']'):
                    # Parse list of floats for bounds, e.g., '[0.1:0.5]'
                    try:
                        parts = value_str[1:-1].split(':')
                        annotations[annotation_type][key] = [float(p.strip()) for p in parts]
                    except ValueError:
                        # Fallback to string if float conversion fails
                        annotations[annotation_type][key] = value_str
                elif value_str.lower() in ['true', 'false']:
                    # Parse boolean literals
                    annotations[annotation_type][key] = value_str.lower() == 'true'
                elif value_str.startswith('"') and value_str.endswith('"'):
                    # Remove quotes for string literals
                    annotations[annotation_type][key] = value_str[1:-1]
                else:
                    # Attempt integer, then float, then fallback to string
                    try:
                        annotations[annotation_type][key] = int(value_str)
                    except ValueError:
                        try:
                            annotations[annotation_type][key] = float(value_str)
                        except ValueError:
                            annotations[annotation_type][key] = value_str
            # Properties without an '=' are currently ignored, but could be added if needed
            # (e.g., a simple flag like "@flag").
    return annotations

# --- Core Parsing Logic ---

class KeYmaeraParser:
    """A parser for KeYmaera models, converting their textual representation
    into a structured Abstract Syntax Tree (AST).

    It uses a combination of regular expressions for tokenization and a
    recursive descent parsing strategy with operator precedence for handling
    terms, formulas, and hybrid programs.
    """
    def __init__(self, text: str):
        """Initializes the KeYmaeraParser with the raw input text.

        Args:
            text: The complete textual content of the KeYmaera model file.
        """
        self.text = text
        self.tokens: List[str] = []
        self.current_token_idx = 0

    def _tokenize(self, expression: str) -> List[str]:
        """Breaks down an input string into a list of atomic tokens.

        This performs the lexical analysis phase. The expression is expected
        to be cleaned of comments by the caller before being passed here.

        Args:
            expression: The string to tokenize (e.g., a term, formula, or HP).

        Returns:
            A list of string tokens.

        Raises:
            ValueError: If any part of the expression cannot be tokenized by the
                        defined patterns.

        Example:
            Input: `expression = "x + 2*y'"`
            Output: `['x', '+', '2', '*', 'y', "'"]`
        """
        # Prioritize multi-character tokens and specific single-character tokens first.
        # Identifiers should be matched before single-character delimiters if they could
        # be part of an identifier.
        patterns = [
            r'@invariant', # Specific keyword that starts with @, without a word boundary
            r'\b(?:Real|Bool|HP|forall|exists|true|false|div|mod|sqrt|abs|min|max|sin|cos|tan|arcsin|arccos|arctan|log|exp|if|then|else|fi)\b', # Other keywords
            r'\d+\.\d+|\d+', # Numbers (float or int) - matched before identifiers to prevent '123x' from being '123' and 'x'
            r'[a-zA-Z_][a-zA-Z0-9_]*', # Identifiers (variables, function names, etc.)
            r':=', # Assignment operator
            r'<->|->|>=|<=|\+\+', # Multi-character formula/HP operators (longer ones first)
            # All other operators and delimiters, longer ones first to avoid partial matches.
            # `re.escape` is used for literal characters that are also regex metacharacters.
            r'\*\*|\*|//|/|\+\+|--|\+|-|\^|==|!=|<=|>=|<|>|=|&&|\|\||&|\||!', 
            re.escape(';'), 
            re.escape(','), 
            re.escape('('), 
            re.escape(')'), 
            re.escape('['), 
            re.escape(']'), 
            re.escape('{'), 
            re.escape('}'), 
            re.escape('`'), 
            re.escape('?'), 
            re.escape("'"), 
            re.escape('"'), 
            re.escape('.'), 
            r'\s+', # Whitespace (to be ignored, but matched to ensure no gaps)
        ]
        
        # Combine all patterns into a single regex, using capturing groups for each pattern.
        combined_pattern = '|'.join(f'({p})' for p in patterns)

        tokens_found = []
        last_end = 0
        # Iterate over all non-overlapping matches in the expression
        for match in re.finditer(combined_pattern, expression):
            start, end = match.span()
            # If there's a gap between the last match and the current one,
            # it means some characters were not recognized by any pattern.
            if start > last_end:
                skipped_text = expression[last_end:start]
                raise ValueError(f"Tokenizer error: Skipped untokenizable text '{skipped_text}' at index {last_end} in expression: '{expression}'")
            
            # Find which specific pattern group matched to get the token.
            for i in range(1, len(patterns) + 1):
                if match.group(i) is not None:
                    token = match.group(i)
                    if not token.isspace(): # Ignore whitespace tokens
                        tokens_found.append(token)
                    break
            last_end = end
        
        # After iterating through all matches, check if any trailing text was left untokenized.
        if last_end < len(expression):
            skipped_text = expression[last_end:]
            raise ValueError(f"Tokenizer error: Skipped untokenizable text '{skipped_text}' at end of expression: '{expression}'")

        return tokens_found

    def _peek_token(self) -> Optional[str]:
        """Returns the next token in the stream without advancing the token index.

        Useful for lookahead to decide parsing paths.

        Returns:
            The next token as a string, or None if at the end of the token stream.
        """
        if self.current_token_idx < len(self.tokens):
            return self.tokens[self.current_token_idx]
        return None

    def _consume_token(self) -> str:
        """Returns the current token and advances the token index.

        Effectively "eats" the current token from the stream.

        Returns:
            The consumed token as a string.

        Raises:
            ValueError: If attempting to consume a token past the end of the stream.
        """
        if self.current_token_idx < len(self.tokens):
            token = self.tokens[self.current_token_idx]
            self.current_token_idx += 1
            return token
        raise ValueError(f"Unexpected end of expression. Current token index: {self.current_token_idx}. Full tokens: {self.tokens}")

    def _expect_token(self, expected_token: str) -> None:
        """Consumes the next token and asserts that it matches the expected token.

        Args:
            expected_token: The string token that is expected next.

        Raises:
            ValueError: If the end of the expression is reached unexpectedly,
                        or if the actual token does not match the expected token.
        """
        try:
            actual_token = self._consume_token()
        except ValueError as e:
            # Provide a more informative error when end of expression is reached unexpectedly.
            raise ValueError(f"Expected '{expected_token}', but reached end of expression. "
                             f"Current token index: {self.current_token_idx}. "
                             f"Full tokens: {self.tokens}") from e
        if actual_token != expected_token:
            raise ValueError(f"Expected '{expected_token}', but got '{actual_token}' at token index {self.current_token_idx-1}. Full tokens: {self.tokens}")

    # --- Term Parsing ---
    # Defines the precedence of term operators. Higher numbers mean higher precedence.
    # Used in the Pratt parser algorithm for terms.
    term_precedence = {
        '^': 4, # Right-associative (e.g., 2^3^2 = 2^(3^2))
        '*': 3, '/': 3, # Multiplication and division
        '+': 2, '-': 2, # Addition and subtraction
        "'": 5 # Postfix derivative operator, highest precedence
    }
    # Specifies which operators are right-associative. Others are left-associative by default.
    term_right_associative = {'^'}

    def _parse_primary_term(self) -> Term:
        """Parses the most basic, indivisible components of a term expression.

        These include parenthesized expressions, numeric literals, variable references,
        and function calls.

        Returns:
            An AST node representing the parsed primary term.

        Raises:
            ValueError: If an unexpected token is encountered or the expression ends prematurely.
        """
        token = self._peek_token()
        if token is None:
            raise ValueError("Unexpected end of term expression: expected a primary term.")

        if token == '(':
            self._consume_token() # Consume '('
            term = self._parse_term_expression() # Recursively parse the inner expression
            self._expect_token(')') # Expect and consume ')'
            return term
        elif re.match(r'^\d+(\.\d+)?$', token):
            self._consume_token() # Consume the number token
            return Literal(float(token)) # Create a Literal AST node
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            name = self._consume_token() # Consume the identifier
            if self._peek_token() == '(': # Check if it's a function call
                self._consume_token() # Consume '('
                args = []
                if self._peek_token() != ')': # Check for arguments
                    while True:
                        args.append(self._parse_term_expression()) # Parse each argument
                        if self._peek_token() == ',':
                            self._consume_token() # Consume ',' if more arguments
                        else:
                            break # No more commas, end of argument list
                self._expect_token(')') # Expect and consume ')'
                return FunctionCall(name, args) # Create a FunctionCall AST node
            else: # It's a simple variable reference
                return VarRef(name) # Create a VarRef AST node
        else:
            # Enhanced error message for better debugging
            raise ValueError(f"Unexpected token in primary term: '{token}' (at index {self.current_token_idx}). "
                             f"Full token list: {self.tokens}. "
                             f"Context: {self.tokens[max(0, self.current_token_idx-3):self.current_token_idx+3]}")

    def _parse_unary_term(self) -> Term:
        """Parses unary operators that precede a term, such as unary minus.

        Returns:
            An AST node representing the parsed unary term or the primary term.
        """
        token = self._peek_token()
        if token == '-': # Unary minus
            self._consume_token() # Consume '-'
            operand = self._parse_unary_term() # Recursively parse the operand
            return UnaryTermOp('-', operand) # Create a UnaryTermOp AST node
        else:
            return self._parse_primary_term() # No unary operator, parse as primary term

    def _parse_term_expression(self, min_precedence=0) -> Term:
        """Parses a term expression using a Pratt parser (operator-precedence parsing).

        This method handles binary operators and postfix operators based on their
        defined precedence and associativity.

        Args:
            min_precedence: The minimum precedence level for operators to be considered
                            at the current parsing stage. This is key for handling
                            operator precedence and associativity.

        Returns:
            An AST node representing the parsed term expression.
        """
        # Start by parsing the left-hand side, which can be a unary or primary term.
        left = self._parse_unary_term()

        # Continuously look for operators with precedence greater than or equal to `min_precedence`.
        while self._peek_token() in self.term_precedence and \
              self.term_precedence[self._peek_token()] >= min_precedence:
            op = self._consume_token() # Consume the operator
            precedence = self.term_precedence[op]

            if op == "'": # Postfix derivative operator (e.g., x')
                left = UnaryTermOp(op, left) # Apply derivative to the left operand
                continue # Continue to check for more postfix operators or higher precedence
            
            # For binary operators, parse the right-hand side.
            if op in self.term_right_associative:
                # For right-associative operators (e.g., ^), parse with current precedence.
                right = self._parse_term_expression(precedence)
            else:
                # For left-associative operators (most common), parse with higher precedence
                # to ensure left-to-right grouping (e.g., 1+2+3 = (1+2)+3).
                right = self._parse_term_expression(precedence + 1)
            
            # Combine the left operand, operator, and right operand into a BinaryTermOp AST node.
            left = BinaryTermOp(op, left, right)
        return left

    def parse_term(self, expression: str, tokens: Optional[List[str]] = None) -> Term:
        """Public method to parse a standalone term expression string.

        This method is designed to be called for parsing term sub-expressions
        (e.g., from definitions or ODE equations) without affecting the main
        parser's state. It manages its own tokenization and token index.

        Args:
            expression: The string representing the term to parse.
            tokens: An optional pre-tokenized list of strings. If provided,
                    `expression` is ignored.

        Returns:
            An AST node representing the parsed term.

        Raises:
            ValueError: If parsing fails or unconsumed tokens remain.

        Example:
            Input: `expression = "a + b * c - d^2"`
            Output: `BinaryTermOp('-', BinaryTermOp('+', VarRef('a'), BinaryTermOp('*', VarRef('b'), VarRef('c'))), BinaryTermOp('^', VarRef('d'), Literal(2.0)))`
        """
        # Save the main parser's state to restore it later.
        original_tokens_state = (self.tokens, self.current_token_idx)
        
        if tokens is not None:
            self.tokens = tokens
            self.current_token_idx = 0
        else:
            # Clean comments from the expression before tokenizing it.
            expression_cleaned = re.sub(r'/\*.*?\*/', '', expression, flags=re.DOTALL)
            expression_cleaned = re.sub(r'//.*', '', expression_cleaned).strip()
            self.tokens = self._tokenize(expression_cleaned)
            self.current_token_idx = 0
        
        term = self._parse_term_expression()
        # Ensure all tokens for this specific term expression were consumed.
        if self.current_token_idx != len(self.tokens):
            raise ValueError(f"Unconsumed tokens after parsing term: {self.tokens[self.current_token_idx:]} from '{expression}'")
        
        # Restore the main parser's state.
        self.tokens, self.current_token_idx = original_tokens_state
        return term

    # --- Formula Parsing ---
    # Defines the precedence of formula operators. Higher numbers mean higher precedence.
    # Used in the Pratt parser algorithm for formulas.
    formula_precedence = {
        '<->': 1, # Equivalence (lowest precedence)
        '->': 2,  # Implication
        '|': 3,   # Logical OR
        '&': 4,   # Logical AND
        '/': 4,   # Also sometimes used for logical AND in some contexts
        '=': 5, '>': 5, '<': 5, '>=': 5, '<=': 5, '!=': 5, # Comparison operators
        '!': 6,   # Unary NOT
        '[': 7, '<': 7 # Modal operators (e.g., [hp]F, <hp>F), highest precedence as prefix
    }
    # Specifies which formula operators are right-associative.
    formula_right_associative = {'->', '<->'}

    def _parse_primary_formula(self) -> Formula:
        """Parses the most basic components of a formula expression.

        These include parenthesized formulas, boolean literals, modal formulas,
        and comparisons/predicate calls derived from terms.

        Returns:
            An AST node representing the parsed primary formula.

        Raises:
            ValueError: If an unexpected token is encountered or the expression ends prematurely.
        """
        token = self._peek_token()
        if token is None:
            raise ValueError("Unexpected end of formula expression: expected a primary formula.")

        if token == '(':
            self._consume_token() # Consume '('
            formula = self._parse_formula_expression() # Recursively parse inner formula
            self._expect_token(')') # Expect and consume ')'
            return formula
        elif token in ['true', 'false']:
            self._consume_token() # Consume 'true' or 'false'
            return BooleanLiteral(token == 'true') # Create BooleanLiteral AST node
        elif token == '[': # Box modality: [hp]F
            self._consume_token() # Consume '['
            
            # The content inside the modal operator is a hybrid program.
            program = self._parse_hp_choice_expression()
            
            self._expect_token(']') # Expect and consume ']'
            
            # The formula part that the modality applies to.
            formula_part = self._parse_unary_formula()
            return ModalFormula('box', program, formula_part) # Create ModalFormula AST node
        elif token == '<': # Diamond modality: <hp>F
            self._consume_token() # Consume '<'
            
            # The content inside the modal operator is a hybrid program.
            program = self._parse_hp_choice_expression()
            
            self._expect_token('>') # Expect and consume '>'
            
            formula_part = self._parse_unary_formula()
            return ModalFormula('diamond', program, formula_part) # Create ModalFormula AST node
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token) or re.match(r'^\d+(\.\d+)?$', token):
            # If it's an identifier or a number, it could be the start of a term
            # that forms a predicate or comparison. Parse it as a term first.
            left_term = self._parse_term_expression()

            op = self._peek_token()
            
            if op in ['=', '>', '<', '>=', '<=', '!=']:
                # If a comparison operator follows, it's a comparison formula.
                self._consume_token() # Consume the comparison operator
                right_term = self._parse_term_expression() # Parse the right-hand term
                return Comparison(op, left_term, right_term) # Create Comparison AST node
            else:
                # If no comparison operator follows, `left_term` must represent a boolean value
                # (i.e., a predicate call or a boolean variable).
                if isinstance(left_term, VarRef):
                    # A bare identifier (e.g., 'P') is treated as a 0-ary predicate (boolean variable).
                    return PredicateCall(left_term.name, [])
                elif isinstance(left_term, FunctionCall):
                    # A function call (e.g., 'P(x,y)') without a comparison implies it's a predicate call.
                    return PredicateCall(left_term.name, left_term.args)
                else:
                    # A numerical literal or a complex term (e.g., '5', 'x+y')
                    # without a comparison operator is an error in a formula context.
                    raise ValueError(f"Unexpected numerical term '{left_term}' without a comparison operator in primary formula. Current token: '{self._peek_token()}', idx: {self.current_token_idx}")
        else:
            # If the token is not any recognized primary formula start, it's an error.
            raise ValueError(f"Unexpected token at start of primary formula: '{token}' (at index {self.current_token_idx})")

    def _parse_unary_formula(self) -> Formula:
        """Parses unary logical NOT (`!`).

        Returns:
            An AST node representing the parsed unary formula or the primary formula.
        """
        token = self._peek_token()
        if token == '!':
            self._consume_token() # Consume '!'
            operand = self._parse_unary_formula() # Recursively parse the operand
            return UnaryFormulaOp('!', operand) # Create UnaryFormulaOp AST node
        else:
            return self._parse_primary_formula() # No unary operator, parse as primary formula

    def _parse_formula_expression(self, min_precedence=0) -> Formula:
        """Parses a formula expression using a Pratt parser (operator-precedence parsing).

        This method handles quantifiers, binary logical operators, and unary operators
        based on their defined precedence and associativity.

        Args:
            min_precedence: The minimum precedence level for operators to be considered
                            at the current parsing stage.

        Returns:
            An AST node representing the parsed formula expression.
        """
        # First, check for quantifiers, as they bind to the entire subsequent formula.
        token = self._peek_token()
        if token in ['forall', 'exists']:
            quantifier_type = self._consume_token() # Consume 'forall' or 'exists'
            self._expect_token('(') # Expect '(' for variable list
            variables: List[VarRef] = []
            while self._peek_token() != ')':
                # Quantified variables are typically just identifiers (e.g., 'x', 'y').
                var_name = self._consume_token()
                variables.append(VarRef(var_name))
                if self._peek_token() == ',':
                    self._consume_token() # Consume ',' if more variables
                else:
                    break # End of variable list
            self._expect_token(')') # Expect ')' after variable list
            self._expect_token('.') # Expect '.' separating variables from formula body
            body_formula = self._parse_formula_expression() # Recursively parse the formula under the quantifier
            return Quantifier(quantifier_type, variables, body_formula) # Create Quantifier AST node
        
        # If no quantifier, proceed with standard formula parsing (unary and binary operators).
        left = self._parse_unary_formula()

        # Continuously look for binary operators with precedence greater than or equal to `min_precedence`.
        while self._peek_token() in self.formula_precedence and \
              self.formula_precedence[self._peek_token()] >= min_precedence:
            op = self._consume_token() # Consume the operator
            precedence = self.formula_precedence[op]

            # For binary operators, parse the right-hand side.
            if op in self.formula_right_associative:
                # For right-associative operators (e.g., ->, <->), parse with current precedence.
                right = self._parse_formula_expression(precedence)
            else:
                # For left-associative operators, parse with higher precedence.
                right = self._parse_formula_expression(precedence + 1)
            
            # Combine the left operand, operator, and right operand into a BinaryFormulaOp AST node.
            left = BinaryFormulaOp(op, left, right)
        return left

    def parse_formula(self, expression: str, tokens: Optional[List[str]] = None) -> Formula:
        """Public method to parse a standalone formula expression string.

        Similar to `parse_term`, this method manages its own tokenization and
        token index, allowing it to parse sub-expressions independently.

        Args:
            expression: The string representing the formula to parse.
            tokens: An optional pre-tokenized list of strings. If provided,
                    `expression` is ignored.

        Returns:
            An AST node representing the parsed formula.

        Raises:
            ValueError: If the formula content is empty, parsing fails, or
                        unconsumed tokens remain.

        Example:
            Input: `expression = "forall(x). x >= 0"`
            Output: `Quantifier('forall', [VarRef('x')], Comparison('>=', VarRef('x'), Literal(0.0)))`
        """
        # Save the main parser's state to restore it later.
        original_tokens_state = (self.tokens, self.current_token_idx)
        
        if tokens is not None:
            self.tokens = tokens
            self.current_token_idx = 0
        else:
            # Clean comments from the expression before tokenizing it.
            expression_cleaned = re.sub(r'/\*.*?\*/', '', expression, flags=re.DOTALL)
            expression_cleaned = re.sub(r'//.*', '', expression_cleaned).strip()
            self.tokens = self._tokenize(expression_cleaned)
            self.current_token_idx = 0
        
        # Check for empty token list after tokenization, which indicates empty or unparseable input.
        if not self.tokens and (expression or tokens):
            raise ValueError(f"Formula content is empty or contains only untokenizable characters: '{expression}'")

        formula = self._parse_formula_expression()
        # Ensure all tokens for this specific formula expression were consumed.
        if self.current_token_idx != len(self.tokens):
            raise ValueError(f"Unconsumed tokens after parsing formula: {self.tokens[self.current_token_idx:]} from '{expression}'")
        
        # Restore the main parser's state.
        self.tokens, self.current_token_idx = original_tokens_state
        return formula

    # --- Hybrid Program Parsing ---
    
    def _parse_ode_block_content(self, ode_tokens: List[str]) -> ODE:
        """Parses the content inside an ODE block `{...}`.

        This helper function takes a list of tokens representing the ODE block's
        interior and extracts ODE equations and domain constraints.

        Args:
            ode_tokens: A list of tokens representing the content within the ODE block.

        Returns:
            An ODE AST node containing the parsed equations and domain constraint.

        Raises:
            ValueError: If the ODE block content is empty or unparseable.

        Example:
            Input: `ode_tokens = ['x', "'", '=', 'v', ',', 't', '<=', 'ep']` (representing `x' = v, t <= ep`)
            Output: `ODE(equations=[ODEEquation(VarRef('x'), VarRef('v'))], domain_constraint=Comparison('<=', VarRef('t'), VarRef('ep')))`
        """
        equations: List[ODEEquation] = []
        domain_constraint: Optional[Formula] = None

        # Create a temporary parser instance to manage the token stream for this ODE block.
        # This allows independent parsing of the ODE content without affecting the main parser's state.
        temp_parser = KeYmaeraParser("")
        temp_parser.tokens = ode_tokens
        temp_parser.current_token_idx = 0

        current_part_tokens = []
        parts_to_parse_token_lists = [] # Stores lists of tokens for each equation/constraint part
        
        balance = 0 # Tracks parenthesis/brace/bracket balance to correctly split parts
        while temp_parser._peek_token() is not None:
            token = temp_parser._peek_token()
            if token in ['(', '[', '{']:
                balance += 1
            elif token in [')', ']', '}']:
                balance -= 1

            # Split by ';', '&', or ',' only if outside of any nested parentheses, brackets, or braces.
            if (token in [';', '&', ',']) and balance == 0:
                if current_part_tokens:
                    parts_to_parse_token_lists.append(current_part_tokens)
                    current_part_tokens = []
                temp_parser._consume_token() # Consume the separator
            else:
                current_part_tokens.append(temp_parser._consume_token())
        # Add any remaining tokens as the last part
        if current_part_tokens:
            parts_to_parse_token_lists.append(current_part_tokens)

        for part_token_list in parts_to_parse_token_lists:
            part_str = ' '.join(part_token_list).strip()
            if not part_str:
                continue

            # Check for an ODE equation: `var' = expr`
            eq_match = re.match(r"(\w+)\s*'\s*=\s*(.+)", part_str)
            if eq_match:
                var_name = eq_match.group(1)
                expr_str = eq_match.group(2).strip()
                # Use the main parser's `parse_term` method, which handles tokenization internally.
                equations.append(ODEEquation(VarRef(var_name), self.parse_term(expr_str)))
            else:
                # If it's not an equation, it must be a domain constraint (a formula).
                new_domain_formula = self.parse_formula(part_str)
                if domain_constraint is not None:
                    # If multiple domain constraints are found, combine them with logical AND.
                    domain_constraint = BinaryFormulaOp('&', domain_constraint, new_domain_formula)
                else:
                    domain_constraint = new_domain_formula
        
        # An ODE block must contain at least one equation or a domain constraint.
        if not equations and domain_constraint is None:
             raise ValueError(f"Empty or unparseable ODE block content: '{' '.join(ode_tokens)}'")

        return ODE(equations, domain_constraint)


    def _parse_hp_atom(self) -> HybridProgram:
        """Parses the smallest, atomic units of a hybrid program.

        These include ODE blocks, HP blocks, test statements, assignments,
        and references to defined hybrid programs. It also handles iteration
        (`*`) and its invariant.

        Returns:
            An AST node representing the parsed hybrid program atom.

        Raises:
            ValueError: If an unexpected token is encountered or delimiters are unmatched.
        """
        token = self._peek_token()
        if token is None:
            raise ValueError("Unexpected end of hybrid program: expected an atom.")

        if token == '{': # ODE or HP Block: `{...}`
            self._consume_token() # Consume the opening '{'

            # Find the corresponding closing '}' for this block.
            # This is done by scanning tokens without consuming them from the main stream,
            # to determine the extent of the inner content.
            temp_idx_scan = self.current_token_idx
            brace_balance_scan = 1
            inner_block_end_token_idx = -1 # Index of the closing '}'

            while temp_idx_scan < len(self.tokens):
                t_scan = self.tokens[temp_idx_scan]
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

            # Extract the tokens for the content *inside* the braces.
            inner_content_tokens = self.tokens[self.current_token_idx:inner_block_end_token_idx]
            
            # Advance the main parser's index past the inner content and its closing '}'.
            self.current_token_idx = inner_block_end_token_idx + 1

            # Heuristic to distinguish between an ODE block and a general HP block:
            # Look for patterns like `var' = expr` or `t` followed by comparison operators.
            is_ode = False
            for i in range(len(inner_content_tokens)):
                if inner_content_tokens[i] == '=':
                    if i > 0 and inner_content_tokens[i-1] == "'":
                        is_ode = True
                        break
                if inner_content_tokens[i] == 't' and i + 1 < len(inner_content_tokens) and inner_content_tokens[i+1] in ['<=', '>=', '<', '>']:
                    is_ode = True
                    break

            if is_ode:
                return self._parse_ode_block_content(inner_content_tokens)

            # If not an ODE, parse the content as a regular hybrid program.
            # A temporary parser is used to parse this nested HP content.
            temp_parser_for_inner_hp = KeYmaeraParser("") 
            temp_parser_for_inner_hp.tokens = inner_content_tokens
            temp_parser_for_inner_hp.current_token_idx = 0
            
            parsed_inner_hp_content = temp_parser_for_inner_hp._parse_hp_choice_expression()

            # After parsing the block, check for an iteration operator (`*`).
            if self._peek_token() == '*':
                self._consume_token() # Consume '*'
                invariant = None
                if self._peek_token() == '@invariant': # Check for an optional invariant
                    self._consume_token() # Consume '@invariant'
                    self._expect_token('(') # Expect '(' for invariant formula
                    
                    # Find the closing ')' for the invariant formula.
                    inv_start_idx = self.current_token_idx
                    paren_balance_inv = 1
                    temp_idx_inv_scan = self.current_token_idx
                    inv_end_token_idx = -1

                    while temp_idx_inv_scan < len(self.tokens):
                        t_scan = self.tokens[temp_idx_inv_scan]
                        if t_scan == '(':
                            paren_balance_inv += 1
                        elif t_scan == ')':
                            paren_balance_inv -= 1
                        
                        if paren_balance_inv == 0:
                            inv_end_token_idx = temp_idx_inv_scan
                            break
                        temp_idx_inv_scan += 1
                    
                    if inv_end_token_idx == -1:
                        raise ValueError("Unexpected end of expression: Unmatched '(' in @invariant.")

                    invariant_tokens = self.tokens[inv_start_idx:inv_end_token_idx]
                    self.current_token_idx = inv_end_token_idx + 1 # Advance main parser past ')'

                    # Parse the invariant formula using a temporary parser.
                    invariant = KeYmaeraParser("").parse_formula("", tokens=invariant_tokens)
                
                # The Iteration wraps the HPBlock, and the HPBlock wraps the parsed_inner_hp_content.
                return Iteration(HPBlock(parsed_inner_hp_content), invariant)
            
            return HPBlock(parsed_inner_hp_content) # If no iteration, just return the HPBlock

        elif token == '?': # Test statement: `?(F)`
            self._consume_token() # Consume '?'
            self._expect_token('(') # Expect opening '('
            
            # Find the closing ')' for the condition formula.
            cond_start_idx = self.current_token_idx
            paren_balance_cond = 1
            temp_idx_cond_scan = self.current_token_idx
            cond_end_token_idx = -1

            while temp_idx_cond_scan < len(self.tokens):
                t_scan = self.tokens[temp_idx_cond_scan]
                if t_scan == '(':
                    paren_balance_cond += 1
                elif t_scan == ')':
                    paren_balance_cond -= 1
                
                if paren_balance_cond == 0:
                    cond_end_token_idx = temp_idx_cond_scan
                    break
                temp_idx_cond_scan += 1
            
            if cond_end_token_idx == -1:
                raise ValueError("Unexpected end of expression: Unmatched '(' in Test condition.")

            condition_tokens = self.tokens[cond_start_idx:cond_end_token_idx]
            self.current_token_idx = cond_end_token_idx + 1 # Advance main parser past ')'

            # Parse the condition formula using a temporary parser.
            condition = KeYmaeraParser("").parse_formula("", tokens=condition_tokens)
            return Test(condition)

        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token): # Assignment or HP reference
            var_name = self._consume_token() # Consume the identifier
            if self._peek_token() == ':=': # If followed by ':=', it's an assignment
                self._consume_token() # Consume ':='
                
                # Collect tokens for the expression until a separator or end of HP.
                expr_tokens = []
                balance = 0 # Track balance for nested structures within the expression
                # Stop collecting if a sequence separator (';'), choice operator ('++'),
                # or modal closing bracket (']', '>') is encountered at the top level.
                while self._peek_token() and self._peek_token() not in [';', ']', '>', '++'] and balance >= 0:
                    t = self._peek_token()
                    if t in ['(', '[', '{']:
                        balance += 1
                    elif t in [')', ']', '}']:
                        balance -= 1
                    
                    if balance < 0: # Unmatched closing delimiter indicates end of expression
                        break
                    
                    expr_tokens.append(self._consume_token())
                
                expr_str = ' '.join(expr_tokens)
                # Parse the right-hand side as a term.
                return Assignment(VarRef(var_name), self.parse_term(expr_str))
            else: # Otherwise, it's a reference to a defined hybrid program (e.g., 'dwwpdl')
                return VarRef(var_name)

        else:
            raise ValueError(f"Unexpected token in HP atom: '{token}'")

    def _parse_hp_sequence_expression(self) -> HybridProgram:
        """Parses a sequence of hybrid program atoms (e.g., `A; B; C` or `A B C`).

        Returns:
            An AST node representing the parsed HP sequence (or a single HP if no sequence).
        """
        programs: List[HybridProgram] = []
        
        # Parse the first HP atom.
        programs.append(self._parse_hp_atom())

        # Continue parsing HP atoms as long as the next token is not a choice operator ('++'),
        # a modal closing bracket (']', '>'), or the end of the expression.
        while True:
            next_token = self._peek_token()
            # If end of tokens, or a choice operator, or a modal closing bracket, stop.
            if next_token is None or next_token in ['++', ']', '>']:
                break 
            
            if next_token == ';':
                self._consume_token() # Consume the semicolon separator
                # If a semicolon is followed by nothing or a choice/modal op, break.
                # Otherwise, continue to parse the next atom.
                if self._peek_token() is None or self._peek_token() in ['++', ']', '>']:
                    break
            
            # If we reach here, it means there's another HP atom to parse,
            # either implicitly (separated by space/newline) or explicitly after a semicolon.
            programs.append(self._parse_hp_atom())
        
        if len(programs) == 1:
            return programs[0] # If only one program, return it directly
        else:
            return Sequence(programs) # Otherwise, wrap them in a Sequence node

    def _parse_hp_choice_expression(self) -> HybridProgram:
        """Parses a choice expression of hybrid programs (e.g., `A ++ B ++ C`).

        The `++` operator is right-associative.

        Returns:
            An AST node representing the parsed HP choice.
        """
        # A choice expression starts with a sequence.
        left_hp = self._parse_hp_sequence_expression()

        if self._peek_token() == '++':
            self._consume_token() # Consume '++'
            # Recursively parse the right-hand side, making '++' right-associative.
            right_hp = self._parse_hp_choice_expression()
            return Choice(left_hp, right_hp) # Create a Choice AST node
        
        return left_hp # If no '++', return the single HP sequence

    def parse_hybrid_program(self, expression: str, tokens: Optional[List[str]] = None) -> HybridProgram:
        """Public method to parse a standalone hybrid program string.

        Similar to `parse_term` and `parse_formula`, this method manages its
        own tokenization and token index for parsing HP sub-expressions.

        Args:
            expression: The string representing the hybrid program to parse.
            tokens: An optional pre-tokenized list of strings. If provided,
                    `expression` is ignored.

        Returns:
            An AST node representing the parsed hybrid program.

        Raises:
            ValueError: If the hybrid program content is empty, parsing fails,
                        or unconsumed tokens remain.

        Example:
            Input: `expression = "{x := x + 1}*@invariant(x < 10)"`
            Output: `Iteration(HPBlock(Assignment(VarRef('x'), BinaryTermOp('+', VarRef('x'), Literal(1.0)))), Comparison('<', VarRef('x'), Literal(10.0)))`
        """
        # Save the main parser's state to restore it later.
        original_tokens_state = (self.tokens, self.current_token_idx)
        
        if tokens is not None:
            self.tokens = tokens
            self.current_token_idx = 0
        else:
            # Clean comments from the expression before tokenizing it.
            expression_cleaned = re.sub(r'/\*.*?\*/', '', expression, flags=re.DOTALL)
            expression_cleaned = re.sub(r'//.*', '', expression_cleaned).strip()
            self.tokens = self._tokenize(expression_cleaned)
            self.current_token_idx = 0
        
        # Check for empty token list after tokenization.
        if not self.tokens and (expression or tokens):
            raise ValueError(f"Hybrid program content is empty or contains only untokenizable characters: '{expression}'")

        hp = self._parse_hp_choice_expression()
        # Ensure all tokens for this specific hybrid program expression were consumed.
        if self.current_token_idx != len(self.tokens):
            raise ValueError(f"Unconsumed tokens after parsing hybrid program: {self.tokens[self.current_token_idx:]} from '{expression}'")
        
        # Restore the main parser's state.
        self.tokens, self.current_token_idx = original_tokens_state
        return hp

    def parse(self) -> KeYmaeraModel:
        """Parses the entire KeYmaera model text into a KeYmaeraModel AST.

        This is the top-level entry point for parsing a complete model file.
        It identifies and parses the main sections: Theorem, Definitions,
        ProgramVariables, and Problem.

        Returns:
            A KeYmaeraModel AST node representing the parsed model.

        Raises:
            ValueError: If required sections are missing, theorem name is not found,
                        or any definition/declaration is malformed.

        Example:
            Input (simplified for brevity, full example would be a complete .key file content):
            ```
            Theorem "MyTheorem"
            Definitions
                Real x;
                Bool P(y) <-> y > 0;
                HP my_hp ::= {x := x + 1};
            End.
            ProgramVariables
                Real x;
            End.
            Problem
                P(x) -> [my_hp]P(x);
            End.
            ```
            Output (simplified AST representation):
            ```
            KeYmaeraModel([
                ArchiveEntry(
                    kind='Theorem',
                    name='MyTheorem',
                    definitions=DefinitionsBlock(defs=[
                        FuncDef(name='x', type='Real', body=VarRef('x'), args=[], annotations={}),
                        PredDef(name='P', type='Bool', body=Comparison('>', VarRef('y'), Literal(0.0)), args=[VarRef('y')], annotations={}),
                        HPDef(name='my_hp', body=HPBlock(Assignment(VarRef('x'), BinaryTermOp('+', VarRef('x'), Literal(1.0)))), annotations={})
                    ]),
                    program_variables=ProgramVariablesBlock(decls=[
                        ProgramVariableDecl(type='Real', name='x', annotations={})
                    ]),
                    problem=ProblemBlock(formula=BinaryFormulaOp('->',
                        PredicateCall('P', [VarRef('x')]),
                        ModalFormula('box', VarRef('my_hp'), PredicateCall('P', [VarRef('x')]))
                    )),
                    tactics=[]
                )
            ])
            ```
        """
        entries = [] # Placeholder for multiple archive entries, though typically one per file.
        
        # 1. Extract Theorem Metadata
        theorem_match = re.search(r'Theorem\s+"([^"]+)"', self.text)
        if not theorem_match:
            raise ValueError("Theorem name not found in the model file.")
        
        full_theorem_string_in_quotes = theorem_match.group(1)
        
        # Default values for kind and name.
        kind = "Theorem"
        name = full_theorem_string_in_quotes.strip()
        
        # Check for an explicit "Kind/Name" prefix (e.g., "Invariant/MyTheorem")
        kind_prefix_match = re.match(r'(\w+)/(.+)', full_theorem_string_in_quotes)
        if kind_prefix_match:
            kind = kind_prefix_match.group(1)
            name = kind_prefix_match.group(2).strip()
        # If no prefix, 'kind' remains "Theorem" and 'name' is the full string.
        
        # 2. Locate and Extract Raw Text for Main Sections
        # Use re.DOTALL to allow '.' to match newlines, capturing multi-line sections.
        definitions_section_match = re.search(r'Definitions\s*(.*?)\s*End\.', self.text, re.DOTALL)
        program_variables_section_match = re.search(r'ProgramVariables\s*(.*?)\s*End.', self.text, re.DOTALL)
        # The Problem section regex is refined to ensure it ends exactly with "\nEnd.\n"
        # to prevent it from greedily consuming subsequent sections if they exist.
        problem_section_match = re.search(r'Problem\s*(.*?)\s*\nEnd\.\n', self.text, re.DOTALL)

        if not (definitions_section_match and program_variables_section_match and problem_section_match):
            raise ValueError("Missing 'Definitions', 'ProgramVariables', or 'Problem' sections in the model file.")

        raw_definitions_text = definitions_section_match.group(1)
        raw_program_variables_text = program_variables_section_match.group(1)
        
        # The problem text needs to be fully cleaned of comments before parsing as a formula.
        raw_problem_text_with_comments = problem_section_match.group(1)
        problem_text_without_c_comments = re.sub(r'/\*.*?\*/', '', raw_problem_text_with_comments, flags=re.DOTALL)
        problem_text = re.sub(r'//.*', '', problem_text_without_c_comments).strip()


        # 3. Parse Definitions Section
        definitions_list: List[Definition] = []
        current_annotations: Dict[str, Any] = {} # Stores annotations for the next definition
        
        def_lines = raw_definitions_text.splitlines()
        i = 0
        while i < len(def_lines):
            line = def_lines[i].strip()
            
            # Check for annotation comments preceding a definition.
            comment_match = re.match(r'//\s*@(\w+):.*', line)
            if comment_match:
                current_annotations = parse_annotations(line)
                i += 1
                continue # Move to the next line to find the actual definition
            
            # Clean C-style (/* ... */) and single-line (//) comments from the current line
            # before checking if it's empty or calculating balance.
            line_cleaned_c_style = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL) 
            line_without_comments = re.sub(r'//.*', '', line_cleaned_c_style).strip()

            if not line_without_comments:
                i += 1
                continue # Skip empty lines

            # --- Accumulate Full Multi-line Definition String ---
            # Definitions can span multiple lines. Accumulate them until a semicolon
            # is found and all parentheses/braces are balanced.
            full_definition_str_builder = [line_without_comments]
            
            paren_balance = line_without_comments.count('(') - line_without_comments.count(')')
            brace_balance = line_without_comments.count('{') - line_without_comments.count('}')
            
            # A definition is considered terminated if it ends with a semicolon AND
            # all delimiters (parentheses, braces) are balanced.
            is_terminated = line_without_comments.endswith(';') and paren_balance == 0 and brace_balance == 0

            current_line_idx_for_def = i + 1
            while not is_terminated and current_line_idx_for_def < len(def_lines):
                next_line = def_lines[current_line_idx_for_def].strip()
                
                # Clean comments from the next line before processing.
                next_line_cleaned_c_style = re.sub(r'/\*.*?\*/', '', next_line, flags=re.DOTALL) 
                next_line_without_comments = re.sub(r'//.*', '', next_line_cleaned_c_style).strip()
                
                if not next_line_without_comments:
                    current_line_idx_for_def += 1
                    continue # Skip empty continuation lines

                full_definition_str_builder.append(next_line_without_comments)
                
                # Update balance for delimiters across lines.
                paren_balance += next_line_without_comments.count('(') - next_line_without_comments.count(')')
                brace_balance += next_line_without_comments.count('{') - next_line_without_comments.count('}')
                
                is_terminated = next_line_without_comments.endswith(';') and paren_balance == 0 and brace_balance == 0
                current_line_idx_for_def += 1
            
            if not is_terminated:
                raise ValueError(f"Unterminated definition starting at line {i+1}: '{line}'")

            full_definition_str = ' '.join(full_definition_str_builder).strip()
            
            # --- Parse the complete definition string ---
            # Variable declaration (e.g., `Real x;` or `Bool P;`)
            if var_decl_match := re.match(r'(Real|Bool)\s+(\w+)\s*;', full_definition_str):
                var_type = var_decl_match.group(1)
                var_name = var_decl_match.group(2)
                if var_type == "Real":
                    # For a simple Real variable declaration, it's a FuncDef where the body is just a VarRef.
                    definitions_list.append(FuncDef(var_name, "Real", VarRef(var_name), [], current_annotations.copy()))
                else: # Bool
                    # For a simple Bool variable declaration, it's a PredDef where the body is a PredicateCall.
                    definitions_list.append(PredDef(var_name, "Bool", PredicateCall(var_name, []), [], current_annotations.copy()))
            # Function definition (e.g., `Real f(x) = x^2;`)
            elif func_def_match := re.match(r'Real\s+(\w+)\s*(\(([^)]*)\))?\s*=\s*(.*);', full_definition_str):
                def_name = func_def_match.group(1)
                args_str = func_def_match.group(3) # Captured arguments string (e.g., "x,y")
                body_str = func_def_match.group(4).strip()
                
                args = []
                if args_str:
                    individual_args_with_types = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                    for arg_with_type in individual_args_with_types:
                        # Extract only the variable name, discarding optional type (e.g., "Real x" -> "x").
                        arg_name_match = re.match(r'(?:Real|Bool)?\s*(\w+)', arg_with_type)
                        if arg_name_match:
                            args.append(VarRef(arg_name_match.group(1)))
                        else:
                            raise ValueError(f"Could not parse argument '{arg_with_type}' in function definition '{def_name}'.")
                
                body_term = self.parse_term(body_str) # Parse the function body as a term.
                definitions_list.append(FuncDef(def_name, "Real", body_term, args, current_annotations.copy()))
            # Predicate definition (e.g., `Bool P(x) <-> x > 0;`)
            elif pred_def_match := re.match(r'Bool\s+(\w+)\s*(\(([^)]*)\))?\s*<->\s*(.*);', full_definition_str):
                def_name = pred_def_match.group(1)
                args_str = pred_def_match.group(3)
                body_str = pred_def_match.group(4).strip()

                args = []
                if args_str:
                    individual_args_with_types = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                    for arg_with_type in individual_args_with_types:
                        # Extract only the variable name.
                        arg_name_match = re.match(r'(?:Real|Bool)?\s*(\w+)', arg_with_type)
                        if arg_name_match:
                            args.append(VarRef(arg_name_match.group(1)))
                        else:
                            raise ValueError(f"Could not parse argument '{arg_with_type}' in predicate definition '{def_name}'.")
                
                body_formula = self.parse_formula(body_str) # Parse the predicate body as a formula.
                definitions_list.append(PredDef(def_name, "Bool", body_formula, args, current_annotations.copy()))
            # Hybrid Program definition (e.g., `HP dwwpdl ::= {x'=v, v'=-g};`)
            elif hp_def_match := re.match(r'HP\s+(\w+)\s*::=\s*(.*);', full_definition_str):
                def_name = hp_def_match.group(1)
                body_str = hp_def_match.group(2).strip()
                
                body_hp = self.parse_hybrid_program(body_str) # Parse the HP body as a hybrid program.
                definitions_list.append(HPDef(def_name, body_hp, annotations=current_annotations.copy()))
            else:
                raise ValueError(f"Unrecognized definition pattern for: '{full_definition_str}' (starting line {i+1})")
            
            # Reset annotations for the next definition and advance the loop counter.
            current_annotations = {}
            i = current_line_idx_for_def # Advance 'i' past the consumed multi-line definition
        
        definitions_block = DefinitionsBlock(definitions_list)

        # 4. Parse ProgramVariables Section
        program_vars_list: List[ProgramVariableDecl] = []
        current_annotations = {} # Stores annotations for the next program variable
        pv_lines = raw_program_variables_text.splitlines()
        for line in pv_lines:
            line = line.strip()
            # Check for annotation comments.
            comment_match = re.match(r'//\s*@(\w+):.*', line)
            if comment_match:
                current_annotations = parse_annotations(line)
                continue
            
            # Clean comments from the line before parsing.
            line_for_parsing = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
            line_for_parsing = re.sub(r'//.*', '', line_for_parsing).strip()

            if not line_for_parsing:
                continue # Skip empty lines

            # Program variable declaration (e.g., `Real x;` or `Bool P;`)
            var_decl_match = re.match(r'(Real|Bool)\s+(\w+)\s*;', line_for_parsing)
            if var_decl_match:
                var_type = var_decl_match.group(1)
                var_name = var_decl_match.group(2)
                program_vars_list.append(ProgramVariableDecl(var_type, var_name, current_annotations.copy()))
                current_annotations = {} # Reset annotations
            elif line_for_parsing:
                raise ValueError(f"Unrecognized program variable declaration: '{line}'")
        
        program_variables_block = ProgramVariablesBlock(program_vars_list)

        # 5. Parse Problem Section
        # The problem_text was already fully cleaned of comments at the beginning of the parse method.
        problem_formula = self.parse_formula(problem_text) # Parse the main problem statement as a formula.
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
        
        # A KeYmaera model can contain multiple archive entries, but typically one per file.
        return KeYmaeraModel([archive_entry])


if __name__ == "__main__":
    # This section demonstrates how to use the KeYmaeraParser and provides
    # a comprehensive example for breakpoint debugging and understanding the code flow.

    print("--- Starting KeYmaera Parser Test Section ---")

    # --- Test parse_annotations helper function ---
    print("\nTesting parse_annotations...")
    annotation_line1 = '// @const: bound=[0.1:0.5]; def="time limit"; is_active=true'
    parsed_anno1 = parse_annotations(annotation_line1)
    print(f"Input: '{annotation_line1}'")
    print(f"Output: {parsed_anno1}")
    assert parsed_anno1 == {'const': {'bound': [0.1, 0.5], 'def': 'time limit', 'is_active': True}}

    annotation_line2 = '// @param: value=10, type=int'
    parsed_anno2 = parse_annotations(annotation_line2)
    print(f"Input: '{annotation_line2}'")
    print(f"Output: {parsed_anno2}")
    assert parsed_anno2 == {'param': {'value': 10, 'type': 'int'}}

    annotation_line3 = '// Not an annotation'
    parsed_anno3 = parse_annotations(annotation_line3)
    print(f"Input: '{annotation_line3}'")
    print(f"Output: {parsed_anno3}")
    assert parsed_anno3 == {}
    print("parse_annotations tests passed.")

    # --- Sample KeYmaera Model Text for comprehensive parsing ---
    sample_keymaera_model_text = """
    Theorem "IJRR17/Theorem 14: Reach waypoint with deadline"

/*
 *	Robot must stop within distance delta at goal.
 *
 *  Robot
 * 	- must stop within distance delta of goal
 *  - can only drive straight and forward
 *  - ensures progress towards goal
 *
 *  Liveness property:
 *  - Robot can stop at goal
 *
 */

Definitions
    // @const: bound=[0.1:0.5]; def="time limit for control decisions"
    Real ep;      /* time limit for control decisions */
    // @const: bound=[0.5:5.0]; def="minimum braking capability of the robot"
    Real b;       /* minimum braking capability of the robot */
    // @const: bound=[1.0:10.0]; def="maximum acceleration"
    Real A;       /* maximum acceleration -b <= a <= A */
    // @const: bound=[1.0:10.0]; def="goal area size"
    Real GDelta;	 /* goal area size */
    // @const: bound=[5.0:30.0]; def="maximum robot velocity"
    Real Vmax;    /* robot cannot go faster than this */

    Real waypointStartDist(Real xg) = xg-GDelta;
    Real waypointEndDist(Real xg)   = xg+GDelta;

    Real minV = A*ep;
    Real maxTravelTime(Real g) = waypointStartDist(g)/minV;
    Real stopTime(Real v)      = v/b;
    Real speedUpTime(Real v)   = ep-v/A;

    Real stopDist(Real v) = v^2/(2*b);
    Real accComp(Real v)  = ( (A/b + 1)*(A/2*ep^2 + ep*v) );

    Bool bounds() <-> (                                                                 
        A > 0                                                                                         /* Working engine */
        & b > 0                                                                                         /* Working brakes */
        & ep > 0                                                                              /* Controller reaction time */
        & Vmax >= 2*A*ep
        & GDelta > Vmax*ep + Vmax^2/(2*b)  /* waypoint is large enough that robot can start driving and still stop inside */
    );

    Bool initialState(Real xr, Real vr, Real xg, Real T) <-> (                      
        vr = 0
        & xr < waypointStartDist(xg)
        & T > ep + maxTravelTime(xg-xr) + ep + stopTime(Vmax)
    );

    Bool assumptions(Real xr, Real vr, Real xg, Real T) <->              
        bounds() & initialState(xr, vr, xg, T);

    // @safety: penalty=-1000, on_failure=truncate:status=failure
    Bool loopinv(Real xr, Real vr, Real xg, Real T) <-> ( // Added T to parameters as it's used in the invariant
        0 <= vr & vr <= Vmax & xr + stopDist(vr) < waypointEndDist(xg)
                & (waypointStartDist(xg) < xr -> (vr = 0 | T >= stopTime(vr)))
                & (xr <= waypointStartDist(xg) ->
                    (vr >= minV & T > maxTravelTime(xg-xr) + ep + stopTime(Vmax))
                    | (vr <= minV & T > speedUpTime(vr) + maxTravelTime(xg-xr) + ep + stopTime(Vmax)))
    );

    HP controlFunc ::= {
        // Choice 1: If in the goal area, decide to brake
        { ? (xr > waypointStartDist(xg));  ar := -b; } 
        ++
        // Choice 2: If in the goal area AND already stopped, decide to stay stopped
        { ? (xr > waypointStartDist(xg) & vr = 0);  ar := 0; }
        ++
        // Choice 3: If outside the goal area AND can accelerate, decide to accelerate
        { ? (xr <= waypointStartDist(xg) & (xr + stopDist(vr) + accComp(vr) < waypointEndDist(xg) & vr+A*ep <= Vmax));  ar := A; }
        ++
        // Choice 4: If outside the goal area AND cannot accelerate (must coast), decide to coast
        { ? (xr <= waypointStartDist(xg) & !(xr + stopDist(vr) + accComp(vr) < waypointEndDist(xg) & vr+A*ep <= Vmax));  ar := 0; }
    };

    HP stepFunc ::= {xr' = vr, vr' = ar, t' = 1, T'=-1 & t <= ep & vr >= 0};

    HP dwwpdl ::= {
        {	{ controlFunc; t := 0; }
            { stepFunc; }
        }*@invariant(loopinv(xr, vr, xg, T))
    };

    // @progress: reward=1000, on_success=terminate:status=success, on_failure=terminate:status=failure
    Bool ensure(Real xr, Real vr, Real xg) <-> (
        waypointStartDist(xg) < xr & vr = 0
    );

End.

ProgramVariables
    // @var: bound=[0:100], def="robot position: x"
    Real xr;
    // @var: bound=[0:20], def="robot translational velocity"
    Real vr;
    // @var: bound=[-10:10], def="robot translational acceleration"
    Real ar;
    // @var: bound=[0:100], def="goal position: x"
    Real xg;
    // @var: bound=[0:1], def="time elapsed in current control step"
    Real t;
    // @var: bound=[0:1000], def="global time remaining until deadline"
    Real T;
End.

Problem
    assumptions(xr, vr, xg, T)
        -> [ dwwpdl; ](xr < waypointEndDist(xg) & (T <= 0 -> ensure(xr, vr, xg))) // Safety postcondition
        & < dwwpdl; > ensure(xr, vr, xg)                                         // Liveness postcondition
End.
    """

    print("\n--- Parsing full sample KeYmaera model ---")
    parser = KeYmaeraParser(sample_keymaera_model_text)
    try:
        model_ast = parser.parse()
        print("Full model parsing successful!")
        # Breakpoint 1: After successful parsing of the entire model.
        # Inspect `model_ast` here to see the complete AST.
        breakpoint() 
        
        print(f"Model Kind: {model_ast.entries[0].kind}")
        print(f"Model Name: {model_ast.entries[0].name}")
        
        print("\n--- Definitions ---")
        for definition in model_ast.entries[0].definitions.defs:
            print(f"- {definition.__class__.__name__}: Name='{definition.name}', Type='{getattr(definition, 'type', 'N/A')}', Annotations={definition.annotations}")
            if isinstance(definition, FuncDef):
                print(f"  Body: {definition.body}")
            elif isinstance(definition, PredDef):
                print(f"  Body: {definition.body}")
            elif isinstance(definition, HPDef):
                print(f"  Body: {definition.body}")

        print("\n--- Program Variables ---")
        for pv_decl in model_ast.entries[0].program_variables.decls:
            print(f"- {pv_decl.type} {pv_decl.name}, Annotations={pv_decl.annotations}")

        print("\n--- Problem Formula ---")
        print(model_ast.entries[0].problem.formula)

    except ValueError as e:
        print(f"Error parsing full model: {e}")
        # Breakpoint 2: When an error occurs during the main model parsing.
        # This is useful to see the exact error message and stack trace.
        breakpoint() 

    # --- Specific Test Cases for Individual Parsers ---
    print("\n--- Testing parse_term ---")
    term_parser = KeYmaeraParser("")
    try:
        term_expr1 = "x + y * (z - 5)^2 + f(a, b')"
        parsed_term1 = term_parser.parse_term(term_expr1)
        print(f"Term '{term_expr1}' parsed as: {parsed_term1}")
        # Breakpoint 3: After successful parsing of a complex term.
        # Inspect `parsed_term1` to see the term's AST.
        breakpoint()
        
        term_expr2 = "-10 + x'"
        parsed_term2 = term_parser.parse_term(term_expr2)
        print(f"Term '{term_expr2}' parsed as: {parsed_term2}")
        # Breakpoint 4: After successful parsing of another term.
        breakpoint()

    except ValueError as e:
        print(f"Error parsing term: {e}")
        # Breakpoint 5: When an error occurs during term parsing.
        breakpoint()

    print("\n--- Testing parse_formula ---")
    formula_parser = KeYmaeraParser("")
    try:
        formula_expr1 = "forall(x, y). P(x) -> (x > 0 && y < 10)"
        parsed_formula1 = formula_parser.parse_formula(formula_expr1)
        print(f"Formula '{formula_expr1}' parsed as: {parsed_formula1}")
        # Breakpoint 6: After successful parsing of a complex formula.
        # Inspect `parsed_formula1` here.
        breakpoint()

        formula_expr2 = "![hp_choice] (x = 5) <-> <simple_hp> true"
        parsed_formula2 = formula_parser.parse_formula(formula_expr2)
        print(f"Formula '{formula_expr2}' parsed as: {parsed_formula2}")
        # Breakpoint 7: After successful parsing of another formula.
        breakpoint()

    except ValueError as e:
        print(f"Error parsing formula: {e}")
        # Breakpoint 8: When an error occurs during formula parsing.
        breakpoint()

    print("\n--- Testing parse_hybrid_program ---")
    hp_parser = KeYmaeraParser("")
    try:
        hp_expr1 = "{x' = v, t <= 10} ; ?(x > 0) ++ {y := y + 1}*@invariant(y < 5)"
        parsed_hp1 = hp_parser.parse_hybrid_program(hp_expr1)
        print(f"Hybrid Program '{hp_expr1}' parsed as: {parsed_hp1}")
        # Breakpoint 9: After successful parsing of a complex hybrid program.
        # Inspect `parsed_hp1` here.
        breakpoint()

        hp_expr2 = "{x := x + 1; y := y - 1} ; complex_ode"
        parsed_hp2 = hp_parser.parse_hybrid_program(hp_expr2)
        print(f"Hybrid Program '{hp_expr2}' parsed as: {parsed_hp2}")
        # Breakpoint 10: After successful parsing of another hybrid program.
        breakpoint()

    except ValueError as e:
        print(f"Error parsing hybrid program: {e}")
        # Breakpoint 11: When an error occurs during hybrid program parsing.
        breakpoint()

    print("\n--- KeYmaera Parser Test Section Complete ---")