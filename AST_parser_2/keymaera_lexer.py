import re
from typing import List, Optional, Dict, Any

class KeYmaeraLexer:
    """
    Performs lexical analysis for KeYmaera model text, converting it into a stream of tokens.
    """
    def __init__(self, text: str):
        """
        Initializes the KeYmaeraLexer with the raw input text.
        """
        self.text = text
        self.tokens: List[str] = []
        self.current_token_idx = 0
        self._full_tokenize() # Tokenize the entire input text on initialization

    def _full_tokenize(self):
        """
        Breaks down the input text into a list of atomic tokens.
        This is called once during initialization for the entire model text.
        """
        # Clean comments from the expression before tokenizing it.
        # Remove C-style comments /* ... */
        cleaned_text = re.sub(r'/\*.*?\*/', '', self.text, flags=re.DOTALL)
        # Remove single-line comments // ...
        cleaned_text = re.sub(r'//.*', '', cleaned_text)

        # Prioritize multi-character tokens and specific single-character tokens first.
        # Identifiers should be matched before single-character delimiters if they could
        # be part of an identifier.
        patterns = [
            r'"[^"]*"', # 1. Added: Quoted strings (e.g., "Theorem 14: ...")
            r'@invariant', # 2. Specific keyword that starts with @, without a word boundary
            r'\b(?:Real|Bool|HP|forall|exists|true|false|div|mod|sqrt|abs|min|max|sin|cos|tan|arcsin|arccos|arctan|log|exp|if|then|else|fi)\b', # 3. Other keywords
            r'\d+\.\d+|\d+', # 4. Numbers (float or int) - matched before identifiers to prevent '123x' from being '123' and 'x'
            r'[a-zA-Z_][a-zA-Z0-9_]*', # 5. Identifiers (variables, function names, etc.)
            r':=', # 6. Assignment operator
            r'<->|->|>=|<=|\+\+', # 7. Multi-character formula/HP operators (longer ones first)
            # 8. All other operators and delimiters, longer ones first to avoid partial matches.
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
            re.escape('.'),
            re.escape(':'), # Added: Colon as a standalone token
            r'\s+', # Whitespace (to be ignored, but matched to ensure no gaps)
        ]

        # Combine all patterns into a single regex, using capturing groups for each pattern.
        combined_pattern = '|'.join(f'({p})' for p in patterns)

        tokens_found = []
        last_end = 0
        for match in re.finditer(combined_pattern, cleaned_text):
            start, end = match.span()
            if start > last_end:
                skipped_text = cleaned_text[last_end:start]
                raise ValueError(f"Tokenizer error: Skipped untokenizable text '{skipped_text}' at index {last_end} in expression: '{cleaned_text}'")

            for i in range(1, len(patterns) + 1):
                if match.group(i) is not None:
                    token = match.group(i)
                    if not token.isspace(): # Ignore whitespace tokens
                        tokens_found.append(token)
                    break
            last_end = end

        if last_end < len(cleaned_text):
            skipped_text = cleaned_text[last_end:]
            raise ValueError(f"Tokenizer error: Skipped untokenizable text '{skipped_text}' at end of expression: '{cleaned_text}'")

        self.tokens = tokens_found

    def peek_token(self) -> Optional[str]:
        """Returns the next token in the stream without advancing the token index."""
        if self.current_token_idx < len(self.tokens):
            return self.tokens[self.current_token_idx]
        return None

    def consume_token(self) -> str:
        """Returns the current token and advances the token index."""
        if self.current_token_idx < len(self.tokens):
            token = self.tokens[self.current_token_idx]
            self.current_token_idx += 1
            return token
        raise ValueError(f"Unexpected end of token stream. Current token index: {self.current_token_idx}. Full tokens: {self.tokens}")

    def expect_token(self, expected_token: str) -> None:
        """Consumes the next token and asserts that it matches the expected token."""
        try:
            actual_token = self.consume_token()
        except ValueError as e:
            raise ValueError(f"Expected '{expected_token}', but reached end of token stream. "
                             f"Current token index: {self.current_token_idx}. "
                             f"Full tokens: {self.tokens}") from e
        if actual_token != expected_token:
            raise ValueError(f"Expected '{expected_token}', but got '{actual_token}' at token index {self.current_token_idx-1}. Full tokens: {self.tokens}")

    def get_remaining_tokens(self) -> List[str]:
        """Returns the list of tokens from the current position to the end."""
        return self.tokens[self.current_token_idx:]

    def get_all_tokens(self) -> List[str]:
        """Returns all tokens generated by the lexer."""
        return self.tokens

    def set_tokens(self, new_tokens: List[str], start_index: int = 0):
        """Sets a new token list and resets the current index. Useful for sub-parsing."""
        self.tokens = new_tokens
        self.current_token_idx = start_index

    def get_state(self) -> tuple[List[str], int]:
        """Returns the current state of the lexer (tokens and index)."""
        return self.tokens, self.current_token_idx

    def restore_state(self, state: tuple[List[str], int]):
        """Restores the lexer state."""
        self.tokens, self.current_token_idx = state

# --- Helper for parsing annotations ---
def parse_annotations(comment_line: str) -> Dict[str, Any]:
    """Parses annotations from a comment string.
    ... (rest of the parse_annotations function as is) ...
    """
    annotations = {}
    match = re.search(r'//\s*@(\w+):(.*)', comment_line)
    if match:
        annotation_type = match.group(1)
        annotation_str = match.group(2).strip()
        annotations[annotation_type] = {}

        if ';' in annotation_str:
            individual_properties = [p.strip() for p in annotation_str.split(';')]
        else:
            individual_properties = [p.strip() for p in annotation_str.split(',')]

        for prop_pair_str in individual_properties:
            if '=' in prop_pair_str:
                key, value_str = prop_pair_str.split('=', 1)
                key = key.strip()
                value_str = value_str.strip()

                if value_str.startswith('[') and value_str.endswith(']'):
                    try:
                        parts = value_str[1:-1].split(':')
                        annotations[annotation_type][key] = [float(p.strip()) for p in parts]
                    except ValueError:
                        annotations[annotation_type][key] = value_str
                elif value_str.lower() in ['true', 'false']:
                    annotations[annotation_type][key] = value_str.lower() == 'true'
                elif value_str.startswith('"') and value_str.endswith('"'):
                    annotations[annotation_type][key] = value_str[1:-1]
                else:
                    try:
                        annotations[annotation_type][key] = int(value_str)
                    except ValueError:
                        try:
                            annotations[annotation_type][key] = float(value_str)
                        except ValueError:
                            annotations[annotation_type][key] = value_str
    return annotations