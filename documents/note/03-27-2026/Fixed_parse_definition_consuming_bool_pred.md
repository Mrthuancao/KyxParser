You're looking for an alternative parsing strategy that doesn't explicitly prioritize Hybrid Programs (HPs) over `Real`/`Bool` definitions. The previous approach parsed HPs first, then removed them from the text, and finally parsed the remaining `Real`/`Bool` definitions. While robust, this creates a two-pass structure.

The challenge with parsing both in a single pass using a simple combined regex is that HPs have a block structure defined by nested braces (`{...}`), which requires careful brace matching (like your `_find_matching_brace_span` function). `Real`/`Bool` definitions, on the other hand, are typically terminated by a semicolon (`;`). A single regex with `re.DOTALL` would struggle to differentiate these structures correctly, especially with nested braces.

A better "single pass" approach that processes definitions in the order they appear, without an explicit "HP first" phase, is an **iterative search and consume** method.

**Here's the refined strategy:**

1.  **Initialize a mutable representation of the `definitions_block`** (e.g., a list of characters) and a `current_search_global_offset` to track your position.
2.  **Loop indefinitely** until no more definitions are found.
3.  **Inside the loop:**
    a.  **Convert the mutable character list back to a string** for regex searching. This string will contain spaces where previous definitions were "removed".
    b.  **Search for the *next* occurrence of an HP header** using `HP_START_PATTERN` from the `current_search_global_offset`.
    c.  **Search for the *next* occurrence of a full `Real`/`Bool` definition** using `DEF_FULL_PATTERN` from the `current_search_global_offset`.
    d.  **Compare their starting indices:**
        *   If an HP header is found *before* a `Real`/`Bool` definition, process the HP.
        *   If a `Real`/`Bool` definition is found *before* an HP header, process the `Real`/`Bool` definition.
        *   If neither is found, break the loop.
    e.  **Process the matched definition:**
        *   **For HPs:** Extract metadata and name. Use `_find_matching_brace_span` to find the correct end of the HP body. Extract the body and any terminators (`;;` or `;`). Create the `HybridProgram` node.
        *   **For `Real`/`Bool` definitions:** The `DEF_FULL_PATTERN` already captures the entire definition (metadata, type, name, params, value, semicolon). Extract these groups and create the `Definition` node.
    f.  **"Consume" the processed text:** Replace the characters of the matched definition (from its start to its end) with spaces in the `mutable_definitions_block_chars` list.
    g.  **Advance `current_search_global_offset`:** Set it to the end index of the *just processed* definition. This ensures the next search starts immediately after the consumed block.

This approach ensures that definitions are processed in their natural order of appearance in the text, without one type being strictly parsed "first" in a separate pass. It also correctly handles the distinct structural requirements of HPs (brace matching) and `Real`/`Bool` definitions (regex-based capture up to semicolon).

Here's the modified `_parse_definitions` function incorporating this strategy:

```python
import re
import logging

# --- Mock classes for demonstration (assuming these are defined elsewhere in the actual code) ---
# These are included to make the code runnable and testable independently.
class ASTNode:
    def __init__(self, node_type, value=None):
        self.node_type = node_type
        self.value = value
        self.children = []
        self.metadata = {}

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        # Basic representation for debugging
        meta_str = ', '.join([f"{k}={v}" for k, v in self.metadata.items()])
        return f"<{self.node_type}{f' name={self.value}' if self.value else ''}{f' meta=[{meta_str}]' if meta_str else ''} children={len(self.children)}>"

class HybridProgram(ASTNode):
    def __init__(self, name, body):
        super().__init__("HybridProgram", name)
        self.name = name
        self.body = body

class Definition(ASTNode):
    def __init__(self, name, def_type, value=None):
        super().__init__("Definition", name)
        self.name = name
        self.def_type = def_type
        self.value = value

class Parser:
    def __init__(self, text):
        self.text = text
        self.ast_root = ASTNode("Root")
        # Configure logging for demonstration
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    def _extract_block(self, block_name):
        # Simplified block extraction for demonstration
        start_marker = f"{block_name}\n"
        
        start_idx = self.text.find(start_marker)
        if start_idx == -1:
            return None
        
        start_idx += len(start_marker)
        
        # Find the end of the block.
        # This is a simplified approach and might need more robust logic
        # for real-world scenarios, especially if 'End.' can appear within blocks.
        end_idx = self.text.find("\nEnd.", start_idx)
        if end_idx == -1:
            # If no "End." found, assume it goes to the end of the text
            # Or find the next block header
            next_block_headers = ["\nProgramVariables\n", "\nProblem\n"]
            found_next_block = False
            for header in next_block_headers:
                next_header_idx = self.text.find(header, start_idx)
                if next_header_idx != -1 and (end_idx == -1 or next_header_idx < end_idx):
                    end_idx = next_header_idx
                    found_next_block = True
            
            if not found_next_block:
                end_idx = len(self.text)

        block_content = self.text[start_idx:end_idx].strip()
        return block_content

    def _parse_metadata_string(self, raw_meta_comments):
        metadata = {}
        meta_pattern = re.compile(r'//\s*@(?P<key>\w+):\s*(?P<value>[^\n]*)')
        if raw_meta_comments: # Only search if there are comments
            for match in meta_pattern.finditer(raw_meta_comments):
                key = match.group('key').strip()
                value = match.group('value').strip()
                metadata[key] = value
        return metadata

    def _find_matching_brace_span(self, text, open_brace_idx):
        balance = 0
        for i in range(open_brace_idx, len(text)):
            if text[i] == '{':
                balance += 1
            elif text[i] == '}':
                balance -= 1
            
            if balance == 0:
                return i
        return None

    def _parse_program_fragment(self, body_content, parent_node):
        logging.debug(f"Parsing controlFunc body: {body_content[:50]}...")
        parent_node.add_child(ASTNode("ProgramFragmentBody", body_content))

    def _parse_step_func_structure(self, body_content, parent_node):
        logging.debug(f"Parsing stepFunc body: {body_content[:50]}...")
        parent_node.add_child(ASTNode("StepFuncBody", body_content))

    def split_by_operator_outside_braces(self, text, operator):
        # A more robust split for parameters, considering parentheses
        parts = []
        balance = 0
        last_split = 0
        for i, char in enumerate(text):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            elif char == operator and balance == 0:
                parts.append(text[last_split:i])
                last_split = i + 1
        parts.append(text[last_split:])
        return [p.strip() for p in parts if p.strip()]


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
                    hp_body_raw = current_definitions_text_for_search[open_brace_idx : closing_brace_idx + 1].strip()
                    logging.debug(f"Extracted HP '{hp_name}' body (raw): '{hp_body_raw[:100]}...'")

                    full_hp_block_end_idx = closing_brace_idx + 1
                    
                    # Check for optional HP terminators (;; or ;) immediately after the closing brace
                    # We need to find the *absolute* index of these terminators from the start of the full string
                    temp_str_after_brace = current_definitions_text_for_search[full_hp_block_end_idx:].lstrip()
                    if temp_str_after_brace.startswith(';;'):
                        terminator_idx = current_definitions_text_for_search.find(';;', full_hp_block_end_idx)
                        if terminator_idx != -1:
                            full_hp_block_end_idx = terminator_idx + 2
                    elif temp_str_after_brace.startswith(';'):
                        terminator_idx = current_definitions_text_for_search.find(';', full_hp_block_end_idx)
                        if terminator_idx != -1:
                            full_hp_block_end_idx = terminator_idx + 1
                    
                    hp_node = HybridProgram(hp_name, hp_body_raw)
                    hp_node.metadata = metadata_dict
                    defs_node.add_child(hp_node)
                    logging.debug(f"Added HybridProgram ASTNode for '{hp_name}' to DefinitionsBlock.")

                    # Parse HP body structure if specific functions are defined
                    body_content = hp_body_raw.strip()
                    if hp_name == 'controlFunc':
                        self._parse_program_fragment(body_content, hp_node)
                        logging.debug(f"Parsed controlFunc structure for '{hp_name}'.")
                    elif hp_name == 'stepFunc':
                        self._parse_step_func_structure(hp_body_raw, hp_node)
                    
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

                def_node = Definition(d_name, d_type, value=d_val)
                def_node.metadata = m_data
                
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

# --- Example Usage ---
if __name__ == "__main__":
    example_input = """
Theorem "IJRR17/Theorem 1: Static safety"

Definitions
  // Physical Constants
  Real ep;      // time limit for control decisions
  Real b;       // minimum braking capability of the robot
  Real A;       // maximum acceleration -b <= a <= A
  Real W;       // maximum steering

  // Distance Calculations
  Real stopDist(Real v) = v^2 / (2*b);
  Real accelComp(Real v) = ((A/b + 1) * (A/2 * ep^2 + ep*v));
  Real admissibleSeparation(Real v) = stopDist(v) + accelComp(v);

  // Directional Integrity:
  // Ensures the orientation vector (dx, dy) remains normalized.
  // @progress: shape={-((dx^2 + dy^2) - 1)^2}
  Bool isWellformedDir(Real dx, Real dy) <-> dx^2 + dy^2 = 1;

  // Physical Validity:
  // Prevents the agent from entering physically impossible states (negative velocity).
  // @safety: penalty=-500, on_fail=truncate, status=failure
  Bool isPhysicallyValid(Real v, Real r) <-> (
    v >= 0 & r != 0
  );

  // Collision Avoidance (Core Safety):
  // The primary safety invariant. The shape encourages maximizing the gap 
  // between current distance and required stopping distance.
  // @safety: penalty=-1000, shape={(abs(x-xo) - stopDist(v)) + (abs(y-yo) - stopDist(v))}, on_fail=truncate, status=failure
  Bool isSafe(Real x, Real y, Real v, Real xo, Real yo) <-> (
    abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v)
  );

  // Velocity Efficiency:
  // Encourages the agent to maintain a productive velocity.
  // @progress: shape={v > 0 ? 1/v : v}
  Bool isEfficient(Real v) <-> ( v >= 0 );

  // Global Bounds
  Bool bounds() <-> (
      A >= 0
    & b > 0
    & ep > 0
  );

  // Initial Conditions
  Bool initialState(Real x, Real y, Real v, Real dx, Real dy, Real xo, Real yo) <-> (        
    v = 0
    & (x-xo)^2 + (y-yo)^2 > 0
    & isWellformedDir(dx, dy)
    & r != 0
  );

  Bool assumptions(Real x, Real y, Real v, Real dx, Real dy, Real xo, Real yo) <->
    bounds() & initialState(x, y, v, dx, dy, xo, yo);

  // Safety Invariant (Aggregate):
  // Combines sub-predicates for the formal proof.
  Bool loopinv(Real x, Real y, Real v, Real dx, Real dy, Real xo, yo, Real r) <-> (
    isPhysicallyValid(v, r) 
    & isWellformedDir(dx, dy) 
    & isSafe(x, y, v, xo, yo)
    & isEfficient(v)
  );
  
  // Goal Condition:
  // @progress: reward=500, on_fail=terminate, status=success
  Bool ensure(Real x, Real y, Real xo, Real yo) <-> (
    (x - xo)^2 + (y - yo)^2 > 0
  );
  
  // Control Logic
  HP controlFunc ::= {
      // Option 1: Brake on current curve or remain stopped
      { a := -b; }
      ++
      { ?v = 0; a := 0; w := 0; }
      ++
      // Option 2: Choose a new safe curve
      { a := A;
        w := *; ?-W<=w & w<=W;
        r := *;
        xo := *; yo := *;
        ?r!=0 & r*w = v;
        ? abs(x-xo) > admissibleSeparation(v)
        | abs(y-yo) > admissibleSeparation(v);
      }
    };;

    // Continuous Dynamics
    HP stepFunc ::= { 
        x' = v * dx, y' = v * dy, v' = a,
        dx' = -w * dy, dy' = w * dx, w' = a/r,
        t' = 1 & t <= ep & v >= 0
      };
End.

ProgramVariables
  Real x;    // robot position: x
  Real y;    // robot position: y
  Real v;    // robot translational velocity
  Real a;    // robot translational acceleration
  Real dx;   // robot orientation: x
  Real dy;   // robot orientation: y
  Real w;    // robot rotational velocity
  Real r;    // robot curve radius
  Real xo;   // position of closest obstacle on curve: x
  Real yo;   // position of closest obstacle on curve: y
  Real t;    // time
End.

Problem
assumptions(x, y, v, dx, dy, xo, yo) ->
  [
    {
      {
        controlFunc;
        t := 0;
      }

      // dynamics with local invariants
      {
      stepFunc;@invariant(t>=0, isWellformedDir(dx, dy))
      }
    }*@invariant(loopinv(x, y, v, dx, dy, xo, yo, r))
  ]ensure(x, y, xo, yo)
End.
"""
    parser = Parser(example_input)
    parser._parse_definitions()

    # You can inspect the AST here
    defs_block = parser.ast_root.children[0]
    print("\n--- Parsed Definitions ---")
    for node in defs_block.children:
        if isinstance(node, Definition):
            print(f"  Definition: {node.name} ({node.def_type}), Value: {node.value[:50]}..., Metadata: {node.metadata}")
        elif isinstance(node, HybridProgram):
            print(f"  HybridProgram: {node.name}, Body: {node.body[:50]}..., Metadata: {node.metadata}")
            for child in node.children:
                print(f"    - Child: {child.node_type}")

    # Verify a specific metadata
    found_is_safe = False
    for node in defs_block.children:
        if isinstance(node, Definition) and node.name == 'isSafe':
            print(f"\nMetadata for isSafe: {node.metadata}")
            assert 'safety' in node.metadata
            assert 'penalty' in node.metadata
            assert node.metadata['penalty'] == '-1000'
            found_is_safe = True
            break
    assert found_is_safe, "isSafe definition not found or metadata missing"

    found_control_func = False
    for node in defs_block.children:
        if isinstance(node, HybridProgram) and node.name == 'controlFunc':
            print(f"\nControlFunc body children: {node.children}")
            assert len(node.children) > 0 and node.children[0].node_type == "ProgramFragmentBody"
            found_control_func = True
            break
    assert found_control_func, "controlFunc not found"

    print("\nParsing successful and metadata/structure checks passed!")
```


This is an excellent and sophisticated parsing strategy! Let's break down the `_parse_definitions` function and the overall approach so you can understand it thoroughly.

### 1. The Overall Goal: A Single-Pass, Order-Preserving Parser

The main objective of this `_parse_definitions` function is to read through the `Definitions` block of your KeYmaera X input and extract all `Real`/`Bool` definitions and Hybrid Programs (HPs) **in the order they appear** in the text.

The previous approach had two passes: first HPs, then everything else. This new "single-pass" strategy aims to be more natural by processing whatever comes next in the text, regardless of whether it's an HP or a `Real`/`Bool` definition.

### 2. The Core Strategy: Iterative Search and Consume

Imagine you have a long piece of string. You want to find specific patterns (like "HP" or "Real" definitions) and, once you find and process one, you want to remove it from the string so you don't find it again. This is precisely what the "iterative search and consume" strategy does:

1.  **Start at the beginning** of the `Definitions` block.
2.  **Look for the *next* occurrence** of *any* type of definition you're interested in (HP or `Real`/`Bool`).
3.  **Choose the one that appears earliest** in the text.
4.  **Parse and process** that definition.
5.  **"Consume" it:** Effectively remove it from the text you're searching in the future.
6.  **Repeat** from step 2, starting *after* the consumed definition, until nothing more can be found.

This ensures that definitions are processed sequentially as they appear in the input.

### 3. Key Components and Variables

Let's look at the important parts of the `_parse_definitions` function:

*   **`definitions_block = self._extract_block("Definitions")`**:
    *   This first line gets the raw text content specifically from the "Definitions" section of your KeYmaera X input. This is the chunk of text we'll be parsing.

*   **`defs_node = ASTNode("DefinitionsBlock")`**:
    *   This creates a new node in your Abstract Syntax Tree (AST) to hold all the definitions found within this block. All parsed `Definition` and `HybridProgram` nodes will become children of this `defs_node`.

*   **`mutable_definitions_block_chars = list(definitions_block)`**:
    *   **Why a list of characters?** This is crucial for the "consume" part of the strategy. Strings in Python are immutable, meaning you can't change individual characters. If you had a string, every time you "removed" a definition, you'd have to create a *new* string, which can be inefficient for many iterations.
    *   By converting it to a list of characters, you can directly modify (replace with spaces) the characters that correspond to a parsed definition. This makes the "removal" process efficient.

*   **`HP_START_PATTERN` and `DEF_FULL_PATTERN`**:
    *   These are pre-compiled regular expressions (regex). Pre-compiling them (`re.compile`) makes them faster to use in a loop.
    *   **`HP_START_PATTERN`**: This regex is designed to find the *beginning* of a Hybrid Program definition.
        *   `(?P<metadata>(?://\s*@[^\n]*\n)*)?`: This part is clever. It captures any number of comment lines (`// @...`) that appear *before* the `HP` keyword. These comments often contain metadata. The `?` at the end makes the entire metadata group optional.
        *   `\s*HP\s+(?P<name>\w+)\s*::=\s*\{`: This matches the literal "HP", captures the HP's name (e.g., `controlFunc`), and then looks for the `::=` and the opening brace `{`.
        *   `re.DOTALL`: Allows the `.` (dot) in the regex to match newline characters, which is important for metadata spanning multiple lines.
    *   **`DEF_FULL_PATTERN`**: This regex is designed to find an *entire* `Real` or `Bool` definition, from its optional metadata comments to its terminating semicolon.
        *   `(?P<raw_comments>(?:^[ \t]*//[^\n]*\n)*)`: Similar to HP metadata, this captures leading comment lines. `^` with `re.MULTILINE` ensures it matches the start of a line.
        *   `[ \t]*(?P<type>Real|Bool)\s+(?P<name>\w+)`: Captures the type (`Real` or `Bool`) and the definition's name.
        *   `(?:\s*\((?P<params>.*?)\))?`: Optionally captures parameters within parentheses (e.g., `(Real v)`). `.*?` is a non-greedy match, meaning it matches as few characters as possible.
        *   `(?:\s*(?:<->|=)\s*(?P<value>.*?))?`: Optionally captures the value or body of the definition, which can be introduced by `<->` or `=`. Again, `.*?` for non-greedy matching.
        *   `;`: Matches the mandatory semicolon that terminates `Real`/`Bool` definitions.
        *   `re.DOTALL | re.MULTILINE`: `DOTALL` for multi-line values/parameters, `MULTILINE` for `^` to work correctly at the start of lines.

*   **`current_search_global_offset = 0`**:
    *   This integer variable keeps track of where in the `definitions_block` (and `mutable_definitions_block_chars`) we should start our *next* search. After processing a definition, this offset is updated to point just past the end of that definition, ensuring we don't re-parse it.

### 4. The Main Parsing Loop (`while True`)

This is the heart of the "iterative search and consume" strategy:

1.  **`current_definitions_text_for_search = "".join(mutable_definitions_block_chars)`**:
    *   At the start of each iteration, the `mutable_definitions_block_chars` list (which might have spaces from previous "consumptions") is joined back into a string. Regex `search` functions will naturally ignore these spaces.

2.  **`hp_match = HP_START_PATTERN.search(...)`** and **`def_match = DEF_FULL_PATTERN.search(...)`**:
    *   Two separate searches are performed: one for the next HP header, and one for the next `Real`/`Bool` definition.
    *   Both searches start from `current_search_global_offset`.

3.  **`next_hp_idx = hp_match.start() if hp_match else float('inf')`** and **`next_def_idx = def_match.start() if def_match else float('inf')`**:
    *   We get the starting index of each potential match. If a match isn't found, we assign `float('inf')` (infinity) to its index. This ensures that an unmatched type will never be considered "first."

4.  **`if next_hp_idx == float('inf') and next_def_idx == float('inf'): break`**:
    *   If neither an HP nor a `Real`/`Bool` definition can be found in the remaining text, it means we've parsed everything, so the loop breaks.

5.  **`if next_hp_idx < next_def_idx:` (Process Hybrid Program)**:
    *   This is the core decision point: if the HP header appears *earlier* in the text than the `Real`/`Bool` definition, we process the HP.
    *   **Extracting HP details**: `hp_name`, `raw_meta`, `metadata_dict` are extracted from the `hp_match`.
    *   **Finding the HP Body (Brace Matching)**: This is crucial for HPs. Since HPs are block-structured with `{...}`, a simple regex won't work for the whole body.
        *   `open_brace_idx = hp_match.end() - 1`: Gets the index of the `{` character.
        *   `closing_brace_idx = self._find_matching_brace_span(...)`: This helper function (explained below) is called to find the *correct* matching closing brace `}` for the HP, handling nested braces.
        *   `hp_body_raw = current_definitions_text_for_search[open_brace_idx : closing_brace_idx + 1].strip()`: Once the closing brace is found, the entire raw body of the HP is extracted.
    *   **Handling HP Terminators (`;;` or `;`)**: HPs often end with `;;` or `;` after the closing brace. The code explicitly checks for these to ensure the *entire* HP definition is consumed.
    *   **Creating the AST Node**: A `HybridProgram` object is created with its name, raw body, and metadata.
    *   **Deeper HP Parsing (Placeholders)**: `_parse_program_fragment` and `_parse_step_func_structure` are called. In a real parser, these would recursively break down the HP body into its components (choices, sequences, tests, assignments, ODEs, etc.). Here, they just add a simple `ProgramFragmentBody` or `StepFuncBody` node.
    *   **"Consuming" the HP**:
        *   `for i in range(hp_header_start_idx, full_hp_block_end_idx): mutable_definitions_block_chars[i] = ' '`: All characters belonging to the just-parsed HP (from its start to its terminator) are replaced with spaces in the `mutable_definitions_block_chars` list.
        *   `current_search_global_offset = full_hp_block_end_idx`: The search offset is moved past the end of the consumed HP.

6.  **`else:` (Process `Real`/`Bool` Definition)**:
    *   If the `Real`/`Bool` definition appears *earlier* (or if no HP was found), we process it.
    *   **Extracting definition details**: `d_type`, `d_name`, `d_params`, `d_val`, `raw_comments` are all directly extracted using named groups from `def_match`. This is much simpler than HPs because the regex captures the whole thing.
    *   **Parsing Metadata**: `_parse_metadata_string` is used for comments.
    *   **Handling `Bool` values**: `if d_type == "Bool" and d_val.startswith('(') and d_val.endswith(')'): d_val = d_val[1:-1].strip()`: This removes outer parentheses from boolean expressions, which are often used for grouping.
    *   **Creating the AST Node**: A `Definition` object is created.
    *   **Parsing Parameters**: If parameters exist (`d_params`), they are split by commas (using `split_by_operator_outside_braces` to handle nested parentheses) and added as a child node.
    *   **"Consuming" the Definition**:
        *   `for i in range(def_start_idx, def_end_idx): mutable_definitions_block_chars[i] = ' '`: Characters of the parsed definition are replaced with spaces.
        *   `current_search_global_offset = def_end_idx`: The search offset is moved.

### 5. Helper Functions

*   **`_parse_metadata_string(self, raw_meta_comments)`**:
    *   Takes a string of raw comments (e.g., `// @key: value\n// @another: thing`).
    *   Uses a regex (`meta_pattern`) to find lines starting with `// @` and extracts the `key` and `value` into a dictionary.

*   **`_find_matching_brace_span(self, text, open_brace_idx)`**:
    *   This is a crucial utility for parsing block-structured languages.
    *   It takes the full text and the index of an *opening* brace `{`.
    *   It then iterates through the text, keeping a `balance` counter.
        *   Increment `balance` for every `{`.
        *   Decrement `balance` for every `}`.
    *   When `balance` returns to `0` (and we've passed the initial `{`), it means we've found the matching closing brace.
    *   This correctly handles nested braces (e.g., `{ { ... } ... }`).

*   **`split_by_operator_outside_braces(self, text, operator)`**:
    *   This is a robust way to split a string (like a parameter list `Real x, Real y`) by an operator (like `,`) but *only* if that operator is not inside parentheses.
    *   It uses a `balance` counter for parentheses, similar to `_find_matching_brace_span`, to ensure the split only happens at the top level.

### 6. Mock Classes

*   `ASTNode`, `HybridProgram`, `Definition`, `Parser`: These are simplified versions of what your actual AST nodes and parser class would look like. They are included here so that the `_parse_definitions` function can be run and tested independently without needing the full KeYmaera X parser implementation. They provide basic structure and methods like `add_child` and `__repr__`.

### 7. Example Usage (`if __name__ == "__main__":`)

This block demonstrates how to use the `_parse_definitions` function:

1.  It defines a sample `example_input` string containing the KeYmaera X specification.
2.  It creates an instance of the `Parser`.
3.  It calls `parser._parse_definitions()` to execute the parsing logic.
4.  Finally, it iterates through the generated AST (`defs_block.children`) and prints out the parsed definitions and HPs, including their metadata, to verify that the parsing worked correctly.
5.  It includes `assert` statements to programmatically check if specific elements (like `isSafe`'s metadata or `controlFunc`'s body structure) were parsed as expected.

### Why this approach is good:

*   **Order Preservation**: Definitions are parsed in the exact order they appear in the source text.
*   **Flexibility**: It can easily be extended to parse new types of definitions by adding another `PATTERN.search()` and a new `if/else if` branch.
*   **Efficiency**: Using a mutable character list for "consumption" is more efficient than repeatedly creating new strings.
*   **Robustness**: The brace-matching logic correctly handles nested structures in HPs, which is a common challenge in parsing.
*   **Single Pass (Conceptually)**: While it involves multiple regex searches in a loop, it processes the input text sequentially from start to finish without requiring a full re-scan or separate passes for different definition types.

This is a well-designed and effective parsing strategy for the given problem!