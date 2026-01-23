import unittest
import os
from pathlib import Path
import difflib
import sys

# --- Path Calculations ---
# SCRIPT_DIR is the directory containing test.py (e.g., 'project_root/test/')
SCRIPT_DIR = Path(__file__).parent

# PROJECT_ROOT is one level up from SCRIPT_DIR (e.g., 'project_root/')
PROJECT_ROOT = SCRIPT_DIR.parent

# Define the directory containing 'keymaerax_parser.py' and 'ast_config.py'
# (e.g., 'project_root/AST_parser/')
parser_module_dir = PROJECT_ROOT / "AST_parser"

# Add the parser module directory to Python's system path
if str(parser_module_dir) not in sys.path:
    sys.path.insert(0, str(parser_module_dir))
    print(f"Added '{parser_module_dir}' to sys.path.")
else:
    print(f"'{parser_module_dir}' already in sys.path.")

# Define test input and expected output directories relative to SCRIPT_DIR
# Based on the file explorer, 'input' and 'output_AST' are direct subdirectories of 'test'.
TEST_INPUT_DIR = SCRIPT_DIR / "input"
# CORRECTED: Pointing to 'output_AST' as per your instruction
TEST_OUTPUT_DIR = SCRIPT_DIR / "output_AST" # This directory holds the expected KeYmaeraX AST string outputs

# --- Debugging prints for paths ---
print(f"Project Root: {PROJECT_ROOT.resolve()}")
print(f"Script Directory: {SCRIPT_DIR.resolve()}")
print(f"Parser Module Directory: {parser_module_dir.resolve()}")
print(f"Test Input Directory: {TEST_INPUT_DIR.resolve()}")
print(f"Test Output Directory: {TEST_OUTPUT_DIR.resolve()}")
# --- End Path Calculations ---


# Import your parser and converter functions
try:
    from keymaerax_parser import KeYmaeraXParser, convert_keymaerax_ast_to_intermediate_pyast
    # It's good practice to also import the PyAST classes themselves if their __repr__
    # methods are being tested, to ensure they are properly loaded.
    # Based on common AST structures, these might be defined in ast_config.py
    # NOTE: These imports are still useful if convert_keymaerax_ast_to_intermediate_pyast
    # is called elsewhere or if you plan to add tests for the PyAST conversion later.
    # For *this specific test* comparing to output_AST, they are not strictly needed
    # but don't hurt.
    from ast_config import GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig, \
                           ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig, \
                           HybridProgramConfig, ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig, \
                           VariableDeclarationConfig, PreconditionConfig, PostconditionConfig, LoopExecutionConfig, \
                           HybridProgramStepConfig, ActionConfig, ChoiceActionConfig, SequenceActionConfig, \
                           DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig, ActionConstraintConfig, \
                           CallHybridProgramConfig
    print("Successfully imported parser components and AST configurations.")
except ImportError as e:
    print(f"Error importing parser components: {e}")
    print(f"Please ensure 'keymaerax_parser.py' and 'ast_config.py' are in '{parser_module_dir}'")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


class TestKeYmaeraXParser(unittest.TestCase):

    def _load_test_case(self, test_name):
        """Loads source code and expected AST from files."""
        # Use the / operator for Path objects to join components
        source_code_path = TEST_INPUT_DIR / f"{test_name}.txt"
        expected_output_path = TEST_OUTPUT_DIR / f"{test_name}.txt"

        # Ensure paths exist before trying to open
        if not source_code_path.exists():
            raise FileNotFoundError(f"Source code file not found: {source_code_path.resolve()}")
        if not expected_output_path.exists():
            raise FileNotFoundError(f"Expected output file not found: {expected_output_path.resolve()}")

        # Path objects can be directly passed to open()
        with open(source_code_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        with open(expected_output_path, 'r', encoding='utf-8') as f:
            # .strip() is important to handle any leading/trailing newlines
            # or whitespace differences that might exist in the file
            expected_ast_str = f.read().strip()

        return source_code, expected_ast_str

    def _assert_parsed_ast_matches_expected(self, test_name):
        """
        Helper method to load a test case, parse the source, and assert
        that the generated AST's string representation matches the expected output.
        Includes detailed diffing if they don't match.
        """
        print(f"\n--- Running Test: {test_name} ---")
        source_code, expected_ast_str = self._load_test_case(test_name)
        parser = KeYmaeraXParser(source_code)
        
        # Parse the KeYmaera X source code into its AST
        kx_ast_root = parser.parse()
        
        # CORRECTED: Use the __repr__() of the KeYmaeraX AST directly for comparison
        actual_ast_str = kx_ast_root.__repr__().strip()

        # print("\n--- Actual Output ---")
        # print(actual_ast_str)
        # print("\n--- Expected Output ---")
        # print(expected_ast_str)
        # print("---------------------\n")

        if actual_ast_str != expected_ast_str:
            print("--- DETAILED DIFFERENCES FOUND ---")
            # difflib.unified_diff generates a diff in the unified format (like 'diff -u')
            # It expects lists of lines, so we split the strings.
            # `keepends=True` ensures that newlines are part of the lines, which is often
            # better for diffing tools to correctly show changes at line endings.
            diff = difflib.unified_diff(
                expected_ast_str.splitlines(keepends=True),
                actual_ast_str.splitlines(keepends=True),
                fromfile=f'Expected KeYmaeraX AST ({test_name})', # Label for the expected content
                tofile=f'Actual KeYmaeraX AST ({test_name})',     # Label for the actual content
                lineterm=''              # Prevent extra newlines in output if lines already have them
            )
            # Join the diff lines back into a single string for printing
            print("".join(diff))
            print("----------------------------------\n")

        # Finally, assert for equality. If the test fails, the diff will already be printed.
        self.assertEqual(actual_ast_str, expected_ast_str,
                         f"AST representations for '{test_name}' do not match. See detailed diff above.")

    def test_case_1_static_safety(self):
        self._assert_parsed_ast_matches_expected("test1")

    def test_case_2_velocity_controlled_passive_safety(self):
        self._assert_parsed_ast_matches_expected("test2")

    def test_case_3_static_safety_imperfect_trajectories(self):
        self._assert_parsed_ast_matches_expected("test3")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)