# test_PyAST.py
import unittest
import os
from pathlib import Path
import difflib
import sys

# --- Path Setup ---
# SCRIPT_DIR should be the directory containing test_PyAST.py
# Path(__file__) gives the full path to test_PyAST.py
# .parent gives the directory of test_PyAST.py
SCRIPT_DIR = Path(__file__).parent

# Calculate PROJECT_ROOT
# SCRIPT_DIR is 'project_root/test/'
# SCRIPT_DIR.parent is 'project_root/'
PROJECT_ROOT = SCRIPT_DIR.parent

# Define TEST_INPUT_DIR and TEST_OUTPUT_DIR relative to SCRIPT_DIR
TEST_INPUT_DIR = SCRIPT_DIR / "input"
TEST_OUTPUT_DIR = SCRIPT_DIR / "output_PyAST" # This directory holds the expected PyAST string outputs

# --- FIX FOR IMPORT START ---
# Add the directory containing 'keymaerax_parser.py' and 'ast_config.py'
# to Python's system path so they can be imported.
# This directory is 'project_root/AST_parser/'.
parser_module_dir = PROJECT_ROOT / "AST_parser"
if str(parser_module_dir) not in sys.path:
    sys.path.insert(0, str(parser_module_dir)) # Insert at the beginning to prioritize it
# --- FIX FOR IMPORT END ---

# Import your parser and converter functions
try:
    from keymaerax_parser import KeYmaeraXParser, convert_keymaerax_ast_to_intermediate_pyast
    # It's good practice to also import the PyAST classes themselves if their __repr__
    # methods are being tested, to ensure they are properly loaded.
    # For example, if GymEnvironmentConfig is in ast_config.py:
    # from ast_config import GymEnvironmentConfig
except ImportError as e:
    print(f"Error importing parser components: {e}")
    print(f"Please ensure 'keymaerax_parser.py' and 'ast_config.py' are in '{parser_module_dir}'")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


class TestPyASTGeneration(unittest.TestCase):
    """
    Tests the generation of the GymEnvironmentConfig (PyAST) from KeYmaera X files.
    """

    def _load_test_case_files(self, test_name):
        """
        Loads KeYmaera X source code from input/ and expected PyAST string from output_PyAST/.
        """
        source_code_path = TEST_INPUT_DIR / f"{test_name}.txt"
        expected_pyast_output_path = TEST_OUTPUT_DIR / f"{test_name}.txt"

        # --- Debugging prints (uncomment if you have path issues) ---
        # print(f"\n--- Debugging Paths for '{test_name}.txt' ---")
        # print(f"Script Directory: {SCRIPT_DIR.resolve()}")
        # print(f"Project Root (calculated): {PROJECT_ROOT.resolve()}")
        # print(f"Parser Module Directory: {parser_module_dir.resolve()}")
        # print(f"Calculated Input Directory: {TEST_INPUT_DIR.resolve()}")
        # print(f"Calculated Output Directory: {TEST_OUTPUT_DIR.resolve()}")
        # print(f"Attempting to open source file: {source_code_path.resolve()}")
        # print(f"Attempting to open expected PyAST output file: {expected_pyast_output_path.resolve()}")
        # print(f"--- End Debugging Paths ---")
        # --- End Debugging prints ---

        if not source_code_path.exists():
            raise FileNotFoundError(f"KeYmaera X source file not found: {source_code_path.resolve()}")
        if not expected_pyast_output_path.exists():
            raise FileNotFoundError(f"Expected PyAST output file not found: {expected_pyast_output_path.resolve()}")

        with open(source_code_path, 'r', encoding='utf-8') as f:
            kx_source_code = f.read()

        with open(expected_pyast_output_path, 'r', encoding='utf-8') as f:
            # .strip() is crucial to handle any leading/trailing newlines or whitespace
            # differences that might exist in the expected output file.
            expected_pyast_str = f.read().strip()

        return kx_source_code, expected_pyast_str

    def _assert_generated_pyast_matches_expected(self, test_name):
        """
        Helper method to load a test case, parse the source, convert to PyAST,
        and assert that the generated GymEnvironmentConfig's string representation
        matches the expected output. Includes detailed diffing if they don't match.
        """
        print(f"\n--- Running PyAST Generation Test: {test_name} ---")
        kx_source_code, expected_pyast_str = self._load_test_case_files(test_name)

        # 1. Parse KeYmaera X source into internal Kx AST
        kx_parser = KeYmaeraXParser(kx_source_code)
        kx_ast = kx_parser.parse()

        # 2. Convert internal Kx AST to PyAST (GymEnvironmentConfig)
        generated_pyast_config = convert_keymaerax_ast_to_intermediate_pyast(kx_ast)

        # 3. Get the string representation of the generated PyAST
        actual_pyast_str = generated_pyast_config.__repr__().strip()

        # Optional: Print actual and expected for manual inspection during development
        # print("\n--- Actual PyAST Output ---")
        # print(actual_pyast_str)
        # print("\n--- Expected PyAST Output ---")
        # print(expected_pyast_str)
        # print("---------------------------\n")

        if actual_pyast_str != expected_pyast_str:
            print(f"--- DETAILED DIFFERENCES FOUND for {test_name} ---")
            diff = difflib.unified_diff(
                expected_pyast_str.splitlines(keepends=True),
                actual_pyast_str.splitlines(keepends=True),
                fromfile=f'Expected PyAST ({test_name})',
                tofile=f'Actual PyAST ({test_name})',
                lineterm='' # Prevent extra newlines in output if lines already have them
            )
            print("".join(diff))
            print("--------------------------------------------------\n")

        self.assertEqual(actual_pyast_str, expected_pyast_str,
                         f"PyAST representations for '{test_name}' do not match. See detailed diff above.")

    # --- Test Cases ---
    # Add a test method for each KeYmaera X input file you want to test.
    # The name 'test1' corresponds to 'test1.txt' in input/ and output_PyAST/.
    def test_case_1_static_safety_pyast(self):
        self._assert_generated_pyast_matches_expected("test1")

    # You can add more test cases here, e.g.:
    # def test_case_2_velocity_controlled_pyast(self):
    #     self._assert_generated_pyast_matches_expected("test2")

    # def test_case_3_imperfect_trajectories_pyast(self):
    #     self._assert_generated_pyast_matches_expected("test3")


if __name__ == '__main__':
    # Use argv=['first-arg-is-ignored'] to prevent unittest from trying to
    # interpret command-line arguments meant for the script itself.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)