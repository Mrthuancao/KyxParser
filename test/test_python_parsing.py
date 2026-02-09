# test_PythonGeneration.py
import unittest
import os
from pathlib import Path
import difflib
import sys

# --- Path Setup ---
# SCRIPT_DIR should be the directory containing test_PythonGeneration.py
SCRIPT_DIR = Path(__file__).parent

# Calculate PROJECT_ROOT
# Assuming SCRIPT_DIR is 'project_root/test/'
# SCRIPT_DIR.parent is 'project_root/'
PROJECT_ROOT = SCRIPT_DIR.parent

# Define TEST_INPUT_DIR and TEST_OUTPUT_DIR relative to SCRIPT_DIR
TEST_INPUT_DIR = SCRIPT_DIR / "input"
# IMPORTANT CHANGE: Point to the directory for expected Python code
TEST_OUTPUT_DIR = SCRIPT_DIR / "output_Python" 

# --- FIX FOR IMPORT START ---
# Add the directory containing 'keymaerax_parser.py' and 'ast_config.py'
# to Python's system path so they can be imported.
# This directory is 'project_root/AST_parser/'.
parser_module_dir = PROJECT_ROOT / "AST_parser"
if str(parser_module_dir) not in sys.path:
    sys.path.insert(0, str(parser_module_dir))

# Add the directory containing 'gym_generator.py'
# This directory is 'project_root/generator/'.
generator_module_dir = PROJECT_ROOT / "generator"
if str(generator_module_dir) not in sys.path:
    sys.path.insert(0, str(generator_module_dir))
# --- FIX FOR IMPORT END ---

# Import your parser, converter, and generator functions
try:
    from keymaerax_parser import KeYmaeraXParser, convert_keymaerax_ast_to_intermediate_pyast
    from gym_generator import generate_gym_environment_code # NEW IMPORT
except ImportError as e:
    print(f"Error importing components: {e}")
    print(f"Please ensure 'keymaerax_parser.py' and 'ast_config.py' are in '{parser_module_dir}'")
    print(f"And 'gym_generator.py' is in '{generator_module_dir}'")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


class TestPythonGeneration(unittest.TestCase): # Renamed class for clarity
    """
    Tests the generation of Gymnasium environment Python code from KeYmaera X files.
    """

    def _load_test_case_files(self, test_name):
        """
        Loads KeYmaera X source code from input/ and expected Python code string from output_Python/.
        """
        source_code_path = TEST_INPUT_DIR / f"{test_name}.txt"
        # IMPORTANT CHANGE: Expecting a .py file for the output
        expected_python_output_path = TEST_OUTPUT_DIR / f"{test_name}.py" 

        # --- Debugging prints (uncomment if you have path issues) ---
        # print(f"\n--- Debugging Paths for '{test_name}' ---")
        # print(f"Script Directory: {SCRIPT_DIR.resolve()}")
        # print(f"Project Root (calculated): {PROJECT_ROOT.resolve()}")
        # print(f"Parser Module Directory: {parser_module_dir.resolve()}")
        # print(f"Generator Module Directory: {generator_module_dir.resolve()}")
        # print(f"Calculated Input Directory: {TEST_INPUT_DIR.resolve()}")
        # print(f"Calculated Output Directory: {TEST_OUTPUT_DIR.resolve()}")
        # print(f"Attempting to open source file: {source_code_path.resolve()}")
        # print(f"Attempting to open expected Python output file: {expected_python_output_path.resolve()}")
        # print(f"--- End Debugging Paths ---")
        # --- End Debugging prints ---

        if not source_code_path.exists():
            raise FileNotFoundError(f"KeYmaera X source file not found: {source_code_path.resolve()}")
        if not expected_python_output_path.exists():
            raise FileNotFoundError(f"Expected Python output file not found: {expected_python_output_path.resolve()}")

        with open(source_code_path, 'r', encoding='utf-8') as f:
            kx_source_code = f.read()

        with open(expected_python_output_path, 'r', encoding='utf-8') as f:
            # .strip() is crucial to handle any leading/trailing newlines or whitespace
            # differences that might exist in the expected output file.
            expected_python_str = f.read().strip()

        return kx_source_code, expected_python_str

    def _assert_generated_python_matches_expected(self, test_name): # Renamed helper for clarity
        """
        Helper method to load a test case, parse the source, convert to PyAST,
        generate Python code, and assert that the generated Python code
        matches the expected output. Includes detailed diffing if they don't match.
        """
        print(f"\n--- Running Python Code Generation Test: {test_name} ---")
        kx_source_code, expected_python_str = self._load_test_case_files(test_name)

        # 1. Parse KeYmaera X source into internal Kx AST
        kx_parser = KeYmaeraXParser(kx_source_code)
        kx_ast = kx_parser.parse()

        # 2. Convert internal Kx AST to PyAST (GymEnvironmentConfig)
        py_ast_config = convert_keymaerax_ast_to_intermediate_pyast(kx_ast)

        # 3. Generate the actual Python code from the PyAST config
        actual_python_str = generate_gym_environment_code(py_ast_config).strip() # NEW STEP

        # Optional: Print actual and expected for manual inspection during development
        # print("\n--- Actual Python Output ---")
        # print(actual_python_str)
        # print("\n--- Expected Python Output ---")
        # print(expected_python_str)
        # print("---------------------------\n")

        if actual_python_str != expected_python_str:
            print(f"--- DETAILED DIFFERENCES FOUND for {test_name} ---")
            diff = difflib.unified_diff(
                expected_python_str.splitlines(keepends=True),
                actual_python_str.splitlines(keepends=True),
                fromfile=f'Expected Python ({test_name})',
                tofile=f'Actual Python ({test_name})',
                lineterm='' # Prevent extra newlines in output if lines already have them
            )
            print("".join(diff))
            print("--------------------------------------------------\n")

        self.assertEqual(actual_python_str, expected_python_str,
                         f"Python code for '{test_name}' does not match expected output. See detailed diff above.")

    # --- Test Cases ---
    # Add a test method for each KeYmaera X input file you want to test.
    # The name 'test1' corresponds to 'test1.txt' in input/ and 'test1.py' in output_Python/.
    def test_case_1_static_safety_python(self): # Renamed test method
        self._assert_generated_python_matches_expected("test1")

    # This corresponds to the kx_input you provided (Corollary 1.1)
    def test_case_2_velocity_controlled_python(self): # Renamed test method
        self._assert_generated_python_matches_expected("test2")

    # This corresponds to Corollary 1.2
    def test_case_3_imperfect_trajectories_python(self): # Renamed test method
        self._assert_generated_python_matches_expected("test3")


if __name__ == '__main__':
    # Use argv=['first-arg-is-ignored'] to prevent unittest from trying to
    # interpret command-line arguments meant for the script itself.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)