# test.py
import unittest
import os
from pathlib import Path # Import Path from pathlib
import difflib # Import difflib for detailed comparison

# Import your parser from the separate file
# Make sure 'keymaerax_parser.py' is in the same directory or accessible via PYTHONPATH
from keymaerax_parser import KeYmaeraXParser

# Define constants for your test directories using pathlib
# Use Path(__file__).parent to get the directory of the current script.
# Then navigate relative to that.
# Assuming 'test' directory is a sibling of 'AST_parser' (i.e., one level up from test.py's directory)
SCRIPT_DIR = Path(__file__)
# Assuming your project structure is something like:
# project_root/
# ├── AST_parser/
# │   └── keymaerax_parser.py
# └── test/
#     ├── input/
#     │   ├── test1.txt
#     │   ├── test2.txt
#     │   └── test3.txt
#     ├── output/
#     │   ├── test1.txt
#     │   ├── test2.txt
#     │   └── test3.txt
#     └── test.py
# If 'test.py' is directly in 'project_root', adjust PROJECT_ROOT accordingly.
# For the structure above, if test.py is in 'test/', then SCRIPT_DIR.parent is 'test/',
# and SCRIPT_DIR.parent.parent would be 'project_root'.
# Let's assume 'test.py' is in 'project_root/test/'
PROJECT_ROOT = SCRIPT_DIR.parent # Go up two levels from test.py to project_root

TEST_DIR = PROJECT_ROOT / "test" / "input"
TEST_OUTPUT_DIR = PROJECT_ROOT / "test" / "output"

class TestKeYmaeraXParser(unittest.TestCase):

    def _load_test_case(self, test_name):
        """Loads source code and expected AST from files."""
        # Use the / operator for Path objects to join components
        source_code_path = TEST_DIR / f"{test_name}.txt"
        expected_output_path = TEST_OUTPUT_DIR / f"{test_name}.txt"

        # --- Debugging prints (highly recommended to verify paths) ---
        # You can uncomment these if you are having trouble with file not found errors
        # print(f"\n--- Debugging Paths for '{test_name}.txt' ---")
        # print(f"Script Directory: {SCRIPT_DIR.resolve()}")
        # print(f"Project Root (calculated): {PROJECT_ROOT.resolve()}")
        # print(f"Calculated Input Directory: {TEST_DIR.resolve()}")
        # print(f"Calculated Output Directory: {TEST_OUTPUT_DIR.resolve()}")
        # print(f"Attempting to open source file: {source_code_path.resolve()}")
        # print(f"Attempting to open output file: {expected_output_path.resolve()}")
        # print(f"--- End Debugging Paths ---")
        # --- End Debugging prints ---

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
        ast_root = parser.parse()
        actual_ast_str = ast_root.__repr__().strip()

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
                fromfile=f'Expected AST ({test_name})', # Label for the expected content
                tofile=f'Actual AST ({test_name})',     # Label for the actual content
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