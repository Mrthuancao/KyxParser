import sys
from pathlib import Path

# Ensure project and AST_parser_2 are importable when running this script directly
proj_root = Path(__file__).resolve().parents[1]
ast2 = proj_root / 'AST_parser_2'
sys.path.insert(0, str(ast2))
sys.path.insert(0, str(proj_root))

from keymaera_parser import KeYmaeraParser

s = Path('test/input/withBound/car2.txt').read_text()
m = KeYmaeraParser(s).parse()
print(m._repr_indented(0))