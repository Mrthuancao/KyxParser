import sys
import json
import dataclasses
from pathlib import Path

# Ensure parser package is importable
proj_root = Path(__file__).resolve().parents[1]
ast2 = proj_root / 'AST_parser_2'
if str(ast2) not in sys.path:
    sys.path.insert(0, str(ast2))
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from keymaera_parser import KeYmaeraParser

infile = proj_root / 'test' / 'input' / 'withBound' / 'car2.txt'
text = infile.read_text(encoding='utf-8')
model = KeYmaeraParser(text).parse()


def to_obj(o):
    basic_types = (str, int, float, bool, type(None))
    if isinstance(o, basic_types):
        return o
    if isinstance(o, list):
        return [to_obj(x) for x in o]
    if isinstance(o, dict):
        return {k: to_obj(v) for k, v in o.items()}
    try:
        if dataclasses.is_dataclass(o):
            d = dataclasses.asdict(o)
            return {'_type': type(o).__name__, 'fields': to_obj(d)}
    except Exception:
        pass
    if hasattr(o, '__dict__'):
        d = {}
        for k, v in o.__dict__.items():
            if k.startswith('_'):
                continue
            d[k] = to_obj(v)
        return {'_type': type(o).__name__, 'fields': d}
    return str(o)


ast_json = to_obj(model)
print(json.dumps(ast_json, indent=2))
