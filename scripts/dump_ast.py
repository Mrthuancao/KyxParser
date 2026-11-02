"""Parse a KeYmaera-style model and save its AST as JSON.

Usage:
    python scripts/dump_ast.py                      # Test 1 -> logs/test1/test1_ast.json
    python scripts/dump_ast.py --model test/input/withBound/car2.txt --out logs/car2/car2_ast.json
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "AST_parser_2", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from keymaera_parser import KeYmaeraParser


def to_obj(o):
    if isinstance(o, (str, int, float, bool, type(None))):
        return o
    if isinstance(o, (list, tuple)):
        return [to_obj(x) for x in o]
    if isinstance(o, dict):
        return {str(k): to_obj(v) for k, v in o.items()}
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        fields = {f.name: to_obj(getattr(o, f.name)) for f in dataclasses.fields(o)}
        return {"_type": type(o).__name__, "fields": fields}
    if hasattr(o, "__dict__"):
        fields = {k: to_obj(v) for k, v in vars(o).items() if not k.startswith("_")}
        return {"_type": type(o).__name__, "fields": fields}
    return str(o)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "test" / "input" / "withBound" / "test1.txt"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "logs" / "test1" / "test1_ast.json"
    )
    args = parser.parse_args()

    model = KeYmaeraParser(args.model.read_text(encoding="utf-8")).parse()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(to_obj(model), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved AST: {args.out}")


if __name__ == "__main__":
    main()
