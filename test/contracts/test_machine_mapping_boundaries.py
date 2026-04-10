import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MACHINE_MAPPING_DIR = ROOT / "vaft" / "machine_mapping"
FORBIDDEN_MODULES = {
    "vaft.builders",
    "vaft.loaders",
    "vaft.models",
}


def iter_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                relative = "." * node.level + (node.module or "")
                yield relative
            elif node.module:
                yield node.module


class MachineMappingBoundaryTests(unittest.TestCase):
    def test_machine_mapping_does_not_depend_on_removed_packages(self):
        failures = []

        for path in sorted(MACHINE_MAPPING_DIR.glob("*.py")):
            for imported in iter_imports(path):
                if imported in FORBIDDEN_MODULES:
                    failures.append(f"{path.name}: {imported}")
                if imported in {".uncertainty", "._common"}:
                    failures.append(f"{path.name}: {imported}")

        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
