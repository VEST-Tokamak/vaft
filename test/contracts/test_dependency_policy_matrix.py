import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


class DependencyPolicyMatrixTests(unittest.TestCase):
    def test_core_dependencies_use_numpy2_and_current_hdf5_stack(self):
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        deps = set(data["project"]["dependencies"])

        expected_specs = {
            "h5py==3.16.0",
            "h5pyd==0.20.0",
            "numpy>=2.0.0,<3",
            "scipy>=1.13.0,<2",
            "matplotlib>=3.7.3,<4",
            "imas_core>=5.6.0,<6",
            "imas_python>=2.1.0,<3",
        }
        for spec in expected_specs:
            self.assertIn(spec, deps)

        removed_dependency = "au" "rorafusion"
        self.assertFalse(any(dep.lower().startswith(removed_dependency) for dep in deps))

    def test_default_hsds_stack_selects_numpy2_without_uv_override(self):
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        overrides = set(data.get("tool", {}).get("uv", {}).get("override-dependencies", []))
        self.assertNotIn("numpy>=2,<3", overrides)
        dependencies = set(data["project"]["dependencies"])
        self.assertIn("numpy>=2.0.0,<3", dependencies)
        self.assertIn("h5pyd==0.20.0", dependencies)


if __name__ == "__main__":
    unittest.main()
