import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


class DependencyPolicyMatrixTests(unittest.TestCase):
    def test_core_dependencies_allow_legacy_and_latest_axes(self):
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        deps = set(data["project"]["dependencies"])

        expected_specs = {
            "numpy>=1.26.4,<3",
            "scipy>=1.13.0,<2",
            "matplotlib>=3.7.3,<4",
            "imas_core>=5.6.0,<6",
            "imas_python>=2.1.0,<3",
        }
        for spec in expected_specs:
            self.assertIn(spec, deps)

    def test_uv_override_allows_dual_compat_numpy(self):
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        overrides = set(data.get("tool", {}).get("uv", {}).get("override-dependencies", []))
        self.assertIn("numpy>=1.26.4,<3", overrides)


if __name__ == "__main__":
    unittest.main()
