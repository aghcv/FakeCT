"""Exercise import-free API contracts, source validation and offline documentation."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/dev"))
from check_quality import SCOPE, check_repository, contract_errors, iter_functions, load_scope
from build_api_docs import build_reference, render_reference


class DeveloperToolsTests(unittest.TestCase):
    def test_contracts_require_real_argument_names_and_return_sections(self):
        tree = ast.parse('''
def f(mask, /, spacing, *, strict=False):
    """Measure a physical mask.

    Parameters
    ----------
    mask : array
        Foreground membership.
    wrong_name : tuple
        Positive voxel spacing.

    Returns
    -------
    float
        Physical volume.
    """
''')
        errors = contract_errors(tree.body[0])
        self.assertIn("Parameters does not document spacing", errors)
        self.assertIn("Parameters does not document strict", errors)
        self.assertIn("Parameters documents nonexistent argument wrong_name", errors)
        tree.body[0].body[0].value.value = "Measure a physical mask."
        self.assertIn("missing or empty Returns section", contract_errors(tree.body[0]))

    def test_class_methods_are_checked_without_requiring_local_callbacks(self):
        tree = ast.parse('''
class Loader:
    def read(self, path):
        def callback(value):
            return value
        return callback(path)
''')
        functions = list(iter_functions(tree))
        self.assertEqual([name for name, _ in functions], ["Loader.read"])
        self.assertEqual(contract_errors(functions[0][1]), ["missing function docstring"])

    def test_scope_rejects_escaping_duplicate_and_invalid_module_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scope.json"
            baseline = {"schema_version": "fakect.development-api/1",
                        "modules": ["src/api.py"], "syntax_roots": ["src"],
                        "language_paths": ["README.md"]}
            for field, bad in (("modules", ["../outside.py"]),
                               ("modules", ["/outside.py"]),
                               ("modules", ["src/api.py", "src/api.py"]),
                               ("modules", ["src/payload.txt"]),
                               ("syntax_roots", [])):
                with self.subTest(field=field, value=bad):
                    path.write_text(json.dumps({**baseline, field: bad}))
                    with self.assertRaises(ValueError):
                        load_scope(path)

    def test_api_generation_escapes_text_and_never_executes_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "source.py"
            path.write_text('''
raise AssertionError("source must not execute")
import unavailable_backend
def compare(image, *, mode="<unsafe>"):
    """Compare <image> safely.

    Parameters
    ----------
    image : object
        An image with <script> text.
    mode : str
        Comparison selector.

    Returns
    -------
    bool
        Whether the image passes.
    """
    return True
''')
            scope = {"modules": ["source.py"]}
            output = root / "docs/api.html"
            result = build_reference(root, scope, output)
            document = output.read_text()
            self.assertIn("&lt;script&gt;", document)
            self.assertNotIn("An image with <script>", document)
            self.assertIn("compare(image, *, mode=", document)
            self.assertEqual(result["bytes"], len(output.read_bytes()))
            self.assertEqual(document, render_reference(root, scope))
            self.assertEqual(output.read_bytes(), document.encode("utf-8"))

    def test_source_checker_reports_syntax_and_contract_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "syntax.py").write_text("def broken(:\n")
            (root / "api.py").write_text("def undocumented(value):\n    return value\n")
            (root / "README.md").write_text("Algorithm development framework\n")
            scope = {"modules": ["api.py"], "syntax_roots": ["api.py", "syntax.py"],
                     "language_paths": ["README.md"]}
            result = check_repository(root, scope)
            self.assertFalse(result["passed"])
            messages = [error["message"] for error in result["errors"]]
            self.assertTrue(any("syntax/read failure" in message for message in messages))
            self.assertIn("undocumented: missing function docstring", messages)

    def test_checked_in_reference_matches_source_and_real_scope_passes(self):
        scope = load_scope(SCOPE)
        result = check_repository(ROOT, scope)
        self.assertTrue(result["passed"], json.dumps(result["errors"], indent=2))
        self.assertGreater(result["documented_functions"], 300)
        self.assertEqual((ROOT / "docs/developer/api.html").read_text(),
                         render_reference(ROOT, scope))

    def test_quality_cli_uses_only_the_standard_library(self):
        script = ROOT / "scripts/dev/check_quality.py"
        result = subprocess.run([sys.executable, "-S", str(script), "--json"],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertTrue(json.loads(result.stdout)["passed"])


if __name__ == "__main__":
    unittest.main()
