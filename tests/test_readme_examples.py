import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKECT = REPO_ROOT / "src" / "fakect.py"


class ReadmeExampleSmokeTests(unittest.TestCase):
    def run_fakect_example(self, cli_args, expected_filename, expected_keys):
        with tempfile.TemporaryDirectory(prefix="fakect-readme-") as tmp:
            out_dir = Path(tmp)
            cmd = [
                sys.executable,
                str(FAKECT),
                *cli_args,
                "--out",
                str(out_dir),
                "--no-show",
            ]
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                "Command failed:\n"
                f"{' '.join(cmd)}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}",
            )

            output_path = out_dir / expected_filename
            self.assertTrue(
                output_path.exists(),
                f"Expected output was not created: {output_path}",
            )
            with np.load(output_path) as npz:
                for key in expected_keys:
                    self.assertIn(key, npz.files)
                    self.assertGreater(npz[key].size, 0)

    def test_stl_readme_examples_run(self):
        examples = [
            (
                "cube",
                ["--in", "data/cube.stl", "--n", "8"],
                "cube_masks.npz",
            ),
            (
                "sphere",
                ["--in", "data/sphere.stl", "--n", "8"],
                "sphere_masks.npz",
            ),
            (
                "carotid",
                ["--in", "data/carotid.stl", "--n", "8", "--margin", "0.10"],
                "carotid_masks.npz",
            ),
        ]
        for name, cli_args, expected_filename in examples:
            with self.subTest(example=name):
                self.run_fakect_example(
                    cli_args,
                    expected_filename,
                    expected_keys=["inside", "on", "out", "spacing", "origin"],
                )

    def test_vti_readme_example_runs(self):
        self.run_fakect_example(
            ["--in", "data/arm.vti", "--vti-max-dim", "64"],
            "arm_masks.npz",
            expected_keys=[
                "scalar_values",
                "spacing",
                "origin",
                "layer_ids",
                "layer_names",
                "source_type",
                "vti_array",
                "vti_volume_kind",
            ],
        )

    def test_vti_fixture_smoke_example_runs(self):
        self.run_fakect_example(
            ["--in", "tests/activity_grid_000_001_005.vti", "--vti-max-dim", "64"],
            "activity_grid_000_001_005_masks.npz",
            expected_keys=[
                "scalar_values",
                "spacing",
                "origin",
                "layer_ids",
                "source_type",
                "vti_array",
                "vti_volume_kind",
            ],
        )


if __name__ == "__main__":
    unittest.main()
