"""Validate the user-editable ROI preview contract before accessing volumes."""
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fakect_config import load_preview_config


class PreviewConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "external-input.ini"
        self.template = (ROOT / "configs/examples/xcat-roi.ini").read_text()

    def load(self, text=None, **values):
        text = self.template if text is None else text
        for key, value in values.items():
            text, replacements = re.subn(r"^" + re.escape(key) + r" = .*?$",
                                         lambda _: key + " = " + value, text, flags=re.MULTILINE)
            self.assertEqual(replacements, 1)
        self.path.write_text(text)
        return load_preview_config(self.path, repo_root=self.root)

    def test_example_normalizes_types_and_repository_relative_paths(self):
        result = self.load()
        self.assertEqual(result["study"]["schema_version"], "fakect.preview/1")
        self.assertEqual(result["input"]["case_id"], "260602")
        self.assertEqual(result["input"]["frame"], 1)
        self.assertEqual(result["input"]["audit"], self.root / "docs/integration/2026-09-16/xcat-results.json")
        self.assertEqual(result["input"]["root"], Path("/home/aghorban/slurm/xcat"))
        self.assertEqual(result["selection"]["source_ids"], (1185,))
        self.assertEqual(result["roi"]["center_ijk"], (404, 363, 1584))
        self.assertEqual(result["roi"]["radius_mm"], 12.0)
        self.assertIs(result["roi"]["coordinate_reviewed"], False)
        self.assertIsNone(result["preview"]["slice_ijk"])
        self.assertEqual(result["preview"]["context_tissues"], ("bone", "vein"))
        self.assertEqual(result["preview"]["volume_stride"], 2)
        self.assertEqual(result["output"]["directory"], self.root / "outputs/roi/carotid-v2")
        self.assertFalse(result["output"]["directory"].exists())
        # The input file's own directory must not change its embedded path base.
        external = self.root / "different-folder"
        external.mkdir()
        moved = external / "moved.ini"
        moved.write_text(self.template)
        self.assertEqual(load_preview_config(moved, repo_root=self.root), result)

    def test_blank_selections_explicit_sections_and_signed_identity(self):
        result = self.load(source_ids="", context_tissues="", slice_ijk="404, 362, 1583",
                           coordinate_reviewed="true", case_id="000040")
        self.assertEqual(result["selection"]["source_ids"], ())
        self.assertEqual(result["preview"]["context_tissues"], ())
        self.assertEqual(result["preview"]["slice_ijk"], (404, 362, 1583))
        self.assertIs(result["roi"]["coordinate_reviewed"], True)
        self.assertEqual(result["input"]["case_id"], "000040")
        result = self.load(source_ids="-2147483648, -1185, 1185, 2147483647")
        self.assertEqual(result["selection"]["source_ids"], (-2147483648, -1185, 1185, 2147483647))

    def test_rejects_unused_settings_sections_duplicates_and_schema_versions(self):
        malformed = [self.template.replace("[study]", "[study]\nunused = 4"),
                     self.template + "\n[morphology]\nscale = 1.2\n",
                     self.template.replace("[study]", "[study]\nname = duplicate"),
                     self.template.replace("[study]", "[DEFAULT]\nframe = 1\n[study]"),
                     self.template.replace("frame = 1 ", "Frame = 1 "),
                     self.template.replace("schema_version = fakect.preview/1", "schema_version = fakect.preview/2"),
                     self.template.replace("[selection]", "[selection_typo]"),
                     self.template.replace("context_tissues = bone, vein", "missing_key = bone, vein")]
        for text in malformed:
            with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                self.load(text)

    def test_rejects_bad_coordinates_numbers_and_booleans(self):
        invalid = [("center_ijk", "1, 2"), ("center_ijk", "-1, 2, 3"),
                   ("center_ijk", "1.5, 2, 3"), ("slice_ijk", "ROi"),
                   ("slice_ijk", "1, 2, 3,"), ("radius_mm", "0"), ("radius_mm", "nan"),
                   ("radius_mm", "inf"), ("crop_half_width_mm", "10"),
                   ("overlay_opacity", "1.01"), ("volume_opacity", "-0.1"),
                   ("context_opacity", "nan"), ("volume_stride", "0"), ("volume_stride", "1.0"),
                   ("frame", "0"), ("frame", "1_000"), ("coordinate_reviewed", "yes"),
                   ("coordinate_reviewed", "True")]
        for key, value in invalid:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load(**{key: value})
        result = self.load(overlay_opacity="0", volume_opacity="1", context_opacity="0")
        self.assertEqual(result["preview"]["overlay_opacity"], 0)
        self.assertEqual(result["preview"]["volume_opacity"], 1)

    def test_rejects_invalid_ids_names_and_ambiguous_lists(self):
        invalid = [("source_ids", "2147483648"), ("source_ids", "-2147483649"),
                   ("source_ids", "1185,1185"), ("source_ids", "1185,,1186"),
                   ("source_ids", "1.0"), ("context_tissues", "bone,"),
                   ("context_tissues", "bone,bone"), ("context_tissues", "Bone"),
                   ("tissue", "artery OR vein"), ("tissue", ""), ("case_id", "../260602"),
                   ("name", "title with spaces")]
        for key, value in invalid:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load(**{key: value})

    def test_paths_allow_spaces_and_tilde_without_shell_or_interpolation(self):
        result = self.load(catalog="outputs/my atlas/labels.json # NOTE: inline comment",
                           root="~/xcat", directory="outputs/100% complete")
        self.assertEqual(result["input"]["catalog"], self.root / "outputs/my atlas/labels.json")
        self.assertEqual(result["input"]["root"], Path.home() / "xcat")
        self.assertEqual(result["output"]["directory"], self.root / "outputs/100% complete")
        for value in ("", "$HOME/xcat", "data/$(date)"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(root=value)
        with self.assertRaisesRegex(ValueError, "one line"):
            self.load(self.template.replace("root = /home/aghorban/slurm/xcat", "root = data\n    extra"))

    def test_each_parameter_links_to_a_numbered_note(self):
        notes = set(re.findall(r"^# NOTE ([0-9]+) --", self.template, re.MULTILINE))
        assignments = [line for line in self.template.splitlines()
                       if line and not line.startswith(("#", "["))]
        self.assertEqual(len(assignments), 20)
        for line in assignments:
            match = re.search(r"# NOTE ([0-9]+):", line)
            self.assertIsNotNone(match, line)
            self.assertIn(match.group(1), notes)


if __name__ == "__main__":
    unittest.main()
