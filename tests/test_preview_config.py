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
        # The tube INI is a user-editable experiment. Keep test geometry fixed
        # without changing the user's saved radii, path, or output directory.
        self.tube_template = (ROOT / "configs/examples/xcat-roi-tube.ini").read_text()
        fixture_values = {"center_ijk": "404, 360, 1544 ; 404, 363, 1584 ; 401, 362, 1624",
                          "radius_mm": "2.5, 2.5, 2.3", "source_ids": "",
                          "context_tissues": "artery, bone, vein", "crop_half_width_mm": "40"}
        for key, value in fixture_values.items():
            self.tube_template = re.sub(r"^(" + re.escape(key) + r" = ).*?(\s+# NOTE .*)$",
                                       lambda match: match.group(1) + value + " " + match.group(2),
                                       self.tube_template, flags=re.MULTILINE)

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
        self.assertEqual(result["roi"]["shape"], "sphere")
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

    def test_original_id_filter_can_be_omitted_for_preview_and_edit(self):
        templates = (self.template, self.tube_template,
                     (ROOT / "configs/examples/xcat-edit.ini").read_text())
        for text in templates:
            with self.subTest(schema=text.split("schema_version = ")[1].splitlines()[0]):
                omitted = re.sub(r"^source_ids\s*=.*\n", "", text, flags=re.MULTILINE)
                blank = re.sub(r"^source_ids\s*=.*$", "source_ids =", text, flags=re.MULTILINE)
                self.assertEqual(self.load(omitted), self.load(blank))
                self.assertEqual(self.load(omitted)["selection"]["source_ids"], ())
                with self.assertRaisesRegex(ValueError, "unknown"):
                    self.load(omitted.replace("[selection]", "[selection]\nsource_id = 7"))

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

    def test_tube_example_preserves_order_radii_and_tissue_only_selection(self):
        text = self.tube_template
        result = self.load(text)
        self.assertEqual(result["study"]["schema_version"], "fakect.preview/2")
        self.assertEqual(result["selection"]["source_ids"], ())
        self.assertEqual(result["roi"]["shape"], "tube")
        self.assertEqual(result["roi"]["center_ijk"],
                         ((404., 360., 1544.), (404., 363., 1584.), (401., 362., 1624.)))
        self.assertEqual(result["roi"]["radius_mm"], (2.5, 2.5, 2.3))
        self.assertEqual(result["preview"]["context_tissues"], ("artery", "bone", "vein"))
        fractional = self.load(text, center_ijk="10.5, 20, 30 ; 9.25, 21, 28", radius_mm="1, 2")
        self.assertEqual(fractional["roi"]["center_ijk"], ((10.5, 20., 30.), (9.25, 21., 28.)))
        notes = set(re.findall(r"^# NOTE ([0-9]+) --", text, re.MULTILINE))
        assignments = [line for line in text.splitlines() if line and not line.startswith(("#", "["))]
        self.assertEqual(len(assignments), 21)
        for line in assignments:
            match = re.search(r"# NOTE ([0-9]+):", line)
            self.assertIsNotNone(match, line)
            self.assertIn(match.group(1), notes)

    def test_v2_sphere_accepts_fractional_center_but_legacy_schema_rejects_shape(self):
        text = self.template.replace("fakect.preview/1", "fakect.preview/2")
        text = text.replace("[roi]\n", "[roi]\nshape = sphere\n")
        result = self.load(text, center_ijk="404.25, 363.5, 1584")
        self.assertEqual(result["roi"]["shape"], "sphere")
        self.assertEqual(result["roi"]["center_ijk"], (404.25, 363.5, 1584.))
        with self.assertRaises(ValueError):
            self.load(text.replace("fakect.preview/2", "fakect.preview/1"))
        with self.assertRaises(ValueError):
            self.load(text.replace("shape = sphere\n", ""))

    def test_tube_rejects_inconsistent_or_ambiguous_geometry(self):
        text = self.tube_template
        invalid = [("shape", "cylinder"), ("center_ijk", "1,2,3"),
                   ("center_ijk", "1,2,3;1,2,3;4,5,6"),
                   ("center_ijk", "1,2,3;;4,5,6"), ("center_ijk", "1,2,3;4,5,6;"),
                   ("center_ijk", "1,2,3;4,5;7,8,9"),
                   ("center_ijk", "1,2,3;-4,5,6;7,8,9"),
                   ("center_ijk", "1,2,3;nan,5,6;7,8,9"),
                   ("radius_mm", "4"), ("radius_mm", ""), ("radius_mm", "4,4,"),
                   ("radius_mm", "4,0,3"), ("radius_mm", "4,-2,3"),
                   ("radius_mm", "4,inf,3"), ("radius_mm", "4,nan,3"),
                   ("crop_half_width_mm", "2.4")]
        for key, value in invalid:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load(text, **{key: value})

    def test_edit_example_parses_bounded_morphology_and_reassignment(self):
        text = (ROOT / "configs/examples/xcat-edit.ini").read_text()
        result = self.load(text)
        self.assertEqual(result["study"]["schema_version"], "fakect.edit/1")
        self.assertEqual(result["roi"]["shape"], "tube")
        self.assertEqual(result["selection"]["source_ids"], ())
        self.assertEqual(result["edit"], {"operation": "erosion", "distance_mm": 1.5,
                         "profile": "gaussian", "profile_axis": "tube", "shape_k": 10.,
                         "shape_window": (.25, .75)})
        self.assertEqual(result["reassignment"], {"allowed_tissues": ("soft_tissue", "muscle", "adipose"),
                         "max_distance_mm": 3., "unresolved": "preserve"})
        notes = set(re.findall(r"^# NOTE ([0-9]+) --", text, re.MULTILINE))
        assignments = [line for line in text.splitlines() if line and not line.startswith(("#", "["))]
        self.assertEqual(len(assignments), 30)
        for line in assignments:
            match = re.search(r"# NOTE ([0-9]+):", line)
            self.assertIsNotNone(match, line)
            self.assertIn(match.group(1), notes)

    def test_edit_none_uniform_sphere_and_axis_choices(self):
        text = (ROOT / "configs/examples/xcat-edit.ini").read_text()
        result = self.load(text, operation="none", distance_mm="0", allowed_tissues="")
        self.assertEqual(result["edit"]["distance_mm"], 0.)
        self.assertEqual(result["reassignment"]["allowed_tissues"], ())
        for axis in ("i", "j", "k"):
            result = self.load(text, shape="sphere", center_ijk="404.5,363,1584", radius_mm="4.7",
                               profile_axis=axis, unresolved="error", operation="dilation")
            self.assertEqual(result["roi"]["center_ijk"], (404.5, 363., 1584.))
            self.assertEqual(result["edit"]["profile_axis"], axis)
            self.assertEqual(result["reassignment"]["unresolved"], "error")
        # Uniform profiles do not use the axis; retain the user's unused setting.
        result = self.load(text, shape="sphere", center_ijk="404,363,1584", radius_mm="4.7",
                           profile="uniform", profile_axis="tube", shape_window="0,1")
        self.assertEqual(result["edit"]["shape_window"], (0., 1.))

    def test_optional_erosion_surrounding_assignment_retains_strict_boolean_without_legacy_default(self):
        text = (ROOT / "configs/examples/xcat-edit.ini").read_text()
        legacy = self.load(text)
        self.assertNotIn("assign_surrounding_tissue", legacy["edit"])
        for value, expected in (("true", True), ("false", False)):
            with self.subTest(value=value):
                explicit = text.replace("[edit]", "[edit]\nassign_surrounding_tissue = " + value)
                result = self.load(explicit)
                self.assertIs(result["edit"].pop("assign_surrounding_tissue"), expected)
                self.assertEqual(result, legacy)
        explicit_true = text.replace("[edit]", "[edit]\nassign_surrounding_tissue = true")
        for operation in ("dilation", "none"):
            with self.subTest(operation=operation):
                self.assertIs(self.load(explicit_true, operation=operation)["edit"]["assign_surrounding_tissue"], True)

    def test_surrounding_assignment_rejects_non_boolean_values_and_false_outside_erosion(self):
        text = (ROOT / "configs/examples/xcat-edit.ini").read_text()
        for value in ("", "False", "TRUE", "0", "1", "yes", "no", "nan", "false,true", "false\n    true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(text.replace("[edit]", "[edit]\nassign_surrounding_tissue = " + value))
        explicit = text.replace("[edit]", "[edit]\nassign_surrounding_tissue = false")
        for operation in ("dilation", "none"):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "false requires operation=erosion"):
                self.load(explicit, operation=operation)
        for section in ("reassignment", "roi", "selection"):
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "unknown.*assign_surrounding_tissue"):
                self.load(text.replace("[" + section + "]", "[" + section + "]\nassign_surrounding_tissue = false"))
        with self.assertRaisesRegex(ValueError, "unknown.*assign_surrounding_tissues"):
            self.load(text.replace("[edit]", "[edit]\nassign_surrounding_tissues = false"))
        with self.assertRaisesRegex(ValueError, "missing.*operation"):
            self.load(re.sub(r"^operation = .*\n", "", explicit, flags=re.MULTILINE))

    def test_edit_rejects_invalid_parameters_and_unsafe_implicit_defaults(self):
        text = (ROOT / "configs/examples/xcat-edit.ini").read_text()
        invalid = [("operation", "stenosis"), ("operation", "Erosion"), ("distance_mm", "0"),
                   ("distance_mm", "-1"), ("distance_mm", "nan"), ("distance_mm", "inf"),
                   ("profile", "Gaussian"), ("profile", "linear"), ("profile_axis", "z"),
                   ("shape_k", "0"), ("shape_k", "-1"), ("shape_k", "nan"),
                   ("shape_window", ""), ("shape_window", ".5"), ("shape_window", "0,.5,1"),
                   ("shape_window", ".5,.5"), ("shape_window", ".75,.25"),
                   ("shape_window", "-.1,.5"), ("shape_window", ".5,1.1"),
                   ("shape_window", ".5,inf"), ("shape_window", "0,1,"),
                   ("allowed_tissues", "soft_tissue,muscle,soft_tissue"),
                   ("allowed_tissues", "bone,"), ("allowed_tissues", "Muscle"),
                   ("max_distance_mm", "0"), ("max_distance_mm", "-3"),
                   ("max_distance_mm", "inf"), ("unresolved", "guess"), ("unresolved", "")]
        for key, value in invalid:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load(text, **{key: value})
        with self.assertRaisesRegex(ValueError, "requires roi.shape=tube"):
            self.load(text, shape="sphere", center_ijk="404,363,1584", radius_mm="4.7")
        with self.assertRaises(ValueError):
            self.load(text, operation="none", distance_mm="-1")
        for section in ("edit", "reassignment"):
            with self.subTest(section=section), self.assertRaises(ValueError):
                self.load(text.replace("[" + section + "]", "[unknown_section]"))
        with self.assertRaises(ValueError):
            self.load(text.replace("[edit]", "[edit]\nscale = 1"))
        with self.assertRaises(ValueError):
            self.load(text.replace("[reassignment]", "[reassignment]\nallow_all = true"))
        with self.assertRaises(ValueError):
            self.load(re.sub(r"^max_distance_mm = .*\n", "", text, flags=re.MULTILINE))

    def test_preview_schemas_cannot_silently_accept_edit_settings(self):
        text = (ROOT / "configs/examples/xcat-edit.ini").read_text()
        with self.assertRaises(ValueError):
            self.load(text.replace("fakect.edit/1", "fakect.preview/2"))
        extra_sections = "\n[edit]\noperation = none\n[reassignment]\nallowed_tissues =\n"
        with self.assertRaises(ValueError):
            self.load(self.template + extra_sections)
        result = self.load(self.template)
        self.assertNotIn("edit", result)
        self.assertNotIn("reassignment", result)
        result = self.load(self.tube_template)
        self.assertNotIn("edit", result)
        self.assertNotIn("reassignment", result)


if __name__ == "__main__":
    unittest.main()
