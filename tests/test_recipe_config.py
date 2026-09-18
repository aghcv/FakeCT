"""Named recipe contracts: strict references, ordering and bounded iteration."""
import configparser
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fakect_config import load_preview_config
from fakect_recipe_config import load_recipe_config, parse_recipe_sections


class RecipeConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "recipe.ini"
        self.template = (ROOT / "tests/fixtures/thoracic-aorta-recipe.ini").read_text()

    def parser(self):
        parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",),
                                           strict=True, empty_lines_in_values=False)
        parser.optionxform = str
        parser.read_string(self.template)
        return parser

    def load(self, parser=None, loader=load_recipe_config):
        parser = self.parser() if parser is None else parser
        stream = io.StringIO()
        parser.write(stream)
        self.path.write_text(stream.getvalue())
        return loader(self.path, repo_root=self.root)

    def test_example_types_dispatch_and_paths_without_source_files(self):
        config = self.load()
        self.assertEqual(config, self.load(loader=load_preview_config))
        self.assertEqual(config["study"]["schema_version"], "fakect.recipe/1")
        self.assertNotIn("edit", config)
        self.assertNotIn("train", config)
        self.assertNotIn("model", config)
        self.assertEqual(config["recipe"], {"steps": ("ascending_expand", "descending_narrow", "arch_refine"),
                                            "overlap": "sequential"})
        self.assertEqual(config["selection"]["source_ids"], (2922,))
        self.assertEqual(config["reassignment"]["mode"], "stiffness")
        self.assertEqual(config["reassignment"]["allowed_tissues"], ())
        self.assertEqual(config["reassignment"]["stiffness"]["default"], .05)
        self.assertEqual(config["reassignment"]["stiffness"]["tissues"]["bone"], 1.)
        self.assertEqual(config["reassignment"]["stiffness"]["tissues"]["skin"], .95)
        self.assertEqual(config["reassignment"]["stiffness"]["labels"], {2897: .05})
        self.assertEqual(config["input"]["audit"], self.root / "docs/integration/2026-09-16/xcat-results.json")
        self.assertTrue(config["roi"]["coordinate_reviewed"])
        self.assertFalse(config["rois"]["ascending"]["coordinate_reviewed"])
        self.assertEqual(config["rois"]["ascending"]["center_ijk"][0], (387., 360., 1352.))
        self.assertEqual(config["rois"]["ascending"]["radius_mm"], (10., 10., 10., 12.))
        self.assertNotIn("crop_half_width_mm", config["rois"]["ascending"])
        self.assertEqual(config["edits"]["ascending_expand"], {
            "roi": "ascending", "iterations": 1, "operation": "dilation", "distance_mm": 2.,
            "profile": "gaussian", "profile_axis": "tube", "shape_k": 6., "shape_window": (0., 1.)})
        self.assertFalse(config["output"]["directory"].exists())

    def test_optional_reassignment_mode_defaults_to_legacy_allowlist(self):
        parser = self.parser()
        parser.remove_section("stiffness")
        parser.remove_section("stiffness.labels")
        parser.remove_option("reassignment", "mode")
        parser["reassignment"]["allowed_tissues"] = "soft_tissue,muscle,adipose"
        implicit = self.load(parser)
        self.assertEqual(implicit["reassignment"]["mode"], "allowlist")
        self.assertEqual(implicit["reassignment"]["allowed_tissues"], ("soft_tissue", "muscle", "adipose"))
        parser["reassignment"]["mode"] = "allowlist"
        self.assertEqual(implicit, self.load(parser))
        parser["reassignment"]["allowed_tissues"] = ""
        self.assertEqual(self.load(parser)["reassignment"]["allowed_tissues"], ())
        self.assertNotIn("stiffness", self.load(parser)["reassignment"])
        parser["reassignment"]["mode"] = "permissive_except_bone_skin"
        permissive = self.load(parser)
        self.assertEqual(permissive["reassignment"]["mode"], "permissive_except_bone_skin")
        self.assertNotIn("stiffness", permissive["reassignment"])

    def test_recipe_needs_no_anatomical_id_inputs(self):
        parser = self.parser()
        parser["selection"]["source_ids"] = ""
        parser.remove_section("stiffness.labels")
        blank = self.load(parser)
        parser.remove_option("selection", "source_ids")
        omitted = self.load(parser)
        self.assertEqual(blank, omitted)
        self.assertEqual(omitted["selection"]["source_ids"], ())
        self.assertEqual(omitted["reassignment"]["stiffness"]["labels"], {})

    def test_permissive_mode_rejects_ambiguous_allowlists_unknown_modes_and_missing_fields(self):
        for mode in ("", "permissive", "all", "Allowlist", "permissive_except_bone_skin\nextra"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                parser = self.parser()
                parser["reassignment"]["mode"] = mode
                self.load(parser)
        for allowed in ("soft_tissue", "unknown", "bone", "soft_tissue,muscle,adipose"):
            with self.subTest(allowed=allowed), self.assertRaisesRegex(ValueError, "must be blank"):
                parser = self.parser()
                parser["reassignment"]["mode"] = "permissive_except_bone_skin"
                parser.remove_section("stiffness")
                parser.remove_section("stiffness.labels")
                parser["reassignment"]["allowed_tissues"] = allowed
                self.load(parser)
        for key in ("allowed_tissues", "max_distance_mm", "unresolved"):
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "missing"):
                parser = self.parser()
                parser.remove_option("reassignment", key)
                self.load(parser)

    def test_stiffness_all_values_editable_optional_categories_and_signed_id_overrides(self):
        parser = self.parser()
        parser["stiffness"] = {"default": "1", "bone": "0.25", "skin": "0", "fluid": ".3"}
        parser["stiffness.labels"] = {"-2147483648": "0", "+7": ".6", "2147483647": "1"}
        stiffness = self.load(parser)["reassignment"]["stiffness"]
        self.assertEqual(stiffness, {"default": 1., "tissues": {"bone": .25, "skin": 0., "fluid": .3},
                                    "labels": {-(2**31): 0., 7: .6, 2**31-1: 1.}})
        parser.remove_section("stiffness.labels")
        self.assertEqual(self.load(parser)["reassignment"]["stiffness"]["labels"], {})
        parser["stiffness.labels"] = {}
        self.assertEqual(self.load(parser)["reassignment"]["stiffness"]["labels"], {})

    def test_stiffness_sections_require_matching_mode_and_required_settings(self):
        for mode in ("allowlist", "permissive_except_bone_skin"):
            for remaining in (("stiffness", "stiffness.labels"), ("stiffness",), ("stiffness.labels",)):
                with self.subTest(mode=mode, remaining=remaining), self.assertRaisesRegex(ValueError, "require reassignment.mode"):
                    parser = self.parser()
                    parser["reassignment"]["mode"] = mode
                    for section in {"stiffness", "stiffness.labels"} - set(remaining):
                        parser.remove_section(section)
                    self.load(parser)
        parser = self.parser()
        parser.remove_section("stiffness")
        with self.assertRaisesRegex(ValueError, "Missing.*stiffness"):
            self.load(parser)
        for key in ("default", "bone", "skin"):
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "missing"):
                parser = self.parser()
                parser.remove_option("stiffness", key)
                self.load(parser)
        parser = self.parser()
        parser["reassignment"]["allowed_tissues"] = "soft_tissue"
        with self.assertRaisesRegex(ValueError, "must be blank"):
            self.load(parser)

    def test_stiffness_rejects_invalid_categories_factors_id_keys_and_normalized_duplicates(self):
        for name in ("Bone", "background", "peri", "stifness", "2897"):
            with self.subTest(category=name), self.assertRaisesRegex(ValueError, "unknown"):
                parser = self.parser()
                parser["stiffness"][name] = ".5"
                self.load(parser)
        for section, key in (("stiffness", "default"), ("stiffness", "bone"),
                             ("stiffness", "skin"), ("stiffness.labels", "2897")):
            for value in ("-0.01", "1.01", "nan", "inf", "", "0.5\n0.6"):
                with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                    parser = self.parser()
                    parser[section][key] = value
                    self.load(parser)
        for key in ("-2147483649", "2147483648", "1.0", "bone", "1_000"):
            with self.subTest(label=key), self.assertRaises(ValueError):
                parser = self.parser()
                parser["stiffness.labels"][key] = ".2"
                self.load(parser)
        for first, second in (("7", "+7"), ("7", "007"), ("0", "-0")):
            with self.subTest(first=first, second=second), self.assertRaisesRegex(ValueError, "duplicate original ID"):
                parser = self.parser()
                parser["stiffness.labels"] = {first: ".2", second: ".3"}
                self.load(parser)

    def test_steps_order_is_explicit_and_parser_is_not_mutated(self):
        parser = self.parser()
        parser["recipe"]["steps"] = "arch_refine, descending_narrow, ascending_expand"
        before = {section: dict(parser[section]) for section in parser.sections()}
        config = parse_recipe_sections(parser, self.root)
        self.assertEqual(config["recipe"]["steps"], ("arch_refine", "descending_narrow", "ascending_expand"))
        self.assertEqual({section: dict(parser[section]) for section in parser.sections()}, before)

    def test_optional_recipe_roi_role_preserves_legacy_and_retains_explicit_choice(self):
        legacy = self.load()
        self.assertNotIn("roi_role", legacy["recipe"])
        for role in ("boundary", "selection"):
            with self.subTest(role=role):
                parser = self.parser()
                parser["recipe"]["roi_role"] = role
                before = {section: dict(parser[section]) for section in parser.sections()}
                config = self.load(parser)
                self.assertEqual(config, self.load(parser, loader=load_preview_config))
                self.assertEqual(config["recipe"].pop("roi_role"), role)
                self.assertEqual(config, legacy)
                self.assertEqual({section: dict(parser[section]) for section in parser.sections()}, before)

    def test_recipe_roi_role_rejects_unknown_case_and_ambiguous_values(self):
        for value in ("", "Boundary", "Selection", "seed", "crop", "all", "boundary,selection",
                      "selection\nboundary"):
            with self.subTest(role=value), self.assertRaises(ValueError):
                parser = self.parser()
                parser["recipe"]["roi_role"] = value
                self.load(parser)
        parser = self.parser()
        parser["recipe"]["roi_role"] = "seed"
        with self.assertRaisesRegex(ValueError, "boundary.*inside the ROI.*selection.*initial targets"):
            self.load(parser)

    def test_recipe_roi_role_is_global_and_does_not_relax_required_recipe_fields(self):
        for section in ("roi", "roi.ascending", "edit.ascending_expand"):
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "unknown.*roi_role"):
                parser = self.parser()
                parser[section]["roi_role"] = "selection"
                self.load(parser)
        for key in ("steps", "overlap"):
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "missing"):
                parser = self.parser()
                parser["recipe"]["roi_role"] = "selection"
                parser.remove_option("recipe", key)
                self.load(parser)

    def test_multiple_steps_can_share_roi_and_unused_roi_is_allowed(self):
        parser = self.parser()
        parser["edit.arch_refine"]["roi"] = "ascending"
        parser["edit.arch_refine"]["operation"] = "erosion"
        parser["recipe"]["overlap"] = "error"
        config = self.load(parser)
        self.assertIn("arch", config["rois"])
        self.assertEqual(config["edits"]["arch_refine"]["roi"], "ascending")
        self.assertEqual(config["recipe"]["overlap"], "error")

    def test_main_roi_can_be_shared_by_edits_without_any_named_rois(self):
        parser = self.parser()
        for section in list(parser.sections()):
            if section.startswith("roi."):
                parser.remove_section(section)
            elif section.startswith("edit."):
                parser[section]["roi"] = "main"
        before = {section: dict(parser[section]) for section in parser.sections()}
        config = self.load(parser)
        self.assertEqual(config["rois"], {})
        self.assertEqual(config, self.load(parser, loader=load_preview_config))
        self.assertEqual(config["edits"]["ascending_expand"], {
            "roi": "main", "iterations": 1, "operation": "dilation", "distance_mm": 2.,
            "profile": "gaussian", "profile_axis": "tube", "shape_k": 6., "shape_window": (0., 1.)})
        self.assertTrue(all(edit["roi"] == "main" for edit in config["edits"].values()))
        self.assertEqual({section: dict(parser[section]) for section in parser.sections()}, before)

    def test_explicit_named_main_roi_is_reserved(self):
        parser = self.parser()
        parser["roi.main"] = dict(parser["roi.ascending"])
        with self.assertRaisesRegex(ValueError, r"\[roi.main\] is reserved.*top-level \[roi\]"):
            self.load(parser)

    def test_point_range_normalizes_1_based_inclusive_main_or_named_node_interval(self):
        for roi_name, interval, expected in (("main", "2,8", (2, 8)),
                                             ("ascending", " 2, 4 ", (2, 4)),
                                             ("ascending", "1,4", (1, 4))):
            with self.subTest(roi=roi_name, interval=interval):
                parser = self.parser()
                parser["edit.ascending_expand"].update(roi=roi_name, point_range=interval)
                config = self.load(parser)
                self.assertEqual(config["edits"]["ascending_expand"]["point_range"], expected)
                self.assertNotIn("path_percent", config["edits"]["ascending_expand"])
                self.assertNotIn("point_range", config["edits"]["descending_narrow"])
                self.assertEqual(len(config["roi"]["center_ijk"]), 14)
                self.assertEqual(len(config["rois"]["ascending"]["center_ijk"]), 4)

    def test_path_percent_normalizes_main_or_named_physical_path_interval(self):
        for roi_name, interval, expected in (("main", "0,100", (0., 100.)),
                                             ("main", "30,75", (30., 75.)),
                                             ("ascending", "2.5,70.125", (2.5, 70.125))):
            with self.subTest(roi=roi_name, interval=interval):
                parser = self.parser()
                parser["edit.ascending_expand"].update(roi=roi_name, path_percent=interval)
                config = self.load(parser)
                self.assertEqual(config["edits"]["ascending_expand"]["path_percent"], expected)
                self.assertNotIn("point_range", config["edits"]["ascending_expand"])

    def test_selector_rejects_malformed_values_bounds_and_nonincreasing_intervals(self):
        invalid = {
            "point_range": ("", "2", "1,2,3", "1,,3", "1,4,", "0,3", "-1,3",
                            "1,5", "3,3", "4,2", "1.0,3", "1,inf", "1,nan", "1;4", "1,\n4"),
            "path_percent": ("", "2", "1,2,3", "1,,3", "0,100,", "-1,70", "1,101",
                             "30,30", "70,30", "nan,40", "1,inf", "-inf,40", "0;100", "0,\n100"),
        }
        for selector, values in invalid.items():
            for value in values:
                with self.subTest(selector=selector, value=value), self.assertRaises(ValueError):
                    parser = self.parser()
                    parser["edit.ascending_expand"][selector] = value
                    self.load(parser)

    def test_selectors_are_mutually_exclusive_and_require_a_tube_base(self):
        parser = self.parser()
        parser["edit.ascending_expand"].update(point_range="1,4", path_percent="0,100")
        with self.assertRaisesRegex(ValueError, "point_range and path_percent are mutually exclusive"):
            self.load(parser)
        for roi_name in ("main", "ascending"):
            for selector, value in (("point_range", "1,2"), ("path_percent", "0,100")):
                with self.subTest(roi=roi_name, selector=selector):
                    parser = self.parser()
                    section = "roi" if roi_name == "main" else "roi.ascending"
                    parser[section].update(shape="sphere", center_ijk="387,360,1352", radius_mm="10")
                    parser["edit.ascending_expand"].update(roi=roi_name, profile="uniform")
                    parser["edit.ascending_expand"][selector] = value
                    with self.assertRaisesRegex(ValueError, "requires the referenced ROI to have shape=tube"):
                        self.load(parser)

    def test_optional_selectors_preserve_required_edit_fields_and_reject_unknown_fields(self):
        for key in ("roi", "iterations", "operation", "distance_mm", "profile",
                    "profile_axis", "shape_k", "shape_window"):
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "missing"):
                parser = self.parser()
                parser["edit.ascending_expand"]["point_range"] = "1,4"
                parser.remove_option("edit.ascending_expand", key)
                self.load(parser)
        parser = self.parser()
        parser["edit.ascending_expand"].update(point_range="1,4", point_ranges="1,4")
        with self.assertRaisesRegex(ValueError, "unknown=.*point_ranges"):
            self.load(parser)

    def test_main_sphere_edit_validates_profile_against_main_not_named_roi(self):
        parser = self.parser()
        parser["roi"].update(shape="sphere", center_ijk="387,360,1352", radius_mm="10")
        parser["edit.ascending_expand"]["roi"] = "main"
        with self.assertRaisesRegex(ValueError, "edit.ascending_expand.*requires roi.shape=tube"):
            self.load(parser)
        parser["edit.ascending_expand"]["profile_axis"] = "k"
        config = self.load(parser)
        self.assertEqual(config["edits"]["ascending_expand"]["roi"], "main")
        self.assertEqual(config["edits"]["ascending_expand"]["profile_axis"], "k")
        self.assertNotIn("point_range", config["edits"]["ascending_expand"])

    def test_optional_centerline_defaults_and_partial_overrides_leave_legacy_edits_unchanged(self):
        legacy = self.load()
        self.assertNotIn("centerline", legacy)
        for edit in legacy["edits"].values():
            self.assertNotIn("direction", edit)
            self.assertNotIn("angular_width_deg", edit)
        for values, expected in (
                ({}, {"smoothing_mm": 1., "sample_step_mm": 1., "min_curvature_per_mm": .002}),
                ({"smoothing_mm": "0"},
                 {"smoothing_mm": 0., "sample_step_mm": 1., "min_curvature_per_mm": .002}),
                ({"smoothing_mm": "2.5", "sample_step_mm": ".4", "min_curvature_per_mm": ".01"},
                 {"smoothing_mm": 2.5, "sample_step_mm": .4, "min_curvature_per_mm": .01})):
            with self.subTest(values=values):
                parser = self.parser()
                parser["centerline"] = values
                before = {section: dict(parser[section]) for section in parser.sections()}
                config = self.load(parser)
                self.assertEqual(config, self.load(parser, loader=load_preview_config))
                self.assertEqual(config.pop("centerline"), expected)
                self.assertEqual(config, legacy)
                self.assertEqual({section: dict(parser[section]) for section in parser.sections()}, before)

    def test_directional_main_or_named_edits_supply_angle_and_centerline_defaults(self):
        for roi_name in ("main", "ascending"):
            for direction in ("inner", "outer"):
                with self.subTest(roi=roi_name, direction=direction):
                    parser = self.parser()
                    parser["edit.ascending_expand"].update(roi=roi_name, direction=direction)
                    config = self.load(parser)
                    self.assertEqual(config["edits"]["ascending_expand"]["direction"], direction)
                    self.assertEqual(config["edits"]["ascending_expand"]["angular_width_deg"], 180.)
                    self.assertEqual(config["centerline"], {
                        "smoothing_mm": 1., "sample_step_mm": 1., "min_curvature_per_mm": .002})
                    self.assertNotIn("direction", config["edits"]["descending_narrow"])
                    self.assertNotIn("angular_width_deg", config["edits"]["descending_narrow"])

    def test_directional_range_validates_original_parent_and_retains_explicit_settings(self):
        for selector, interval, expected in (("point_range", "2,3", (2, 3)),
                                             ("path_percent", "30,75", (30., 75.))):
            with self.subTest(selector=selector):
                parser = self.parser()
                parser["edit.ascending_expand"].update(direction="inner", angular_width_deg="75.5")
                parser["edit.ascending_expand"][selector] = interval
                parser["centerline"] = {"sample_step_mm": ".25"}
                before = {section: dict(parser[section]) for section in parser.sections()}
                config = self.load(parser)
                edit = config["edits"]["ascending_expand"]
                # Named parent has four nodes; point_range retains only two.
                self.assertEqual(len(config["rois"]["ascending"]["center_ijk"]), 4)
                self.assertEqual(edit[selector], expected)
                self.assertEqual(edit["direction"], "inner")
                self.assertEqual(edit["angular_width_deg"], 75.5)
                self.assertEqual(config["centerline"], {
                    "smoothing_mm": 1., "sample_step_mm": .25, "min_curvature_per_mm": .002})
                self.assertEqual({section: dict(parser[section]) for section in parser.sections()}, before)

    def test_explicit_all_preserves_ordinary_sphere_edits_without_centerline_defaults(self):
        parser = self.parser()
        parser["roi.ascending"].update(shape="sphere", center_ijk="387,360,1352", radius_mm="10")
        parser["edit.ascending_expand"]["profile_axis"] = "k"
        baseline = self.load(parser)
        parser["edit.ascending_expand"]["direction"] = "all"
        config = self.load(parser)
        self.assertEqual(config["edits"]["ascending_expand"].pop("direction"), "all")
        self.assertEqual(config, baseline)
        self.assertNotIn("centerline", config)

    def test_directional_edits_reject_spheres_and_parents_with_fewer_than_four_nodes(self):
        for roi_name in ("main", "ascending"):
            for shape, centers, radii in (
                    ("sphere", "387,360,1352", "10"),
                    ("tube", "387,360,1352 ; 378,365,1363", "10,10"),
                    ("tube", "387,360,1352 ; 378,365,1363 ; 373,362,1374", "10,10,10")):
                with self.subTest(roi=roi_name, shape=shape, centers=centers):
                    parser = self.parser()
                    section = "roi" if roi_name == "main" else "roi.ascending"
                    parser[section].update(shape=shape, center_ijk=centers, radius_mm=radii)
                    parser["edit.ascending_expand"].update(roi=roi_name, direction="outer", profile_axis="k")
                    with self.assertRaisesRegex(ValueError, "parent tube.*at least four"):
                        self.load(parser)

    def test_direction_and_angular_width_reject_invalid_or_unused_settings(self):
        for direction in ("", "Inner", "inside", "both", "inner,outer", "inner\nouter"):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                parser = self.parser()
                parser["edit.ascending_expand"]["direction"] = direction
                self.load(parser)
        for width in ("", "0", "-1", "180.01", "nan", "inf", "-inf", "90,180", "90\n180"):
            with self.subTest(width=width), self.assertRaises(ValueError):
                parser = self.parser()
                parser["edit.ascending_expand"].update(direction="inner", angular_width_deg=width)
                self.load(parser)
        for direction in (None, "all"):
            with self.subTest(unused_width_direction=direction):
                parser = self.parser()
                parser["edit.ascending_expand"]["angular_width_deg"] = "90"
                if direction is not None:
                    parser["edit.ascending_expand"]["direction"] = direction
                with self.assertRaisesRegex(ValueError, "angular_width_deg requires direction=inner or outer"):
                    self.load(parser)
        for width in (".1", "180"):
            parser = self.parser()
            parser["edit.ascending_expand"].update(direction="outer", angular_width_deg=width)
            self.assertEqual(self.load(parser)["edits"]["ascending_expand"]["angular_width_deg"], float(width))

    def test_centerline_rejects_unknown_nonfinite_negative_and_nonpositive_settings(self):
        for key in ("smoothing_mm", "sample_step_mm", "min_curvature_per_mm"):
            invalid = ("", "-1", "nan", "inf", "-inf", "1,2", "1\n2")
            if key != "smoothing_mm":
                invalid += ("0",)
            for value in invalid:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    parser = self.parser()
                    parser["centerline"] = {key: value}
                    self.load(parser)
        for section, key in (("centerline", "smoothing"), ("centerline", "direction"),
                             ("edit.ascending_expand", "angular_width")):
            with self.subTest(section=section, key=key):
                parser = self.parser()
                if not parser.has_section(section):
                    parser.add_section(section)
                parser[section][key] = "1"
                with self.assertRaisesRegex(ValueError, "unknown"):
                    self.load(parser)

    def test_named_sphere_is_validated_against_its_own_profile_axis(self):
        parser = self.parser()
        parser["roi.ascending"].update({"shape": "sphere", "center_ijk": "387.5,360,1352", "radius_mm": "50"})
        # Named ROIs may extend beyond the outer ROI; the engine clips them.
        parser["edit.ascending_expand"]["profile_axis"] = "k"
        config = self.load(parser)
        self.assertEqual(config["rois"]["ascending"]["radius_mm"], 50.)
        parser["edit.ascending_expand"]["profile_axis"] = "tube"
        with self.assertRaisesRegex(ValueError, "edit.ascending_expand.*requires roi.shape=tube"):
            self.load(parser)
        parser["edit.ascending_expand"]["profile"] = "uniform"
        self.assertEqual(self.load(parser)["edits"]["ascending_expand"]["profile_axis"], "tube")

    def test_rejects_unlisted_duplicate_and_unknown_steps_and_roi_references(self):
        for value in ("", "ascending_expand,descending_narrow", "ascending_expand,descending_narrow,missing",
                      "ascending_expand,descending_narrow,arch_refine,arch_refine", "ascending_expand,,arch_refine"):
            with self.subTest(steps=value), self.assertRaises(ValueError):
                parser = self.parser()
                parser["recipe"]["steps"] = value
                self.load(parser)
        for value in ("missing", "Ascending", "", "bad name", "../ascending"):
            with self.subTest(roi=value), self.assertRaises(ValueError):
                parser = self.parser()
                parser["edit.ascending_expand"]["roi"] = value
                self.load(parser)

    def test_rejects_unbounded_or_fractional_iterations_and_counts_inactive_steps(self):
        for value in ("0", "11", "-1", "1.5", "nan", ""):
            with self.subTest(iterations=value), self.assertRaises(ValueError):
                parser = self.parser()
                parser["edit.ascending_expand"]["iterations"] = value
                self.load(parser)
        parser = self.parser()
        parser["edit.ascending_expand"]["iterations"] = "10"
        parser["edit.descending_narrow"]["iterations"] = "9"
        self.load(parser)  # 10 + 9 + one inactive pass is exactly 20.
        parser["edit.descending_narrow"]["iterations"] = "10"
        with self.assertRaisesRegex(ValueError, "20 total iterations"):
            self.load(parser)

    def test_rejects_unknown_sections_keys_and_implicit_defaults(self):
        for section in ("edit", "train", "model", "roi_typo", "edit.", "roi.bad name"):
            with self.subTest(section=section), self.assertRaises(ValueError):
                parser = self.parser()
                parser[section] = {"extra": "true"}
                self.load(parser)
        for section in ("recipe", "roi.ascending", "edit.ascending_expand", "input", "roi", "reassignment"):
            with self.subTest(section=section), self.assertRaises(ValueError):
                parser = self.parser()
                parser[section]["unknown"] = "true"
                self.load(parser)
        parser = self.parser()
        parser["DEFAULT"]["iterations"] = "1"
        with self.assertRaisesRegex(ValueError, "DEFAULT"):
            self.load(parser)
        parser = self.parser()
        parser["roi.ascending"]["crop_half_width_mm"] = "22"
        with self.assertRaisesRegex(ValueError, "crop_half_width_mm"):
            self.load(parser)

    def test_required_named_sections_and_fields(self):
        for sections in (("recipe",), ("reassignment",), ("roi.ascending", "roi.arch", "roi.descending"),
                         ("edit.ascending_expand", "edit.descending_narrow", "edit.arch_refine")):
            with self.subTest(sections=sections), self.assertRaises(ValueError):
                parser = self.parser()
                for section in sections:
                    parser.remove_section(section)
                self.load(parser)
        for section, key in (("recipe", "steps"), ("roi.ascending", "coordinate_reviewed"),
                             ("edit.arch_refine", "iterations"), ("edit.descending_narrow", "roi")):
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                parser = self.parser()
                parser.remove_option(section, key)
                self.load(parser)

    def test_rejects_invalid_shapes_parameters_and_ambiguous_multiline_values(self):
        cases = [("recipe", "overlap", "parallel"), ("roi.ascending", "shape", "box"),
                 ("roi.ascending", "radius_mm", "10,10"),
                 ("roi.ascending", "center_ijk", "1,2,3;1,2,3;2,3,4;4,5,6"),
                 ("roi.ascending", "coordinate_reviewed", "yes"),
                 ("edit.ascending_expand", "distance_mm", "0"),
                 ("edit.ascending_expand", "operation", "dilate"),
                 ("edit.arch_refine", "distance_mm", "-1"),
                 ("edit.descending_narrow", "shape_window", "1,0"),
                 ("edit.ascending_expand", "shape_k", "inf"),
                 ("roi.arch", "center_ijk", "1,2,3;\n4,5,6"),
                 ("recipe", "steps", "ascending_expand,\ndescending_narrow,arch_refine")]
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                parser = self.parser()
                parser[section][key] = value
                self.load(parser)

    def test_old_schemas_cannot_accept_recipe_sections_and_wrong_loader_rejects(self):
        for version in ("fakect.preview/1", "fakect.preview/2", "fakect.edit/1", "fakect.study/1"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                parser = self.parser()
                parser["study"]["schema_version"] = version
                self.load(parser, loader=load_preview_config)
        parser = self.parser()
        parser["study"]["schema_version"] = "fakect.edit/1"
        with self.assertRaisesRegex(ValueError, "fakect.recipe/1"):
            self.load(parser)

    def test_each_starter_parameter_references_an_existing_numbered_note(self):
        starter = (ROOT / "configs/studies/thoracic-aorta-recipe.ini").read_text()
        notes = set(re.findall(r"^# NOTE ([0-9]+) --", starter, re.MULTILINE))
        assignments = [line for line in starter.splitlines()
                       if line and not line.startswith(("#", "["))]
        self.assertGreater(len(assignments), 60)
        for line in assignments:
            match = re.search(r"# NOTE ([0-9]+):", line)
            self.assertIsNotNone(match, line)
            self.assertIn(match.group(1), notes)


if __name__ == "__main__":
    unittest.main()
