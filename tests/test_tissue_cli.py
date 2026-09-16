"""Cross-file contracts: lineage, ambiguous joins and reversible crop outputs."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_tissues as cli


class TissueCLIContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = {"categories": [{"id": 0, "name": "background"}, {"id": 5, "name": "artery"},
                                       {"id": 6, "name": "vein"}, {"id": 255, "name": "unknown"}],
                        "records": [{"original_id": value, "original_name": name,
                                     "classification": {"tissue_id": tissue, "tissue_name": name, "status": "classified"}}
                                    for value, tissue, name in [(0, 0, "background"), (1, 5, "artery"), (-3, 6, "vein")]]}
        self.catalog_path = self.root / "catalog.json"
        cli.write_json(self.catalog_path, self.catalog)

    def test_dictionary_union_has_no_silent_precedence(self):
        a, b = self.root / "a.txt", self.root / "b.txt"
        a.write_text("artery = 1\nvein = -3\n")
        b.write_text("artery = 1\nother = 2\n")
        names, provenance = cli.merge_dictionaries([a, b])
        self.assertEqual(names, {1: "artery", -3: "vein", 2: "other"})
        self.assertEqual(len(provenance[1]), 2)
        b.write_text("bone = 1\n")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            cli.merge_dictionaries([a, b])

    def test_surface_scan_keeps_duplicate_occurrences_and_local_indices(self):
        case = self.root / "123"
        case.mkdir()
        for group in ("body", "bones", "heart", "muscle", "organs", "vessels"):
            (case / f"123_1_{group}.raw").write_text("\nartery\n0 0 0 1 0 0 0 1 0\nartery\nnan 0 0 1 0 0 0 1 0\n")
        inventory = cli.scan_surfaces(case, 1)
        self.assertEqual(len(inventory["files"]), 6)
        for file in inventory["files"]:
            self.assertTrue(file["complete"])
            self.assertEqual([s["name"] for s in file["surfaces"]], ["artery", "artery"])
            self.assertEqual([s["surface_index_in_file"] for s in file["surfaces"]], [0, 1])
            self.assertNotIn("original_id", file["surfaces"][0])

    def reduce_args(self, geometry=None):
        original = np.array([0, 1, -3, 999, 1, -3, 0, 1], dtype=np.int32).reshape(2, 2, 2)
        np.save(self.root / "labels.npy", original)
        geometry = geometry or {"array_order": "zyx", "shape_zyx": [2, 2, 2], "spacing_ijk_mm": [1, 2, 3],
                                "crop_origin_ijk": [3, 4, 5], "source_shape_zyx": [10, 10, 10],
                                "coordinate_status": "unverified anatomical orientation"}
        cli.write_json(self.root / "geometry.json", geometry)
        return argparse.Namespace(catalog=self.catalog_path, labels=self.root / "labels.npy",
                                  geometry=self.root / "geometry.json", out=self.root / "view.npz",
                                  strict=False, max_voxels=100), original, geometry

    def test_crop_roundtrip_keeps_exact_ids_geometry_catalog_and_unknown(self):
        args, original, geometry = self.reduce_args()
        with contextlib.redirect_stdout(io.StringIO()):
            cli.reduce_command(args)
        with np.load(args.out, allow_pickle=False) as result:
            np.testing.assert_array_equal(original, result["original_labels"])
            self.assertEqual(result["tissue_labels"][0, 1, 1], 255)
            metadata = json.loads(str(result["metadata_json"]))
            self.assertEqual(metadata["geometry"], geometry)
            self.assertEqual(metadata["catalog"], self.catalog)
            self.assertEqual(metadata["source_sha256"], cli.digest(args.labels))
        np.testing.assert_array_equal(np.load(args.labels), original)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            cli.reduce_command(args)

    def test_crop_rejects_bad_geometry_unknowns_and_unbounded_export(self):
        for geometry in [{"array_order": "xyz", "shape_zyx": [2, 2, 2], "spacing_ijk_mm": [1, 1, 1]},
                         {"array_order": "zyx", "shape_zyx": [3, 2, 2], "spacing_ijk_mm": [1, 1, 1]},
                         {"array_order": "zyx", "shape_zyx": [2, 2, 2], "spacing_ijk_mm": [1, -1, 1]}]:
            args, _, _ = self.reduce_args(geometry)
            with self.assertRaises(ValueError):
                cli.reduce_command(args)
            self.assertFalse(args.out.exists())
        args, _, _ = self.reduce_args()
        args.max_voxels = 1
        with self.assertRaisesRegex(ValueError, "max-voxels"):
            cli.reduce_command(args)
        args.max_voxels, args.strict = 100, True
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            cli.reduce_command(args)

    def test_profile_preserves_material_ambiguity_and_cross_group_overlap(self):
        log = self.root / "case_log"
        log.write_text("Linear Attenuation Coefficients (1/cm):\n Blood = 0.1700\n Tissue = 0.1700\n"
                       "Linear Attenuation Coefficients (1/pixel):\n Blood = 0.0170\n")
        audit = self.root / "audit.json"
        cli.write_json(audit, {"method": "small paired fixture", "source_metadata_sha256": {str(log): cli.digest(log)},
                              "cases": [{"case_id": "case", "log_path": str(log), "scalar_to_organ_ids_sample":
                                         [{"attenuation_cm_inverse": 0.17, "organ_ids": [1, -3, 999]}]}]})
        args = argparse.Namespace(catalog=self.catalog_path, audit=audit, cases=None, out=self.root / "profile", no_plot=True)
        with contextlib.redirect_stdout(io.StringIO()):
            cli.profile_command(args)
        report = cli.read_json(args.out / "attenuation-profiles.json")
        self.assertEqual(report["unknown_original_ids"], [999])
        self.assertEqual(report["observed_original_ids"], 3)
        self.assertEqual(len(report["shared_attenuation_levels"]), 3)
        self.assertEqual(report["records"][0]["observed_levels"][0]["material_candidates"], ["Blood", "Tissue"])
        self.assertTrue(report["range_is_not_classifier"])
        log.write_text(log.read_text() + "changed\n")
        args.out = self.root / "changed-profile"
        with self.assertRaisesRegex(ValueError, "changed since sampling"):
            cli.profile_command(args)
        self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
