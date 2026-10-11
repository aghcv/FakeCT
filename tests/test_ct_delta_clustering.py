"""Scientific invariants for exploratory CT-difference PCA and clustering."""

import json
import unittest

import numpy as np

from fakect_ct_delta_clustering import (adjusted_rand_index, analyse,
                                       silhouette, transform_features)


def three_groups(seed=1):
    rng = np.random.default_rng(seed)
    centres = np.asarray([[-5., -3., 1.], [0., 6., -1.], [6., -2., 2.]])
    return np.concatenate([centre + rng.normal(scale=.15, size=(24, 3)) for centre in centres])


class CtDeltaClusteringTests(unittest.TestCase):
    def test_three_separated_groups_and_json_metadata(self):
        features = three_groups()
        result = analyse(features, {"k_max": 5})
        self.assertEqual(result["analysis"]["clustering"]["candidate_best_k"], 3)
        self.assertEqual(result["analysis"]["clustering"]["suggested_k"], 3)
        self.assertGreater(adjusted_rand_index(np.repeat(np.arange(3), 24), result["labels"]), .99)
        self.assertEqual(sorted(np.bincount(result["labels"]).tolist()), [24, 24, 24])
        json.dumps(result["analysis"], allow_nan=False)

    def test_repeat_and_row_permutation_preserve_partition(self):
        features = three_groups()
        original = analyse(features, {"k_max": 4, "silhouette_sample_size": 40})
        repeated = analyse(features, {"k_max": 4, "silhouette_sample_size": 40})
        np.testing.assert_array_equal(original["labels"], repeated["labels"])
        np.testing.assert_array_equal(original["scores"], repeated["scores"])
        permutation = np.random.default_rng(55).permutation(len(features))
        permuted = analyse(features[permutation], {"k_max": 4, "silhouette_sample_size": 40})
        np.testing.assert_allclose(original["scores"][permutation], permuted["scores"], atol=1e-12)
        self.assertEqual(adjusted_rand_index(original["labels"][permutation], permuted["labels"]), 1.)
        self.assertAlmostEqual(original["analysis"]["clustering"]["candidates"][0]["silhouette"],
                               permuted["analysis"]["clustering"]["candidates"][0]["silhouette"], places=12)

    def test_magnitude_is_retained_not_per_sibling_normalized(self):
        features = np.zeros((30, 3))
        features[:, 0] = np.arange(30)
        result = analyse(features, {"k_max": 3})
        # Identical direction with different extent remains distinct in PCA.
        distances = np.linalg.norm(result["scores"][1:] - result["scores"][:-1], axis=1)
        np.testing.assert_allclose(distances, np.ones(29))
        self.assertGreater(np.ptp(result["scores"][:, 0]), 28.)
        scaled = analyse(features * 7., {"k_max": 3})
        np.testing.assert_allclose(result["scores"] * 7, scaled["scores"], atol=1e-10)

    def test_tiny_noise_columns_are_not_promoted_by_standardization(self):
        rng = np.random.default_rng(8)
        main = np.repeat([-4., 0., 6.], 30) + rng.normal(0, .03, 90)
        noise = rng.normal(0, 1e-9, (90, 70))
        result = analyse(np.column_stack((main, noise)), {"k_max": 3})
        self.assertGreater(result["analysis"]["pca"]["explained_variance_ratio"][0], .999999)
        self.assertEqual(result["analysis"]["clustering"]["candidate_best_k"], 3)

    def test_clustering_in_retained_space_not_only_plot_axes(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(150, 3)) * [8, 7, 5]
        result = analyse(features, {"k_max": 2, "variance_target": .95})
        self.assertEqual(result["scores"].shape[1], 3)
        labels, centres = result["candidate_labels"], result["cluster_centers"]
        expected = np.sum((result["scores"] - centres[labels]) ** 2)
        two_dimensional = np.sum((result["scores"][:, :2] - centres[labels, :2]) ** 2)
        candidate = result["analysis"]["clustering"]["candidates"][0]
        self.assertAlmostEqual(candidate["inertia"], expected)
        self.assertGreater(expected, two_dimensional + 100)

    def test_pca_cap_reports_unmet_variance_target(self):
        features = np.random.default_rng(5).normal(size=(100, 50))
        result = analyse(features, {"max_components": 2, "variance_target": .90, "k_max": 2})
        self.assertFalse(result["analysis"]["pca"]["target_reached"])
        self.assertLess(result["analysis"]["pca"]["retained_variance_ratio"], .90)
        self.assertEqual(result["scores"].shape[1], 2)

    def test_randomized_pca_and_baseline_transform_without_refit(self):
        rng = np.random.default_rng(19)
        features = rng.normal(size=(150, 4)) @ rng.normal(size=(4, 256)) + 3.
        result = analyse(features, {"k_max": 2})
        self.assertEqual(result["analysis"]["pca"]["solver"], "randomized_svd")
        self.assertTrue(result["analysis"]["pca"]["target_reached"])
        projected = transform_features(result, np.zeros((2, features.shape[1])))
        np.testing.assert_allclose(projected[0], -result["mean"] @ result["components"].T)
        np.testing.assert_allclose(transform_features(result, features), result["scores"])
        with self.assertRaises(ValueError):
            transform_features(result, np.zeros((2, 4)))

    def test_identical_and_single_rows_have_no_forced_groups(self):
        for features in (np.zeros((30, 4)), np.ones((30, 4)), np.zeros((1, 4))):
            with self.subTest(shape=features.shape):
                result = analyse(features)
                self.assertEqual(result["analysis"]["clustering"]["suggested_k"], 1)
                self.assertEqual(result["analysis"]["clustering"]["candidate_best_k"], 1)
                np.testing.assert_array_equal(result["labels"], np.zeros(len(features)))
                json.dumps(result["analysis"], allow_nan=False)

    def test_evidence_fallback_preserves_exploratory_candidate(self):
        result = analyse(three_groups(), {"k_max": 3, "silhouette_threshold": 1.})
        clustering = result["analysis"]["clustering"]
        self.assertEqual(clustering["candidate_best_k"], 3)
        self.assertEqual(clustering["suggested_k"], 1)
        self.assertIn("silhouette_below_threshold", clustering["fallback_reasons"])
        self.assertEqual(np.unique(result["candidate_labels"]).size, 3)
        self.assertEqual(np.unique(result["labels"]).size, 1)

    def test_silhouette_queries_use_all_reference_rows_and_singletons(self):
        scores = np.asarray([[0., 0.], [1., 0.], [10., 0.], [11., 0.]])
        expected = ((10.5 - 1) / 10.5 + (9.5 - 1) / 9.5) / 2
        self.assertAlmostEqual(silhouette(scores, [0, 0, 1, 1], [0, 1]), expected)
        self.assertEqual(silhouette(scores, [0, 1, 1, 1], [0]), 0.)
        self.assertEqual(silhouette(scores, [0, 0, 0, 0]), 0.)

    def test_adjusted_rand_is_name_independent_and_adjusted(self):
        self.assertEqual(adjusted_rand_index([0, 0, 1, 1], [7, 7, 3, 3]), 1.)
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]), -.5)
        self.assertEqual(adjusted_rand_index([0, 0], [1, 1]), 1.)
        with self.assertRaises(ValueError):
            adjusted_rand_index([0, 1], [0])

    def test_finite_input_and_settings_validation(self):
        for invalid in (np.empty((0, 3)), np.empty((3, 0)), [1, 2], [[np.nan, 0]], [[np.inf, 0]]):
            with self.subTest(invalid=str(invalid)), self.assertRaises(ValueError):
                analyse(invalid)
        for settings in ({"variance_target": 0}, {"k_min": 1}, {"k_max": 1},
                         {"max_components": 1}, {"n_init": 0}, {"seed": -1},
                         {"silhouette_sample_size": 0}, {"unexpected": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                analyse(three_groups(), settings)


if __name__ == "__main__":
    unittest.main()
