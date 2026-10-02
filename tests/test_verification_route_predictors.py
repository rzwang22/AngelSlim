"""Synthetic LOGO validation, preprocessing, and held-out policy checks."""

import copy
import importlib.util
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "verification_route_predictors.py"
SPEC = importlib.util.spec_from_file_location("verification_route_predictors_tests", SCRIPT)
predictors = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predictors)
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
GROUPS = {
    "context_only": [
        "current_prefix_length",
        "current_round_id",
        "current_generated_tokens_before_round_available",
    ],
    "accept_only": ["prev_accepted_draft_tokens"],
    "accept_reject": ["prev_accepted_draft_tokens", "prev_first_reject_position"],
    "verification_probability": ["prev_gap_mean", "prev_gap_available"],
    "verification_full": [
        "prev_accepted_draft_tokens",
        "prev_first_reject_position",
        "prev_gap_mean",
    ],
    "context_plus_verification": [
        "current_prefix_length",
        "prev_accepted_draft_tokens",
        "prev_gap_mean",
    ],
}


def make_row(sample, index, original=4, deep=6, mid_deep=3, oracle=None):
    oracle = max(original, deep, mid_deep) if oracle is None else oracle
    return {
        "sample_id": sample,
        "turn_id": index // 4,
        "round_id": index % 4 + 1,
        "current_prefix_length": 100 + 10 * index,
        "current_round_id": index % 4 + 1,
        "current_generated_tokens_before_round_available": True,
        "prev_accepted_draft_tokens": 10 if deep >= original else 1,
        "prev_first_reject_position": None if index % 3 == 0 else index,
        "prev_gap_mean": None if index % 3 == 0 else -float(index),
        "prev_gap_available": index % 3 != 0,
        "accept_original": original,
        "accept_deep": deep,
        "accept_mid_deep": mid_deep,
        "oracle_accept": oracle,
        "any_restricted_beats_original": oracle > original,
        "deep_matches_or_beats_original": deep >= original,
        "mid_deep_matches_or_beats_original": mid_deep >= original,
    }


def synthetic_rows():
    return [
        make_row(sample, index, deep=6 if index % 2 else 2, mid_deep=5 if (index // 2) % 2 else 3)
        for sample in range(4)
        for index in range(8)
    ]


class GroupedPreprocessingTests(unittest.TestCase):
    def test_groups_never_split_requests_or_turns(self):
        rows = synthetic_rows()
        splits = list(predictors.grouped_splits(rows))
        self.assertEqual(len(splits), 4)
        seen = []
        for train, test in splits:
            train_groups = {rows[index]["sample_id"] for index in train}
            test_groups = {rows[index]["sample_id"] for index in test}
            self.assertTrue(train_groups.isdisjoint(test_groups))
            self.assertEqual(len(test_groups), 1)
            self.assertEqual({rows[index]["turn_id"] for index in test}, {0, 1})
            self.assertEqual(sorted(train + test), list(range(len(rows))))
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(len(rows))))

    def test_at_least_two_groups_are_required(self):
        for rows in ([], [make_row(0, 0), make_row(0, 4)]):
            with self.subTest(rows=len(rows)):
                with self.assertRaisesRegex(ValueError, "at least 2 independent request groups"):
                    list(predictors.grouped_splits(rows))

    def test_preprocessing_fits_training_median_mean_and_scale_only(self):
        rows = [
            {"prev_value": value, "prev_missing": None}
            for value in (0.0, 2.0, None, 100000.0, None)
        ]
        columns = ["prev_value", "prev_missing"]
        train, test = predictors._prepare_features(rows, columns, [0, 1, 2], [3, 4])
        scale = math.sqrt(2 / 3)
        self.assertAlmostEqual(train[0][0], -1 / scale)
        self.assertAlmostEqual(train[1][0], 1 / scale)
        self.assertEqual(train[2][0], 0)
        self.assertEqual(test[1][0], 0)
        self.assertGreater(test[1][2], 0)  # Explicit missing flag survives imputation.
        modified = copy.deepcopy(rows)
        modified[3]["prev_value"] = -999999.0
        modified[3]["prev_missing"] = 42.0
        changed_train, changed_test = predictors._prepare_features(
            modified, columns, [0, 1, 2], [3, 4]
        )
        self.assertEqual(train, changed_train)
        self.assertEqual(changed_test[0][1], 42.0)
        self.assertEqual(changed_test[0][3], -1.0)

    def test_current_outcomes_and_identity_cannot_enter_features(self):
        for column in (
            "accept_deep",
            "oracle_accept",
            "oracle_gain",
            "round_id",
            "sample_id",
            "route_results",
            "current_logprob_gaps",
            "current_first_reject_position",
            "draft_token_ids",
            "target_token_ids_for_proposals",
        ):
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, "Unsafe predictor"):
                    predictors._validate_inputs(synthetic_rows(), {"bad": [column]})
        predictors._validate_inputs(synthetic_rows(), GROUPS)

    def test_module_import_works_without_optional_site_packages(self):
        code = (
            "import importlib.util,sys;"
            f"spec=importlib.util.spec_from_file_location('p',{str(SCRIPT)!r});"
            "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
            "assert 'sklearn' not in sys.modules;assert 'numpy' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-c", code], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_missing_sklearn_has_actionable_optional_install_hint(self):
        real_import = __import__

        def without_sklearn(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ImportError("synthetic missing dependency")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=without_sklearn):
            with self.assertRaisesRegex(
                ImportError, "pip install scikit-learn.*omit --run-predictors"
            ):
                predictors._load_sklearn()


@unittest.skipUnless(SKLEARN_AVAILABLE, "optional scikit-learn is not installed")
class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        cls.result = predictors.run_predictors(cls.rows, GROUPS)

    def test_all_targets_feature_groups_and_both_models_are_reported(self):
        metrics = self.result["prediction_metrics"]
        self.assertEqual(len(metrics), 3 * (1 + 2 * len(GROUPS)))
        self.assertEqual({row["target"] for row in metrics}, set(predictors.TARGETS))
        for target in predictors.TARGETS:
            for group in GROUPS:
                matching = [
                    row
                    for row in metrics
                    if row["target"] == target and row["feature_group"] == group
                ]
                self.assertEqual(
                    {row["model"] for row in matching}, {"logistic_regression", "decision_tree"}
                )
                for metric in matching:
                    self.assertEqual(metric["valid_roc_auc_folds"], 4)
                    self.assertEqual(metric["feature_count"], len(GROUPS[group]))
                    self.assertEqual(metric["expanded_feature_count"], 2 * len(GROUPS[group]))
        signal = next(
            row
            for row in metrics
            if row["target"] == "deep_matches_or_beats_original"
            and row["model"] == "logistic_regression"
            and row["feature_group"] == "accept_only"
        )
        self.assertEqual(signal["fold_mean_roc_auc"], 1.0)
        self.assertEqual(signal["pooled_oof_roc_auc"], 1.0)

    def test_every_output_marks_actual_group_count_and_is_json_serializable(self):
        self.assertEqual(self.result["summary"]["num_groups"], 4)
        for key in ("prediction_metrics", "policy_metrics", "oof_predictions"):
            for row in self.result[key]:
                self.assertEqual(
                    row["preliminary_warning"], "PRELIMINARY — only 4 independent request groups"
                )
        self.assertIn("Trapezoidal", self.result["summary"]["pr_auc_definition"])
        json.dumps(self.result, allow_nan=False)

    def test_each_state_has_exactly_one_oof_prediction_per_model_target(self):
        identities = {(row["sample_id"], row["turn_id"], row["round_id"]) for row in self.rows}
        for metric in self.result["prediction_metrics"]:
            matching = [
                row
                for row in self.result["oof_predictions"]
                if all(row[key] == metric[key] for key in ("target", "model", "feature_group"))
            ]
            self.assertEqual(len(matching), len(self.rows))
            self.assertEqual(
                {(row["sample_id"], row["turn_id"], row["round_id"]) for row in matching},
                identities,
            )
            for row in matching:
                training = [item for item in self.rows if item["sample_id"] != row["sample_id"]]
                expected_base_rate = sum(item[row["target"]] for item in training) / len(training)
                self.assertEqual(row["train_base_rate"], expected_base_rate)
                if row["model"] == "base_rate":
                    self.assertEqual(row["probability"], expected_base_rate)

    def test_policy_uses_only_held_out_prediction_and_all_baselines_are_present(self):
        policies = self.result["policy_metrics"]
        self.assertTrue(
            {
                "always_original",
                "always_deep",
                "always_mid_deep",
                "oracle_binary_policy",
                "full_six_route_oracle",
            }.issubset({row["policy"] for row in policies})
        )
        lookup = {(row["sample_id"], row["turn_id"], row["round_id"]): row for row in self.rows}
        for policy in policies:
            if policy["policy"] not in ("learned_binary_policy", "base_rate_binary_policy"):
                continue
            alternative = policy["alternative_route"]
            target = f"{alternative}_matches_or_beats_original"
            matching = [
                row
                for row in self.result["oof_predictions"]
                if row["target"] == target
                and row["model"] == policy["model"]
                and row["feature_group"] == policy["feature_group"]
            ]
            total = 0
            for prediction in matching:
                source = lookup[
                    (prediction["sample_id"], prediction["turn_id"], prediction["round_id"])
                ]
                chosen = alternative if prediction["probability"] >= 0.5 else "original"
                total += source[f"accept_{chosen}"]
            self.assertEqual(policy["policy_mean_accepted_tokens"], total / len(self.rows))
            expected_gain = (policy["policy_mean_accepted_tokens"] - policy["original_mean"]) / (
                policy["full_oracle_mean"] - policy["original_mean"]
            )
            self.assertEqual(policy["captured_oracle_gain"], expected_gain)

    def test_one_class_train_fallback_skipped_folds_and_negative_gain_are_explicit(self):
        rows = [
            make_row(sample, index, deep=7 if sample == 0 else 1, mid_deep=3)
            for sample in range(2)
            for index in range(2)
        ]
        result = predictors.run_predictors(rows, {"accept_only": ["prev_accepted_draft_tokens"]})
        for metric in result["prediction_metrics"]:
            self.assertEqual(metric["one_class_train_folds"], 2)
            self.assertEqual(metric["skipped_single_class_test_folds"], 2)
            self.assertEqual(metric["valid_roc_auc_folds"], 0)
            self.assertIsNone(metric["fold_mean_roc_auc"])
            self.assertIsNone(metric["fold_mean_pr_auc"])
            self.assertIsNone(metric["fold_mean_balanced_accuracy"])
        deep = next(
            row
            for row in result["policy_metrics"]
            if row["policy"] == "learned_binary_policy" and row["alternative_route"] == "deep"
        )
        self.assertEqual(deep["policy_mean_accepted_tokens"], 2.5)
        self.assertEqual(deep["captured_oracle_gain"], -1.0)
        metric = next(
            row
            for row in result["prediction_metrics"]
            if row["target"] == "deep_matches_or_beats_original"
        )
        self.assertEqual(metric["pooled_oof_roc_auc"], 0)
        self.assertEqual(metric["pooled_oof_balanced_accuracy"], 0)

    def test_zero_oracle_gain_has_undefined_captured_fraction(self):
        rows = [make_row(sample, 0, deep=2, mid_deep=3) for sample in range(2)]
        result = predictors.run_predictors(rows, {"accept_only": ["prev_accepted_draft_tokens"]})
        for policy in result["policy_metrics"]:
            self.assertIsNone(policy["captured_oracle_gain"])
        for metric in result["prediction_metrics"]:
            self.assertIsNone(metric["pooled_oof_roc_auc"])
        json.dumps(result, allow_nan=False)

    def test_repeated_runs_are_deterministic_and_inputs_unchanged(self):
        rows = synthetic_rows()
        original = copy.deepcopy(rows)
        groups = {"accept_only": GROUPS["accept_only"]}
        first = predictors.run_predictors(rows, groups)
        second = predictors.run_predictors(rows, groups)
        self.assertEqual(first, second)
        self.assertEqual(rows, original)


if __name__ == "__main__":
    unittest.main()
