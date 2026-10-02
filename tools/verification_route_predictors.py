"""Optional grouped predictive baselines for offline verification-route analysis.

The module imports only the standard library until run_predictors() is called.
It consumes the already aligned P4 dataset and never interacts with inference.
"""

from __future__ import annotations

import math
from statistics import mean, median, pstdev

TARGETS = (
    "any_restricted_beats_original",
    "deep_matches_or_beats_original",
    "mid_deep_matches_or_beats_original",
)
CONTEXT_COLUMNS = {
    "current_prefix_length",
    "current_round_id",
    "current_generated_tokens_before_round",
    "current_generated_tokens_before_round_available",
}


def _load_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            auc,
            balanced_accuracy_score,
            precision_recall_curve,
            roc_auc_score,
        )
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as error:
        raise ImportError(
            "Optional predictor analysis requires scikit-learn. Install it in your analysis "
            "environment with `python -m pip install scikit-learn`, or omit --run-predictors. "
            "It is not an AngelSlim inference dependency."
        ) from error
    return {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "auc": auc,
        "balanced_accuracy_score": balanced_accuracy_score,
        "precision_recall_curve": precision_recall_curve,
        "roc_auc_score": roc_auc_score,
    }


def grouped_splits(rows):
    """Yield LOGO (train_indices, test_indices), grouping all turns by sample_id."""
    groups = []
    for row in rows:
        group = row.get("sample_id")
        if type(group) is not int or group < 0:
            raise ValueError("sample_id must be a nonnegative integer")
        if group not in groups:
            groups.append(group)
    if len(groups) < 2:
        raise ValueError(
            "Predictor validation requires at least 2 independent request groups; "
            f"got {len(groups)}"
        )
    for held_out in groups:
        train = [index for index, row in enumerate(rows) if row["sample_id"] != held_out]
        test = [index for index, row in enumerate(rows) if row["sample_id"] == held_out]
        assert train and test
        assert {rows[index]["sample_id"] for index in train}.isdisjoint(
            rows[index]["sample_id"] for index in test
        )
        yield train, test


def _numeric(value, column):
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"Predictor {column!r} must be numeric or missing")
    value = float(value)
    if math.isnan(value):
        return None
    if not math.isfinite(value):
        raise ValueError(f"Predictor {column!r} must be finite or missing")
    return value


def _prepare_features(rows, columns, train, test):
    """Fit median imputation and mean/std scaling on train only; append masks.

    A missing indicator is reserved for every source column, including features
    that happen to be complete in the training fold. Fully missing train columns
    use a constant 0 plus their explicit mask. No held-out values enter fitting.
    """
    raw = [[_numeric(row.get(column), column) for column in columns] for row in rows]
    medians = []
    for column in range(len(columns)):
        observed = [raw[index][column] for index in train if raw[index][column] is not None]
        medians.append(median(observed) if observed else 0.0)

    def imputed(index):
        values = raw[index]
        return [
            medians[column] if value is None else value for column, value in enumerate(values)
        ] + [float(value is None) for value in values]

    training = [imputed(index) for index in train]
    means = [mean(column) for column in zip(*training)]
    scales = [pstdev(column) or 1.0 for column in zip(*training)]

    def transform(index):
        return [
            (value - center) / scale for value, center, scale in zip(imputed(index), means, scales)
        ]

    return [transform(index) for index in train], [transform(index) for index in test]


def _validate_inputs(rows, feature_groups):
    if not isinstance(feature_groups, dict) or not feature_groups:
        raise ValueError("feature_groups must be a nonempty mapping")
    for group, columns in feature_groups.items():
        if not isinstance(group, str) or not group or not isinstance(columns, (list, tuple)):
            raise ValueError("feature_groups must map names to feature lists")
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"Feature group {group!r} must contain distinct nonempty features")
        for column in columns:
            if not isinstance(column, str) or not (
                column.startswith("prev_") or column in CONTEXT_COLUMNS
            ):
                raise ValueError(
                    f"Unsafe predictor {column!r}: current outcomes/identity are labels"
                )

    for row_index, row in enumerate(rows):
        for column in ("accept_original", "accept_deep", "accept_mid_deep", "oracle_accept"):
            value = row.get(column)
            if type(value) is not int or value < 0:
                raise ValueError(f"row {row_index}: {column} must be a nonnegative integer")
        if row["oracle_accept"] < max(
            row["accept_original"], row["accept_deep"], row["accept_mid_deep"]
        ):
            raise ValueError(f"row {row_index}: oracle_accept is smaller than a route acceptance")
        expected = (
            row["oracle_accept"] > row["accept_original"],
            row["accept_deep"] >= row["accept_original"],
            row["accept_mid_deep"] >= row["accept_original"],
        )
        for target, truth in zip(TARGETS, expected):
            value = row.get(target)
            if type(value) not in (bool, int) or value not in (0, 1):
                raise ValueError(f"row {row_index}: {target} must be a binary label")
            if bool(value) != truth:
                raise ValueError(f"row {row_index}: {target} disagrees with route outcomes")


def _classification_metrics(truth, probabilities, sklearn):
    # Balanced accuracy requires both class recalls. A single-class held-out
    # prompt cannot estimate this, even if a library returns its observed recall.
    if len(set(truth)) < 2:
        return {"roc_auc": None, "pr_auc": None, "balanced_accuracy": None}
    precision, recall, _ = sklearn["precision_recall_curve"](truth, probabilities)
    return {
        "roc_auc": float(sklearn["roc_auc_score"](truth, probabilities)),
        "pr_auc": float(sklearn["auc"](recall, precision)),
        "balanced_accuracy": float(
            sklearn["balanced_accuracy_score"](
                truth, [int(value >= 0.5) for value in probabilities]
            )
        ),
    }


def _policy_row(rows, selected, *, policy, alternative, model, feature_group, common):
    policy_mean = mean(selected)
    original_mean = mean(row["accept_original"] for row in rows)
    full_mean = mean(row["oracle_accept"] for row in rows)
    gain = full_mean - original_mean
    return {
        **common,
        "policy": policy,
        "alternative_route": alternative,
        "model": model,
        "feature_group": feature_group,
        "policy_mean_accepted_tokens": policy_mean,
        "original_mean": original_mean,
        "full_oracle_mean": full_mean,
        "binary_oracle_mean": (
            mean(max(row["accept_original"], row[f"accept_{alternative}"]) for row in rows)
            if alternative is not None
            else None
        ),
        "captured_oracle_gain": (policy_mean - original_mean) / gain if gain else None,
    }


def run_predictors(rows, feature_groups):
    """Return grouped classification metrics, held-out policies, and OOF rows.

    Each sample_id is held out exactly once, including every turn of the request.
    Fold means give each evaluable request equal weight; pooled OOF metrics and
    policy means instead pool states and are labeled separately. Single-class
    training folds use the constant training base rate. Undefined single-class
    test metrics are excluded from fold means with explicit counts.
    """
    _validate_inputs(rows, feature_groups)
    splits = list(grouped_splits(rows))
    sklearn = _load_sklearn()
    num_groups = len(splits)
    warning = f"PRELIMINARY — only {num_groups} independent request groups"
    common = {"num_states": len(rows), "num_groups": num_groups, "preliminary_warning": warning}
    prediction_metrics = []
    policy_metrics = []
    oof_predictions = []

    for route in ("original", "deep", "mid_deep"):
        policy_metrics.append(
            _policy_row(
                rows,
                [row[f"accept_{route}"] for row in rows],
                policy=f"always_{route}",
                alternative=route if route != "original" else None,
                model="static",
                feature_group="none",
                common=common,
            )
        )
    for route in ("deep", "mid_deep"):
        policy_metrics.append(
            _policy_row(
                rows,
                [max(row["accept_original"], row[f"accept_{route}"]) for row in rows],
                policy="oracle_binary_policy",
                alternative=route,
                model="oracle",
                feature_group="none",
                common=common,
            )
        )
    policy_metrics.append(
        _policy_row(
            rows,
            [row["oracle_accept"] for row in rows],
            policy="full_six_route_oracle",
            alternative=None,
            model="oracle",
            feature_group="none",
            common=common,
        )
    )

    configurations = [("base_rate", "none", [])] + [
        (model, group, columns)
        for group, columns in feature_groups.items()
        for model in ("logistic_regression", "decision_tree")
    ]
    prepared = {
        group: [_prepare_features(rows, columns, train, test) for train, test in splits]
        for group, columns in feature_groups.items()
    }
    for target in TARGETS:
        labels = [int(row[target]) for row in rows]
        for model_name, group, columns in configurations:
            probabilities = [None] * len(rows)
            fold_scores = []
            one_class_train_folds = 0
            for fold_id, (train, test) in enumerate(splits):
                train_labels = [labels[index] for index in train]
                test_labels = [labels[index] for index in test]
                train_base_rate = mean(train_labels)
                single_class_train = len(set(train_labels)) == 1
                one_class_train_folds += int(single_class_train)
                if model_name == "base_rate" or single_class_train:
                    fold_probabilities = [train_base_rate] * len(test)
                else:
                    train_features, test_features = prepared[group][fold_id]
                    if model_name == "logistic_regression":
                        model = sklearn[model_name](
                            solver="liblinear", max_iter=1000, random_state=0
                        )
                    else:
                        model = sklearn[model_name](
                            max_depth=3, min_samples_leaf=2, random_state=0
                        )
                    model.fit(train_features, train_labels)
                    positive_index = list(model.classes_).index(1)
                    fold_probabilities = [
                        float(value[positive_index])
                        for value in model.predict_proba(test_features)
                    ]
                fold_scores.append(
                    _classification_metrics(test_labels, fold_probabilities, sklearn)
                )
                for index, probability in zip(test, fold_probabilities):
                    assert (
                        probabilities[index] is None
                    ), "state received multiple held-out predictions"
                    probabilities[index] = probability
                    oof_predictions.append(
                        {
                            **common,
                            "sample_id": rows[index]["sample_id"],
                            "turn_id": rows[index].get("turn_id"),
                            "round_id": rows[index].get("round_id"),
                            "fold_id": fold_id,
                            "target": target,
                            "model": model_name,
                            "feature_group": group,
                            "true_label": labels[index],
                            "probability": probability,
                            "predicted_label": int(probability >= 0.5),
                            "train_base_rate": train_base_rate,
                            "training_class_count": len(set(train_labels)),
                        }
                    )
            assert all(
                value is not None for value in probabilities
            ), "missing held-out predictions"
            pooled = _classification_metrics(labels, probabilities, sklearn)
            metric_row = {
                **common,
                "target": target,
                "model": model_name,
                "feature_group": group,
                "feature_count": len(columns),
                "expanded_feature_count": 2 * len(columns),
                "num_folds": num_groups,
                "one_class_train_folds": one_class_train_folds,
                "skipped_single_class_test_folds": sum(
                    score["roc_auc"] is None for score in fold_scores
                ),
            }
            for metric in ("roc_auc", "pr_auc", "balanced_accuracy"):
                valid = [score[metric] for score in fold_scores if score[metric] is not None]
                metric_row[f"fold_mean_{metric}"] = mean(valid) if valid else None
                metric_row[f"valid_{metric}_folds"] = len(valid)
                metric_row[f"pooled_oof_{metric}"] = pooled[metric]
            prediction_metrics.append(metric_row)
            if target in ("deep_matches_or_beats_original", "mid_deep_matches_or_beats_original"):
                alternative = target.removesuffix("_matches_or_beats_original")
                selected = [
                    row[f"accept_{alternative}"] if probability >= 0.5 else row["accept_original"]
                    for row, probability in zip(rows, probabilities)
                ]
                policy_metrics.append(
                    _policy_row(
                        rows,
                        selected,
                        policy=(
                            "base_rate_binary_policy"
                            if model_name == "base_rate"
                            else "learned_binary_policy"
                        ),
                        alternative=alternative,
                        model=model_name,
                        feature_group=group,
                        common=common,
                    )
                )

    return {
        "prediction_metrics": prediction_metrics,
        "policy_metrics": policy_metrics,
        "oof_predictions": oof_predictions,
        "summary": {
            **common,
            "validation": "Leave-One-Group-Out by sample_id, with all turns kept together",
            "targets": list(TARGETS),
            "feature_groups": feature_groups,
            "preprocessing": (
                "Training-fold median imputation + all-column missing indicators "
                "+ training mean/std scaling"
            ),
            "fold_metric_weighting": (
                "Unweighted mean over folds with both test classes; undefined folds skipped"
            ),
            "pooled_oof_metric_weighting": (
                "All held-out states pooled; distinct from equal-request fold means"
            ),
            "pr_auc_definition": (
                "Trapezoidal area under the precision-recall curve; "
                "undefined for single-class tests"
            ),
            "single_class_train_fallback": "Constant training-fold positive-label prevalence",
            "policy_threshold": 0.5,
            "policy_weighting": "Mean over all states using exclusively held-out predictions",
            "captured_oracle_gain_definition": (
                "(policy_mean-original_mean)/(full_oracle_mean-original_mean); "
                "no clipping; null if denominator zero"
            ),
            "random_state": 0,
            "tree_max_depth": 3,
            "tree_min_samples_leaf": 2,
        },
    }
