"""Synthetic P1/P3 fixtures: strict temporal alignment, labels, and no leakage."""

import copy
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROUTES = ("original", "shallow", "middle", "deep", "spread", "mid_deep")


@pytest.fixture(scope="module")
def analysis():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "analyze_verification_route_predictability.py"
    )
    spec = importlib.util.spec_from_file_location("verification_predictability_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route_result(accepted, proposal_count):
    target = list(range(100, 100 + proposal_count))
    draft = target.copy()
    if accepted < proposal_count:
        draft[accepted] += 1000
    return {
        "accepted_draft_tokens": accepted,
        "reported_acceptance_length": accepted + 1,
        "draft_token_ids": draft,
        "target_token_ids_for_proposals": target,
    }


def _trace(sample, turn, round_id, accepted, *, start=10, proposal_count=15, probability=False):
    result = _route_result(accepted, proposal_count)
    record = {
        "sample_id": sample,
        "turn_id": turn,
        "round_id": round_id,
        "num_input_tokens": 10,
        "round_start": start,
        "generated_tokens_before_round": start - 10,
        "block_size": proposal_count + 1,
        "proposal_count": proposal_count,
        "anchor_token_id": 9,
        **result,
        "match_mask": [
            draft == target
            for draft, target in zip(
                result["draft_token_ids"], result["target_token_ids_for_proposals"]
            )
        ],
        "first_reject_position": accepted if accepted < proposal_count else None,
        "all_draft_tokens_accepted": accepted == proposal_count,
    }
    if probability:
        record.update(
            logprob_gaps=[float(i - 2) for i in range(proposal_count)],
            target_entropies=[float(i + 1) for i in range(proposal_count)],
            draft_entropies=[float(i + 2) for i in range(proposal_count)],
        )
    return record


def _oracle(trace, acceptance=None):
    accepted, proposal_count = trace["accepted_draft_tokens"], trace["proposal_count"]
    if acceptance is None:
        acceptance = dict(
            zip(
                ROUTES,
                [
                    accepted,
                    0,
                    accepted,
                    min(accepted + 1, proposal_count),
                    max(accepted - 1, 0),
                    accepted,
                ],
            )
        )
    assert acceptance["original"] == accepted
    return {
        **{key: trace[key] for key in ("sample_id", "turn_id", "round_id")},
        "prefix_length": trace["round_start"] + 1,
        "canonical_route": "original",
        "canonical_accepted_draft_tokens": accepted,
        "canonical_reported_acceptance_length": accepted + 1,
        "original_replay_matches": True,
        # Deliberately omit top-level canonical IDs: the actual P3 schema keeps
        # them inside route_results.original and alignment must still check them.
        "route_results": {
            route: _route_result(value, proposal_count) for route, value in acceptance.items()
        },
    }


def _chain(sample=0, turn=0, accepted=(1, 3, 2), *, probability=False, proposal_count=15):
    traces, start = [], 10
    for round_id, count in enumerate(accepted):
        traces.append(
            _trace(
                sample,
                turn,
                round_id,
                count,
                start=start,
                proposal_count=proposal_count,
                probability=probability,
            )
        )
        start += count + 1
    return traces, [_oracle(trace) for trace in traces]


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    return path


def test_current_round_uses_only_exact_previous_round_within_same_sample_and_turn(analysis):
    traces, oracles, expected = [], [], {}
    for sample, turn, counts in [(7, 0, [1, 4]), (8, 0, [2, 3]), (7, 1, [0, 2])]:
        chain, outcomes = _chain(sample, turn, counts, probability=True)
        chain[0]["logprob_gaps"] = [-float(sample + turn)] * 15
        chain[1]["logprob_gaps"] = [123.0] * 15
        traces.extend(chain)
        oracles.extend(outcomes)
        expected[(sample, turn, 1)] = counts[0]
    rows, summary = analysis.build_dataset(list(reversed(traces)), list(reversed(oracles)))
    assert len(rows) == 3
    for row in rows:
        key = row["sample_id"], row["turn_id"], row["round_id"]
        assert row["prev_round_id"] == 0
        assert row["prev_accepted_draft_tokens"] == expected[key]
        assert row["prev_gap_mean"] == -(key[0] + key[1])
        assert row["current_prefix_length"] == 12 + expected[key]
        assert row["current_generated_tokens_before_round"] == expected[key] + 1
        assert row["current_round_id"] == 1
    assert summary["excluded_first_round_states"] == 3
    assert summary["usable_states"] == 3
    assert summary["alignment_mismatches"] == 0


@pytest.mark.parametrize("other_sample,other_turn", [(1, 0), (0, 1)])
def test_missing_previous_round_never_borrows_from_another_request_or_turn(
    analysis, other_sample, other_turn
):
    current = _trace(0, 0, 1, 2, start=12)
    wrong_previous = _trace(other_sample, other_turn, 0, 1)
    with pytest.raises(ValueError, match="previous|round|missing"):
        analysis.build_dataset([wrong_previous, current], [_oracle(current)])


def test_missing_previous_round_never_falls_back_across_a_round_gap(analysis):
    traces, oracles = _chain(accepted=[1, 3, 2])
    with pytest.raises(ValueError, match="previous|missing"):
        analysis.build_dataset([traces[0], traces[2]], [oracles[2]])


def test_missing_current_trace_is_an_alignment_error(analysis):
    traces, oracles = _chain()
    with pytest.raises(ValueError, match="current|missing|align"):
        analysis.build_dataset(traces[:-1], oracles)


def test_round_zero_is_validated_before_exclusion(analysis):
    traces, oracles = _chain()
    oracles[0]["prefix_length"] += 1
    with pytest.raises(ValueError, match="prefix|align"):
        analysis.build_dataset(traces, oracles)


@pytest.mark.parametrize(
    "mismatch", ["accepted", "draft_ids", "target_ids", "prefix", "round_start"]
)
def test_current_canonical_mismatches_fail_fast(analysis, mismatch):
    traces, oracles = _chain(accepted=[1, 2])
    current = oracles[1]
    if mismatch == "accepted":
        current["canonical_accepted_draft_tokens"] = 1
        current["canonical_reported_acceptance_length"] = 2
        current["route_results"]["original"] = _route_result(1, 15)
    elif mismatch == "draft_ids":
        current["route_results"]["original"]["draft_token_ids"][-1] += 5
    elif mismatch == "target_ids":
        current["route_results"]["original"]["target_token_ids_for_proposals"][-1] += 5
    elif mismatch == "prefix":
        current["prefix_length"] += 1
    else:
        current["round_start"] = traces[1]["round_start"] + 1
    with pytest.raises(ValueError):
        analysis.build_dataset(traces, oracles)


@pytest.mark.parametrize("which", ["trace", "oracle"])
def test_duplicate_identity_is_rejected(analysis, which):
    traces, oracles = _chain()
    (traces if which == "trace" else oracles).append(
        copy.deepcopy((traces if which == "trace" else oracles)[0])
    )
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        analysis.build_dataset(traces, oracles)


def test_previous_probability_features_use_exact_prefix_and_reject_regions(analysis):
    previous = _trace(0, 0, 0, 1, proposal_count=3, probability=True)
    previous["logprob_gaps"] = [-3.0, -1.0, 2.0]
    features = analysis.previous_features(previous, large_gap_threshold=-2.0)
    expected = {
        "prev_accepted_draft_tokens": 1,
        "prev_reported_acceptance_length": 2,
        "prev_acceptance_fraction": 1 / 3,
        "prev_first_reject_position": 1,
        "prev_first_reject_fraction": 1 / 3,
        "prev_gap_mean": -2 / 3,
        "prev_gap_std": math.sqrt(38 / 9),
        "prev_gap_min": -3,
        "prev_gap_max": 2,
        "prev_gap_at_first_reject": -1,
        "prev_gap_accepted_prefix_mean": -3,
        "prev_gap_post_reject_mean": 0.5,
        "prev_abs_gap_mean": 2,
        "prev_large_negative_gap_fraction": 1 / 3,
        "prev_target_entropy_mean": 2,
        "prev_target_entropy_std": math.sqrt(2 / 3),
        "prev_target_entropy_min": 1,
        "prev_target_entropy_max": 3,
        "prev_target_entropy_at_first_reject": 2,
        "prev_draft_entropy_mean": 3,
        "prev_draft_entropy_std": math.sqrt(2 / 3),
        "prev_draft_entropy_min": 2,
        "prev_draft_entropy_max": 4,
        "prev_draft_entropy_at_first_reject": 3,
    }
    for key, value in expected.items():
        assert features[key] == pytest.approx(value)
    for key in (
        "prev_gap_mean",
        "prev_gap_at_first_reject",
        "prev_target_entropy_at_first_reject",
    ):
        assert features[key + "_available"]


@pytest.mark.parametrize("accepted", [0, 2, 3])
def test_legacy_missing_first_reject_is_inferred_without_confusing_all_accepted(
    analysis, accepted
):
    previous = _trace(0, 0, 0, accepted, proposal_count=3, probability=True)
    del previous["first_reject_position"]
    features = analysis.previous_features(previous)
    assert features["prev_all_draft_tokens_accepted"] == int(accepted == 3)
    if accepted == 3:
        assert features["prev_first_reject_position"] is None
        assert features["prev_first_reject_fraction"] is None
        assert features["prev_gap_at_first_reject"] is None
        assert features["prev_gap_post_reject_mean"] is None
        assert not features["prev_first_reject_position_available"]
        assert not features["prev_gap_at_first_reject_available"]
    else:
        assert features["prev_first_reject_position"] == accepted
        assert features["prev_first_reject_position_available"]
    if accepted == 0:
        assert features["prev_gap_accepted_prefix_mean"] is None
        assert not features["prev_gap_accepted_prefix_mean_available"]


def test_explicit_null_reject_position_is_invalid_for_partial_acceptance(analysis):
    previous = _trace(0, 0, 0, 1, proposal_count=3)
    previous["first_reject_position"] = None
    with pytest.raises(ValueError):
        analysis.previous_features(previous)


def test_route_labels_preserve_gains_binary_outcomes_and_tied_winners(analysis):
    trace = _trace(0, 0, 1, 3, start=15, proposal_count=5)
    counts = dict(zip(ROUTES, [3, 0, 2, 4, 3, 4]))
    labels = analysis.route_labels(_oracle(trace, counts))
    for route, count in counts.items():
        assert labels[f"accept_{route}"] == count
        if route != "original":
            assert labels[f"gain_{route}"] == count - 3
    assert labels["oracle_accept"] == 4
    assert labels["oracle_gain"] == 1
    assert set(labels["best_route_set"]) == {"deep", "mid_deep"}
    assert labels["any_restricted_beats_original"]
    assert labels["any_restricted_matches_or_beats_original"]
    for route in ("deep", "mid_deep"):
        assert labels[f"{route}_beats_original"]
        assert not labels[f"{route}_matches_original"]
        assert labels[f"{route}_within_one"]
    ties = analysis.route_labels(_oracle(trace, dict(zip(ROUTES, [3, 1, 2, 3, 0, 3]))))
    assert set(ties["best_route_set"]) == {"original", "deep", "mid_deep"}
    assert not ties["any_restricted_beats_original"]
    assert ties["any_restricted_matches_or_beats_original"]
    assert ties["oracle_gain"] == 0


def test_all_six_routes_are_required_for_comparable_labels(analysis):
    traces, oracles = _chain()
    del oracles[1]["route_results"]["shallow"]
    with pytest.raises(ValueError, match="route|six"):
        analysis.build_dataset(traces, oracles)


def test_predictor_whitelist_and_all_ablation_groups_exclude_current_outcome_leakage(analysis):
    analysis.assert_predictor_columns()
    assert set(analysis.PREDICTOR_COLUMNS).isdisjoint(analysis.LABEL_COLUMNS)
    required_groups = {
        "context_only",
        "accept_only",
        "accept_reject",
        "verification_probability",
        "verification_full",
        "context_plus_verification",
    }
    assert required_groups <= analysis.FEATURE_GROUPS.keys()
    context = {
        "current_prefix_length",
        "current_round_id",
        "current_generated_tokens_before_round",
        "current_generated_tokens_before_round_available",
    }
    assert set(analysis.FEATURE_GROUPS["context_only"]) <= context
    for group in analysis.FEATURE_GROUPS.values():
        analysis.assert_predictor_columns(group)
        assert set(group) <= set(analysis.PREDICTOR_COLUMNS)
    assert set(analysis.FEATURE_GROUPS["context_plus_verification"]) == (
        set(analysis.FEATURE_GROUPS["context_only"])
        | set(analysis.FEATURE_GROUPS["verification_full"])
    )


@pytest.mark.parametrize(
    "column",
    [
        "accepted_draft_tokens",
        "first_reject_position",
        "logprob_gaps",
        "target_entropies",
        "draft_token_ids",
        "target_token_ids_for_proposals",
        "route_results",
        "oracle_gain",
        "accept_deep",
        "gain_deep",
        "current_accepted_draft_tokens",
        "prev_oracle_gain",
    ],
)
def test_unknown_or_leaking_predictor_columns_fail_explicitly(analysis, column):
    with pytest.raises((AssertionError, ValueError)):
        analysis.assert_predictor_columns([column])


def test_alignment_failure_produces_no_analysis_outputs(analysis, tmp_path):
    traces, oracles = _chain()
    oracles[1]["prefix_length"] += 1
    trace_path = _write_jsonl(tmp_path / "trace.jsonl", traces)
    oracle_path = _write_jsonl(tmp_path / "oracle.jsonl", oracles)
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError):
        analysis.analyze([trace_path], [oracle_path], output_dir)
    assert not output_dir.exists()


def test_basic_trace_without_probabilities_writes_dataset_and_explicit_warning(analysis, tmp_path):
    traces, oracles = _chain()
    trace_path = _write_jsonl(tmp_path / "trace.jsonl", traces)
    oracle_path = _write_jsonl(tmp_path / "oracle.jsonl", oracles)
    output_dir = tmp_path / "output"
    summary = analysis.analyze([trace_path], [oracle_path], output_dir)
    for name in (
        "aligned_verification_route_dataset.csv",
        "summary.json",
        "bucket_by_prev_acceptance.csv",
        "bucket_by_prev_reject_position.csv",
        "feature_correlations.csv",
    ):
        assert (output_dir / name).is_file()
    assert any("probab" in warning.lower() for warning in summary["warnings"])
    with (output_dir / "aligned_verification_route_dataset.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["prev_gap_mean"] == "NaN" for row in rows)
    assert all(row["prev_gap_mean_available"] in {"False", "0"} for row in rows)
    assert [int(row["prev_accepted_draft_tokens"]) for row in rows] == [1, 3]
    saved = json.loads((output_dir / "summary.json").read_text())
    assert saved == json.loads(json.dumps(summary))


def test_descriptive_correlations_use_average_ranks_for_ties_and_skip_missing_values(analysis):
    traces, oracles = [], []
    for sample, (previous_acceptance, gain) in enumerate(zip([1, 1, 2, 3], [0, 1, 1, 2])):
        chain, _ = _chain(sample, accepted=[previous_acceptance, 1])
        counts = dict(zip(ROUTES, [1, 0, 0, 1 + gain, 0, 1]))
        traces.extend(chain)
        oracles.append(_oracle(chain[1], counts))
    rows, _ = analysis.build_dataset(traces, oracles)
    description = analysis.describe_dataset(rows)
    correlations = {(row["feature"], row["target"]): row for row in description["correlations"]}
    acceptance = correlations["prev_accepted_draft_tokens", "oracle_gain"]
    assert acceptance["num_pairs"] == 4
    assert acceptance["pearson"] == pytest.approx(2 / math.sqrt(5.5))
    assert acceptance["spearman"] == pytest.approx(5 / 6)
    constant = correlations["prev_proposal_count", "oracle_gain"]
    assert constant["pearson"] is None
    assert constant["spearman"] is None
    missing = correlations["prev_gap_mean", "oracle_gain"]
    assert missing["num_pairs"] == 0
    assert missing["pearson"] is None
    assert missing["spearman"] is None
    defined = [
        abs(row["spearman"]) for row in description["correlations"] if row["spearman"] is not None
    ]
    assert defined == sorted(defined, reverse=True)
    assert all(row["feature"].startswith("prev_") for row in description["correlations"])
    assert description["overall"]["mean_oracle_gain"] == 1
    assert description["overall"]["p_any_restricted_beats_original"] == 0.75


def test_descriptive_acceptance_and_rejection_bucket_boundaries(analysis):
    traces, oracles = [], []
    for sample, previous in enumerate([0, 2, 3, 5, 6, 9, 10, 14, 15]):
        chain, _ = _chain(sample, accepted=[previous, 1])
        traces.extend(chain)
        oracles.append(_oracle(chain[1]))
    rows, _ = analysis.build_dataset(traces, oracles)
    description = analysis.describe_dataset(rows)
    acceptance = {row["bucket"]: row for row in description["acceptance_buckets"]}
    rejection = {row["bucket"]: row for row in description["rejection_buckets"]}
    for name, count in {"0-2": 2, "3-5": 2, "6-9": 2, "10-14": 2, "15": 1}.items():
        assert acceptance[name]["num_states"] == count
        assert acceptance[name]["mean_oracle_gain"] == 1
        assert acceptance[name]["p_any_restricted_beats_original"] == 1
    assert {name: row["num_states"] for name, row in rejection.items()} == {
        "early": 3,
        "middle": 2,
        "late": 3,
        "all_accepted": 1,
    }


@pytest.mark.parametrize("values", [[], [0.0], [float("nan")] * 15, [True] * 15, ["0"] * 15])
def test_malformed_probability_vectors_are_not_treated_as_missing(analysis, values):
    previous = _trace(0, 0, 0, 1)
    previous["logprob_gaps"] = values
    with pytest.raises(ValueError, match="finite|proposal_count"):
        analysis.previous_features(previous)


def test_previous_and_current_boundaries_must_be_contiguous(analysis):
    traces, oracles = _chain(accepted=[1, 3])
    traces[1]["round_start"] += 1
    traces[1]["generated_tokens_before_round"] += 1
    oracles[1]["prefix_length"] += 1
    with pytest.raises(ValueError, match="nonconsecutive|continuity"):
        analysis.build_dataset(traces, oracles)


def test_changing_current_verification_and_labels_never_changes_predictor_values(analysis):
    traces, oracles = _chain(accepted=[1, 3], probability=True)
    original_rows, _ = analysis.build_dataset(traces, oracles)
    changed_current = _trace(0, 0, 1, 0, start=traces[1]["round_start"], probability=True)
    changed_current["logprob_gaps"] = [999.0] * 15
    changed_current["target_entropies"] = [777.0] * 15
    changed_current["draft_entropies"] = [555.0] * 15
    changed_rows, _ = analysis.build_dataset(
        [traces[0], changed_current], [oracles[0], _oracle(changed_current)]
    )
    for column in analysis.PREDICTOR_COLUMNS:
        assert changed_rows[0][column] == original_rows[0][column]
    assert changed_rows[0]["accept_original"] != original_rows[0]["accept_original"]


@pytest.mark.parametrize("record_kind", ["trace", "oracle"])
def test_duplicate_json_keys_are_rejected_before_output(analysis, tmp_path, record_kind):
    traces, oracles = _chain()
    paths = {
        "trace": _write_jsonl(tmp_path / "trace.jsonl", traces),
        "oracle": _write_jsonl(tmp_path / "oracle.jsonl", oracles),
    }
    path = paths[record_kind]
    path.write_text(
        path.read_text().replace('"sample_id": 0', '"sample_id": 0, "sample_id": 0', 1)
    )
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        analysis.analyze([paths["trace"]], [paths["oracle"]], output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("error_kind", ["dataset", "invalid_oracle"])
def test_metadata_mismatch_or_invalid_oracle_status_prevents_output(
    analysis, tmp_path, error_kind
):
    traces, oracles = _chain()
    trace_path = _write_jsonl(tmp_path / "trace.jsonl", traces)
    oracle_path = _write_jsonl(tmp_path / "oracle.jsonl", oracles)
    trace_metadata = {"dataset": "gsm8k", "draft_arch": "dflare", "target_route": "original"}
    oracle_metadata = dict(
        trace_metadata, evaluation_status="complete", original_replay_mismatch_count=0
    )
    if error_kind == "dataset":
        oracle_metadata["dataset"] = "mt-bench"
    else:
        oracle_metadata["evaluation_status"] = "invalid"
    Path(str(trace_path) + ".meta.json").write_text(json.dumps(trace_metadata))
    Path(str(oracle_path) + ".meta.json").write_text(json.dumps(oracle_metadata))
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError):
        analysis.analyze([trace_path], [oracle_path], output_dir)
    assert not output_dir.exists()


def test_existing_outputs_are_not_overwritten(analysis, tmp_path):
    traces, oracles = _chain()
    trace_path = _write_jsonl(tmp_path / "trace.jsonl", traces)
    oracle_path = _write_jsonl(tmp_path / "oracle.jsonl", oracles)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    original_summary = '{"preserve":true}\n'
    (output_dir / "summary.json").write_text(original_summary)
    with pytest.raises(ValueError, match="already exist|fresh"):
        analysis.analyze([trace_path], [oracle_path], output_dir)
    assert (output_dir / "summary.json").read_text() == original_summary
    assert list(output_dir.iterdir()) == [output_dir / "summary.json"]


def test_cli_accepts_multiple_trace_and_oracle_shards_and_prints_alignment_summary(
    analysis, tmp_path, capsys
):
    trace_paths, oracle_paths = [], []
    for sample in range(2):
        traces, oracles = _chain(sample, accepted=[1, 3], probability=True)
        trace_paths.append(_write_jsonl(tmp_path / f"trace{sample}.jsonl", traces))
        oracle_paths.append(_write_jsonl(tmp_path / f"oracle{sample}.jsonl", oracles))
    output_dir = tmp_path / "output"
    summary = analysis.main(
        [
            "--trace",
            *map(str, trace_paths),
            "--oracle",
            *map(str, oracle_paths),
            "--output-dir",
            str(output_dir),
            "--large-gap-threshold",
            "-1.5",
        ]
    )
    assert summary["oracle_states"] == summary["aligned_current_states"] == 4
    assert summary["usable_states"] == summary["excluded_first_round_states"] == 2
    printed = capsys.readouterr().out
    assert "Alignment mismatches: 0" in printed
    assert "PRELIMINARY" in printed
    assert "2 independent request groups" in printed


def test_full_cli_predictor_analysis_connects_all_feature_groups_and_output_artifacts(
    analysis, tmp_path, capsys
):
    pytest.importorskip("sklearn")
    traces, oracles = [], []
    for sample, accepted in enumerate(([1, 2, 3, 1], [2, 3, 1, 2], [3, 1, 2, 3])):
        chain, _ = _chain(sample, accepted=accepted, probability=True)
        traces.extend(chain)
        for current in chain:
            count = current["accepted_draft_tokens"]
            gain = 1 if (sample + current["round_id"]) % 2 else -1
            counts = dict(
                zip(
                    ROUTES,
                    [count, max(count - 2, 0), count - 1, count + gain, count - 1, count + gain],
                )
            )
            oracles.append(_oracle(current, counts))
    trace_path = _write_jsonl(tmp_path / "trace.jsonl", traces)
    oracle_path = _write_jsonl(tmp_path / "oracle.jsonl", oracles)
    output_dir = tmp_path / "predictions"
    summary = analysis.main(
        [
            "--trace",
            str(trace_path),
            "--oracle",
            str(oracle_path),
            "--output-dir",
            str(output_dir),
            "--run-predictors",
        ]
    )
    assert summary["analysis_status"] == "complete"
    assert summary["predictors"]["enabled"] is True
    assert summary["independent_request_groups"] == 3
    assert summary["usable_states"] == 9
    groups = summary["predictors"]["feature_groups"]
    assert groups.keys() == analysis.FEATURE_GROUPS.keys()
    for columns in groups.values():
        analysis.assert_predictor_columns(columns)
        assert set(columns).isdisjoint(analysis.LABEL_COLUMNS)
    required_files = {
        "aligned_verification_route_dataset.csv",
        "summary.json",
        "report.txt",
        "bucket_by_prev_acceptance.csv",
        "bucket_by_prev_reject_position.csv",
        "feature_correlations.csv",
        "prediction_metrics.csv",
        "policy_metrics.csv",
        "oof_predictions.csv",
    }
    assert required_files <= {path.name for path in output_dir.iterdir()}
    with (output_dir / "prediction_metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    assert {row["feature_group"] for row in metrics if row["model"] != "base_rate"} == set(groups)
    assert {row["model"] for row in metrics} == {
        "base_rate",
        "logistic_regression",
        "decision_tree",
    }
    assert all(int(row["num_folds"]) == 3 for row in metrics)
    assert all("fold_mean_roc_auc" in row and "pooled_oof_roc_auc" in row for row in metrics)
    with (output_dir / "oof_predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    for metric in metrics:
        selected = [
            row
            for row in predictions
            if all(row[key] == metric[key] for key in ("target", "model", "feature_group"))
        ]
        identities = {(row["sample_id"], row["turn_id"], row["round_id"]) for row in selected}
        assert len(selected) == len(identities) == 9
        for sample in range(3):
            assert (
                len({row["fold_id"] for row in selected if row["sample_id"] == str(sample)}) == 1
            )
    with (output_dir / "policy_metrics.csv").open() as stream:
        policies = list(csv.DictReader(stream))
    assert {
        "always_original",
        "always_deep",
        "always_mid_deep",
        "learned_binary_policy",
        "oracle_binary_policy",
        "full_six_route_oracle",
    } <= {row["policy"] for row in policies}
    report = (output_dir / "report.txt").read_text()
    assert "ROC-AUC" in report
    assert "held-out" in report
    assert "PRELIMINARY" in report
    assert "3 independent request groups" in report
    assert report == capsys.readouterr().out
    saved = json.loads((output_dir / "summary.json").read_text())
    assert saved == json.loads(json.dumps(summary))


def test_descriptive_cli_runs_without_site_packages(analysis, tmp_path):
    traces, oracles = _chain()
    trace_path = _write_jsonl(tmp_path / "trace.jsonl", traces)
    oracle_path = _write_jsonl(tmp_path / "oracle.jsonl", oracles)
    output_dir = tmp_path / "stdlib-output"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            analysis.__file__,
            "--trace",
            str(trace_path),
            "--oracle",
            str(oracle_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["usable_states"] == 2
    assert summary["alignment_mismatches"] == 0
    assert summary["predictors"]["enabled"] is False
