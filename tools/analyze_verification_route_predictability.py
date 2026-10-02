"""Offline previous-verification -> next-round route analysis (stdlib by default).

Current-round traces are used only to validate the canonical trajectory. Features
come from exactly r-1 in the same request/turn, plus pre-decision context. No model
or inference code is imported. Optional sklearn predictors live in a separate module.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import tempfile
from pathlib import Path


def _load_sibling(name):
    # Works both as a CLI and when loaded by file path, without changing sys.path.
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_oracle = _load_sibling("analyze_dflare_route_oracle")
ROUTES = ("original", "shallow", "middle", "deep", "spread", "mid_deep")
IDENTITY_COLUMNS = ("sample_id", "turn_id", "round_id", "prev_round_id")
CONTEXT_COLUMNS = (
    "current_prefix_length",
    "current_round_id",
    "current_generated_tokens_before_round",
    "current_generated_tokens_before_round_available",
)
BASIC_COLUMNS = (
    "prev_accepted_draft_tokens",
    "prev_reported_acceptance_length",
    "prev_acceptance_fraction",
    "prev_first_reject_position",
    "prev_first_reject_fraction",
    "prev_all_draft_tokens_accepted",
    "prev_proposal_count",
    "prev_first_reject_position_available",
    "prev_first_reject_fraction_available",
)
PROBABILITY_VALUE_COLUMNS = (
    "prev_gap_mean",
    "prev_gap_std",
    "prev_gap_min",
    "prev_gap_max",
    "prev_gap_at_first_reject",
    "prev_gap_accepted_prefix_mean",
    "prev_gap_post_reject_mean",
    "prev_target_entropy_mean",
    "prev_target_entropy_std",
    "prev_target_entropy_min",
    "prev_target_entropy_max",
    "prev_target_entropy_at_first_reject",
    "prev_draft_entropy_mean",
    "prev_draft_entropy_std",
    "prev_draft_entropy_min",
    "prev_draft_entropy_max",
    "prev_draft_entropy_at_first_reject",
    "prev_abs_gap_mean",
    "prev_large_negative_gap_fraction",
)
PROBABILITY_COLUMNS = PROBABILITY_VALUE_COLUMNS + tuple(
    name + "_available" for name in PROBABILITY_VALUE_COLUMNS
)
PREVIOUS_COLUMNS = BASIC_COLUMNS + PROBABILITY_COLUMNS
PREDICTOR_COLUMNS = CONTEXT_COLUMNS + PREVIOUS_COLUMNS
LABEL_COLUMNS = (
    *("accept_" + route for route in ROUTES),
    *("gain_" + route for route in ROUTES[1:]),
    "oracle_accept",
    "oracle_gain",
    "any_restricted_beats_original",
    "any_restricted_matches_or_beats_original",
    *(
        route + "_" + outcome
        for route in ("deep", "mid_deep")
        for outcome in (
            "beats_original",
            "matches_original",
            "matches_or_beats_original",
            "within_one",
        )
    ),
    "best_route_set",
)
_ACCEPT_COLUMNS = (
    "prev_accepted_draft_tokens",
    "prev_acceptance_fraction",
    "prev_all_draft_tokens_accepted",
)
_ACCEPT_REJECT_COLUMNS = _ACCEPT_COLUMNS + (
    "prev_first_reject_position",
    "prev_first_reject_fraction",
    "prev_first_reject_position_available",
    "prev_first_reject_fraction_available",
)
FEATURE_GROUPS = {
    "context_only": CONTEXT_COLUMNS,
    "accept_only": _ACCEPT_COLUMNS,
    "accept_reject": _ACCEPT_REJECT_COLUMNS,
    "verification_probability": _ACCEPT_REJECT_COLUMNS + PROBABILITY_COLUMNS,
    "verification_full": PREVIOUS_COLUMNS,
    "context_plus_verification": PREDICTOR_COLUMNS,
}
ACCEPTANCE_BINS = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-9", 6, 9),
    ("10-14", 10, 14),
    ("15", 15, 15),
    ("16+", 16, None),
)
REJECTION_BINS = (("early", 0, 3), ("middle", 4, 8), ("late", 9, None))
BUCKET_COLUMNS = (
    "bucket",
    "num_states",
    "p_any_restricted_beats_original",
    "mean_oracle_gain",
    "mean_gain_deep",
    "mean_gain_mid_deep",
    "p_deep_matches_or_beats_original",
    "p_mid_deep_matches_or_beats_original",
)
CORRELATION_COLUMNS = ("feature", "target", "num_pairs", "pearson", "spearman")


def assert_predictor_columns(columns=PREDICTOR_COLUMNS):
    """A closed whitelist; a 'prev_' prefix alone does not make a feature safe."""
    allowed = frozenset(CONTEXT_COLUMNS + PREVIOUS_COLUMNS)
    assert len(columns) == len(set(columns)), "duplicate predictor columns"
    assert set(columns) <= allowed, "predictor contains current/future leakage or unknown feature"
    assert not set(columns) & (
        set(LABEL_COLUMNS) | set(IDENTITY_COLUMNS)
    ), "label/identity leakage"


assert_predictor_columns()
for _columns in FEATURE_GROUPS.values():
    assert_predictor_columns(_columns)


def _identity(record):
    if not isinstance(record, dict):
        raise ValueError("trace must be a JSON object")
    return tuple(
        _oracle._nonnegative_integer(record.get(k), k)
        for k in ("sample_id", "turn_id", "round_id")
    )


def _validate_trace(record):
    identity = _identity(record)
    _oracle._validate_markers(record)
    if record.get("target_route", "original") != "original":
        raise ValueError(f"trace {identity}: canonical target_route must be original")
    accepted = _oracle._nonnegative_integer(
        record.get("accepted_draft_tokens"), "accepted_draft_tokens"
    )
    proposals = _oracle._nonnegative_integer(record.get("proposal_count"), "proposal_count")
    if proposals == 0 or accepted > proposals:
        raise ValueError("trace must have 0 <= accepted_draft_tokens <= positive proposal_count")
    if (
        type(record.get("reported_acceptance_length")) is not int
        or record["reported_acceptance_length"] != accepted + 1
    ):
        raise ValueError("trace reported_acceptance_length must equal accepted + 1")
    all_accepted = accepted == proposals
    if record.get("all_draft_tokens_accepted") is not all_accepted:
        raise ValueError("trace all_draft_tokens_accepted disagrees with accepted count")
    reject = None if all_accepted else accepted
    # Older traces may omit this redundant field; an explicit null is valid only
    # for all-accepted rounds. Never substitute zero for an absent rejection.
    if "first_reject_position" in record and (
        record["first_reject_position"] != reject
        or (reject is not None and type(record["first_reject_position"]) is not int)
    ):
        raise ValueError("trace first_reject_position disagrees with accepted prefix")
    draft = _oracle._tokens(record.get("draft_token_ids"), "trace draft_token_ids")
    if len(draft) != proposals:
        raise ValueError("trace draft token count differs from proposal_count")
    if "target_token_ids_for_proposals" in record:
        target = _oracle._tokens(record["target_token_ids_for_proposals"], "trace target IDs")
        if len(target) != proposals:
            raise ValueError("trace target token count differs from proposal_count")
        matches = [x == y for x, y in zip(draft, target)]
        prefix = next((i for i, matches_here in enumerate(matches) if not matches_here), proposals)
        if prefix != accepted:
            raise ValueError("trace token matches disagree with accepted prefix")
        if "match_mask" in record and record["match_mask"] != matches:
            raise ValueError("trace match_mask disagrees with token matches")
    start = _oracle._nonnegative_integer(record.get("round_start"), "trace round_start")
    if "block_size" in record and record["block_size"] != proposals + 1:
        raise ValueError("trace block_size must equal proposal_count + 1")
    if "num_input_tokens" in record:
        prompt = _oracle._nonnegative_integer(record["num_input_tokens"], "num_input_tokens")
        if start < prompt or (identity[2] == 0 and start != prompt):
            raise ValueError("trace round_start inconsistent with num_input_tokens")
    if "generated_tokens_before_round" in record:
        generated = _oracle._nonnegative_integer(
            record["generated_tokens_before_round"], "generated_tokens_before_round"
        )
        if "num_input_tokens" in record and generated != start - record["num_input_tokens"]:
            raise ValueError("trace generated_tokens_before_round inconsistent with prefix")
    for field in (
        "logprob_gaps",
        "target_entropies",
        "draft_entropies",
        "target_logprobs",
        "draft_logprobs",
    ):
        if field not in record:
            continue
        values = record[field]
        if (
            not isinstance(values, list)
            or len(values) != proposals
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise ValueError(f"trace {field} must contain proposal_count finite numbers")
    return identity


def _nullable(features, name, value):
    features[name] = value
    features[name + "_available"] = int(value is not None)


def previous_features(record, large_gap_threshold=-2.0):
    """Statistics of one already validated previous round; missing values stay None."""
    if not math.isfinite(large_gap_threshold):
        raise ValueError("large_gap_threshold must be finite")
    _validate_trace(record)
    accepted, proposals = record["accepted_draft_tokens"], record["proposal_count"]
    reject = accepted if accepted < proposals else None
    result = {
        "prev_accepted_draft_tokens": accepted,
        "prev_reported_acceptance_length": accepted + 1,
        "prev_acceptance_fraction": accepted / proposals,
        "prev_all_draft_tokens_accepted": int(reject is None),
        "prev_proposal_count": proposals,
    }
    _nullable(result, "prev_first_reject_position", reject)
    _nullable(
        result, "prev_first_reject_fraction", reject / proposals if reject is not None else None
    )
    for field, prefix in (
        ("logprob_gaps", "prev_gap"),
        ("target_entropies", "prev_target_entropy"),
        ("draft_entropies", "prev_draft_entropy"),
    ):
        values = record.get(field)
        for suffix, reducer in (
            ("mean", statistics.fmean),
            ("std", statistics.pstdev),
            ("min", min),
            ("max", max),
        ):
            _nullable(result, prefix + "_" + suffix, reducer(values) if values else None)
        _nullable(
            result,
            prefix + "_at_first_reject",
            values[reject] if values is not None and reject is not None else None,
        )
    gaps = record.get("logprob_gaps")
    _nullable(
        result,
        "prev_gap_accepted_prefix_mean",
        statistics.fmean(gaps[:accepted]) if gaps is not None and accepted else None,
    )
    _nullable(
        result,
        "prev_gap_post_reject_mean",
        statistics.fmean(gaps[accepted:]) if gaps is not None and reject is not None else None,
    )
    _nullable(
        result,
        "prev_abs_gap_mean",
        statistics.fmean(abs(x) for x in gaps) if gaps is not None else None,
    )
    _nullable(
        result,
        "prev_large_negative_gap_fraction",
        sum(x < large_gap_threshold for x in gaps) / proposals if gaps is not None else None,
    )
    assert set(result) == set(PREVIOUS_COLUMNS)
    return result


def route_labels(record):
    """Current outcomes, never predictor inputs. Ties remain a set of route names."""
    _, values = _oracle._validate_record(record)
    if set(values) != set(ROUTES):
        raise ValueError(f"P4 requires exactly the six routes {ROUTES}; got {tuple(values)}")
    original, maximum = values["original"], max(values.values())
    restricted = max(values[route] for route in ROUTES[1:])
    result = {"accept_" + route: values[route] for route in ROUTES}
    result.update({"gain_" + route: values[route] - original for route in ROUTES[1:]})
    result.update(
        oracle_accept=maximum,
        oracle_gain=maximum - original,
        any_restricted_beats_original=int(restricted > original),
        any_restricted_matches_or_beats_original=int(restricted >= original),
        best_route_set=[route for route in ROUTES if values[route] == maximum],
    )
    for route in ("deep", "mid_deep"):
        result.update(
            {
                route + "_beats_original": int(values[route] > original),
                route + "_matches_original": int(values[route] == original),
                route + "_matches_or_beats_original": int(values[route] >= original),
                route + "_within_one": int(values[route] >= original - 1),
            }
        )
    assert set(result) == set(LABEL_COLUMNS)
    return result


def _check_current_alignment(oracle, trace):
    original = oracle["route_results"]["original"]
    pairs = [
        (
            "canonical acceptance",
            oracle["canonical_accepted_draft_tokens"],
            trace["accepted_draft_tokens"],
        ),
        (
            "canonical reported acceptance",
            oracle["canonical_reported_acceptance_length"],
            trace["reported_acceptance_length"],
        ),
        ("original draft_token_ids", original["draft_token_ids"], trace["draft_token_ids"]),
        ("prefix_length / round_start", oracle["prefix_length"], trace["round_start"] + 1),
    ]
    for field in ("target_token_ids_for_proposals",):
        if field in original and field in trace:
            pairs.append((field, original[field], trace[field]))
    for field in (
        "round_start",
        "num_input_tokens",
        "generated_tokens_before_round",
        "block_size",
        "proposal_count",
        "anchor_token_id",
        "active_target_layer_ids",
        "target_layer_ids",
    ):
        if field in oracle and field in trace:
            pairs.append((field, oracle[field], trace[field]))
    for name, expected, actual in pairs:
        if expected != actual:
            raise ValueError(
                f"alignment mismatch at {_identity(trace)}: {name}: "
                f"oracle={expected!r}, trace={actual!r}"
            )


def _check_previous_alignment(previous, current):
    if current["round_start"] != previous["round_start"] + previous["reported_acceptance_length"]:
        raise ValueError(f"alignment mismatch at {_identity(current)}: nonconsecutive round_start")
    for field in ("num_input_tokens", "proposal_count", "block_size", "active_target_layer_ids"):
        if field in previous and field in current and previous[field] != current[field]:
            raise ValueError(
                f"alignment mismatch at {_identity(current)}: previous/current {field}"
            )
    if "generated_tokens_before_round" in previous and "generated_tokens_before_round" in current:
        if (
            current["generated_tokens_before_round"]
            != previous["generated_tokens_before_round"] + previous["reported_acceptance_length"]
        ):
            raise ValueError(
                f"alignment mismatch at {_identity(current)}: generated token continuity"
            )


def build_dataset(trace_records, oracle_records, *, large_gap_threshold=-2.0):
    """Validate *all* current states before joining r-1. Missing links fail closed."""
    if not math.isfinite(large_gap_threshold):
        raise ValueError("large_gap_threshold must be finite")
    traces = {}
    for record in trace_records:
        identity = _validate_trace(record)
        if identity in traces:
            raise ValueError(f"duplicate trace identity {identity}")
        traces[identity] = record
    aligned, seen = [], set()
    for oracle in oracle_records:
        identity, _ = _oracle._validate_record(oracle)
        labels = route_labels(oracle)
        if identity in seen:
            raise ValueError(f"duplicate oracle identity {identity}")
        seen.add(identity)
        if identity not in traces:
            raise ValueError(
                f"alignment mismatch: missing current trace for oracle state {identity}"
            )
        current = traces[identity]
        _check_current_alignment(oracle, current)
        aligned.append((identity, oracle, current, labels))
    if not aligned:
        raise ValueError("no oracle states found")
    rows, excluded = [], 0
    for identity, oracle, current, labels in sorted(aligned, key=lambda item: item[0]):
        sample, turn, round_id = identity
        if round_id == 0:
            excluded += 1
            continue
        previous_id = (sample, turn, round_id - 1)
        if previous_id not in traces:
            raise ValueError(
                f"missing previous trace {previous_id} for oracle state {identity}; "
                "no cross-request, cross-turn or gap fallback"
            )
        previous = traces[previous_id]
        _check_previous_alignment(previous, current)
        # Only this projection may reach model feature matrices. In particular,
        # no current verification value is copied into a predictor column.
        row = dict(zip(IDENTITY_COLUMNS, (*identity, round_id - 1)))
        row.update(current_prefix_length=oracle["prefix_length"], current_round_id=round_id)
        _nullable(
            row,
            "current_generated_tokens_before_round",
            current.get("generated_tokens_before_round"),
        )
        row.update(previous_features(previous, large_gap_threshold))
        row.update(labels)
        assert set(row) == set(IDENTITY_COLUMNS + PREDICTOR_COLUMNS + LABEL_COLUMNS)
        rows.append(row)
    summary = {
        "oracle_states": len(aligned),
        "trace_states": len(traces),
        "aligned_current_states": len(aligned),
        "alignment_mismatches": 0,
        "excluded_first_round_states": excluded,
        "usable_states": len(rows),
        "usable_previous_feedback_pairs": len(rows),
        "missing_previous_strategy": "fail; exact same sample_id and turn_id, round_id - 1 only",
    }
    return rows, summary


def _mean(rows, field):
    return statistics.fmean(row[field] for row in rows) if rows else None


def _outcomes(rows):
    return {
        "num_states": len(rows),
        "p_any_restricted_beats_original": _mean(rows, "any_restricted_beats_original"),
        "mean_oracle_gain": _mean(rows, "oracle_gain"),
        "mean_gain_deep": _mean(rows, "gain_deep"),
        "mean_gain_mid_deep": _mean(rows, "gain_mid_deep"),
        "p_deep_matches_or_beats_original": _mean(rows, "deep_matches_or_beats_original"),
        "p_mid_deep_matches_or_beats_original": _mean(rows, "mid_deep_matches_or_beats_original"),
    }


def _ranks(values):
    """Average 1-based ranks for ties, independent of input order."""
    indices = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indices):
        j = i + 1
        while j < len(indices) and values[indices[i]] == values[indices[j]]:
            j += 1
        for k in range(i, j):
            ranks[indices[k]] = (i + 1 + j) / 2
        i = j
    return ranks


def _pearson(x, y):
    if len(x) < 2:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return (
        max(-1.0, min(1.0, sum(a * b for a, b in zip(dx, dy)) / denominator))
        if denominator
        else None
    )


def describe_dataset(rows):
    """State-weighted exploratory associations, not causal or significance tests."""
    acceptance = [
        {
            "bucket": name,
            **_outcomes(
                [
                    r
                    for r in rows
                    if r["prev_accepted_draft_tokens"] >= low
                    and (high is None or r["prev_accepted_draft_tokens"] <= high)
                ]
            ),
        }
        for name, low, high in ACCEPTANCE_BINS
    ]
    rejection = [
        {
            "bucket": name,
            **_outcomes(
                [
                    r
                    for r in rows
                    if r["prev_first_reject_position"] is not None
                    and r["prev_first_reject_position"] >= low
                    and (high is None or r["prev_first_reject_position"] <= high)
                ]
            ),
        }
        for name, low, high in REJECTION_BINS
    ]
    rejection.append(
        {
            "bucket": "all_accepted",
            **_outcomes([r for r in rows if r["prev_all_draft_tokens_accepted"]]),
        }
    )
    correlations = []
    for feature in PREVIOUS_COLUMNS:
        for target in ("oracle_gain", "gain_deep", "gain_mid_deep"):
            pairs = [(r[feature], r[target]) for r in rows if r[feature] is not None]
            x, y = zip(*pairs) if pairs else ([], [])
            correlations.append(
                {
                    "feature": feature,
                    "target": target,
                    "num_pairs": len(pairs),
                    "pearson": _pearson(x, y),
                    "spearman": _pearson(_ranks(x), _ranks(y)),
                }
            )
    correlations.sort(
        key=lambda r: (r["spearman"] is None, -abs(r["spearman"] or 0), r["feature"], r["target"])
    )
    return {
        "overall": _outcomes(rows),
        "acceptance_buckets": acceptance,
        "rejection_buckets": rejection,
        "correlations": correlations,
    }


def _read_traces(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(
                        line,
                        object_pairs_hook=_oracle._unique_object,
                        parse_constant=_oracle._reject_constant,
                    )
                    _validate_trace(record)
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                yield record


def _check_metadata(trace_paths, oracle_paths):
    """Compare shared provenance when sidecars exist; never require fake metadata."""
    metadata, missing = [], []
    for kind, paths in (("trace", trace_paths), ("oracle", oracle_paths)):
        for path in paths:
            sidecar = Path(str(path) + ".meta.json")
            if not sidecar.exists():
                missing.append(str(sidecar))
                continue
            record = json.loads(
                sidecar.read_text(encoding="utf-8"),
                object_pairs_hook=_oracle._unique_object,
                parse_constant=_oracle._reject_constant,
            )
            if not isinstance(record, dict):
                raise ValueError(f"{sidecar}: metadata must be an object")
            _oracle._validate_markers(record)
            for key, expected in (
                ("draft_arch", "dflare"),
                ("target_route", "original"),
                ("temperature", 0),
            ):
                if key in record and record[key] != expected:
                    raise ValueError(f"{sidecar}: {key} must be {expected!r}")
            metadata.append((kind, sidecar, record))
    for key in (
        "dataset",
        "max_samples",
        "num_samples",
        "max_new_tokens",
        "temperature",
        "seed",
        "draft_arch",
        "block_size",
        "target_layer_ids",
        "model_name_or_path",
        "draft_name_or_path",
        "target_route",
        "active_target_layer_ids",
    ):
        supplied = [(path, record[key]) for _, path, record in metadata if key in record]
        if supplied and any(value != supplied[0][1] for _, value in supplied[1:]):
            raise ValueError(
                f"alignment mismatch: metadata {key} differs across inputs: {supplied}"
            )
    return missing


def _write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        "NaN"
                        if row.get(key) is None
                        else (
                            json.dumps(row[key])
                            if isinstance(row[key], (list, dict))
                            else row[key]
                        )
                    )
                    for key in columns
                }
            )


def _paths(paths):
    return [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]


def analyze(
    trace_paths, oracle_paths, output_dir, *, large_gap_threshold=-2.0, run_predictors=False
):
    trace_paths, oracle_paths = _paths(trace_paths), _paths(oracle_paths)
    if not trace_paths or not oracle_paths:
        raise ValueError("at least one trace file and one oracle file are required")
    missing_metadata = _check_metadata(trace_paths, oracle_paths)
    rows, alignment = build_dataset(
        _read_traces(trace_paths),
        (r for _, r in _oracle._read_jsonl(oracle_paths)),
        large_gap_threshold=large_gap_threshold,
    )
    description = describe_dataset(rows)
    availability = {
        field: sum(row[field + "_available"] for row in rows)
        for field in PROBABILITY_VALUE_COLUMNS
    }
    warnings = []
    for field in ("prev_gap_mean", "prev_target_entropy_mean", "prev_draft_entropy_mean"):
        if availability[field] < len(rows) or not rows:
            warnings.append(
                f"Probability feature {field}: available in {availability[field]}/{len(rows)} "
                "previous-feedback pairs; missing probability stats stay NaN with "
                "availability=0. Basic analysis remains valid."
            )
    if missing_metadata:
        warnings.append(
            "Some metadata sidecars are missing; provenance validation is limited to "
            "supplied fields and per-state trajectory checks."
        )
    group_count = len({row["sample_id"] for row in rows})
    preliminary = f"PRELIMINARY — only {group_count} independent request groups"
    summary = {
        **alignment,
        "analysis_status": "complete",
        "metric": "accepted_draft_tokens (anchor excluded)",
        "independent_request_groups": group_count,
        "result_scope": preliminary,
        "trace_inputs": [str(p.resolve()) for p in trace_paths],
        "oracle_inputs": [str(p.resolve()) for p in oracle_paths],
        "predictor_columns": PREDICTOR_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "feature_groups": FEATURE_GROUPS,
        "probability_feature_available_counts": availability,
        "missing_metadata_sidecars": missing_metadata,
        "warnings": warnings,
        "definitions": {
            "current_prefix_length": "round_start + 1, includes current anchor",
            "first_reject_fraction": "zero-based first_reject_position / proposal_count",
            "gap_std": "population standard deviation (also for entropy)",
            "gap_accepted_prefix_mean": "mean(gaps[:accepted_draft_tokens])",
            "gap_post_reject_mean": "mean(gaps[accepted_draft_tokens:]), includes first rejection",
            "large_negative_gap": f"gap < {large_gap_threshold}",
            "missing_values": "NaN in CSV / null in JSON, explicit *_available indicators",
            "correlations": (
                "pairwise complete observations; average ranks for ties; undefined for "
                "constant or <2 observations; state-weighted exploratory associations, "
                "not causation or significance"
            ),
            "gain": "current route acceptance - current original acceptance",
            "oracle_gain": "max acceptance across six current routes - current original",
            "best_route_set": "all routes tied for maximum, no tie-breaking",
            "within_one": "route acceptance >= original acceptance - 1",
        },
        "acceptance_bin_boundaries": ACCEPTANCE_BINS,
        "rejection_bin_boundaries": REJECTION_BINS,
        "overall": description["overall"],
        "acceptance_buckets": description["acceptance_buckets"],
        "rejection_buckets": description["rejection_buckets"],
        "top_oracle_gain_correlations": [
            r
            for r in description["correlations"]
            if r["target"] == "oracle_gain" and r["spearman"] is not None
        ][:10],
        "predictors": {"enabled": False},
    }
    predictions = None
    if run_predictors:
        for columns in FEATURE_GROUPS.values():
            assert_predictor_columns(columns)
        predictions = _load_sibling("verification_route_predictors").run_predictors(
            rows, FEATURE_GROUPS
        )
        summary["predictors"] = {"enabled": True, **predictions["summary"]}
    artifacts = {
        "aligned_verification_route_dataset.csv": (
            rows,
            IDENTITY_COLUMNS + PREDICTOR_COLUMNS + LABEL_COLUMNS,
        ),
        "bucket_by_prev_acceptance.csv": (description["acceptance_buckets"], BUCKET_COLUMNS),
        "bucket_by_prev_reject_position.csv": (description["rejection_buckets"], BUCKET_COLUMNS),
        "feature_correlations.csv": (description["correlations"], CORRELATION_COLUMNS),
    }
    if predictions is not None:
        for key in ("prediction_metrics", "policy_metrics", "oof_predictions"):
            if predictions.get(key):
                artifacts[key + ".csv"] = (predictions[key], tuple(predictions[key][0]))
    summary["output_files"] = [*artifacts, "report.txt", "summary.json"]
    output = Path(output_dir)
    protected = {p.resolve() for p in trace_paths + oracle_paths}
    protected |= {Path(str(p) + ".meta.json").resolve() for p in trace_paths + oracle_paths}
    if any((output / name).resolve() in protected for name in summary["output_files"]):
        raise ValueError("output files must not overwrite inputs or their metadata")
    # Validation and optional fitting complete before any result file is written.
    # Refuse a stale prior report, so a failed rerun cannot look like a valid new run.
    if any((output / name).exists() for name in summary["output_files"]):
        raise ValueError("output artifacts already exist; use a fresh --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".p4-", dir=output) as temporary:
        staging = Path(temporary)
        for name, (records, columns) in artifacts.items():
            _write_csv(staging / name, records, columns)
        (staging / "report.txt").write_text(
            format_summary(summary, predictions) + "\n", encoding="utf-8"
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for name in summary["output_files"]:  # summary.json is the final completion marker.
            (staging / name).replace(output / name)
    return summary


def _format(value):
    return "undefined" if value is None else f"{value:.6f}"


def format_summary(summary, predictions=None):
    lines = [
        f"Oracle states: {summary['oracle_states']}",
        f"Aligned current states: {summary['aligned_current_states']}",
        f"Alignment mismatches: {summary['alignment_mismatches']}",
        f"Excluded round-0 states: {summary['excluded_first_round_states']}",
        f"Usable previous-feedback pairs: {summary['usable_states']}",
        summary["result_scope"],
        "Overall (usable states only):",
        "  P(restricted beats original): "
        + _format(summary["overall"]["p_any_restricted_beats_original"]),
        "  Mean oracle gain: " + _format(summary["overall"]["mean_oracle_gain"]),
    ]
    for title, key in (
        ("By previous acceptance", "acceptance_buckets"),
        (
            "By previous rejection (early 0-3, middle 4-8, late 9+, all accepted)",
            "rejection_buckets",
        ),
    ):
        lines.append(title + ":")
        for row in summary[key]:
            lines.append(
                f"  {row['bucket']}: n={row['num_states']}, "
                f"P(win)={_format(row['p_any_restricted_beats_original'])}, "
                f"oracle gain={_format(row['mean_oracle_gain'])}, "
                f"deep gain={_format(row['mean_gain_deep'])}, "
                f"mid_deep gain={_format(row['mean_gain_mid_deep'])}, "
                f"P(deep>=original)={_format(row['p_deep_matches_or_beats_original'])}, "
                f"P(mid_deep>=original)={_format(row['p_mid_deep_matches_or_beats_original'])}"
            )
    lines.append(
        "Top previous verification features correlated with oracle gain (exploratory, not causal):"
    )
    for row in summary["top_oracle_gain_correlations"]:
        lines.append(
            f"  {row['feature']}: n={row['num_pairs']}, "
            f"Spearman={_format(row['spearman'])}, Pearson={_format(row['pearson'])}"
        )
    lines.extend("Notice: " + warning for warning in summary["warnings"])
    if predictions is not None:
        lines.append(
            "Predictive baselines (sample_id-held-out; equal-request fold means; "
            "PR-AUC uses trapezoidal area):"
        )
        for row in predictions["prediction_metrics"]:
            lines.append(
                f"  {row['target']} / {row['model']} / {row['feature_group']}: "
                f"ROC-AUC={_format(row['fold_mean_roc_auc'])}, "
                f"PR-AUC={_format(row['fold_mean_pr_auc'])}, "
                f"balanced accuracy={_format(row['fold_mean_balanced_accuracy'])}, "
                f"valid folds={row['valid_roc_auc_folds']}/{row['num_folds']}, "
                f"single-class train folds={row['one_class_train_folds']}"
            )
        lines.append(
            "Binary routing policies (held-out predictions; captured gain is not clipped):"
        )
        for row in predictions["policy_metrics"]:
            lines.append(
                f"  {row['policy']} / {row['alternative_route'] or 'all'} / "
                f"{row['model']} / {row['feature_group']}: "
                f"mean acceptance={_format(row['policy_mean_accepted_tokens'])}, "
                f"captured oracle gain={_format(row['captured_oracle_gain'])}"
            )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", nargs="+", required=True, help="Canonical original-route P1 JSONL file(s)"
    )
    parser.add_argument(
        "--oracle", nargs="+", required=True, help="Validated six-route P3 JSONL file(s)"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory without existing P4 result files"
    )
    parser.add_argument("--large-gap-threshold", type=float, default=-2.0)
    parser.add_argument(
        "--run-predictors", action="store_true", help="Optional sklearn grouped baselines"
    )
    args = parser.parse_args(argv)
    try:
        summary = analyze(
            args.trace,
            args.oracle,
            args.output_dir,
            large_gap_threshold=args.large_gap_threshold,
            run_predictors=args.run_predictors,
        )
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    # Reuse the saved report so console and artifact include identical policy metrics.
    print((Path(args.output_dir) / "report.txt").read_text(encoding="utf-8"), end="")
    return summary


if __name__ == "__main__":
    main()
