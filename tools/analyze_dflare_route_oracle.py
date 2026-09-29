"""Analyze validated same-state DFlare route results using only the standard library.

All metrics use accepted draft tokens, excluding the anchor. Input files can be
rank-sharded JSONL outputs, but every state must occur exactly once. No malformed
or invalid states are skipped. Accumulators stream the input; only state identity
keys are retained to detect duplicates across files.
Existing metadata sidecars must mark evaluation complete with zero replay mismatches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _nonnegative_integer(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _tokens(value, field):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty token ID list")
    for token in value:
        _nonnegative_integer(token, field)
    return value


def _validate_markers(record):
    for key in ("valid", "is_valid"):
        if key in record and record[key] is not True:
            raise ValueError(f"{key} must be true; invalid states cannot be analyzed")
    for key in ("invalid", "original_replay_mismatch"):
        if key in record and record[key] is not False:
            raise ValueError(f"{key} must be false; invalid states cannot be analyzed")
    for key in ("mismatch_count", "original_replay_mismatch_count"):
        if key in record and (type(record[key]) is not int or record[key] != 0):
            raise ValueError(f"{key} must be zero; replay mismatches invalidate the input")


def _validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("each state must be a JSON object")
    _validate_markers(record)
    identity = tuple(
        _nonnegative_integer(record.get(key), key) for key in ("sample_id", "turn_id", "round_id")
    )
    if _nonnegative_integer(record.get("prefix_length"), "prefix_length") == 0:
        raise ValueError("prefix_length must include at least the anchor token")
    if record.get("canonical_route") != "original":
        raise ValueError("canonical_route must be 'original'")
    if record.get("original_replay_matches") is not True:
        raise ValueError("original_replay_matches must be true; replay is missing or invalid")
    canonical = _nonnegative_integer(
        record.get("canonical_accepted_draft_tokens"), "canonical_accepted_draft_tokens"
    )
    canonical_reported = _nonnegative_integer(
        record.get("canonical_reported_acceptance_length"),
        "canonical_reported_acceptance_length",
    )
    if canonical_reported != canonical + 1:
        raise ValueError("canonical_reported_acceptance_length must equal canonical + 1")
    results = record.get("route_results")
    if not isinstance(results, dict) or "original" not in results:
        raise ValueError("route_results must be an object containing 'original'")

    values = {}
    proposal_count = None
    for route, result in results.items():
        if not isinstance(route, str) or not route.strip():
            raise ValueError("route_results names must be nonempty strings")
        if not isinstance(result, dict):
            raise ValueError(f"route_results.{route} must be an object")
        _validate_markers(result)
        accepted = _nonnegative_integer(
            result.get("accepted_draft_tokens"), f"{route}.accepted_draft_tokens"
        )
        reported = _nonnegative_integer(
            result.get("reported_acceptance_length"), f"{route}.reported_acceptance_length"
        )
        draft = _tokens(result.get("draft_token_ids"), f"{route}.draft_token_ids")
        if accepted > len(draft):
            raise ValueError(f"{route}.accepted_draft_tokens exceeds its proposal count")
        if reported != accepted + 1:
            raise ValueError(f"{route}.reported_acceptance_length must equal accepted + 1")
        if proposal_count is not None and len(draft) != proposal_count:
            raise ValueError("all routes in a state must have the same proposal count")
        proposal_count = len(draft)
        if "target_token_ids_for_proposals" in result:
            target = _tokens(
                result["target_token_ids_for_proposals"],
                f"{route}.target_token_ids_for_proposals",
            )
            if len(target) != len(draft):
                raise ValueError(f"{route} target and draft proposal counts differ")
            prefix_acceptance = 0
            for proposed, verified in zip(draft, target):
                if proposed != verified:
                    break
                prefix_acceptance += 1
            if prefix_acceptance != accepted:
                raise ValueError(f"{route} accepted count disagrees with token-prefix matches")
        values[route] = accepted

    if values["original"] != canonical:
        raise ValueError("original replay acceptance differs from canonical acceptance")
    if "canonical_draft_token_ids" in record:
        canonical_draft = _tokens(record["canonical_draft_token_ids"], "canonical_draft_token_ids")
        if canonical_draft != results["original"]["draft_token_ids"]:
            raise ValueError("original replay draft tokens differ from canonical draft tokens")
    if "block_size" in record:
        block_size = _nonnegative_integer(record["block_size"], "block_size")
        if block_size != proposal_count + 1:
            raise ValueError("block_size must equal proposal count + 1")
    return identity, values


def _count_fraction(count, num_states):
    return {"count": count, "fraction": count / num_states}


def _gain(oracle, baseline):
    absolute = oracle - baseline
    return {
        "absolute": absolute,
        "relative_fraction": absolute / baseline if baseline else None,
    }


def _summarize_located_records(located_records):
    seen = set()
    routes = None
    num_states = 0
    oracle_total = 0
    optimality = {
        "original_optimal_or_tied": 0,
        "any_restricted_beats_original": 0,
        "any_restricted_matches_original": 0,
        "all_restricted_worse_than_original": 0,
    }
    for source, record in located_records:
        try:
            identity, values = _validate_record(record)
            if identity in seen:
                raise ValueError(f"duplicate state (sample_id, turn_id, round_id) = {identity}")
            if routes is None:
                routes = ["original", *(route for route in values if route != "original")]
                totals = dict.fromkeys(routes, 0)
                unique_wins = dict.fromkeys(routes, 0)
                max_or_tied = dict.fromkeys(routes, 0)
                opportunity = {
                    route: {
                        "matches_original": 0,
                        "within_one_of_original": 0,
                        "beats_original": 0,
                    }
                    for route in routes
                    if route != "original"
                }
            elif set(values) != set(routes):
                raise ValueError(
                    f"inconsistent route set: expected {routes}, received {list(values)}"
                )
        except ValueError as error:
            raise ValueError(f"{source}: {error}") from error

        seen.add(identity)
        num_states += 1
        maximum = max(values.values())
        oracle_total += maximum
        winners = [route for route in routes if values[route] == maximum]
        for route in routes:
            totals[route] += values[route]
            max_or_tied[route] += int(route in winners)
        if len(winners) == 1:
            unique_wins[winners[0]] += 1
        original = values["original"]
        restricted = [values[route] for route in opportunity]
        optimality["original_optimal_or_tied"] += int(original == maximum)
        optimality["any_restricted_beats_original"] += int(any(x > original for x in restricted))
        optimality["any_restricted_matches_original"] += int(
            any(x == original for x in restricted)
        )
        optimality["all_restricted_worse_than_original"] += int(
            bool(restricted) and all(x < original for x in restricted)
        )
        for route, counts in opportunity.items():
            counts["matches_original"] += int(values[route] == original)
            counts["within_one_of_original"] += int(values[route] >= original - 1)
            counts["beats_original"] += int(values[route] > original)

    if not num_states:
        raise ValueError("no canonical states found in the input")
    means = {route: totals[route] / num_states for route in routes}
    best_total = max(totals.values())
    best_mean = best_total / num_states
    oracle_mean = oracle_total / num_states
    return {
        "metric": "accepted_draft_tokens",
        "num_states": num_states,
        "routes": routes,
        "num_restricted_routes": len(opportunity),
        "mean_accepted_draft_tokens": means,
        "best_static": {
            "routes": [route for route in routes if totals[route] == best_total],
            "mean_accepted_draft_tokens": best_mean,
        },
        "oracle": {
            "mean_accepted_draft_tokens": oracle_mean,
            "gain_over_best_static": _gain(oracle_mean, best_mean),
            "gain_over_original": _gain(oracle_mean, means["original"]),
        },
        "win_frequency": {
            route: {
                "unique_win_count": unique_wins[route],
                "unique_win_fraction": unique_wins[route] / num_states,
                "max_or_tied_count": max_or_tied[route],
                "max_or_tied_fraction": max_or_tied[route] / num_states,
            }
            for route in routes
        },
        "original_optimality": {
            key: _count_fraction(count, num_states) for key, count in optimality.items()
        },
        "restricted_route_opportunity": {
            route: {key: _count_fraction(count, num_states) for key, count in counts.items()}
            for route, counts in opportunity.items()
        },
    }


def summarize_records(records):
    """Validate and summarize an iterable of decoded oracle JSON objects.

    Relative gains are fractions (0.1 means 10%), or None for a zero baseline.
    Fractions use all states as the denominator and may overlap. When no
    restricted routes are present, all restricted-route opportunity counts are
    zero, including all_restricted_worse_than_original (no vacuous wins).
    """
    return _summarize_located_records(
        (f"record {index}", record) for index, record in enumerate(records, start=1)
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value {value!r}")


def _read_jsonl(paths):
    for path in paths:
        metadata_path = Path(str(path) + ".meta.json")
        if metadata_path.exists():
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8"),
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
                if (
                    not isinstance(metadata, dict)
                    or metadata.get("evaluation_status") != "complete"
                ):
                    raise ValueError(
                        "evaluation_status must be 'complete'; input may be partial or invalid"
                    )
                if (
                    type(metadata.get("original_replay_mismatch_count")) is not int
                    or metadata["original_replay_mismatch_count"] != 0
                ):
                    raise ValueError("original_replay_mismatch_count must be zero")
                _validate_markers(metadata)
            except ValueError as error:
                raise ValueError(f"{metadata_path}: {error}") from error
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                source = f"{path}:{line_number}"
                try:
                    record = json.loads(
                        line,
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                    )
                except ValueError as error:
                    raise ValueError(f"{source}: invalid JSON: {error}") from error
                yield source, record


def summarize_oracle(paths):
    """Stream JSONL files; existing sidecars must confirm complete, valid replay."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return _summarize_located_records(_read_jsonl(paths))


def format_summary(summary):
    """Render the metrics with counts, tie semantics, and zero-baseline handling."""
    lines = [
        f"Canonical states: {summary['num_states']}",
        "Metric: accepted_draft_tokens (anchor excluded)",
        "Per-route mean accepted draft tokens:",
    ]
    for route, mean in summary["mean_accepted_draft_tokens"].items():
        lines.append(f"  {route}: {mean:.6f}")
    best = summary["best_static"]
    oracle = summary["oracle"]
    lines.extend(
        [
            f"Best static route(s): {', '.join(best['routes'])}",
            f"Best static mean accepted draft tokens: {best['mean_accepted_draft_tokens']:.6f}",
            f"Oracle mean accepted draft tokens: {oracle['mean_accepted_draft_tokens']:.6f}",
        ]
    )
    for key, label in (
        ("gain_over_best_static", "best static"),
        ("gain_over_original", "original"),
    ):
        gain = oracle[key]
        relative = gain["relative_fraction"]
        relative_text = "undefined (zero baseline)" if relative is None else f"{relative:.2%}"
        lines.append(
            f"Oracle gain over {label}: absolute={gain['absolute']:.6f}, relative={relative_text}"
        )
    lines.append("Route win frequency (ties count for every maximizing route):")
    for route, counts in summary["win_frequency"].items():
        lines.append(
            f"  {route}: unique_win_fraction={counts['unique_win_fraction']:.2%} "
            f"({counts['unique_win_count']}/{summary['num_states']}), "
            f"max_or_tied_fraction={counts['max_or_tied_fraction']:.2%} "
            f"({counts['max_or_tied_count']}/{summary['num_states']})"
        )
    lines.append("Original optimality (fractions may overlap):")
    for key, metric in summary["original_optimality"].items():
        lines.append(
            f"  {key}: {metric['fraction']:.2%} ({metric['count']}/{summary['num_states']})"
        )
    lines.append("Restricted-route opportunities (fractions may overlap):")
    for route, metrics in summary["restricted_route_opportunity"].items():
        items = [
            f"{key}={metric['fraction']:.2%} ({metric['count']}/{summary['num_states']})"
            for key, metric in metrics.items()
        ]
        lines.append(f"  {route}: {', '.join(items)}")
    if not summary["num_restricted_routes"]:
        lines.append("  No restricted routes; restricted-route counts are zero.")
    lines.append("within_one_of_original means accepted >= original - 1, including better routes.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Oracle JSONL file(s), including shards"
    )
    parser.add_argument(
        "--json-output", type=Path, help="Also write a machine-readable JSON summary"
    )
    args = parser.parse_args(argv)
    try:
        if args.json_output is not None and args.json_output.resolve() in {
            protected.resolve()
            for path in args.inputs
            for protected in (path, Path(str(path) + ".meta.json"))
        }:
            raise ValueError("--json-output must not overwrite an input file or its metadata")
        summary = summarize_oracle(args.inputs)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(format_summary(summary))


if __name__ == "__main__":
    main()
