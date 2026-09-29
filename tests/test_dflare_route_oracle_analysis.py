"""Hand-calculated oracle statistics and strict input validation; CPU/stdlib only."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "analyze_dflare_route_oracle.py"
SPEC = importlib.util.spec_from_file_location("dflare_oracle_analysis_test", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def make_record(round_id=0, values=None):
    values = {"original": 4, "deep": 4, "middle": 1} if values is None else values
    original = values["original"]
    return {
        "sample_id": 0,
        "turn_id": 0,
        "round_id": round_id,
        "prefix_length": 8 + round_id,
        "canonical_route": "original",
        "canonical_accepted_draft_tokens": original,
        "canonical_reported_acceptance_length": original + 1,
        "original_replay_matches": True,
        "route_results": {
            route: {
                "accepted_draft_tokens": accepted,
                "reported_acceptance_length": accepted + 1,
                "draft_token_ids": list(range(10, 16)),
            }
            for route, accepted in values.items()
        },
    }


def example_records():
    return [
        make_record(index, dict(zip(("original", "deep", "middle"), values)))
        for index, values in enumerate(((4, 4, 1), (4, 2, 5), (1, 6, 2), (3, 1, 2)))
    ]


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class OracleAnalysisTests(unittest.TestCase):
    def test_hand_calculated_best_static_oracle_gains_and_wins(self):
        summary = analysis.summarize_records(iter(example_records()))
        self.assertEqual(summary["num_states"], 4)
        self.assertEqual(
            summary["mean_accepted_draft_tokens"], {"original": 3, "deep": 3.25, "middle": 2.5}
        )
        self.assertEqual(
            summary["best_static"], {"routes": ["deep"], "mean_accepted_draft_tokens": 3.25}
        )
        oracle = summary["oracle"]
        self.assertEqual(oracle["mean_accepted_draft_tokens"], 4.5)
        self.assertEqual(oracle["gain_over_best_static"]["absolute"], 1.25)
        self.assertAlmostEqual(oracle["gain_over_best_static"]["relative_fraction"], 5 / 13)
        self.assertEqual(oracle["gain_over_original"], {"absolute": 1.5, "relative_fraction": 0.5})
        self.assertEqual(
            summary["win_frequency"]["original"],
            {
                "unique_win_count": 1,
                "unique_win_fraction": 0.25,
                "max_or_tied_count": 2,
                "max_or_tied_fraction": 0.5,
            },
        )
        self.assertEqual(summary["win_frequency"]["deep"]["unique_win_count"], 1)
        self.assertEqual(summary["win_frequency"]["deep"]["max_or_tied_count"], 2)
        self.assertEqual(summary["win_frequency"]["middle"]["max_or_tied_count"], 1)
        self.assertEqual(
            summary["original_optimality"],
            {
                "original_optimal_or_tied": {"count": 2, "fraction": 0.5},
                "any_restricted_beats_original": {"count": 2, "fraction": 0.5},
                "any_restricted_matches_original": {"count": 1, "fraction": 0.25},
                "all_restricted_worse_than_original": {"count": 1, "fraction": 0.25},
            },
        )
        self.assertEqual(
            summary["restricted_route_opportunity"]["middle"],
            {
                "matches_original": {"count": 0, "fraction": 0},
                "within_one_of_original": {"count": 3, "fraction": 0.75},
                "beats_original": {"count": 2, "fraction": 0.5},
            },
        )

    def test_tied_static_means_and_tied_rounds_are_not_tiebroken(self):
        records = [
            make_record(i, values)
            for i, values in enumerate(
                (
                    {"original": 2, "deep": 4},
                    {"original": 4, "deep": 2},
                    {"original": 3, "deep": 3},
                )
            )
        ]
        summary = analysis.summarize_records(records)
        self.assertEqual(summary["best_static"]["routes"], ["original", "deep"])
        self.assertEqual(summary["best_static"]["mean_accepted_draft_tokens"], 3)
        self.assertEqual(summary["oracle"]["mean_accepted_draft_tokens"], 11 / 3)
        for route in ("original", "deep"):
            self.assertEqual(summary["win_frequency"][route]["unique_win_count"], 1)
            self.assertEqual(summary["win_frequency"][route]["max_or_tied_count"], 2)

    def test_zero_denominators_are_undefined_instead_of_nan_or_infinity(self):
        summary = analysis.summarize_records([make_record(values={"original": 0, "deep": 0})])
        for key in ("gain_over_original", "gain_over_best_static"):
            self.assertEqual(summary["oracle"][key], {"absolute": 0, "relative_fraction": None})
        self.assertIn("undefined (zero baseline)", analysis.format_summary(summary))
        json.dumps(summary, allow_nan=False)
        summary = analysis.summarize_records([make_record(values={"original": 0, "deep": 2})])
        self.assertIsNone(summary["oracle"]["gain_over_original"]["relative_fraction"])
        self.assertEqual(summary["oracle"]["gain_over_best_static"]["relative_fraction"], 0)

    def test_original_only_has_no_vacuous_restricted_route_wins(self):
        summary = analysis.summarize_records([make_record(values={"original": 3})])
        self.assertEqual(summary["num_restricted_routes"], 0)
        self.assertEqual(summary["restricted_route_opportunity"], {})
        self.assertEqual(
            summary["original_optimality"]["all_restricted_worse_than_original"]["count"], 0
        )

    def test_opportunities_can_overlap(self):
        summary = analysis.summarize_records(
            [make_record(values={"original": 3, "deep": 3, "middle": 4})]
        )
        self.assertEqual(
            summary["original_optimality"]["any_restricted_beats_original"]["fraction"], 1
        )
        self.assertEqual(
            summary["original_optimality"]["any_restricted_matches_original"]["fraction"], 1
        )
        self.assertIn("fractions may overlap", analysis.format_summary(summary))

    def test_missing_required_fields_are_rejected(self):
        for key in make_record():
            with self.subTest(key=key):
                record = make_record()
                del record[key]
                with self.assertRaises(ValueError):
                    analysis.summarize_records([record])
        for key in make_record()["route_results"]["deep"]:
            with self.subTest(route_key=key):
                record = make_record()
                del record["route_results"]["deep"][key]
                with self.assertRaises(ValueError):
                    analysis.summarize_records([record])

    def test_invalid_markers_and_malformed_state_fields_are_rejected(self):
        changes = (
            ("canonical_route", "deep"),
            ("original_replay_matches", False),
            ("original_replay_matches", 1),
            ("valid", False),
            ("is_valid", False),
            ("invalid", True),
            ("original_replay_mismatch", True),
            ("original_replay_mismatch_count", 1),
            ("mismatch_count", None),
            ("sample_id", -1),
            ("turn_id", True),
            ("round_id", 0.0),
            ("prefix_length", 0),
            ("canonical_accepted_draft_tokens", 3),
            ("canonical_reported_acceptance_length", 4),
            ("route_results", []),
            ("block_size", 6),
            ("canonical_draft_token_ids", [999] * 6),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value):
                record = make_record()
                record[key] = value
                with self.assertRaises(ValueError):
                    analysis.summarize_records([record])
        with self.assertRaisesRegex(ValueError, "JSON object"):
            analysis.summarize_records([[]])

    def test_invalid_route_values_are_rejected(self):
        changes = (
            ("accepted_draft_tokens", -1),
            ("accepted_draft_tokens", True),
            ("accepted_draft_tokens", 7),
            ("reported_acceptance_length", 4),
            ("draft_token_ids", []),
            ("draft_token_ids", [1, 2]),
            ("draft_token_ids", [1, 2, 3, 4, 5, "6"]),
            ("target_token_ids_for_proposals", [10, 11]),
            ("target_token_ids_for_proposals", [10, 11, 12, 13, 14, 15]),
            ("invalid", True),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value):
                record = make_record()
                record["route_results"]["deep"][key] = value
                with self.assertRaises(ValueError):
                    analysis.summarize_records([record])

    def test_optional_debug_tokens_confirm_acceptance_and_canonical_match(self):
        record = make_record()
        record["block_size"] = 7
        record["canonical_draft_token_ids"] = list(range(10, 16))
        for result in record["route_results"].values():
            accepted = result["accepted_draft_tokens"]
            result["target_token_ids_for_proposals"] = result["draft_token_ids"][:accepted] + [
                99
            ] * (6 - accepted)
        self.assertEqual(analysis.summarize_records([record])["num_states"], 1)
        record["route_results"]["original"]["accepted_draft_tokens"] = 3
        record["route_results"]["original"]["reported_acceptance_length"] = 4
        del record["route_results"]["original"]["target_token_ids_for_proposals"]
        with self.assertRaisesRegex(ValueError, "differs from canonical"):
            analysis.summarize_records([record])

    def test_duplicate_state_identity_and_inconsistent_route_sets_are_rejected(self):
        record = make_record()
        duplicate = copy.deepcopy(record)
        duplicate["prefix_length"] += 1
        with self.assertRaisesRegex(ValueError, "duplicate state"):
            analysis.summarize_records([record, duplicate])
        other = make_record(1)
        del other["route_results"]["middle"]
        with self.assertRaisesRegex(ValueError, "inconsistent route set"):
            analysis.summarize_records([record, other])
        other = make_record(1)
        other["route_results"] = dict(reversed(list(other["route_results"].items())))
        self.assertEqual(analysis.summarize_records([record, other])["num_states"], 2)

    def test_streaming_aggregation_across_files_and_duplicate_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "rank0.jsonl", Path(directory) / "rank1.jsonl"
            records = example_records()
            write_jsonl(first, records[:2])
            write_jsonl(second, records[2:])
            self.assertEqual(
                analysis.summarize_oracle([first, second]), analysis.summarize_records(records)
            )
            with self.assertRaisesRegex(ValueError, "rank0.jsonl:1: duplicate state"):
                analysis.summarize_oracle([first, first])

    def test_malformed_partial_and_blank_lines_are_never_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.jsonl"
            for bad in ('{"route_results":', "\n", "NaN\n", '{"sample_id":0,"sample_id":1}\n'):
                with self.subTest(bad=bad):
                    path.write_text(json.dumps(make_record()) + "\n" + bad, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "oracle.jsonl:2"):
                        analysis.summarize_oracle(path)
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no canonical states"):
                analysis.summarize_oracle(path)
        with self.assertRaisesRegex(ValueError, "no canonical states"):
            analysis.summarize_records([])

    def test_existing_sidecars_require_complete_evaluation_without_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.jsonl"
            metadata = Path(str(path) + ".meta.json")
            write_jsonl(path, [make_record()])
            invalid = (
                [
                    {"evaluation_status": status, "original_replay_mismatch_count": 0}
                    for status in ("running", "invalid", None)
                ]
                + [
                    {"evaluation_status": "complete", "original_replay_mismatch_count": value}
                    for value in (None, 1, False)
                ]
                + [{"evaluation_status": "complete"}, [], {}]
            )
            for value in invalid:
                with self.subTest(value=value):
                    metadata.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "oracle.jsonl.meta.json"):
                        analysis.summarize_oracle(path)
            metadata.write_text(
                json.dumps({"evaluation_status": "complete", "original_replay_mismatch_count": 0}),
                encoding="utf-8",
            )
            self.assertEqual(analysis.summarize_oracle(path)["num_states"], 1)
            metadata.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "oracle.jsonl.meta.json"):
                analysis.summarize_oracle(path)

    def test_cli_runs_without_site_packages_and_writes_json_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = (
                Path(directory) / "oracle.jsonl",
                Path(directory) / "results" / "summary.json",
            )
            write_jsonl(path, example_records())
            completed = subprocess.run(
                [sys.executable, "-S", str(SCRIPT), str(path), "--json-output", str(output)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Best static route(s): deep", completed.stdout)
            self.assertEqual(completed.stdout.count("Best static route(s):"), 1)
            self.assertEqual(completed.stdout.count("Best static mean accepted draft tokens:"), 1)
            self.assertIn("relative=50.00%", completed.stdout)
            self.assertEqual(json.loads(output.read_text())["num_states"], 4)
            invalid = subprocess.run(
                [sys.executable, "-S", str(SCRIPT), str(path), "--json-output", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("must not overwrite an input", invalid.stderr)
            self.assertEqual(invalid.stdout, "")

    def test_cli_cannot_overwrite_any_input_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "rank0.jsonl", Path(directory) / "rank1.jsonl"
            write_jsonl(first, [make_record(0)])
            write_jsonl(second, [make_record(1)])
            metadata = Path(str(second) + ".meta.json")
            original_metadata = json.dumps(
                {"evaluation_status": "complete", "original_replay_mismatch_count": 0}
            )
            metadata.write_text(original_metadata, encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(SCRIPT),
                    str(first),
                    str(second),
                    "--json-output",
                    str(metadata),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("must not overwrite an input file or its metadata", completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertEqual(metadata.read_text(encoding="utf-8"), original_metadata)


if __name__ == "__main__":
    unittest.main()
