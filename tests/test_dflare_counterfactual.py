"""CPU checks for exact original-history reconstruction and isolated route probes."""

import copy
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1] / "tools" / "dflash_benchmark.py"
    spec = importlib.util.spec_from_file_location("dflare_counterfactual_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryWriter:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(json.loads(json.dumps(record, allow_nan=False)))


class ToyEnvironment:
    """Model-visible cache contents depend on past routes, not just cache length.

    Draft context entries remain after crop; draft noise entries are discarded.
    This reproduces the lag between draft-cache length and the current anchor.
    A route-contaminated historical cache changes subsequent draft proposals.
    """

    def __init__(self, benchmark, monkeypatch):
        self.benchmark = benchmark
        self.calls, self.caches, self.route_changes = [], [], []
        self.vocab_size = 11
        environment = self

        class Cache:
            def __init__(self):
                self.entries = []
                self.index = len(environment.caches)
                environment.caches.append(self)

            def get_seq_length(self):
                return len(self.entries)

            def crop(self, length):
                del self.entries[length:]

        class Draft:
            device = torch.device("cpu")
            target_layer_ids = [1, 5, 9, 13, 17, 21, 25, 29, 33]
            block_size = 4
            mask_token_id = environment.vocab_size
            training = False

            def __init__(self):
                self._target_route_mask = None
                self.route = "original"

            def set_target_layer_route(self, active_layer_ids=None):
                self.route = "original"
                self._target_route_mask = None
                if active_layer_ids is not None:
                    self.route = next(
                        name
                        for name, ids in benchmark.TARGET_ROUTES.items()
                        if ids == list(active_layer_ids)
                    )
                    self._target_route_mask = torch.tensor(
                        [layer_id in active_layer_ids for layer_id in self.target_layer_ids]
                    )
                environment.route_changes.append(self.route)

            def __call__(
                self,
                *,
                target_hidden,
                noise_embedding,
                position_ids,
                past_key_values,
                use_cache,
                is_causal,
            ):
                before = copy.deepcopy(past_key_values.entries)
                environment.calls.append(
                    {
                        "kind": "draft",
                        "route": self.route,
                        "cache": past_key_values.index,
                        "before": before,
                        "positions": position_ids.tolist(),
                        "hidden": target_hidden.tolist(),
                        "noise": noise_embedding.tolist(),
                    }
                )
                code = ["original", *benchmark.TARGET_ROUTES].index(self.route)
                # Only past context entries contribute here. Evaluating a route on
                # an already-mutated prior route cache will therefore change logits.
                historical_route_bias = sum(entry[2] for entry in before) % 10
                for token, position in target_hidden[0].tolist():
                    past_key_values.entries.append((int(position), int(token), code))
                positions = position_ids[:, -noise_embedding.shape[1] :]
                for position, token in zip(
                    positions[0].tolist(), noise_embedding.argmax(-1)[0].tolist()
                ):
                    past_key_values.entries.append((position, token, code))
                mismatch = (positions % 5 == 0).long()
                if self.route == "deep":
                    mismatch = torch.zeros_like(mismatch)
                elif self.route != "original":
                    mismatch = mismatch + code
                preferred = (positions + mismatch + historical_route_bias) % environment.vocab_size
                return environment.logits(preferred)

        class Target:
            device = torch.device("cpu")
            model = SimpleNamespace(
                embed_tokens=lambda ids: torch.nn.functional.one_hot(
                    ids, environment.vocab_size + 1
                ).float()
            )
            lm_head = staticmethod(lambda hidden: hidden)

            def __call__(
                self,
                input_ids,
                *,
                position_ids,
                past_key_values,
                use_cache,
                output_hidden_states,
                logits_to_keep=None,
            ):
                environment.calls.append(
                    {
                        "kind": "target",
                        "cache": past_key_values.index,
                        "before": copy.deepcopy(past_key_values.entries),
                        "positions": position_ids.tolist(),
                        "tokens": input_ids.tolist(),
                        "prefill": logits_to_keep is not None,
                    }
                )
                for position, token in zip(position_ids[0].tolist(), input_ids[0].tolist()):
                    past_key_values.entries.append((position, token, 0))
                logits = environment.logits((position_ids + 1) % environment.vocab_size)
                if logits_to_keep is not None:
                    logits = logits[:, -logits_to_keep:]
                return SimpleNamespace(
                    logits=logits,
                    hidden_states=torch.stack((input_ids.float(), position_ids.float()), dim=-1),
                )

        self.model, self.target = Draft(), Target()
        monkeypatch.setattr(benchmark, "DynamicCache", Cache)
        ticks = itertools.count()
        monkeypatch.setattr(benchmark, "cuda_time", lambda: float(next(ticks)))

    def logits(self, preferred):
        return -(torch.arange(self.vocab_size).float() - preferred.unsqueeze(-1)).square()

    @staticmethod
    def sample(logits, temperature=0.0):
        assert temperature == 0.0
        return logits.argmax(dim=-1)

    @staticmethod
    def extract(hidden_states, layer_ids):
        assert layer_ids == [1, 5, 9, 13, 17, 21, 25, 29, 33]
        return hidden_states

    def generate(self, **kwargs):
        arguments = dict(
            model=self.model,
            target=self.target,
            input_ids=torch.tensor([[1, 2, 3]]),
            mask_token_id=self.model.mask_token_id,
            max_new_tokens=16,
            block_size=4,
            stop_token_ids=None,
            sample_fn=self.sample,
            extract_context_feature_fn=self.extract,
            temperature=0.0,
            trace_context={"sample_id": 73, "turn_id": 2},
        )
        arguments.update(kwargs)
        return self.benchmark.dflash_generate(**arguments)

    def evaluate(self, state, routes=None):
        return self.benchmark.evaluate_counterfactual_state(
            state,
            model=self.model,
            target=self.target,
            sample_fn=self.sample,
            extract_context_feature_fn=self.extract,
            routes=routes or ["original", "shallow", "deep"],
        )


def _capture(benchmark, monkeypatch):
    environment = ToyEnvironment(benchmark, monkeypatch)
    writer = MemoryWriter()
    output = environment.generate(state_writer=writer)
    return environment, writer.records, output


def test_canonical_manifest_streams_round_start_prefix_and_actual_draft_context(
    benchmark, monkeypatch, tmp_path
):
    environment = ToyEnvironment(benchmark, monkeypatch)
    path = tmp_path / "states.jsonl"
    with benchmark.VerificationTraceWriter(path) as writer:
        output = environment.generate(state_writer=writer)
        states = [json.loads(line) for line in path.read_text().splitlines()]
    draft_calls = [call for call in environment.calls if call["kind"] == "draft"]
    assert len(states) == len(output.acceptance_lengths) == len(draft_calls)
    assert len(states) > 2
    for round_id, state in enumerate(states):
        benchmark._validate_canonical_state(state)
        start = 3 + sum(output.acceptance_lengths[:round_id])
        assert (state["sample_id"], state["turn_id"], state["round_id"]) == (73, 2, round_id)
        assert state["state_schema_version"] == 1
        assert state["round_start"] == start
        assert state["num_input_tokens"] == 3
        assert state["generated_tokens_before_round"] == start - 3
        prefix = state["canonical_prefix_token_ids"]
        assert len(prefix) == start + 1
        assert prefix[:3] == [1, 2, 3]
        assert prefix[-1] == start % environment.vocab_size
        assert state["draft_context_start"] == len(draft_calls[round_id]["before"])
        assert state["draft_context_start"] < start
        assert state["canonical_route"] == "original"
        assert state["canonical_reported_acceptance_length"] == output.acceptance_lengths[round_id]
        assert state["canonical_accepted_draft_tokens"] + 1 == output.acceptance_lengths[round_id]
        assert len(state["canonical_draft_token_ids"]) == state["block_size"] - 1 == 3
        assert len(state["canonical_target_token_ids_for_proposals"]) == 3
        assert state["target_layer_ids"] == environment.model.target_layer_ids
        assert state["temperature"] == 0.0
        assert state["max_new_tokens"] == 16
        assert state["mask_token_id"] == environment.model.mask_token_id
        assert state["stop_token_ids"] is None


def test_state_capture_keeps_generation_outputs_caches_and_rng_unchanged(benchmark, monkeypatch):
    torch.manual_seed(42)
    plain = ToyEnvironment(benchmark, monkeypatch)
    plain_result = plain.generate()
    plain_rng = torch.get_rng_state().clone()
    traced, states, traced_result = _capture(benchmark, monkeypatch)
    assert states
    assert torch.equal(plain_result.output_ids, traced_result.output_ids)
    assert plain_result.acceptance_lengths == traced_result.acceptance_lengths
    assert plain.calls == traced.calls
    assert [cache.entries for cache in plain.caches] == [cache.entries for cache in traced.caches]
    assert torch.equal(plain_rng, torch.get_rng_state())


def test_original_replay_matches_every_canonical_round(benchmark, monkeypatch):
    environment, states, _ = _capture(benchmark, monkeypatch)
    for state in states:
        result = environment.evaluate(state, routes=["original"])
        assert result["original_replay_matches"] is True
        assert result["prefix_length"] == len(state["canonical_prefix_token_ids"])
        assert result["canonical_route"] == "original"
        for key in (
            "sample_id",
            "turn_id",
            "round_id",
            "canonical_accepted_draft_tokens",
            "canonical_reported_acceptance_length",
        ):
            assert result[key] == state[key]
        original = result["route_results"]["original"]
        for key in (
            "accepted_draft_tokens",
            "reported_acceptance_length",
            "draft_token_ids",
            "target_token_ids_for_proposals",
        ):
            assert original[key] == state[f"canonical_{key}"]


def test_routes_reconstruct_original_history_with_fresh_caches_before_single_probe(
    benchmark, monkeypatch
):
    environment, states, _ = _capture(benchmark, monkeypatch)
    state = states[2]
    canonical_copy = copy.deepcopy(states)
    calls_before, caches_before = len(environment.calls), len(environment.caches)
    result = environment.evaluate(state, routes=["deep", "original", "shallow"])
    assert list(result["route_results"]) == ["original", "deep", "shallow"]
    calls = environment.calls[calls_before:]
    fresh_caches = environment.caches[caches_before:]
    assert len(fresh_caches) == 6
    assert len({id(cache) for cache in fresh_caches}) == 6
    groups = []
    for offset, route in enumerate(result["route_results"]):
        target_cache, draft_cache = fresh_caches[2 * offset : 2 * offset + 2]
        group = [
            call for call in calls if call["cache"] in {target_cache.index, draft_cache.index}
        ]
        drafts = [call for call in group if call["kind"] == "draft"]
        targets = [call for call in group if call["kind"] == "target"]
        assert len(drafts) == state["round_id"] + 1
        assert len(targets) == state["round_id"] + 2
        assert targets[0]["prefill"]
        assert targets[0]["tokens"] == [[1, 2, 3]]
        assert [call["route"] for call in drafts] == ["original"] * state["round_id"] + [route]
        assert len(drafts[-1]["before"]) == state["draft_context_start"]
        assert all(entry[2] == 0 for entry in drafts[-1]["before"])
        groups.append(group)

    def normalized(call):
        return {key: value for key, value in call.items() if key != "cache"}

    for alternative in groups[1:]:
        # Both previous original rounds (including target KV and hidden values)
        # must be identical, not just the round counter or cache lengths.
        assert list(map(normalized, alternative[:-2])) == list(map(normalized, groups[0][:-2]))
        final_draft = dict(normalized(alternative[-2]), route="original")
        assert final_draft == normalized(groups[0][-2])
        assert alternative[-1]["before"] == groups[0][-1]["before"]
    assert states == canonical_copy
    assert (
        result["route_results"]["deep"]["draft_token_ids"]
        != result["route_results"]["shallow"]["draft_token_ids"]
    )
    assert environment.model._target_route_mask is None
    assert environment.model.route == "original"


def test_one_round_probe_does_not_commit_token_buffers_or_advance_round(benchmark, monkeypatch):
    environment, states, _ = _capture(benchmark, monkeypatch)
    state = states[1]
    snapshot = environment.generate(_stop_before_round=state["round_id"])
    assert snapshot._replay_state is True
    assert snapshot.start == state["round_start"]
    assert (
        snapshot.output_ids[0, : snapshot.start + 1].tolist()
        == state["canonical_prefix_token_ids"]
    )
    before = snapshot.output_ids.clone()
    count = len(environment.calls)
    result = benchmark._run_counterfactual_round(
        environment.model,
        environment.target,
        snapshot,
        environment.sample,
        environment.extract,
        block_size=4,
    )
    assert [call["kind"] for call in environment.calls[count:]] == ["draft", "target"]
    assert torch.equal(snapshot.output_ids, before)
    assert snapshot.start == state["round_start"]
    assert result["draft_token_ids"] == state["canonical_draft_token_ids"]
    assert result["accepted_draft_tokens"] == state["canonical_accepted_draft_tokens"]


@pytest.mark.parametrize(
    "field", ["canonical_draft_token_ids", "canonical_target_token_ids_for_proposals"]
)
def test_original_mismatch_fails_before_any_alternative_and_restores_original(
    benchmark, monkeypatch, field
):
    environment, states, _ = _capture(benchmark, monkeypatch)
    state = copy.deepcopy(states[0])
    state[field][-1] = (state[field][-1] + 3) % environment.vocab_size
    start = len(environment.calls)
    with pytest.raises(ValueError, match="[Mm]ismatch|reproduce|replay") as error:
        environment.evaluate(state)
    assert "73" in str(error.value)
    assert all(
        call["route"] == "original"
        for call in environment.calls[start:]
        if call["kind"] == "draft"
    )
    assert environment.model._target_route_mask is None


def test_original_acceptance_mismatch_invalidates_oracle(benchmark, monkeypatch):
    environment, states, _ = _capture(benchmark, monkeypatch)
    real_round = benchmark._run_counterfactual_round

    def mismatched_round(*args, **kwargs):
        result = real_round(*args, **kwargs)
        result["accepted_draft_tokens"] += 1
        result["reported_acceptance_length"] += 1
        return result

    monkeypatch.setattr(benchmark, "_run_counterfactual_round", mismatched_round)
    with pytest.raises(ValueError, match="Original replay mismatch"):
        environment.evaluate(states[0])
    assert environment.model._target_route_mask is None


def test_exact_prefix_mismatch_fails_before_measured_round(benchmark, monkeypatch):
    environment, states, _ = _capture(benchmark, monkeypatch)
    state = copy.deepcopy(states[1])
    state["canonical_prefix_token_ids"][-1] = 10
    count = len(environment.calls)
    with pytest.raises(ValueError, match="prefix|[Mm]ismatch"):
        environment.evaluate(state)
    drafts = [call for call in environment.calls[count:] if call["kind"] == "draft"]
    assert len(drafts) == state["round_id"]
    assert environment.model._target_route_mask is None


def test_counterfactual_failure_restores_original_route(benchmark, monkeypatch):
    environment, states, _ = _capture(benchmark, monkeypatch)
    real_round = benchmark._run_counterfactual_round

    def failing_round(*args, **kwargs):
        if environment.model.route == "deep":
            raise RuntimeError("probe failed")
        return real_round(*args, **kwargs)

    monkeypatch.setattr(benchmark, "_run_counterfactual_round", failing_round)
    with pytest.raises(RuntimeError, match="probe failed"):
        environment.evaluate(states[1], routes=["original", "deep"])
    assert environment.model._target_route_mask is None
    assert environment.model.route == "original"


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.5),
        ("canonical_route", "deep"),
        ("block_size", 1),
        ("num_input_tokens", 0),
        ("generated_tokens_before_round", 999),
        ("round_start", 999),
        ("canonical_reported_acceptance_length", 99),
        ("canonical_accepted_draft_tokens", 99),
        ("canonical_draft_token_ids", []),
        ("canonical_target_token_ids_for_proposals", []),
    ],
)
def test_canonical_state_validation_rejects_inconsistent_or_unsupported_states(
    benchmark, monkeypatch, field, value
):
    _, states, _ = _capture(benchmark, monkeypatch)
    state = copy.deepcopy(states[0])
    state[field] = value
    with pytest.raises(ValueError):
        benchmark._validate_canonical_state(state)


@pytest.mark.parametrize("routes", [["deep"], ["original", "unknown"], ["original", "original"]])
def test_evaluator_rejects_invalid_route_selection(benchmark, monkeypatch, routes):
    environment, states, _ = _capture(benchmark, monkeypatch)
    with pytest.raises(ValueError):
        environment.evaluate(states[0], routes=routes)


def test_replay_rejects_model_layer_bank_mismatch(benchmark, monkeypatch):
    environment, states, _ = _capture(benchmark, monkeypatch)
    state = copy.deepcopy(states[0])
    state["target_layer_ids"] = list(reversed(state["target_layer_ids"]))
    with pytest.raises(ValueError, match="target_layer_ids|bank"):
        environment.evaluate(state)


def _file_evaluation_fixture(benchmark, environment, states, tmp_path):
    source = tmp_path / "states.jsonl"
    metadata = {
        "output_kind": "canonical_states",
        "state_schema_version": 1,
        "model_name_or_path": "fake-target",
        "draft_name_or_path": "fake-draft",
        "draft_arch": "dflare",
        "temperature": 0.0,
        "target_route": "original",
        "attn_implementation": "eager",
        "block_size": 4,
        "dataset": "gsm8k",
        "target_layer_ids": environment.model.target_layer_ids,
    }
    with benchmark.VerificationTraceWriter(source, metadata=metadata) as writer:
        for state in states:
            writer.write(state)
    return SimpleNamespace(
        counterfactual_state_input=str(source),
        counterfactual_output=str(tmp_path / "oracle.jsonl"),
        model_name_or_path="fake-target",
        draft_name_or_path="fake-draft",
        block_size=None,
        dataset=None,
        counterfactual_state_start=0,
        counterfactual_max_states=None,
        counterfactual_routes=["original", "deep"],
    )


def test_file_evaluator_respects_start_limit_and_marks_complete(benchmark, monkeypatch, tmp_path):
    environment, states, _ = _capture(benchmark, monkeypatch)
    args = _file_evaluation_fixture(benchmark, environment, states, tmp_path)
    args.counterfactual_state_start = 1
    args.counterfactual_max_states = 2
    benchmark._evaluate_counterfactual_file(
        args,
        environment.model,
        environment.target,
        environment.sample,
        environment.extract,
        "eager",
    )
    path = Path(args.counterfactual_output)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["round_id"] for record in records] == [1, 2]
    assert all(record["original_replay_matches"] for record in records)
    metadata = json.loads(Path(str(path) + ".meta.json").read_text())
    assert metadata["evaluation_status"] == "complete"
    assert metadata["selected_states"] == metadata["completed_states"] == 2
    assert metadata["original_replay_mismatch_count"] == 0
    assert metadata["output_kind"] == "route_oracle"


def test_file_evaluator_marks_partial_output_invalid_after_replay_mismatch(
    benchmark, monkeypatch, tmp_path
):
    environment, states, _ = _capture(benchmark, monkeypatch)
    args = _file_evaluation_fixture(benchmark, environment, states[:2], tmp_path)
    real_round = benchmark._run_counterfactual_round

    def mismatched_second_state(*args, **kwargs):
        result = real_round(*args, **kwargs)
        if args[2].start == states[1]["round_start"] and environment.model.route == "original":
            result["accepted_draft_tokens"] = 0
        return result

    monkeypatch.setattr(benchmark, "_run_counterfactual_round", mismatched_second_state)
    with pytest.raises(ValueError, match="Original replay mismatch"):
        benchmark._evaluate_counterfactual_file(
            args,
            environment.model,
            environment.target,
            environment.sample,
            environment.extract,
            "eager",
        )
    path = Path(args.counterfactual_output)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    metadata = json.loads(Path(str(path) + ".meta.json").read_text())
    assert metadata["evaluation_status"] == "invalid"
    assert metadata["completed_states"] == 1
    assert "Original replay mismatch" in metadata["error"]
    assert environment.model._target_route_mask is None


@pytest.mark.parametrize(
    "flags,reason",
    [
        (["--state-output", "unused.jsonl", "--temperature", "0.5"], "temperature"),
        (["--state-output", "unused.jsonl", "--target-route", "deep"], "original"),
        (
            [
                "--counterfactual-state-input",
                "unused.jsonl",
                "--counterfactual-output",
                "out.jsonl",
                "--temperature",
                "0.5",
            ],
            "temperature",
        ),
    ],
)
def test_cli_rejects_unsupported_capture_and_replay_before_cuda(
    benchmark, monkeypatch, capsys, flags, reason
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid P3 mode reached CUDA/model loading")

    monkeypatch.setattr(benchmark, "_dist_init", forbidden)
    monkeypatch.setattr(benchmark, "_resolve_draft_arch", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dflash_benchmark.py",
            "--model-name-or-path",
            "unused-target",
            "--draft-name-or-path",
            "unused-draft",
            "--draft-arch",
            "dflare",
            "--dataset",
            "gsm8k",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert reason in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags,world_size,reason",
    [
        (
            [
                "--counterfactual-state-input",
                "states.meta.json",
                "--counterfactual-output",
                "states",
            ],
            1,
            "overwrite",
        ),
        (["--state-output", "x.meta.json", "--trace-output", "x"], 1, "separate"),
        (["--state-output", "foo", "--trace-output", "foo.jsonl"], 2, "separate"),
    ],
)
def test_cli_rejects_colliding_rank_or_metadata_artifacts_before_loading(
    benchmark, monkeypatch, capsys, flags, world_size, reason
):
    def forbidden(*args, **kwargs):
        pytest.fail("Conflicting artifact paths reached initialization")

    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(benchmark, "_dist_init", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dflash_benchmark.py",
            "--model-name-or-path",
            "unused-target",
            "--draft-name-or-path",
            "unused-draft",
            "--draft-arch",
            "dflare",
            "--dataset",
            "gsm8k",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert reason in capsys.readouterr().err


def test_main_captures_each_turn_only_on_speculative_path_with_trace_compatible(
    benchmark, monkeypatch, tmp_path
):
    environment = ToyEnvironment(benchmark, monkeypatch)
    environment.model.to = lambda device: environment.model
    environment.model.eval = lambda: environment.model
    environment.target.to = lambda device: environment.target
    environment.target.eval = lambda: environment.target
    loaded_draft = SimpleNamespace(from_pretrained=lambda *args, **kwargs: environment.model)
    tokenizer = SimpleNamespace(
        eos_token_id=10,
        apply_chat_template=lambda messages, **kwargs: messages[-1]["content"],
        encode=lambda *args, **kwargs: torch.tensor([[1, 2, 3]]),
        decode=lambda *args, **kwargs: "answer",
    )
    monkeypatch.setattr(
        benchmark,
        "_resolve_draft_arch",
        lambda arch: (loaded_draft, environment.sample, environment.extract),
    )
    monkeypatch.setattr(
        benchmark,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: environment.target),
    )
    monkeypatch.setattr(
        benchmark,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
    )
    monkeypatch.setattr(
        benchmark, "load_and_process_dataset", lambda name: [{"turns": ["first", "second"]}]
    )
    monkeypatch.setattr(benchmark, "_dist_init", lambda: None)
    monkeypatch.setattr(benchmark, "_dist_rank", lambda: 0)
    monkeypatch.setattr(benchmark, "_dist_size", lambda: 1)
    monkeypatch.setattr(benchmark, "_dist_is_main", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: None)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    state_path, trace_path = tmp_path / "states.jsonl", tmp_path / "trace.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dflash_benchmark.py",
            "--model-name-or-path",
            "fake-target",
            "--draft-name-or-path",
            "fake-draft",
            "--draft-arch",
            "dflare",
            "--dataset",
            "mt-bench",
            "--max-new-tokens",
            "6",
            "--state-output",
            str(state_path),
            "--trace-output",
            str(trace_path),
        ],
    )
    benchmark.main()
    states = [json.loads(line) for line in state_path.read_text().splitlines()]
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(states) == len(traces) == 4
    assert [(state["sample_id"], state["turn_id"], state["round_id"]) for state in states] == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    ]
    for state, trace in zip(states, traces):
        benchmark._validate_canonical_state(state)
        assert state["block_size"] == trace["block_size"] == 4
        assert state["canonical_accepted_draft_tokens"] == trace["accepted_draft_tokens"]
        assert state["canonical_draft_token_ids"] == trace["draft_token_ids"]
    metadata = json.loads(Path(str(state_path) + ".meta.json").read_text())
    assert metadata["output_kind"] == "canonical_states"
    assert metadata["target_route"] == "original"
