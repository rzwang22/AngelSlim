"""CPU checks for opt-in tracing; no model downloads or CUDA runtime needed."""

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1] / "tools" / "dflash_benchmark.py"
    spec = importlib.util.spec_from_file_location("dflash_benchmark_trace_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("verified", "accepted", "matches", "reject"),
    [
        ([9, 8, 7, 99], 0, [False, False, False], 0),
        ([1, 2, 7, 99], 2, [True, True, False], 2),
        ([1, 2, 3, 99], 3, [True, True, True], None),
        ([9, 2, 3, 99], 0, [False, True, True], 0),
    ],
)
def test_trace_acceptance_prefix_and_anchor(benchmark, verified, accepted, matches, reject):
    block = torch.tensor([[42, 1, 2, 3]])
    posterior = torch.tensor([verified])
    before = (block.clone(), posterior.clone())
    record = benchmark._build_verification_trace(
        trace_context={"sample_id": 7, "turn_id": 2},
        round_id=3,
        num_input_tokens=11,
        round_start=18,
        block_output_ids=block,
        posterior=posterior,
        accepted_draft_tokens=accepted,
    )
    assert record == {
        "sample_id": 7,
        "turn_id": 2,
        "round_id": 3,
        "num_input_tokens": 11,
        "round_start": 18,
        "generated_tokens_before_round": 7,
        "block_size": 4,
        "proposal_count": 3,
        "anchor_token_id": 42,
        "draft_token_ids": [1, 2, 3],
        "target_token_ids_for_proposals": verified[:-1],
        "match_mask": matches,
        "accepted_draft_tokens": accepted,
        "reported_acceptance_length": accepted + 1,
        "first_reject_position": reject,
        "all_draft_tokens_accepted": accepted == 3,
    }
    assert torch.equal(block, before[0])
    assert torch.equal(posterior, before[1])


def _distribution_stats(logits, tokens):
    """Independent scalar reference for raw distributions (natural logarithms)."""
    logprobs, entropies = [], []
    for row, token in zip(logits[0].tolist(), tokens[0].tolist()):
        log_z = math.log(sum(math.exp(value) for value in row))
        row_logprobs = [value - log_z for value in row]
        logprobs.append(row_logprobs[token])
        entropies.append(-sum(math.exp(value) * value for value in row_logprobs))
    return logprobs, entropies


def test_probability_stats_gather_proposed_tokens_in_each_position(benchmark):
    draft = torch.tensor([[[1, 2, 4], [3, -1, 1], [-2, 4, 0]]], dtype=torch.float16)
    target = torch.tensor([[[4, 1, -2], [-1, 5, 2], [3, 0, 6]]], dtype=torch.float16)
    # None is the target argmax: gathering target's own selected tokens is incorrect.
    tokens = torch.tensor([[2, 0, 1]])
    originals = [value.clone() for value in (draft, target, tokens)]
    result = benchmark._trace_probability_stats(draft, target, tokens)
    draft_lp, draft_entropy = _distribution_stats(draft, tokens)
    target_lp, target_entropy = _distribution_stats(target, tokens)
    expected = {
        "draft_logprobs": draft_lp,
        "target_logprobs": target_lp,
        "logprob_gaps": [p - q for p, q in zip(target_lp, draft_lp)],
        "draft_entropies": draft_entropy,
        "target_entropies": target_entropy,
    }
    assert result.keys() == expected.keys()
    for key in expected:
        assert result[key] == pytest.approx(expected[key], abs=1e-6)
    for original, current in zip(originals, (draft, target, tokens)):
        assert torch.equal(original, current)
        assert original.dtype == current.dtype


def test_probability_stats_entropy_remains_finite_after_underflow(benchmark):
    logits = torch.tensor([[[0.0, -10000.0]]], dtype=torch.bfloat16)
    result = benchmark._trace_probability_stats(logits, logits, torch.tensor([[0]]))
    assert result["draft_entropies"] == [0.0]
    assert result["target_entropies"] == [0.0]
    json.dumps(result, allow_nan=False)


def test_writer_streams_strict_json_and_metadata(benchmark, tmp_path):
    path = tmp_path / "nested" / "trace.jsonl"
    metadata = {"dataset": "gsm8k", "trace_prob_stats": False}
    with benchmark.VerificationTraceWriter(path, metadata=metadata) as writer:
        assert writer.path == path
        writer.write({"round_id": 0, "first_reject_position": None})
        # Reading while open checks that each complete round is immediately available.
        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            {"round_id": 0, "first_reject_position": None}
        ]
        with pytest.raises(ValueError):
            writer.write({"bad": float("nan")})
        writer.write({"round_id": 1, "match_mask": [True, False]})
    assert len(path.read_text().splitlines()) == 2
    saved = json.loads(Path(str(path) + ".meta.json").read_text())
    assert saved == dict(metadata, rank=0, world_size=1)
    assert metadata == {"dataset": "gsm8k", "trace_prob_stats": False}


def test_writer_uses_independent_rank_files(benchmark, tmp_path):
    path = tmp_path / "trace.jsonl"
    for rank in range(2):
        with benchmark.VerificationTraceWriter(
            path, rank=rank, world_size=2, metadata={"dataset": "gsm8k"}
        ) as writer:
            assert writer.path == tmp_path / f"trace.rank{rank}.jsonl"
            writer.write({"rank": rank})
    assert not path.exists()
    for rank in range(2):
        rank_path = tmp_path / f"trace.rank{rank}.jsonl"
        assert json.loads(rank_path.read_text()) == {"rank": rank}
        metadata = json.loads(Path(str(rank_path) + ".meta.json").read_text())
        assert (metadata["rank"], metadata["world_size"]) == (rank, 2)


class _MemoryWriter:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(json.loads(json.dumps(record, allow_nan=False)))


def _run_cpu_generation(
    benchmark,
    monkeypatch,
    *,
    writer=None,
    prob_stats=False,
    temperature=0.0,
    block_size=4,
    max_new_tokens=9,
    stop_token_ids=None,
):
    """Exercise the real loop with deterministic CPU models and inspect its state."""
    events, caches, sampled_logits = [], [], []
    vocabulary_size = 7

    class Cache:
        def __init__(self):
            self.index = len(caches)
            self.length = 0
            caches.append(self)

        def get_seq_length(self):
            return self.length

        def crop(self, length):
            events.append(("crop", self.index, length))
            self.length = min(self.length, length)

    def logits_for(preferred):
        vocabulary = torch.arange(vocabulary_size, dtype=torch.float32)
        return -(vocabulary - preferred.unsqueeze(-1)).square() / 3

    def target_call(
        input_ids,
        *,
        position_ids,
        past_key_values,
        use_cache,
        output_hidden_states,
        logits_to_keep=None,
    ):
        events.append(
            (
                "target",
                input_ids.tolist(),
                position_ids.tolist(),
                past_key_values.length,
                use_cache,
                output_hidden_states,
                logits_to_keep,
            )
        )
        past_key_values.length += input_ids.shape[1]
        logits = logits_for((position_ids + 1) % vocabulary_size)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        features = torch.stack((input_ids.float(), position_ids.float()), dim=-1)
        return SimpleNamespace(logits=logits, hidden_states=(features, features + 1))

    class Target:
        model = SimpleNamespace(
            embed_tokens=lambda ids: torch.nn.functional.one_hot(ids, vocabulary_size + 1).float()
        )
        lm_head = staticmethod(lambda hidden: hidden)
        __call__ = staticmethod(target_call)

    class Draft:
        device = torch.device("cpu")
        target_layer_ids = [0, 1]

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
            events.append(
                (
                    "draft",
                    target_hidden.tolist(),
                    noise_embedding.tolist(),
                    position_ids.tolist(),
                    past_key_values.length,
                    use_cache,
                    is_causal,
                )
            )
            past_key_values.length = int(position_ids[0, -1]) + 1
            positions = position_ids[:, -noise_embedding.shape[1] :]
            # Later matches after this periodic mismatch must not extend acceptance.
            preferred = (positions + (positions % 5 == 0)) % vocabulary_size
            return logits_for(preferred)

    def extract(hidden_states, layer_ids):
        features = torch.cat([hidden_states[i] for i in layer_ids], dim=-1)
        events.append(("extract", features.tolist(), list(layer_ids)))
        return features

    def sample(logits, temperature=0.0):
        sampled_logits.append(logits.clone())
        events.append(("sample", logits.tolist(), temperature))
        if temperature < 1e-5:
            return logits.argmax(dim=-1)
        probabilities = torch.softmax(logits.view(-1, vocabulary_size) / temperature, dim=-1)
        return torch.multinomial(probabilities, 1).view(logits.shape[:-1])

    ticks = iter(range(100))
    monkeypatch.setattr(benchmark, "DynamicCache", Cache)
    monkeypatch.setattr(benchmark, "cuda_time", lambda: next(ticks))
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    output = benchmark.dflash_generate(
        model=Draft(),
        target=Target(),
        input_ids=torch.tensor([[0, 1, 2]]),
        mask_token_id=vocabulary_size,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        stop_token_ids=stop_token_ids,
        sample_fn=sample,
        extract_context_feature_fn=extract,
        temperature=temperature,
        trace_writer=writer,
        trace_context={"sample_id": 8, "turn_id": 2},
        trace_prob_stats=prob_stats,
    )
    state = (
        events,
        [cache.length for cache in caches],
        torch.get_rng_state().tolist(),
        random.getstate(),
        np.random.get_state()[1].tolist(),
    )
    return output, state, sampled_logits


@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("prob_stats", [False, True])
def test_tracing_preserves_generation_cache_hidden_sampling_and_rng(
    benchmark, monkeypatch, temperature, prob_stats
):
    baseline, baseline_state, _ = _run_cpu_generation(
        benchmark, monkeypatch, temperature=temperature
    )
    writer = _MemoryWriter()
    traced, traced_state, sampled = _run_cpu_generation(
        benchmark,
        monkeypatch,
        writer=writer,
        prob_stats=prob_stats,
        temperature=temperature,
    )
    assert torch.equal(baseline.output_ids, traced.output_ids)
    for field in (
        "num_input_tokens",
        "num_output_tokens",
        "acceptance_lengths",
        "time_to_first_token",
        "time_per_output_token",
    ):
        assert getattr(baseline, field) == getattr(traced, field)
    assert baseline_state == traced_state
    assert len(writer.records) == len(traced.acceptance_lengths)
    assert [record["round_id"] for record in writer.records] == list(range(len(writer.records)))
    for index, record in enumerate(writer.records):
        assert (record["sample_id"], record["turn_id"]) == (8, 2)
        assert record["reported_acceptance_length"] == traced.acceptance_lengths[index]
        assert record["round_start"] == 3 + sum(traced.acceptance_lengths[:index])
        if prob_stats:
            # sample calls are prefill, then draft/target pairs. Verify the actual
            # integration slices target logits at :-1 and gathers the draft tokens.
            proposal_tokens = torch.tensor([record["draft_token_ids"]])
            draft_lp, draft_entropy = _distribution_stats(sampled[1 + 2 * index], proposal_tokens)
            target_lp, target_entropy = _distribution_stats(
                sampled[2 + 2 * index][:, :-1], proposal_tokens
            )
            for key, expected in (
                ("draft_logprobs", draft_lp),
                ("target_logprobs", target_lp),
                ("draft_entropies", draft_entropy),
                ("target_entropies", target_entropy),
            ):
                assert record[key] == pytest.approx(expected, abs=1e-6)
        else:
            assert "draft_logprobs" not in record
            assert "target_entropies" not in record


@pytest.mark.parametrize("block_size,writer_enabled", [(4, False), (1, True)])
def test_disabled_trace_and_ar_never_build_records_or_probabilities(
    benchmark, monkeypatch, block_size, writer_enabled
):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled tracing or the AR baseline invoked trace work")

    monkeypatch.setattr(benchmark, "_build_verification_trace", forbidden)
    monkeypatch.setattr(benchmark, "_trace_probability_stats", forbidden)
    writer = _MemoryWriter() if writer_enabled else None
    _run_cpu_generation(
        benchmark, monkeypatch, writer=writer, prob_stats=True, block_size=block_size
    )
    if writer is not None:
        assert writer.records == []


def test_basic_trace_never_computes_probability_statistics(benchmark, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Basic tracing computed probability statistics")

    monkeypatch.setattr(benchmark, "_trace_probability_stats", forbidden)
    writer = _MemoryWriter()
    _run_cpu_generation(benchmark, monkeypatch, writer=writer)
    assert writer.records


def test_round_id_restarts_for_each_generation_with_shared_writer(benchmark, monkeypatch):
    writer = _MemoryWriter()
    _run_cpu_generation(benchmark, monkeypatch, writer=writer)
    first_call_rounds = len(writer.records)
    _run_cpu_generation(benchmark, monkeypatch, writer=writer)
    expected = list(range(first_call_rounds))
    assert [record["round_id"] for record in writer.records] == expected + expected


@pytest.mark.parametrize("max_new_tokens,stop_ids", [(2, None), (9, [6])])
def test_final_trace_describes_full_verification_before_budget_or_eos_truncation(
    benchmark, monkeypatch, max_new_tokens, stop_ids
):
    writer = _MemoryWriter()
    output, _, _ = _run_cpu_generation(
        benchmark,
        monkeypatch,
        writer=writer,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_ids,
    )
    assert all(len(record["draft_token_ids"]) == 3 for record in writer.records)
    if stop_ids:
        assert output.output_ids[0, -1].item() == 6
        assert len(writer.records) == 2
        assert writer.records[-1]["all_draft_tokens_accepted"]
        assert writer.records[-1]["reported_acceptance_length"] == 4
    else:
        assert output.num_output_tokens == 2
        assert len(writer.records) == 1


@pytest.mark.parametrize(
    "trace_enabled,prob_stats,route,custom_layers",
    [
        (False, False, None, None),
        (False, True, None, None),
        (True, True, None, None),
        (True, True, "deep", None),
        (True, True, "shallow", "5,21,33"),
        (True, True, None, "5,21,33"),
        (False, False, "spread", None),
    ],
)
def test_main_plumbs_selected_sample_turn_rank_and_metadata(
    benchmark, monkeypatch, tmp_path, trace_enabled, prob_stats, route, custom_layers
):
    dataset_events, generation_calls, conversations = [], [], []
    route_calls = []
    bank = [1, 5, 9, 13, 17, 21, 25, 29, 33]
    effective_route = "custom" if custom_layers else route or "original"
    active_ids = {
        "original": bank,
        "deep": [25, 29, 33],
        "spread": [1, 17, 33],
        "custom": [5, 21, 33],
    }[effective_route]
    route_context = {"target_route": effective_route, "active_target_layer_ids": active_ids}

    class Dataset:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

        def shuffle(self, seed):
            dataset_events.append(("shuffle", seed))
            return Dataset(self.rows[::-1])

        def select(self, indices):
            dataset_events.append(("select", list(indices)))
            return Dataset([self.rows[index] for index in indices])

    class LoadedModel:
        device = torch.device("cpu")
        block_size = 4
        mask_token_id = 0
        target_layer_ids = bank

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def set_target_layer_route(self, active_layer_ids):
            route_calls.append(list(active_layer_ids))

    class Tokenizer:
        eos_token_id = 6

        def apply_chat_template(self, messages, **kwargs):
            conversations.append([dict(message) for message in messages])
            return messages[-1]["content"]

        def encode(self, text, **kwargs):
            return torch.tensor([[1, 2]])

        def decode(self, ids, **kwargs):
            return "answer"

    def generate(**kwargs):
        generation_calls.append(kwargs)
        writer = kwargs["trace_writer"]
        if writer is not None:
            writer.write(dict(kwargs["trace_context"], round_id=0))
        return SimpleNamespace(
            output_ids=torch.tensor([[1, 2, 3]]),
            num_input_tokens=2,
            time_per_output_token=0.1,
            acceptance_lengths=[2],
        )

    rows = [{"turns": [f"sample{i}-turn0", f"sample{i}-turn1"]} for i in range(5)]
    monkeypatch.setattr(benchmark, "load_and_process_dataset", lambda name: Dataset(rows))
    monkeypatch.setattr(benchmark, "AutoModelForCausalLM", LoadedModel)
    monkeypatch.setattr(
        benchmark,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()),
    )
    monkeypatch.setattr(benchmark, "_resolve_draft_arch", lambda arch: (LoadedModel, None, None))
    monkeypatch.setattr(benchmark, "dflash_generate", generate)
    monkeypatch.setattr(benchmark, "_dist_init", lambda: None)
    monkeypatch.setattr(benchmark, "_dist_rank", lambda: 1)
    monkeypatch.setattr(benchmark, "_dist_size", lambda: 2)
    monkeypatch.setattr(benchmark, "_dist_is_main", lambda: False)
    monkeypatch.setattr(benchmark, "_dist_gather", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: None)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    trace_path = tmp_path / "trace.jsonl"
    argv = [
        "dflash_benchmark.py",
        "--model-name-or-path",
        "fake-target",
        "--draft-name-or-path",
        "fake-draft",
        "--draft-arch",
        "dflare",
        "--dataset",
        "mt-bench",
        "--max-samples",
        "3",
        "--max-new-tokens",
        "10",
    ]
    if trace_enabled:
        argv += ["--trace-output", str(trace_path)]
    if prob_stats:
        argv += ["--trace-prob-stats"]
    if route:
        argv += ["--target-route", route]
    if custom_layers:
        argv += ["--target-route-layers", custom_layers]
    monkeypatch.setattr(sys, "argv", argv)
    benchmark.main()

    assert dataset_events == [("shuffle", 0), ("select", [0, 1, 2])]
    assert route_calls == ([] if effective_route == "original" else [active_ids])
    assert [call["block_size"] for call in generation_calls] == [1, 4, 1, 4]
    assert conversations[0] == [{"role": "user", "content": "sample3-turn0"}]
    assert conversations[1][-2:] == [
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "sample3-turn1"},
    ]
    for index, call in enumerate(generation_calls):
        assert call["trace_prob_stats"] is prob_stats
        if trace_enabled and call["block_size"] > 1:
            assert call["trace_context"] == dict(sample_id=1, turn_id=index // 2, **route_context)
            assert call["trace_writer"] is not None
        else:
            assert call["trace_context"] is None
            assert call["trace_writer"] is None
    if trace_enabled:
        rank_path = tmp_path / "trace.rank1.jsonl"
        assert [json.loads(line) for line in rank_path.read_text().splitlines()] == [
            dict(sample_id=1, turn_id=0, round_id=0, **route_context),
            dict(sample_id=1, turn_id=1, round_id=0, **route_context),
        ]
        metadata = json.loads(Path(str(rank_path) + ".meta.json").read_text())
        for key, value in {
            "dataset": "mt-bench",
            "num_samples": 3,
            "max_samples": 3,
            "max_new_tokens": 10,
            "temperature": 0.0,
            "draft_arch": "dflare",
            "block_size": 4,
            "target_layer_ids": bank,
            **route_context,
            "model_name_or_path": "fake-target",
            "draft_name_or_path": "fake-draft",
            "trace_prob_stats": True,
            "rank": 1,
            "world_size": 2,
        }.items():
            assert metadata[key] == value
    else:
        assert list(tmp_path.iterdir()) == []
