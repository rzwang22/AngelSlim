"""DFlash / DFlare end-to-end speculative decoding benchmark.

A self-contained evaluation entry point for AngelSlim's draft model classes.
Selects the draft architecture via ``--draft-arch``:

    --draft-arch dflash  -> angelslim.compressor.speculative.train.models.draft
                            .qwen_dflash.QwenDFlashDraftModel
    --draft-arch dflare  -> angelslim.compressor.speculative.train.models.draft
                            .qwen_dflare.QwenDFlareDraftModel

Reports decoding speedup vs single-token decoding and per-block acceptance
length distribution. Supports torchrun for multi-GPU sharded evaluation.

Usage (single GPU)::

    python tools/dflash_benchmark.py \\
        --model-name-or-path /path/to/Qwen3-4B \\
        --draft-name-or-path /path/to/dflash_or_dflare_ckpt \\
        --draft-arch dflare \\
        --dataset gsm8k --max-samples 128

Usage (8 GPUs)::

    torchrun --nproc_per_node=8 --master_port=29600 \\
        tools/dflash_benchmark.py \\
        --model-name-or-path /path/to/Qwen3-4B \\
        --draft-name-or-path /path/to/dflare_ckpt \\
        --draft-arch dflare \\
        --dataset gsm8k --max-samples 128
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from contextlib import ExitStack
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import torch
from datasets import Features, Sequence, Value, load_dataset
from loguru import logger
from rich import print
from torch import distributed as torch_dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


# ---------------------------------------------------------------------------
# Distributed helpers (small wrapper over torch.distributed; no extra package
# dependency on AngelSlim's side).
# ---------------------------------------------------------------------------
def _dist_init() -> None:
    if "RANK" not in os.environ:
        warnings.warn(
            "Environment variable `RANK` is not set; running single-process.",
            stacklevel=2,
        )
        return
    torch_dist.init_process_group(backend="nccl", init_method="env://")


def _dist_is_initialized() -> bool:
    return torch_dist.is_initialized()


def _dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _dist_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_gather(obj: Any, dst: int = 0) -> Optional[List[Any]]:
    if not _dist_is_initialized():
        return [obj]
    if _dist_is_main():
        objs: List[Any] = [None for _ in range(_dist_size())]
        torch_dist.gather_object(obj, objs, dst=dst)
        return objs
    torch_dist.gather_object(obj, dst=dst)
    return None


# These are source layer IDs, not positions or percentiles in a checkpoint bank.
# Other checkpoints must contain the requested IDs, or use a custom route.
TARGET_ROUTES = {
    "shallow": [1, 5, 9],
    "middle": [13, 17, 21],
    "deep": [25, 29, 33],
    "spread": [1, 17, 33],
    "mid_deep": [17, 25, 33],
}


def _parse_target_route_layers(value):
    try:
        layers = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Target route must be a nonempty comma-separated list of integer layer IDs."
        ) from exc
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("Target route must not contain duplicate layer IDs.")
    return layers


def _resolve_target_route(
    target_layer_ids, target_route="original", target_route_layers=None, draft_arch="dflare"
):
    """Validate against the actual checkpoint bank; return name and IDs in bank order."""
    if target_route_layers is not None:
        target_route = "custom"
    if target_route != "original" and draft_arch != "dflare":
        raise ValueError("target-layer routing is currently supported only for DFlare")
    bank = list(target_layer_ids)
    if target_route == "original":
        # Preserve existing banks, including repeated IDs from the model's fallback.
        return target_route, bank
    if not bank or any(type(layer_id) is not int or layer_id < 0 for layer_id in bank):
        raise ValueError("Checkpoint target_layer_ids must be nonempty nonnegative integer IDs.")
    if len(set(bank)) != len(bank):
        raise ValueError("Checkpoint target_layer_ids must not contain duplicate layer IDs.")
    if target_route == "custom":
        if not target_route_layers:
            raise ValueError("Custom target route requires nonempty --target-route-layers.")
        active_ids = list(target_route_layers)
    elif target_route in TARGET_ROUTES:
        active_ids = TARGET_ROUTES[target_route]
    else:
        raise ValueError(f"Unknown target route: {target_route}")
    if any(type(layer_id) is not int or layer_id < 0 for layer_id in active_ids):
        raise ValueError("Target route must contain nonnegative integer layer IDs.")
    if len(set(active_ids)) != len(active_ids):
        raise ValueError("Target route must not contain duplicate layer IDs.")
    for layer_id in active_ids:
        if layer_id not in bank:
            raise ValueError(
                f"{layer_id} is not in checkpoint target_layer_ids {bank}. "
                "Use --target-route-layers with IDs from this bank."
            )
    return target_route, [layer_id for layer_id in bank if layer_id in active_ids]


# ---------------------------------------------------------------------------
# Opt-in verification tracing. Only reduced statistics and token IDs leave the
# device; no model/cache tensors are retained by the writer.
# ---------------------------------------------------------------------------
class VerificationTraceWriter:
    """Stream one JSON object per round, flushing each line (without fsync)."""

    @staticmethod
    def path_for_rank(path, rank=0, world_size=1):
        path = Path(path)
        if world_size > 1:
            name = path.stem if path.suffix == ".jsonl" else path.name
            path = path.with_name(f"{name}.rank{rank}.jsonl")
        return path

    def __init__(self, path, *, rank=0, world_size=1, metadata=None):
        self.path = self.path_for_rank(path, rank, world_size)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if metadata is not None:
            metadata = dict(metadata, rank=rank, world_size=world_size)
            Path(str(self.path) + ".meta.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        self._file = self.path.open("w", encoding="utf-8", buffering=1)

    def write(self, record):
        self._file.write(json.dumps(record, allow_nan=False) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()


def _trace_artifact_paths(path, rank=0, world_size=1):
    path = VerificationTraceWriter.path_for_rank(path, rank, world_size)
    return {path.resolve(), Path(str(path) + ".meta.json").resolve()}


def _build_verification_trace(
    *,
    trace_context,
    round_id,
    num_input_tokens,
    round_start,
    block_output_ids,
    posterior,
    accepted_draft_tokens,
):
    """Describe the full verified block, before EOS/length output truncation."""
    block_tokens = block_output_ids[0].tolist()
    draft_tokens = block_tokens[1:]
    target_tokens = posterior[0, :-1].tolist()
    all_accepted = accepted_draft_tokens == len(draft_tokens)
    record = {
        "sample_id": trace_context["sample_id"],
        "turn_id": trace_context["turn_id"],
        "round_id": round_id,
        "num_input_tokens": num_input_tokens,
        "round_start": round_start,
        # Counts the committed prefix, excluding this round's sampled anchor.
        "generated_tokens_before_round": round_start - num_input_tokens,
        "block_size": len(block_tokens),
        "proposal_count": len(draft_tokens),
        "anchor_token_id": block_tokens[0],
        "draft_token_ids": draft_tokens,
        "target_token_ids_for_proposals": target_tokens,
        "match_mask": [draft == target for draft, target in zip(draft_tokens, target_tokens)],
        "accepted_draft_tokens": accepted_draft_tokens,
        "reported_acceptance_length": accepted_draft_tokens + 1,
        "first_reject_position": None if all_accepted else accepted_draft_tokens,
        "all_draft_tokens_accepted": all_accepted,
    }
    # Preserve compatibility with callers that only supply sample/turn identity.
    for key in ("target_route", "active_target_layer_ids"):
        if key in trace_context:
            record[key] = trace_context[key]
    return record


def _trace_probability_stats(draft_logits, target_logits, draft_token_ids):
    """Aligned proposal positions; raw model distributions, natural logs/nats.

    Both logits have shape [1, proposal_count, vocab_size]. Target probabilities
    are gathered for the *draft's* proposed tokens, including rejected proposals.
    Reductions stay on-device in float32; only per-position scalars reach CPU.
    """

    def reduce_logits(logits):
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        proposed_logprobs = logprobs.gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1)
        entropies = torch.special.entr(logprobs.exp()).sum(dim=-1)
        return proposed_logprobs, entropies

    draft_logprobs, draft_entropies = reduce_logits(draft_logits)
    target_logprobs, target_entropies = reduce_logits(target_logits)
    return {
        "draft_logprobs": draft_logprobs[0].tolist(),
        "target_logprobs": target_logprobs[0].tolist(),
        "logprob_gaps": (target_logprobs - draft_logprobs)[0].tolist(),
        "draft_entropies": draft_entropies[0].tolist(),
        "target_entropies": target_entropies[0].tolist(),
    }


# ---------------------------------------------------------------------------
# Dataset loader. Each loaded item must expose a ``turns`` field that is a
# list of user messages (one entry per turn for multi-turn datasets like
# mt-bench).
# ---------------------------------------------------------------------------
def load_and_process_dataset(data_name: str):
    if data_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        fmt = (
            "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        )
        return ds.map(lambda x: {"turns": [fmt.format(**x)]})

    if data_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        fmt = (
            "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        )
        return ds.map(lambda x: {"turns": [fmt.format(**x)]})

    if data_name == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        fmt = (
            "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        )
        return ds.map(lambda x: {"turns": [fmt.format(**x)]})

    if data_name == "aime25":
        ds = load_dataset("MathArena/aime_2025", split="train")
        fmt = (
            "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        )
        return ds.map(lambda x: {"turns": [fmt.format(**x)]})

    if data_name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        ds = ds.map(
            lambda x: {
                "formatted_input": (
                    f"{x['instruction']}\n\nInput:\n{x['input']}"
                    if x["input"]
                    else x["instruction"]
                )
            }
        )
        return ds.map(lambda x: {"turns": [x["formatted_input"]]})

    if data_name == "mt-bench":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        return ds.map(lambda x: {"turns": x["prompt"]})

    if data_name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        fmt = (
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            "```python\n{prompt}\n```"
        )
        return ds.map(lambda x: {"turns": [fmt.format(**x)]})

    if data_name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        return ds.map(lambda x: {"turns": [x["prompt"]]})

    if data_name == "lbpp":
        url = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        ds = load_dataset("parquet", data_files={"test": url})["test"]
        return ds.map(lambda x: {"turns": [x["instruction"]]})

    if data_name == "swe-bench":
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        return ds.map(lambda x: {"turns": [fmt.format(**x)]})

    if data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        files = [
            "test.jsonl",
            "test2.jsonl",
            "test3.jsonl",
            "test4.jsonl",
            "test5.jsonl",
            "test6.jsonl",
        ]
        ds = load_dataset("json", data_files={"test": [base + fn for fn in files]})["test"]

        def _fmt(doc):
            sys = (
                "You are an expert Python programmer. You will be given a question "
                "(problem specification) and will generate a correct Python program "
                "that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            q = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                fmt_msg = "### Format: Use the following code structure:"
                code = f"```python\n{doc['starter_code']}\n```"
            else:
                fmt_msg = "### Format: Write your code in the following format:"
                code = "```python\n# YOUR CODE HERE\n```"
            tail = "### Answer: (use the provided format with backticks)"
            return f"{sys}\n\n{q}\n\n{fmt_msg}\n{code}\n\n{tail}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        return ds.map(
            lambda x: {"turns": [_fmt(x)]},
            remove_columns=ds.column_names,
            features=target_features,
        )

    raise ValueError(f"Unknown dataset: {data_name}")


# ---------------------------------------------------------------------------
# Draft architecture dispatch.
# ---------------------------------------------------------------------------
def _resolve_draft_arch(arch: str):
    """Return (DraftModelClass, sample_fn, extract_context_feature_fn)."""
    arch = arch.lower()
    if arch == "dflash":
        from angelslim.compressor.speculative.train.models.draft.qwen_dflash import (
            QwenDFlashDraftModel,
            extract_context_feature,
            sample,
        )

        return QwenDFlashDraftModel, sample, extract_context_feature
    if arch == "dflare":
        from angelslim.compressor.speculative.train.models.draft.qwen_dflare import (
            QwenDFlareDraftModel,
            extract_context_feature,
            sample,
        )

        return QwenDFlareDraftModel, sample, extract_context_feature
    raise ValueError(f"--draft-arch must be one of {{dflash, dflare}}, got: {arch}")


# ---------------------------------------------------------------------------
# Speculative-decoding loop: block-parallel draft proposal, target
# verification, longest-prefix accept.
# ---------------------------------------------------------------------------
def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    model,
    target,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list,
    sample_fn,
    extract_context_feature_fn,
    temperature: float = 0.0,
    trace_writer=None,
    trace_context=None,
    trace_prob_stats: bool = False,
    state_writer=None,
    _stop_before_round=None,
) -> SimpleNamespace:
    """Generate as before; tracing requires a context with sample_id/turn_id."""
    if (trace_writer is not None or state_writer is not None) and block_size > 1:
        if trace_context is None or not {"sample_id", "turn_id"} <= trace_context.keys():
            raise ValueError("Speculative tracing requires sample_id and turn_id.")
    if state_writer is not None:
        if temperature != 0:
            raise ValueError("Canonical state capture requires temperature=0.")
        if getattr(model, "_target_route_mask", None) is not None or (
            trace_context is not None
            and trace_context.get("target_route", "original") != "original"
        ):
            raise ValueError("Canonical state capture requires target-route original.")
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_fn(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature_fn(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    while start < max_length:
        # Internal offline replay hook: return before *any* work on this round.
        # The caller owns these fresh caches; they are never shared across routes.
        if _stop_before_round is not None and len(acceptance_lengths) == _stop_before_round:
            return SimpleNamespace(
                _replay_state=True,
                output_ids=output_ids,
                position_ids=position_ids,
                start=start,
                num_input_tokens=num_input_tokens,
                past_key_values_target=past_key_values_target,
                past_key_values_draft=past_key_values_draft,
                target_hidden=target_hidden,
            )
        if state_writer is not None and block_size > 1:
            draft_context_start = past_key_values_draft.get_seq_length()
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample_fn(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample_fn(output.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        if trace_writer is not None and block_size > 1:
            record = _build_verification_trace(
                trace_context=trace_context,
                round_id=len(acceptance_lengths),
                num_input_tokens=num_input_tokens,
                round_start=start,
                block_output_ids=block_output_ids,
                posterior=posterior,
                accepted_draft_tokens=acceptance_length,
            )
            if trace_prob_stats:
                # Target logit k predicts block token k+1, the draft proposal k.
                record.update(
                    _trace_probability_stats(
                        draft_logits, output.logits[:, :-1, :], block_output_ids[:, 1:]
                    )
                )
            trace_writer.write(record)
        if state_writer is not None and block_size > 1:
            state_writer.write(
                {
                    "state_schema_version": 1,
                    "sample_id": trace_context["sample_id"],
                    "turn_id": trace_context["turn_id"],
                    "round_id": len(acceptance_lengths),
                    "canonical_route": "original",
                    # Include the sampled anchor; target KV itself ends before it.
                    "canonical_prefix_token_ids": output_ids[0, : start + 1].tolist(),
                    "num_input_tokens": num_input_tokens,
                    "round_start": start,
                    "generated_tokens_before_round": start - num_input_tokens,
                    "draft_context_start": draft_context_start,
                    "canonical_draft_token_ids": block_output_ids[0, 1:].tolist(),
                    "canonical_target_token_ids_for_proposals": posterior[0, :-1].tolist(),
                    "canonical_accepted_draft_tokens": acceptance_length,
                    "canonical_reported_acceptance_length": acceptance_length + 1,
                    "block_size": block_size,
                    "max_new_tokens": max_new_tokens,
                    "mask_token_id": mask_token_id,
                    "stop_token_ids": stop_token_ids,
                    "temperature": temperature,
                    "target_layer_ids": list(model.target_layer_ids),
                }
            )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature_fn(
                output.hidden_states, model.target_layer_ids
            )[:, : acceptance_length + 1, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_indices = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(
            as_tuple=True
        )[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
    )


# ---------------------------------------------------------------------------
# Offline same-state route evaluation. Reuse the canonical generation loop to
# rebuild original-conditioned history, including its exact forward chunking.
# ---------------------------------------------------------------------------
def _validate_canonical_state(state):
    required = {
        "state_schema_version",
        "sample_id",
        "turn_id",
        "round_id",
        "canonical_route",
        "canonical_prefix_token_ids",
        "num_input_tokens",
        "round_start",
        "generated_tokens_before_round",
        "draft_context_start",
        "canonical_draft_token_ids",
        "canonical_target_token_ids_for_proposals",
        "canonical_accepted_draft_tokens",
        "canonical_reported_acceptance_length",
        "block_size",
        "max_new_tokens",
        "mask_token_id",
        "stop_token_ids",
        "temperature",
        "target_layer_ids",
    }
    if not isinstance(state, dict) or required - state.keys():
        raise ValueError("Incomplete canonical state manifest record.")
    if state["state_schema_version"] != 1 or state["canonical_route"] != "original":
        raise ValueError("Replay requires schema version 1 and canonical_route original.")
    if state["temperature"] != 0:
        raise ValueError("Counterfactual evaluation currently requires temperature=0.")
    integer_fields = required - {
        "canonical_route",
        "canonical_prefix_token_ids",
        "canonical_draft_token_ids",
        "canonical_target_token_ids_for_proposals",
        "stop_token_ids",
        "temperature",
        "target_layer_ids",
    }
    for field in integer_fields:
        if type(state[field]) is not int or state[field] < 0:
            raise ValueError(f"Invalid nonnegative integer field: {field}")
    for field in (
        "canonical_prefix_token_ids",
        "canonical_draft_token_ids",
        "canonical_target_token_ids_for_proposals",
        "target_layer_ids",
        "stop_token_ids",
    ):
        values = state[field]
        if field == "stop_token_ids" and values is None:
            continue
        if not isinstance(values, list) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"Invalid ID list: {field}")
    start, prompt, block = state["round_start"], state["num_input_tokens"], state["block_size"]
    accepted = state["canonical_accepted_draft_tokens"]
    if (
        prompt < 1
        or block < 2
        or not state["target_layer_ids"]
        or not prompt <= start < prompt + state["max_new_tokens"]
        or len(state["canonical_prefix_token_ids"]) != start + 1
        or state["generated_tokens_before_round"] != start - prompt
        or not 0 <= accepted < block
        or state["canonical_reported_acceptance_length"] != accepted + 1
        or len(state["canonical_draft_token_ids"]) != block - 1
        or len(state["canonical_target_token_ids_for_proposals"]) != block - 1
        or not 0 <= state["draft_context_start"] < start
    ):
        raise ValueError("Inconsistent canonical state lengths or acceptance fields.")
    if state["round_id"] == 0 and (start != prompt or state["draft_context_start"] != 0):
        raise ValueError("Invalid first-round canonical boundary.")
    prefix_matches = 0
    for draft, target in zip(
        state["canonical_draft_token_ids"], state["canonical_target_token_ids_for_proposals"]
    ):
        if draft != target:
            break
        prefix_matches += 1
    if prefix_matches != accepted:
        raise ValueError("Canonical acceptance does not match its proposal prefix.")


@torch.inference_mode()
def _run_counterfactual_round(
    model, target, replay_state, sample_fn, extract_context_feature_fn, block_size
):
    """Measure one round using exclusively this replay's caches; commit no tokens."""
    start = replay_state.start
    block_ids = replay_state.output_ids[:, start : start + block_size].clone()
    draft_cache = replay_state.past_key_values_draft
    draft_logits = target.lm_head(
        model(
            target_hidden=replay_state.target_hidden,
            noise_embedding=target.model.embed_tokens(block_ids),
            position_ids=replay_state.position_ids[
                :, draft_cache.get_seq_length() : start + block_size
            ],
            past_key_values=draft_cache,
            use_cache=True,
            is_causal=False,
        )[:, -block_size + 1 :, :]
    )
    draft_cache.crop(start)
    block_ids[:, 1:] = sample_fn(draft_logits)
    output = target(
        block_ids,
        position_ids=replay_state.position_ids[:, start : start + block_size],
        past_key_values=replay_state.past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    posterior = sample_fn(output.logits, 0.0)
    accepted = (block_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
    return {
        "accepted_draft_tokens": accepted,
        "reported_acceptance_length": accepted + 1,
        "draft_token_ids": block_ids[0, 1:].tolist(),
        "target_token_ids_for_proposals": posterior[0, :-1].tolist(),
    }


def evaluate_counterfactual_state(
    state, model, target, sample_fn, extract_context_feature_fn, routes
):
    """Each route independently rebuilds the SAME original history before intervention."""
    _validate_canonical_state(state)
    if "original" not in routes or len(set(routes)) != len(routes):
        raise ValueError("Counterfactual routes must include original and contain no duplicates.")
    if state["target_layer_ids"] != list(model.target_layer_ids):
        raise ValueError("Canonical target_layer_ids differ from the loaded checkpoint bank.")
    resolved = {}
    for route in ["original", *(route for route in routes if route != "original")]:
        if route not in {"original", *TARGET_ROUTES}:
            raise ValueError(f"Unsupported counterfactual route: {route}")
        resolved[route] = _resolve_target_route(model.target_layer_ids, route)[1]
    identity = tuple(state[key] for key in ("sample_id", "turn_id", "round_id"))
    results = {}
    try:
        for route, active_ids in resolved.items():
            # Historical draft context must ALWAYS be built with original fusion.
            model.set_target_layer_route(None)
            prompt_ids = torch.tensor(
                [state["canonical_prefix_token_ids"][: state["num_input_tokens"]]],
                dtype=torch.long,
                device=model.device,
            )
            replay = dflash_generate(
                model=model,
                target=target,
                input_ids=prompt_ids,
                mask_token_id=state["mask_token_id"],
                max_new_tokens=state["max_new_tokens"],
                block_size=state["block_size"],
                stop_token_ids=state["stop_token_ids"],
                sample_fn=sample_fn,
                extract_context_feature_fn=extract_context_feature_fn,
                temperature=0.0,
                _stop_before_round=state["round_id"],
            )
            if (
                not getattr(replay, "_replay_state", False)
                or replay.start != state["round_start"]
                or replay.past_key_values_target.get_seq_length() != state["round_start"]
                or replay.past_key_values_draft.get_seq_length() != state["draft_context_start"]
                or replay.output_ids[0, : replay.start + 1].tolist()
                != state["canonical_prefix_token_ids"]
            ):
                raise ValueError(
                    f"Canonical prefix/cache reconstruction mismatch at state {identity}"
                )
            if route != "original":
                model.set_target_layer_route(active_ids)
            result = _run_counterfactual_round(
                model, target, replay, sample_fn, extract_context_feature_fn, state["block_size"]
            )
            if route == "original" and any(
                result[key] != state["canonical_" + key]
                for key in (
                    "draft_token_ids",
                    "target_token_ids_for_proposals",
                    "accepted_draft_tokens",
                    "reported_acceptance_length",
                )
            ):
                raise ValueError(
                    f"Original replay mismatch at state {identity}; oracle is invalid."
                )
            results[route] = result
            del replay  # Release these mutated caches before constructing the next route.
    finally:
        model.set_target_layer_route(None)
    return {
        **{key: state[key] for key in ("sample_id", "turn_id", "round_id")},
        "prefix_length": len(state["canonical_prefix_token_ids"]),
        "canonical_route": "original",
        "canonical_accepted_draft_tokens": state["canonical_accepted_draft_tokens"],
        "canonical_reported_acceptance_length": state["canonical_reported_acceptance_length"],
        "original_replay_matches": True,
        "route_results": results,
    }


def _parse_counterfactual_routes(value):
    routes = [route.strip() for route in value.split(",")]
    if (
        "original" not in routes
        or len(set(routes)) != len(routes)
        or any(route not in {"original", *TARGET_ROUTES} for route in routes)
    ):
        raise argparse.ArgumentTypeError(
            "Counterfactual routes must be distinct predefined routes including original."
        )
    return routes


def _evaluate_counterfactual_file(args, model, target, sample_fn, extract_fn, attn_impl):
    source = Path(args.counterfactual_state_input)
    metadata_path = Path(str(source) + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "output_kind": "canonical_states",
        "model_name_or_path": args.model_name_or_path,
        "draft_name_or_path": args.draft_name_or_path,
        "draft_arch": "dflare",
        "temperature": 0.0,
        "target_route": "original",
        "attn_implementation": attn_impl,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Canonical metadata mismatch for {key}: expected {value!r}.")
    if args.block_size is not None and args.block_size != metadata["block_size"]:
        raise ValueError("--block-size must match the canonical manifest.")
    if args.dataset is not None and args.dataset != metadata["dataset"]:
        raise ValueError("--dataset must match the canonical manifest.")
    # Count lines without loading prefixes into RAM. Blank lines are invalid records.
    with source.open(encoding="utf-8") as stream:
        total = sum(1 for _ in stream)
    count = total - args.counterfactual_state_start
    if args.counterfactual_max_states is not None:
        count = min(count, args.counterfactual_max_states)
    if count <= 0:
        raise ValueError("No canonical states selected for counterfactual evaluation.")
    output_metadata = {
        **metadata,
        "output_kind": "route_oracle",
        "evaluation_status": "running",
        "canonical_state_input": str(source),
        "counterfactual_routes": args.counterfactual_routes,
        "counterfactual_state_start": args.counterfactual_state_start,
        "selected_states": count,
        "completed_states": 0,
        "original_replay_mismatch_count": None,
    }
    seen = set()
    try:
        with VerificationTraceWriter(
            args.counterfactual_output, metadata=output_metadata
        ) as writer:
            with source.open(encoding="utf-8") as stream, tqdm(
                total=count, desc="Counterfactual states"
            ) as progress:
                for index, line in enumerate(stream):
                    if index < args.counterfactual_state_start:
                        continue
                    if progress.n >= count:
                        break
                    state = json.loads(line)
                    _validate_canonical_state(state)
                    identity = tuple(state[key] for key in ("sample_id", "turn_id", "round_id"))
                    if identity in seen:
                        raise ValueError(f"Duplicate canonical state: {identity}")
                    seen.add(identity)
                    progress.set_postfix(sample=identity[0], turn=identity[1], round=identity[2])
                    writer.write(
                        evaluate_counterfactual_state(
                            state, model, target, sample_fn, extract_fn, args.counterfactual_routes
                        )
                    )
                    progress.update(1)
                    output_metadata["completed_states"] += 1
        output_metadata.update(evaluation_status="complete", original_replay_mismatch_count=0)
    except Exception as exc:
        output_metadata.update(evaluation_status="invalid", error=str(exc))
        raise
    finally:
        Path(str(args.counterfactual_output) + ".meta.json").write_text(
            json.dumps(dict(output_metadata, rank=0, world_size=1), indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path", type=str, required=True, help="Path or HF id of the target model."
    )
    parser.add_argument(
        "--draft-name-or-path",
        type=str,
        required=True,
        help="Path of the trained DFlash/DFlare draft checkpoint.",
    )
    parser.add_argument(
        "--draft-arch",
        type=str,
        choices=["dflash", "dflare"],
        required=True,
        help="Which AngelSlim draft architecture to load.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Speculative block size. Defaults to draft model's config value.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (required except in offline counterfactual replay mode).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--trace-output",
        type=str,
        default=None,
        help=(
            "Stream speculative rounds to JSONL "
            "(disabled by default; per-rank files under torchrun)."
        ),
    )
    parser.add_argument(
        "--trace-prob-stats",
        action="store_true",
        help="Include raw-distribution logprobs/entropies; only effective with --trace-output.",
    )
    parser.add_argument(
        "--target-route",
        choices=["original", *TARGET_ROUTES, "custom"],
        default="original",
        help="Fixed DFlare source-layer route for the entire run (default: original, no mask).",
    )
    parser.add_argument(
        "--target-route-layers",
        type=_parse_target_route_layers,
        default=None,
        help="Comma-separated source layer IDs; overrides --target-route and selects custom.",
    )
    parser.add_argument(
        "--state-output", help="Stream canonical original round-start states to JSONL."
    )
    parser.add_argument(
        "--counterfactual-state-input", help="Replay saved canonical states instead of a dataset."
    )
    parser.add_argument("--counterfactual-output", help="Output per-state route results as JSONL.")
    parser.add_argument(
        "--counterfactual-routes",
        type=_parse_counterfactual_routes,
        default=",".join(["original", *TARGET_ROUTES]),
        help="Comma-separated predefined routes; original must be included (default: all six).",
    )
    parser.add_argument("--counterfactual-max-states", type=int, default=None)
    parser.add_argument("--counterfactual-state-start", type=int, default=0)
    args = parser.parse_args()
    if args.counterfactual_state_input is None and args.dataset is None:
        parser.error("--dataset is required for ordinary benchmark/state capture.")
    if args.state_output is not None or args.counterfactual_state_input is not None:
        if (
            args.draft_arch != "dflare"
            or args.target_route != "original"
            or (args.target_route_layers is not None)
        ):
            parser.error("P3 requires --draft-arch dflare and --target-route original.")
        if args.temperature != 0:
            parser.error("counterfactual evaluation currently requires temperature=0")
    if args.counterfactual_state_input is not None:
        if args.counterfactual_output is None:
            parser.error("--counterfactual-state-input requires --counterfactual-output.")
        if args.state_output is not None or args.trace_output is not None or args.trace_prob_stats:
            parser.error("Offline replay cannot be combined with state capture or P1 trace flags.")
        if _dist_size() != 1:
            parser.error("Offline counterfactual replay currently requires a single process.")
        if _trace_artifact_paths(args.counterfactual_output) & _trace_artifact_paths(
            args.counterfactual_state_input
        ):
            parser.error(
                "Counterfactual output must not overwrite the input manifest or metadata."
            )
    elif (
        args.counterfactual_output is not None
        or args.counterfactual_max_states is not None
        or (args.counterfactual_state_start != 0)
    ):
        parser.error("Counterfactual output/state limits require --counterfactual-state-input.")
    if args.counterfactual_state_start < 0 or (
        args.counterfactual_max_states is not None and args.counterfactual_max_states < 1
    ):
        parser.error(
            "Counterfactual state start must be nonnegative; max states must be positive."
        )
    if args.state_output is not None and args.trace_output is not None:
        if _trace_artifact_paths(args.state_output, _dist_rank(), _dist_size()) & (
            _trace_artifact_paths(args.trace_output, _dist_rank(), _dist_size())
        ):
            parser.error("State/trace outputs and their metadata must use separate files.")
    if args.target_route == "custom" and args.target_route_layers is None:
        parser.error("--target-route custom requires --target-route-layers.")
    if args.draft_arch != "dflare" and (
        args.target_route != "original" or args.target_route_layers is not None
    ):
        parser.error("target-layer routing is currently supported only for DFlare")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _dist_init()
    torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}")

    DraftModelCls, sample_fn, extract_context_feature_fn = _resolve_draft_arch(args.draft_arch)

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "flash_attn is not installed; falling back to torch.sdpa. "
                "End-to-end speedup will be lower."
            )
            return False

    installed_flash_attn = has_flash_attn()
    attn_impl = "flash_attention_2" if installed_flash_attn else "sdpa"

    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    draft_model = (
        DraftModelCls.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )

    try:
        target_route, active_target_layer_ids = _resolve_target_route(
            draft_model.target_layer_ids,
            args.target_route,
            args.target_route_layers,
            args.draft_arch,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if target_route != "original":
        draft_model.set_target_layer_route(active_target_layer_ids)
    if args.draft_arch == "dflare" and _dist_is_main():
        print(f"Target layer bank: {draft_model.target_layer_ids}")
        print(f"Target route: {target_route}")
        print(f"Active target layers: {active_target_layer_ids}")
    route_info = {
        "target_route": target_route,
        "active_target_layer_ids": active_target_layer_ids,
    }

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    if args.state_output is not None and block_size < 2:
        parser.error("Canonical state capture requires a speculative block size greater than 1.")
    if args.counterfactual_state_input is not None:
        _evaluate_counterfactual_file(
            args, draft_model, target, sample_fn, extract_context_feature_fn, attn_impl
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    responses = []
    indices = range(_dist_rank(), len(dataset), _dist_size())
    with ExitStack() as stack:
        trace_writer = None
        state_writer = None
        if args.trace_output is not None:
            trace_writer = stack.enter_context(
                VerificationTraceWriter(
                    args.trace_output,
                    rank=_dist_rank(),
                    world_size=_dist_size(),
                    metadata={
                        "trace_schema_version": 1,
                        "dataset": args.dataset,
                        "max_samples": args.max_samples,
                        "num_samples": len(dataset),
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "seed": 0,
                        "draft_arch": args.draft_arch,
                        "block_size": block_size,
                        "target_layer_ids": draft_model.target_layer_ids,
                        "model_name_or_path": args.model_name_or_path,
                        "draft_name_or_path": args.draft_name_or_path,
                        "trace_prob_stats": args.trace_prob_stats,
                        "probability_distribution": "raw_logits_softmax",
                        **route_info,
                    },
                )
            )
        if args.state_output is not None:
            state_writer = stack.enter_context(
                VerificationTraceWriter(
                    args.state_output,
                    rank=_dist_rank(),
                    world_size=_dist_size(),
                    metadata={
                        "output_kind": "canonical_states",
                        "state_schema_version": 1,
                        "dataset": args.dataset,
                        "max_samples": args.max_samples,
                        "num_samples": len(dataset),
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "seed": 0,
                        "draft_arch": args.draft_arch,
                        "block_size": block_size,
                        "target_layer_ids": draft_model.target_layer_ids,
                        "model_name_or_path": args.model_name_or_path,
                        "draft_name_or_path": args.draft_name_or_path,
                        "attn_implementation": attn_impl,
                        **route_info,
                    },
                )
            )
        for idx in tqdm(indices, disable=not _dist_is_main()):
            instance = dataset[idx]
            messages = []
            for turn_id, user_content in enumerate(instance["turns"]):
                messages.append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

                response = {}
                for bs in [1, block_size]:
                    response[bs] = dflash_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=bs,
                        stop_token_ids=[tokenizer.eos_token_id],
                        sample_fn=sample_fn,
                        extract_context_feature_fn=extract_context_feature_fn,
                        temperature=args.temperature,
                        trace_writer=trace_writer if bs > 1 else None,
                        trace_context=(
                            {"sample_id": idx, "turn_id": turn_id, **route_info}
                            if (trace_writer is not None or state_writer is not None) and bs > 1
                            else None
                        ),
                        trace_prob_stats=args.trace_prob_stats,
                        **(
                            {"state_writer": state_writer}
                            if state_writer is not None and bs > 1
                            else {}
                        ),
                    )

                spec_response = response[block_size]
                generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": output_text})
                responses.append(response)

    if _dist_size() > 1:
        gathered = _dist_gather(responses, dst=0)
        if not _dist_is_main():
            return
        responses = list(chain(*gathered))

    if not responses:
        return

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"[draft_arch={args.draft_arch}] Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"[draft_arch={args.draft_arch}] Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [
        acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)
    ]
    print(
        f"[draft_arch={args.draft_arch}] Acceptance length histogram: "
        f"{[f'{x * 100:.1f}%' for x in histogram]}"
    )


if __name__ == "__main__":
    main()
