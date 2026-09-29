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


# ---------------------------------------------------------------------------
# Opt-in verification tracing. Only reduced statistics and token IDs leave the
# device; no model/cache tensors are retained by the writer.
# ---------------------------------------------------------------------------
class VerificationTraceWriter:
    """Stream one JSON object per round, flushing each line (without fsync)."""

    def __init__(self, path, *, rank=0, world_size=1, metadata=None):
        self.path = Path(path)
        if world_size > 1:
            name = self.path.stem if self.path.suffix == ".jsonl" else self.path.name
            self.path = self.path.with_name(f"{name}.rank{rank}.jsonl")
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
    return {
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
) -> SimpleNamespace:
    """Generate as before; tracing requires a context with sample_id/turn_id."""
    if trace_writer is not None and block_size > 1:
        if trace_context is None or not {"sample_id", "turn_id"} <= trace_context.keys():
            raise ValueError("Speculative tracing requires sample_id and turn_id.")
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
        required=True,
        help="Dataset name; see load_and_process_dataset() for the supported list.",
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
    args = parser.parse_args()

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

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    responses = []
    indices = range(_dist_rank(), len(dataset), _dist_size())
    with ExitStack() as stack:
        trace_writer = None
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
                            {"sample_id": idx, "turn_id": turn_id}
                            if trace_writer is not None and bs > 1
                            else None
                        ),
                        trace_prob_stats=args.trace_prob_stats,
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
