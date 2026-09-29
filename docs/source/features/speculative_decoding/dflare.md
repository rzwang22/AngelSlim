# DFlare

**DFlare** is a block-diffusion speculative decoding framework that accelerates large language model inference by predicting an entire block of tokens in one shot for the target model to verify in parallel. It removes the narrow conditioning bottleneck of the prior state-of-the-art DFlash through a lightweight **layer-wise fusion** mechanism: each draft layer attends to its own learnable combination of a broad set of target layers at negligible overhead, simultaneously injecting richer target knowledge and giving every draft layer a distinct input. Combined with training-data scaling, this enhanced per-layer expressiveness allows the draft model to scale to deeper architectures with consistent gains, achieving up to **5.52× end-to-end speedup** without compromising output quality.

This repository contains the official implementation and resources for the paper: **DFLARE: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding**.


:::{image} /assets/dflare/intro.png
:alt: An overview of the DFlare framework.
:::

---

## 🚀 Abstract

Block diffusion speculative decoding accelerates LLM inference by predicting all tokens within a block simultaneously for the target model to verify in parallel. Predicting an entire block at once requires a sufficiently capable draft model and effective utilization of the target model's internal knowledge. However, the state-of-the-art method DFlash constrains all draft layers to share a single fused representation derived from only a few target layers, limiting per-layer expressiveness and hindering further scaling of draft capacity. We present **DFLARE**, which flares out the narrow conditioning bottleneck of DFlash through a lightweight layer-wise fusion mechanism: each draft layer attends to its own learnable combination of a broad set of target layers at negligible overhead, simultaneously injecting richer target knowledge and providing every draft layer with a distinct input. This enhanced per-layer expressiveness enables scaling the draft model to deeper architectures with consistent gains. We further scale training data from 800K to 2.4M samples to fully exploit the enlarged capacity. On six benchmarks spanning mathematical reasoning, code generation, and conversation, DFLARE attains average wall-clock speedups of **5.52× on Qwen3-4B**, **5.46× on Qwen3-8B**, and **3.91× on GPT-OSS-20B**, improving over DFlash by roughly 11%, 8%, and 5% respectively.


## ✨ Key Highlights

- **Layer-wise Fusion for Richer Conditioning**: Replaces DFlash's single fused representation with a lightweight mechanism in which each draft layer attends to its own learnable combination of a broad set of target layers, removing the conditioning bottleneck at negligible overhead.
- **Scalable Draft Capacity**: The enriched per-layer expressiveness lets the draft model scale to deeper architectures with consistent gains, complemented by scaling training data from 800K to 2.4M samples to fully exploit the enlarged capacity.
- **Substantial End-to-End Speedups**: Across six benchmarks covering mathematical reasoning, code generation, and conversation, DFlare delivers average wall-clock speedups of 5.52× on Qwen3-4B, 5.46× on Qwen3-8B, and 3.91× on GPT-OSS-20B — roughly 11%, 8%, and 5% over DFlash respectively.


## ⚡ Quick Start

### Training

DFlare reuses the DFlash training pipeline and selects the layer-wise fusion architecture via `--draft_arch dflare`. Two entry points are provided:

**Online training** (recommended) — runs the target model on the fly to produce hidden states each step. No data pre-generation step needed.

```shell
export TARGET_MODEL_PATH=/path/to/Qwen3-4B
export TRAIN_DATA_PATH=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/output

bash scripts/speculative/run_dflare_online.sh 8 flex_attention
```

**Offline training** — trains from pre-computed hidden-state `.ckpt` files. First generate the cache with `scripts/speculative/generate_dflash_data.sh` using a DFlare-compatible draft config, then:

```shell
export TARGET_MODEL_PATH=/path/to/Qwen3-4B
export TRAIN_HIDDEN_PATH=/path/to/hidden_cache
export OUTPUT_DIR=/path/to/output

bash scripts/speculative/run_dflare_offline.sh 8 flex_attention
```

Both entries use the same defaults: `block_size=16`, `num_anchors=512`, `lr=6e-4`, cosine schedule with 4% warmup, `max_length=3072`, FSDP `shard_grad_op` with FP32 master-weights optimizer, and `flash_attention_2` for the target model. The default draft model config is `configs/qwen3_dflare.json`.

### Inference and Evaluation

To benchmark a trained DFlare draft model on tasks such as GSM8K, MT-Bench, MATH-500, and HumanEval, use `tools/dflash_benchmark.py`. The script supports both DFlash and DFlare draft architectures via the `--draft-arch` flag — for DFlare set `--draft-arch dflare`. It loads the matching `QwenDFlareDraftModel` class, runs block-parallel speculative decoding (one existing anchor token plus `block_size - 1` draft proposals, parallel target verification, and longest-prefix acceptance), and reports decoding speedup, average acceptance length, and the per-block acceptance-length histogram.

**Single-GPU evaluation:**

```shell
python tools/dflash_benchmark.py \
    --model-name-or-path /path/to/Qwen3-4B \
    --draft-name-or-path /path/to/dflare_checkpoint \
    --draft-arch dflare \
    --dataset gsm8k \
    --max-samples 128 \
    --max-new-tokens 2048 \
    --temperature 0.0 \
    --block-size 16
```

**Multi-GPU evaluation** (workload is sharded across ranks; results are gathered to rank 0):

```shell
torchrun --nproc_per_node=8 --master_port=29600 \
    tools/dflash_benchmark.py \
    --model-name-or-path /path/to/Qwen3-4B \
    --draft-name-or-path /path/to/dflare_checkpoint \
    --draft-arch dflare \
    --dataset gsm8k \
    --max-samples 128 \
    --max-new-tokens 2048 \
    --temperature 0.0 \
    --block-size 16
```

Notes:

- `--block-size` is optional; if omitted, the script reads `block_size` directly from the loaded draft checkpoint's config.
- The script runs each prompt twice — once with `block_size=1` (vanilla AR decoding) and once with the speculative `block_size` — so the reported `Decoding speedup` is a self-contained ratio. No external baseline run is required.
- Both target and draft are loaded in `bfloat16` with `flash_attention_2` when `flash-attn` is installed (otherwise it falls back to PyTorch SDPA, which reduces wall-clock speedup but does not affect acceptance length).
- Supported datasets out of the box: `gsm8k`, `math500`, `aime24`, `aime25`, `alpaca`, `mt-bench`, `humaneval`, `mbpp`, `lbpp`, `swe-bench`, `livecodebench`.
- To compare DFlash and DFlare on the same checkpoint format, switch `--draft-arch dflash` and point `--draft-name-or-path` to a DFlash checkpoint — the rest of the command stays identical.

### Static Target-Layer Routes

For inference diagnostics, `--target-route` selects one fixed set of target source layers for the entire benchmark run. The default `original` preserves the checkpoint's fusion behavior. Other routes require `--draft-arch dflare` and mask the learned DFlare fusion to measure acceptance sensitivity without retraining.

The benchmark reads the actual target layer bank from the loaded draft model, honoring the checkpoint's `dflash_config.target_layer_ids`. For the Qwen3-8B DFlare checkpoint with bank `[1, 5, 9, 13, 17, 21, 25, 29, 33]`, the routes are:

| Route | Active target layer IDs |
| --- | --- |
| `original` | Full runtime checkpoint bank |
| `shallow` | `[1, 5, 9]` |
| `middle` | `[13, 17, 21]` |
| `deep` | `[25, 29, 33]` |
| `spread` | `[1, 17, 33]` |
| `mid_deep` | `[17, 25, 33]` |
| `custom` | IDs supplied by `--target-route-layers` |

Preset IDs are fixed, not recalculated as depth percentiles for other checkpoints. Every requested ID must belong to the runtime bank; missing IDs and empty routes fail before evaluation. Use a custom route for a different bank. Layer IDs map to their positions in that bank: for example, `[1, 17, 33]` produces mask `[1, 0, 0, 0, 1, 0, 0, 0, 1]` for the bank above. After loading the model, the main rank prints the bank, effective route, and active layers once.

Example with a deep route and verification tracing:

```shell
python tools/dflash_benchmark.py \
    --model-name-or-path /path/to/Qwen3-8B \
    --draft-name-or-path /path/to/dflare_checkpoint \
    --draft-arch dflare --dataset gsm8k --max-samples 8 \
    --temperature 0.0 --target-route deep \
    --trace-output /path/to/deep.jsonl
```

For custom layers, replace `--target-route deep` with `--target-route custom --target-route-layers 5,21,33`, or simply `--target-route-layers 5,21,33`. Providing `--target-route-layers` always takes precedence over the selected route name and records the effective route as `custom`; for example, `--target-route deep --target-route-layers 5,21,33` also selects `[5, 21, 33]`. Selecting `custom` without supplying layers is an error. Non-original routes on other draft architectures are rejected.

The implementation is in `angelslim/compressor/speculative/train/models/draft/qwen_dflare.py`. `extract_context_feature()` concatenates `hidden_states[layer_id + 1]` for each bank ID, accounting for the embedding output at index 0. `QwenDFlareDraftModel.forward()` reshapes this into the target layer bank and treats `layer_fusion_weights[d, l]` as learned logits `W[d, l]`, where `l` indexes the bank. `set_target_layer_route()` installs an inference-only, nonpersistent source-layer mask shared by all draft layers. Fusion is:

```text
W_route[d, l] = W[d, l] if target_layer_ids[l] is active, else -inf
alpha_route[d, :] = softmax(W_route[d, :])
fused_hidden[b, s, d, h] = sum_l alpha_route[d, l] * target_hidden[b, s, l, h]
```

Each draft layer retains its own learned relative weights among active sources. After `hidden_norm` RMS normalization, its fused context enters that layer's cross-attention through `k_proj_target` and `v_proj_target`. For `original`, the mask is `None` and `forward()` uses the original `softmax(layer_fusion_weights, dim=1)` path directly. The intervention leaves the target forward and full target hidden-state capture intact; it changes neither checkpoint weights, state-dict keys, nor the configured bank. It measures route sensitivity, not target compute savings, and does not change routes between rounds.

### Optional Verification Trace

Add `--trace-output /path/to/verification.jsonl` to either evaluation command to stream one JSON object per speculative round. Add `--trace-prob-stats` to also record per-proposal probabilities and entropies. Tracing is disabled by default: without `--trace-output`, no trace files or extra probability calculations are produced, and `--trace-prob-stats` has no effect. The `block_size=1` AR baseline is never traced.

For a verified block of size `B`, position 0 is an already sampled anchor; the drafter greedily proposes the remaining `B - 1` tokens. Target verification samples one token from each target output position using the CLI temperature (argmax at temperature 0), and acceptance takes the longest matching proposal prefix. The draft remains greedy even when the CLI temperature is nonzero. The trace uses the tensors from these existing forwards and records the following fields:

| Fields | Meaning |
| --- | --- |
| `sample_id`, `turn_id`, `round_id` | Zero-based sample index after the benchmark's optional shuffle/select, before rank sharding; turn index within that sample; round index reset to 0 for each generation call. |
| `target_route`, `active_target_layer_ids` | Effective fixed route name and active target layer IDs. `original` records the full runtime bank; explicit layer IDs record route `custom`. |
| `num_input_tokens`, `round_start`, `generated_tokens_before_round` | Prompt length; absolute token index of the current anchor; `round_start - num_input_tokens`, counting generated tokens before the anchor and excluding the anchor itself. |
| `block_size`, `proposal_count` | `B` and `B - 1`. |
| `anchor_token_id`, `draft_token_ids` | The anchor at block position 0 and the `B - 1` draft proposal IDs at positions 1 through `B - 1`. |
| `target_token_ids_for_proposals`, `match_mask` | Target verification IDs from `posterior[:, :-1]` and their elementwise equality with the draft proposals. All proposal positions are included, even after the first rejection. |
| `accepted_draft_tokens`, `reported_acceptance_length` | Longest matching proposal prefix length `A`; the existing benchmark metric `A + 1`, which includes the anchor. |
| `first_reject_position`, `all_draft_tokens_accepted` | Zero-based proposal index `A` of the first rejection, or JSON `null` if all proposals match; whether `A == B - 1`. |

With `--trace-prob-stats`, each record also contains five arrays of length `B - 1`:

- `draft_logprobs[k] = log q_D(x_k)` and `target_logprobs[k] = log p_T(x_k)`, both evaluated at the token `x_k` actually proposed by the drafter.
- `logprob_gaps[k] = target_logprobs[k] - draft_logprobs[k]`.
- `draft_entropies[k]` and `target_entropies[k]`, computed over each position's complete vocabulary distribution.

Proposal `x_k = block_output_ids[:, k + 1]` aligns with `draft_logits[:, k]` and `output.logits[:, k]`; the corresponding target verification token is `posterior[:, k]`. In particular, `target_logprobs` gathers the proposed token's probability, even when it differs from the target verification token. Target tokens and logits after the first rejection still condition on the original draft proposal prefix from the same causal verification forward; they do not represent a corrected AR continuation. Statistics use FP32 log-softmax of the **raw, unscaled model logits**, natural logarithms, and entropy in nats, regardless of sampling temperature. Reductions stay on the device; only token IDs and per-position scalar statistics are written. Full logits, hidden states, and KV caches are not saved.

Each record describes the complete verified block before EOS or output-length truncation. Thus acceptance fields preserve the benchmark's per-round metric even when the returned generation ends partway through the final block.

On a single GPU, the writer uses the requested path directly. Under `torchrun`, `verification.jsonl` becomes `verification.rank0.jsonl`, `verification.rank1.jsonl`, and so on. Parent directories are created automatically; each JSONL line is flushed through line buffering. A companion `<resolved-trace-path>.meta.json` records benchmark settings, model paths, block size, target layer IDs, probability-statistics setting, rank, and world size. Both metadata and every benchmark trace record include `target_route` and `active_target_layer_ids`.

Trace collection adds device-to-host transfers and file writes inside the timed decoding loop, with additional computation when probability statistics are enabled. Use runs with tracing disabled for baseline speed comparisons. For any selected route, enabling trace collection leaves sampling, acceptance, cache updates, fusion, and generation output unchanged. Selecting a non-original route is a separate intervention that changes draft fusion and may change proposals and acceptance.


## 📈 Results

We evaluate DFlare on six benchmarks spanning mathematical reasoning (GSM8K, MATH-500, AIME), code generation (HumanEval, MBPP, LiveCodeBench), and open-domain conversation (MT-Bench, Alpaca), against DFlash and EAGLE-3 baselines on Qwen3-4B, Qwen3-8B, and GPT-OSS-20B target models.

:::{image} /assets/dflare/speedup.png
:alt: DFlare end-to-end speedup vs DFlash and EAGLE-3 across six benchmarks.
:::


## 📜 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{DFlare2026,
  title={DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding},
  author={Jiebin Zhang and Zhenghan Yu and Song Liu and Eugene J. Yu and Zheng Li and Dawei Zhu and Jiangshan Duo and Weimin Xiong and Yifan Song and Guanghua Yu and Jianchen Zhu and Sujian Li},
  journal={arXiv preprint arXiv},
  year={2026}
}
```
