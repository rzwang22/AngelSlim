"""Static route validation and real, tiny DFlare fusion checks on CPU."""

import argparse
import copy
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BANK = [1, 5, 9, 13, 17, 21, 25, 29, 33]
PRESETS = {
    "shallow": [1, 5, 9],
    "middle": [13, 17, 21],
    "deep": [25, 29, 33],
    "spread": [1, 17, 33],
    "mid_deep": [17, 25, 33],
}


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location(
        "dflare_route_benchmark_tests", ROOT / "tools" / "dflash_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dflare_module():
    # Import the actual model and factory without importing AngelSlim's unrelated
    # top-level compression backends. No model implementation is copied or mocked.
    name = "_dflare_route_test_models"
    package = ModuleType(name)
    package.__path__ = [str(ROOT / "angelslim/compressor/speculative/train/models/draft")]
    sys.modules[name] = package
    try:
        yield importlib.import_module(f"{name}.qwen_dflare")
    finally:
        for module_name in list(sys.modules):
            if module_name == name or module_name.startswith(f"{name}."):
                del sys.modules[module_name]


def _tiny_model(module, bank=None):
    config = module.Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_target_layers=36,
        block_size=4,
        dflash_config={
            "target_layer_ids": list(BANK if bank is None else bank),
            "mask_token_id": 0,
        },
    )
    config._attn_implementation = "eager"
    return module.QwenDFlareDraftModel(config).eval()


def _inputs(model):
    source_size = 3 * model.num_target_layers * model.config.hidden_size
    return {
        "target_hidden": torch.arange(source_size, dtype=torch.float32).reshape(1, 3, -1) / 17,
        "noise_embedding": torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 11,
        "position_ids": torch.arange(5).unsqueeze(0),
        "use_cache": False,
    }


@pytest.mark.parametrize("name,expected", [("original", BANK), *PRESETS.items()])
def test_predefined_routes_use_absolute_layer_ids(benchmark, name, expected):
    bank = BANK.copy()
    assert benchmark._resolve_target_route(bank, target_route=name) == (name, expected)
    assert bank == BANK


def test_routes_follow_checkpoint_bank_order_and_custom_overrides_preset(benchmark):
    bank = [33, 9, 17, 1, 5, 21, 13, 29, 25]
    assert benchmark._resolve_target_route(bank, target_route="spread") == ("spread", [33, 17, 1])
    assert benchmark._resolve_target_route(
        bank, target_route="deep", target_route_layers=[1, 17, 33]
    ) == ("custom", [33, 17, 1])
    assert benchmark._resolve_target_route([6, 2, 18], target_route_layers=[18, 2]) == (
        "custom",
        [2, 18],
    )


def test_preset_does_not_reinterpret_an_unrelated_bank_by_position(benchmark):
    with pytest.raises(ValueError, match="target_layer_ids|checkpoint|bank"):
        benchmark._resolve_target_route(list(range(40, 49)), target_route="shallow")


@pytest.mark.parametrize("text", ["", " ", ",", "1,", ",1", "1,,5", "one,5", "1.5,5", "1,1"])
def test_route_layer_parser_rejects_empty_malformed_and_duplicate_values(benchmark, text):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._parse_target_route_layers(text)


def test_route_layer_parser_accepts_whitespace(benchmark):
    assert benchmark._parse_target_route_layers(" 5, 21 ,33 ") == [5, 21, 33]


@pytest.mark.parametrize("bank", [[], [1, 1], [-1, 1], [1, "5"], [1, 5.0], [True, 5]])
def test_route_resolution_validates_checkpoint_bank(benchmark, bank):
    with pytest.raises(ValueError):
        benchmark._resolve_target_route(bank, target_route_layers=[1])


@pytest.mark.parametrize("draft_arch", ["dflare", "dflash"])
def test_original_preserves_legacy_banks_with_repeated_captured_layers(benchmark, draft_arch):
    assert benchmark._resolve_target_route([1, 1], draft_arch=draft_arch) == ("original", [1, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_route": "custom"},
        {"target_route_layers": []},
        {"target_route_layers": [1, 1]},
        {"target_route_layers": [1, 999]},
        {"target_route": "unknown"},
    ],
)
def test_route_resolution_rejects_invalid_requests(benchmark, kwargs):
    with pytest.raises(ValueError):
        benchmark._resolve_target_route(BANK, **kwargs)


def test_unknown_layer_error_identifies_missing_id(benchmark):
    with pytest.raises(ValueError, match="999.*target_layer_ids|target_layer_ids.*999"):
        benchmark._resolve_target_route(BANK, target_route_layers=[1, 999])


@pytest.mark.parametrize("kwargs", [{"target_route": "spread"}, {"target_route_layers": [1]}])
def test_non_dflare_routes_fail_explicitly(benchmark, kwargs):
    with pytest.raises(ValueError, match="only.*DFlare|DFlare.*only"):
        benchmark._resolve_target_route(BANK, draft_arch="dflash", **kwargs)
    assert benchmark._resolve_target_route(BANK, draft_arch="dflash") == ("original", BANK)


@pytest.mark.parametrize(
    "route_flags", [["--target-route", "spread"], ["--target-route-layers", "1,17,33"]]
)
def test_cli_rejects_non_dflare_route_before_cuda_or_model_loading(
    benchmark, monkeypatch, capsys, route_flags
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid route initialized distributed/CUDA or loaded models")

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
            "dflash",
            "--dataset",
            "gsm8k",
            *route_flags,
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        benchmark.main()
    assert exit_info.value.code == 2
    assert "supported only for DFlare" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bank,active,mask",
    [
        (BANK, [1, 17, 33], [True, False, False, False, True, False, False, False, True]),
        ([33, 5, 17, 1], [1, 17, 33], [True, False, True, True]),
        ([6, 2, 18], [18, 2], [False, True, True]),
    ],
)
def test_model_mask_maps_real_layer_ids_and_tracks_device(dflare_module, bank, active, mask):
    model = _tiny_model(dflare_module, bank)
    model.set_target_layer_route(active)
    assert model._target_route_mask.tolist() == mask
    assert model._target_route_mask.dtype == torch.bool
    assert model._target_route_mask.device == model.layer_fusion_weights.device
    model.to(dtype=torch.float64)
    assert model._target_route_mask.dtype == torch.bool
    assert model._target_route_mask.device == model.layer_fusion_weights.device


@pytest.mark.parametrize("active", [[], [1, 1], [1, 999], ["1"], [1.0], [True], [-1]])
def test_model_rejects_invalid_routes_without_replacing_existing_mask(dflare_module, active):
    model = _tiny_model(dflare_module)
    model.set_target_layer_route([1, 17, 33])
    previous = model._target_route_mask.clone()
    with pytest.raises(ValueError):
        model.set_target_layer_route(active)
    assert torch.equal(model._target_route_mask, previous)


def test_masked_fusion_renormalizes_and_preserves_each_draft_layers_preferences(
    dflare_module, monkeypatch
):
    model = _tiny_model(dflare_module)
    logits = torch.stack([torch.arange(9), torch.arange(8, -1, -1)]).float() / 3
    with torch.no_grad():
        model.layer_fusion_weights.copy_(logits)
    model.set_target_layer_route([1, 17, 33])
    captured = {"probabilities": [], "fused": [], "layer_inputs": []}
    softmax = torch.softmax

    def spy_softmax(value, *args, **kwargs):
        result = softmax(value, *args, **kwargs)
        if value.shape == logits.shape:
            captured["probabilities"].append(result.detach().clone())
        return result

    monkeypatch.setattr(torch, "softmax", spy_softmax)
    handles = [
        model.hidden_norm.register_forward_pre_hook(
            lambda module, args: captured["fused"].append(args[0].detach().clone())
        )
    ]
    for layer in model.layers:
        handles.append(
            layer.register_forward_pre_hook(
                lambda module, args, kwargs: captured["layer_inputs"].append(
                    kwargs["target_hidden"].detach().clone()
                ),
                with_kwargs=True,
            )
        )
    inputs = _inputs(model)
    try:
        with torch.inference_mode():
            output = model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    assert output.shape == (1, 2, 8)
    assert len(captured["probabilities"]) == 1
    probabilities = captured["probabilities"][0]
    inactive = [1, 2, 3, 5, 6, 7]
    assert torch.count_nonzero(probabilities[:, inactive]).item() == 0
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
    assert not torch.equal(probabilities[0], probabilities[1])
    torch.testing.assert_close(
        probabilities[:, 0] / probabilities[:, 4], torch.exp(logits[:, 0] - logits[:, 4])
    )
    sources = inputs["target_hidden"].view(1, 3, 9, 8)
    expected = torch.stack(
        [
            sum(sources[:, :, source, :] * probabilities[draft, source] for source in [0, 4, 8])
            for draft in range(2)
        ],
        dim=2,
    )
    torch.testing.assert_close(captured["fused"][0], expected)
    normalized = model.hidden_norm(expected)
    for layer, consumed in enumerate(captured["layer_inputs"]):
        torch.testing.assert_close(consumed, normalized[:, :, layer, :])
    assert inputs["target_hidden"].shape[-1] == 9 * 8


def test_original_route_keeps_exact_softmax_input_and_restores_identical_output(
    dflare_module, monkeypatch
):
    model = _tiny_model(dflare_module)
    assert model._target_route_mask is None
    inputs = _inputs(model)
    with torch.inference_mode():
        baseline = model(**inputs)
    model.set_target_layer_route([1, 17, 33])
    model.set_target_layer_route(None)
    assert model._target_route_mask is None
    softmax, masked_fill = torch.softmax, torch.Tensor.masked_fill
    fusion_calls = []

    def spy_softmax(value, *args, **kwargs):
        if value.shape == model.layer_fusion_weights.shape:
            fusion_calls.append(value)
            assert value is model.layer_fusion_weights
            assert kwargs.get("dim", args[0] if args else None) == 1
        return softmax(value, *args, **kwargs)

    def reject_fusion_mask(value, *args, **kwargs):
        if value is model.layer_fusion_weights:
            pytest.fail("original route entered the fusion-mask path")
        return masked_fill(value, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", spy_softmax)
    monkeypatch.setattr(torch.Tensor, "masked_fill", reject_fusion_mask)
    with torch.inference_mode():
        restored = model(**inputs)
    assert len(fusion_calls) == 1
    assert torch.equal(restored, baseline)


def test_route_is_nonpersistent_and_does_not_change_weights_or_configuration(dflare_module):
    model = _tiny_model(dflare_module)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    configuration = copy.deepcopy(model.config.to_dict())
    original_ids = model.target_layer_ids.copy()
    model.set_target_layer_route([5, 21, 33])
    assert "_target_route_mask" in dict(model.named_buffers())
    assert "_target_route_mask" not in model.state_dict()
    assert model.state_dict().keys() == weights.keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, weights[name])
    assert model.config.to_dict() == configuration
    assert model.target_layer_ids == original_ids


def test_route_requires_eval_and_cannot_survive_into_training_forward(dflare_module):
    model = _tiny_model(dflare_module)
    model.train()
    with pytest.raises((ValueError, RuntimeError), match="inference|eval|training"):
        model.set_target_layer_route([1])
    model.eval()
    model.set_target_layer_route([1])
    model.train()
    with pytest.raises((ValueError, RuntimeError), match="inference|eval|training"):
        model(**_inputs(model))
    model.set_target_layer_route(None)
    assert model(**_inputs(model)).shape == (1, 2, 8)


def test_trace_route_fields_are_additive_to_existing_verification_schema(benchmark):
    arguments = dict(
        round_id=0,
        num_input_tokens=3,
        round_start=3,
        block_output_ids=torch.tensor([[2, 3, 4]]),
        posterior=torch.tensor([[3, 0, 1]]),
        accepted_draft_tokens=1,
    )
    identity = {"sample_id": 5, "turn_id": 1}
    basic = benchmark._build_verification_trace(trace_context=identity, **arguments)
    route = {"target_route": "spread", "active_target_layer_ids": [1, 17, 33]}
    routed = benchmark._build_verification_trace(
        trace_context=dict(identity, **route), **arguments
    )
    assert routed == dict(basic, **route)
