#!/usr/bin/env python3
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dspark_export", ROOT / "scripts" / "dspark_export.py")
dspark_export = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["dspark_export"] = dspark_export
SPEC.loader.exec_module(dspark_export)


def test_name_mapping():
    cases = {
        "mtp.0.main_proj.weight": "dspark.main_proj.weight",
        "mtp.0.attn.wq_a.weight": "dspark.blk.0.attn_q_a.weight",
        "mtp.1.attn.kv_norm.weight": "dspark.blk.1.attn_kv_a_norm.weight",
        "mtp.2.ffn.experts.17.w3.scale": "dspark.blk.2.ffn_up_exps.expert.17.scale",
        "mtp.2.markov_head.markov_w1.weight": "dspark.markov_embd.weight",
        "mtp.2.confidence_head.proj.weight": "dspark.confidence_proj.weight",
    }
    for source, expected in cases.items():
        actual = dspark_export.ds4_record_name(source)
        assert actual == expected, (source, actual, expected)


def test_flash_metadata_defaults():
    metas = {
        "mtp.0.ffn.experts.0.w1.weight": dspark_export.TensorMeta(
            "mtp.0.ffn.experts.0.w1.weight", "s0", "I8", (16, 128), (0, 2048)
        ),
        "mtp.1.ffn.experts.0.w1.weight": dspark_export.TensorMeta(
            "mtp.1.ffn.experts.0.w1.weight", "s1", "I8", (16, 128), (0, 2048)
        ),
        "mtp.2.ffn.experts.0.w1.weight": dspark_export.TensorMeta(
            "mtp.2.ffn.experts.0.w1.weight", "s2", "I8", (16, 128), (0, 2048)
        ),
    }
    config = {
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "vocab_size": 129280,
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
    }
    model, dspark = dspark_export.model_metadata(config, metas, None)
    assert model["variant"] == "flash"
    assert model["hidden_size"] == 4096
    assert model["expert_count"] == 256
    assert dspark["block_size"] == 5
    assert dspark["draft_layer_ids"] == [0, 1, 2]
    assert dspark["target_layer_ids"] == [40, 41, 42]
    assert dspark["window_size"] == 128


if __name__ == "__main__":
    test_name_mapping()
    test_flash_metadata_defaults()
    print("dspark_export_smoke: ok")
