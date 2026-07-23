from pathlib import Path

import torch

import spladex.encoder as encoder_module
from spladex.encoder import build_static_query_assets


class FakeTokenizer:
    vocab_size = 4
    unk_token = "[UNK]"
    all_special_ids = [0, 1]
    do_lower_case = True

    def get_vocab(self):
        return {
            "[PAD]": 0,
            "[UNK]": 1,
            "search": 2,
            "python": 3,
        }


def test_static_query_assets_match_tokenizer(monkeypatch, tmp_path: Path):
    weights_file = tmp_path / "static_query_weights.pt"
    torch.save(
        {
            "query_weights": torch.tensor([0.0, 0.0, 1.5, 2.5]),
            "special_token_ids": [0, 1],
        },
        weights_file,
    )

    monkeypatch.setattr(
        encoder_module,
        "hf_hub_download",
        lambda **_: str(weights_file),
    )

    assets = build_static_query_assets(
        "example/static-query-model",
        FakeTokenizer(),
    )

    assert assets["kind"] == "learned-static-query-weights"
    assert assets["tokenizer"]["type"] == "wordpiece"
    assert len(assets["weights"]) == 4
    assert assets["weights"][2] == 1.5
    assert assets["special_token_ids"] == [0, 1]