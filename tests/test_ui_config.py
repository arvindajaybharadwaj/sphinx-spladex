import json
from unittest.mock import MagicMock
from spladex.extension import setup, on_build_finished


class ConfigMock:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_ui_config_defaults_in_setup():
    app = MagicMock()
    setup(app)

    # Collect all config calls
    config_keys = [call[0][0] for call in app.add_config_value.call_args_list]

    assert "spladex_results_per_page" in config_keys
    assert "spladex_highlight_color" in config_keys
    assert "spladex_accent_color" in config_keys
    assert "spladex_badge_style" in config_keys
    assert "spladex_enable_keyboard_shortcut" in config_keys
    assert "spladex_enable_live_search" in config_keys


def test_on_build_finished_includes_ui_config(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()

    app = MagicMock()
    app.outdir = str(outdir)
    app.builder.name = "html"
    app.config = ConfigMock(
        spladex_model_name="test-model",
        spladex_device="cpu",
        spladex_max_length=256,
        spladex_top_k_terms=96,
        spladex_min_weight=0.0,
        spladex_encoder_backend="auto",
        spladex_semantic_weight=0.6,
        spladex_bm25_weight=0.4,
        spladex_rrf_k=60,
        spladex_results_per_page=15,
        spladex_highlight_color="#ff0000",
        spladex_accent_color="#00ff00",
        spladex_badge_style="subtle",
        spladex_enable_keyboard_shortcut=True,
        spladex_enable_live_search=False,
    )

    # Simulate environment records
    app.env.model_semantic_records = {
        "index": [
            {
                "id": "page:index",
                "title": "Index Page",
                "url": "index.html",
                "text": "Welcome to documentation",
                "granularity": "page",
                "object_type": "page",
            }
        ]
    }

    on_build_finished(app, None)

    index_path = outdir / "_static" / "model_semantic_index.json"
    assert index_path.exists()

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    assert "ui_config" in index_data
    ui_cfg = index_data["ui_config"]

    assert ui_cfg["results_per_page"] == 15
    assert ui_cfg["highlight_color"] == "#ff0000"
    assert ui_cfg["accent_color"] == "#00ff00"
    assert ui_cfg["badge_style"] == "subtle"
    assert ui_cfg["enable_keyboard_shortcut"] is True
    assert ui_cfg["enable_live_search"] is False
