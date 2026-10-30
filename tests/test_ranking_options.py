import json
from pathlib import Path

from spladex.encoder import _load_idf_json_weights, build_bm25_index, tokenize_for_bm25


def test_identifier_tokenizer_splits_code_names():
    tokens = tokenize_for_bm25("Use DataFrame.dropna or nan_to_num()", mode="identifier")
    for expected in ("dataframe.dropna", "data", "frame", "dropna", "nan_to_num", "nan", "to", "num", "use"):
        assert expected in tokens
    assert tokenize_for_bm25("DataFrame.dropna") == ["dataframe", "dropna"]  # simple mode unchanged


def test_bm25_index_records_tokenizer():
    docs = [{"id": "a", "searchable_text": "numpy.isfinite test"}]
    assert build_bm25_index(docs)["tokenizer"] == "simple"
    index = build_bm25_index(docs, tokenizer="identifier")
    assert index["tokenizer"] == "identifier"
    assert "isfinite" in index["postings"] and "numpy.isfinite" in index["postings"]


class _Tokenizer:
    vocab_size = 4
    all_special_ids = [0]

    def convert_tokens_to_ids(self, token):
        return {"[PAD]": 0, "mean": 1, "array": 2, "sort": 3}.get(token)


def test_idf_json_weights_from_local_model(tmp_path: Path):
    (tmp_path / "idf.json").write_text(json.dumps({"[PAD]": 9.0, "mean": 2.5, "sort": 1.25, "unknown": 7.0}))
    assert _load_idf_json_weights(str(tmp_path), _Tokenizer()) == [0.0, 2.5, 0.0, 1.25]


def test_new_builds_record_ranking_defaults(tmp_path: Path):
    from types import SimpleNamespace

    from spladex.extension import on_build_finished

    app = SimpleNamespace(
        builder=SimpleNamespace(name="html"),
        outdir=str(tmp_path),
        config=SimpleNamespace(spladex_model_name="does-not-exist/model"),
        env=SimpleNamespace(model_semantic_records={"index": [{
            "id": "page:index", "title": "Index", "url": "index.html", "text": "numpy.isfinite tests finiteness",
            "granularity": "page", "object_type": "page"}]}),
    )
    on_build_finished(app, None)
    index = json.loads((tmp_path / "_static" / "model_semantic_index.json").read_text(encoding="utf-8"))
    assert index["kind"] == "spladex-compact"
    assert index["hybrid"]["title_boost"] == "off"
    assert index["hybrid"]["bm25_prefix_match"] is False
    assert index["hybrid"]["dedupe"] == "exact"
    assert index["bm25"]["tokenizer"] == "identifier"
