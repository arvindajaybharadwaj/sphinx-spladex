import json
import shutil
import subprocess
from pathlib import Path

import pytest

from spladex.compact import COMPACT_KIND, build_compact_index, compact_query_assets
from spladex.encoder import build_bm25_index, build_inverted_index

JS = Path(__file__).resolve().parents[1] / "src" / "spladex" / "static" / "model_semantic_search.js"

DOCUMENTS = [
    {"id": "py:function:pkg.mean", "title": "pkg.mean", "url": "api.html#pkg.mean", "text": "Compute the arithmetic mean average of values.",
     "granularity": "object", "object_type": "function", "searchable_text": "pkg.mean function Compute the arithmetic mean average of values.",
     "vector": {"mean": 2.1, "average": 1.4, "values": 0.3}},
    {"id": "py:function:pkg.isfinite", "title": "pkg.isfinite", "url": "api.html#pkg.isfinite", "text": "Test element-wise for finiteness.",
     "granularity": "object", "object_type": "function", "searchable_text": "pkg.isfinite function Test element-wise for finiteness.",
     "vector": {"finite": 2.4, "test": 0.8, "nan": 0.6, "number": 0.5}},
    {"id": "page:guide", "title": "User guide", "url": "guide.html", "text": "How to compute statistics such as the mean.",
     "granularity": "page", "object_type": "page", "searchable_text": "User guide page How to compute statistics such as the mean.",
     "vector": {"guide": 1.0, "mean": 0.9, "statistics": 1.2}},
]


def _decode(entry, ids, scale=None):
    deltas, values = entry
    out, position = [], 0
    for delta, value in zip(deltas, values):
        position += delta
        out.append((ids[position], value * scale if scale else value))
    return out


def test_compact_index_round_trips_postings():
    inverted = build_inverted_index(DOCUMENTS)
    bm25 = build_bm25_index(DOCUMENTS)
    compact = build_compact_index(DOCUMENTS, inverted, bm25, {"version": "0.3"})

    assert compact["kind"] == COMPACT_KIND
    ids = compact["docs"]["id"]
    scale = compact["splade"]["scale"]
    for term, pairs in inverted.items():
        decoded = dict(_decode(compact["splade"]["postings"][term], ids, scale))
        for doc_id, weight in pairs:
            assert decoded[doc_id] == pytest.approx(weight, abs=scale / 2 + 1e-9)
    for term, pairs in bm25["postings"].items():
        assert dict(_decode(compact["bm25"]["postings"][term], ids)) == {d: tf for d, tf in pairs}
    assert compact["bm25"]["document_lengths"] == [bm25["document_lengths"][d] for d in ids]


def test_compact_query_assets_keep_vocabulary_order():
    assets = {"kind": "learned-static-query-weights", "tokenizer": {"vocab": {"[PAD]": 0, "mean": 2, "test": 1}},
              "weights": [0.0, 1.234567, 2.0], "special_token_ids": [0]}
    compact = compact_query_assets(assets)
    assert compact["tokenizer"]["vocab_list"] == ["[PAD]", "test", "mean"]
    assert "vocab" not in compact["tokenizer"]
    assert compact["weights"] == [0.0, 1.2346, 2.0]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for the browser parity test")
def test_browser_ranking_is_identical_for_legacy_and_compact(tmp_path: Path):
    inverted = build_inverted_index(DOCUMENTS)
    bm25 = build_bm25_index(DOCUMENTS)
    header = {"hybrid": {"fusion": "rrf", "semantic_weight": 0.6, "bm25_weight": 0.4, "rrf_k": 60}}
    legacy = {**header, "documents": {d["id"]: {k: d[k] for k in ("title", "url", "text", "granularity", "object_type")} for d in DOCUMENTS},
              "inverted_index": inverted, "bm25_index": bm25}
    compact = build_compact_index(DOCUMENTS, inverted, bm25, header)
    vocab = {"[PAD]": 0, "[UNK]": 1, "mean": 2, "average": 3, "finite": 4, "test": 5, "number": 6, "statistics": 7, "guide": 8, "nan": 9, "values": 10}
    assets = {"tokenizer": {"type": "wordpiece", "do_lower_case": True, "vocab": vocab, "unknown_token": "[UNK]"},
              "weights": [0.0, 0.0] + [1.0] * 9, "special_token_ids": [0, 1]}
    for name, payload in (("legacy.json", legacy), ("compact.json", compact), ("assets.json", assets),
                          ("assets_compact.json", compact_query_assets(assets))):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    script = f"""
const fs = require("fs");
const sx = require({json.dumps(str(JS))});
const read = (n) => JSON.parse(fs.readFileSync({json.dumps(str(tmp_path))} + "/" + n, "utf8"));
const L = sx.normaliseIndex(read("legacy.json")), C = sx.normaliseIndex(read("compact.json"));
const A = sx.normaliseQueryAssets(read("assets.json")), AC = sx.normaliseQueryAssets(read("assets_compact.json"));
const out = {{}};
for (const mode of ["hybrid", "semantic", "lexical"]) {{
  for (const q of ["compute the average", "test if a number is finite", "mean", "statistics guide"]) {{
    const a = sx.searchLocalIndex(L, A, q, 100, mode).map((d) => d.url);
    const b = sx.searchLocalIndex(C, AC, q, 100, mode).map((d) => d.url);
    out[mode + ":" + q] = [a, b];
  }}
}}
console.log(JSON.stringify(out));
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    pairs = json.loads(result.stdout)
    for key, (legacy_urls, compact_urls) in pairs.items():
        assert legacy_urls == compact_urls, key
    assert pairs["semantic:test if a number is finite"][0][0] == "api.html#pkg.isfinite"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for the tokenizer parity test")
def test_browser_identifier_tokens_match_python():
    from spladex.encoder import tokenize_for_bm25

    samples = ["DataFrame.dropna", "nan_to_num()", "getHTTPResponse", "aBcD x.y_z", "numpy.linalg.norm",
               "replace NaN values", "...leading.dots...", "camelCaseWord and snake_case_word"]
    script = (f"const sx = require({json.dumps(str(JS))});"
              f"console.log(JSON.stringify({json.dumps(samples)}.map((s) => sx.identifierTokens(s))));")
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [tokenize_for_bm25(s, mode="identifier") for s in samples]
