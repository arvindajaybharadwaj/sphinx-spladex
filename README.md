# Sphinx SpladeX

SpladeX adds static, client-side hybrid search to Sphinx HTML documentation. At
build time it extracts pages and documented objects, creates SPLADE sparse
vectors plus a BM25 index, and writes assets into `_static`. At search time the
browser loads only those assets: it does not run or download a model.

## Install

```bash
pip install sphinx-spladex
```

## Development

Install the test and release tools, then run the release check:

```bash
pip install -e ".[dev]"
scripts/release-check.sh
```

## Enable

Add the extension to `conf.py`:

```python
extensions = ["spladex"]
```

The default model is `Arvind0101/static-query-splade-code-docs`. The model must
include a `static_query_weights.pt` file containing a one-dimensional
`query_weights` tensor matching the tokenizer vocabulary.

```python
spladex_model_name = "Arvind0101/static-query-splade-code-docs"
spladex_device = "cpu"
spladex_max_length = 256
spladex_top_k_terms = 96
spladex_min_weight = 0.0
spladex_semantic_weight = 0.6
spladex_bm25_weight = 0.4
spladex_rrf_k = 60
```

After `sphinx-build -b html docs docs/_build/html`, the generated site includes
`_static/model_semantic_index.json` and `_static/model_static_query_assets.json`.

SpladeX reads `tokenizer.json`, `tokenizer_config.json`, and
`static_query_weights.pt` during the documentation build. It bundles their
tokenization rules and aligned weights into the local query-assets file; search
never requests those source files or a model from Hugging Face.

SpladeX replaces the standard Sphinx results with semantic hybrid results while
retaining the usual Sphinx search-page presentation. It does not show relevance
scores or provide a mode switch.
