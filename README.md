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

The default model is `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill`
(67M parameters, Apache-2.0). Any SPLADE-style document encoder with an MLM head works if it
ships static query weights, either as `static_query_weights.pt` (a one-dimensional
`query_weights` tensor matching the tokenizer vocabulary) or as an `idf.json` (token to weight).

On documentation projects that none of the encoders was trained on (Flutter and React), the
hybrid search improves nDCG@10 over BM25 from 0.509 to 0.547 and from 0.499 to 0.556 with
the default encoder. `naver/splade-v3-doc` is stronger (0.560 on Flutter) but is licensed
CC BY-NC-SA 4.0, so it is not suitable for commercial sites. The models trained in this project
(`Arvind0101/static-query-splade-code-docs`, `Akshat131/static-splade-trained-pruned`,
`Akshat131/splade-multi-static-pruned-v2`) are fine-tuned from SPLADE-v3-doc, inherit its
licence, and do not beat BM25 on those unseen projects.

```python
spladex_model_name = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
spladex_device = "cpu"
spladex_max_length = 256
spladex_top_k_terms = 96
spladex_min_weight = 0.0
spladex_semantic_weight = 0.6
spladex_bm25_weight = 0.4
spladex_rrf_k = 60

# Index and ranking (defaults shown)
spladex_index_format = "compact"     # or "legacy" (pretty-printed v0.1 format)
spladex_bm25_tokenizer = "identifier"  # also index snake_case / camelCase / dotted parts
spladex_bm25_prefix_match = False    # v0.1 matched words by a crude stem prefix
spladex_title_boost = "off"          # "always" (v0.1), "identifier" (code-like queries only), "off"
spladex_dedupe = "exact"             # "aggressive" (v0.1) also hides methods sharing a short name

# UI Theme & Customization (Optional)
spladex_results_per_page = 10
spladex_highlight_color = "#fff3a0"
spladex_accent_color = "#0066cc"
spladex_badge_style = "pill"  # "pill", "subtle", or "none"
spladex_enable_keyboard_shortcut = True  # Ctrl+K / /
spladex_enable_live_search = True
spladex_show_engine_toggle = False   # Hybrid / SpladeX-only / BM25-only switch (demos)
spladex_show_matched_terms = False   # "matched: ..." line under each result
```

After `sphinx-build -b html docs docs/_build/html`, the generated site includes
`_static/model_semantic_index.json` and `_static/model_static_query_assets.json`.

SpladeX reads the tokenizer files and the static query weights (`static_query_weights.pt`
or `idf.json`) during the documentation build. It bundles their tokenization rules and
aligned weights into the local query-assets file; search never requests those source files
or a model from Hugging Face.

SpladeX replaces the standard Sphinx results with semantic hybrid results while
retaining the usual Sphinx search-page presentation. It does not show relevance
scores; the engine switch and matched terms are opt-in (see above).
