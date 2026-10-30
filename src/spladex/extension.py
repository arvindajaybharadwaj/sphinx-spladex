from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.util import logging
from sphinx.util.display import progress_message

from .compact import build_compact_index, compact_query_assets
from .encoder import (
    SemanticSparseEncoder,
    build_bm25_index,
    build_inverted_index,
    build_static_query_assets,
)

logger = logging.getLogger("spladex")

SUPPORTED_BUILDERS = {"html", "dirhtml"}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_page_title(doctree: nodes.document) -> str:
    title_node = next(iter(doctree.findall(nodes.title)), None)
    return clean_text(title_node.astext()) if title_node else ""


def make_html_url(app: Any, docname: str, anchor: str | None = None) -> str:
    uri = app.builder.get_target_uri(docname)
    return f"{uri}#{anchor}" if anchor else uri


def extract_page_record(app: Any, doctree: nodes.document, docname: str) -> dict[str, Any]:
    title = get_page_title(doctree)
    return {
        "id": f"page:{docname}",
        "title": title or docname,
        "url": make_html_url(app, docname),
        "text": clean_text(doctree.astext()),
        "granularity": "page",
        "object_type": "page",
    }


def extract_object_records(
    app: Any,
    doctree: nodes.document,
    docname: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for desc in doctree.findall(addnodes.desc):
        domain = desc.get("domain", "")
        objtype = desc.get("objtype", "")

        signatures = list(desc.findall(addnodes.desc_signature))
        content_nodes = list(desc.findall(addnodes.desc_content))

        if not signatures:
            continue

        content_text = clean_text(" ".join(node.astext() for node in content_nodes))

        for sig in signatures:
            module = sig.get("module", "")
            fullname = sig.get("fullname", "")

            if module and fullname and not fullname.startswith(module):
                public_name = f"{module}.{fullname}"
            else:
                public_name = fullname or clean_text(sig.astext())

            signature_text = clean_text(sig.astext())
            ids = sig.get("ids", [])
            anchor = ids[0] if ids else None

            object_text = clean_text(
                " ".join(
                    [
                        public_name,
                        signature_text,
                        objtype,
                        content_text,
                    ]
                )
            )

            records.append(
                {
                    "id": f"{domain}:{objtype}:{public_name}",
                    "title": public_name,
                    "url": make_html_url(app, docname, anchor),
                    "text": object_text,
                    "granularity": "object",
                    "object_type": objtype,
                    "domain": domain,
                    "docname": docname,
                    "anchor": anchor,
                }
            )

    return records


def on_builder_inited(app: Any) -> None:
    if app.builder.name not in SUPPORTED_BUILDERS:
        return

    if not hasattr(app.env, "model_semantic_records"):
        app.env.model_semantic_records = {}

    app.add_js_file("model_semantic_search.js")


def on_env_purge_doc(app: Any, env: Any, docname: str) -> None:
    if hasattr(env, "model_semantic_records") and docname in env.model_semantic_records:
        del env.model_semantic_records[docname]


def on_env_merge_info(app: Any, env: Any, docnames: list[str], other_env: Any) -> None:
    """Merge documentation records collected by parallel worker processes."""
    if not hasattr(env, "model_semantic_records"):
        env.model_semantic_records = {}
    if hasattr(other_env, "model_semantic_records"):
        for docname, doc_records in other_env.model_semantic_records.items():
            if docname in docnames or docname not in env.model_semantic_records:
                env.model_semantic_records[docname] = doc_records


def on_doctree_read(app: Any, doctree: nodes.document) -> None:
    docname = app.env.docname

    if app.builder.name not in SUPPORTED_BUILDERS:
        return

    if not hasattr(app.env, "model_semantic_records"):
        app.env.model_semantic_records = {}

    doc_records = []
    doc_records.append(extract_page_record(app, doctree, docname))

    for record in extract_object_records(app, doctree, docname):
        doc_records.append(record)

    app.env.model_semantic_records[docname] = doc_records


def on_build_finished(app: Any, exception: Exception | None) -> None:
    if exception is not None or app.builder.name not in SUPPORTED_BUILDERS:
        return

    # Copy JS script to built outputs _static directory
    out_static = Path(app.outdir) / "_static"
    out_static.mkdir(parents=True, exist_ok=True)

    src_js = Path(__file__).resolve().parent / "static" / "model_semantic_search.js"
    dest_js = out_static / "model_semantic_search.js"

    if src_js.exists():
        dest_js.write_text(src_js.read_text(encoding="utf-8"), encoding="utf-8")

    # Get settings from conf.py or use defaults
    model_name = getattr(app.config, "spladex_model_name", None) or getattr(
        app.config,
        "semantic_search_model_name",
        os.environ.get(
            "SEMANTIC_SEARCH_MODEL",
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
        ),
    )
    device = getattr(app.config, "spladex_device", None) or getattr(
        app.config, "semantic_search_device", "cpu"
    )
    max_length = getattr(app.config, "spladex_max_length", None) or getattr(
        app.config, "semantic_search_max_length", 256
    )
    top_k_terms = getattr(app.config, "spladex_top_k_terms", None) or getattr(
        app.config, "semantic_search_top_k_terms", 96
    )
    min_weight = getattr(app.config, "spladex_min_weight", None) or getattr(
        app.config, "semantic_search_min_weight", 0.0
    )
    backend = getattr(app.config, "spladex_encoder_backend", None) or getattr(
        app.config, "semantic_search_encoder_backend", "auto"
    )
    semantic_weight = float(getattr(app.config, "spladex_semantic_weight", 0.6))
    bm25_weight = float(getattr(app.config, "spladex_bm25_weight", 0.4))
    rrf_k = int(getattr(app.config, "spladex_rrf_k", 60))

    if semantic_weight < 0 or bm25_weight < 0 or (semantic_weight == 0 and bm25_weight == 0):
        logger.warning("Invalid spladex weights (semantic=%s, bm25=%s). Using defaults (0.6, 0.4).", semantic_weight, bm25_weight)
        semantic_weight, bm25_weight = 0.6, 0.4
    if rrf_k <= 0:
        logger.warning("Invalid spladex_rrf_k (%s). Using default 60.", rrf_k)
        rrf_k = 60

    # Ranking heuristics. Defaults were chosen on unseen projects (Flutter, React); indexes built
    # before these options existed keep the original client behaviour.
    bm25_tokenizer = str(getattr(app.config, "spladex_bm25_tokenizer", "identifier"))
    title_boost = str(getattr(app.config, "spladex_title_boost", "off"))
    bm25_prefix_match = bool(getattr(app.config, "spladex_bm25_prefix_match", False))
    dedupe = str(getattr(app.config, "spladex_dedupe", "exact"))
    if bm25_tokenizer not in ("simple", "identifier"):
        logger.warning("Invalid spladex_bm25_tokenizer (%s). Using 'identifier'.", bm25_tokenizer)
        bm25_tokenizer = "identifier"
    if title_boost not in ("always", "identifier", "off"):
        logger.warning("Invalid spladex_title_boost (%s). Using 'off'.", title_boost)
        title_boost = "off"
    if dedupe not in ("aggressive", "exact"):
        logger.warning("Invalid spladex_dedupe (%s). Using 'exact'.", dedupe)
        dedupe = "exact"

    # UI Theme & Customization Configs
    results_per_page = int(getattr(app.config, "spladex_results_per_page", 10))
    highlight_color = str(getattr(app.config, "spladex_highlight_color", "#fff3a0"))
    accent_color = str(getattr(app.config, "spladex_accent_color", "#0066cc"))
    badge_style = str(getattr(app.config, "spladex_badge_style", "pill"))
    enable_keyboard_shortcut = bool(getattr(app.config, "spladex_enable_keyboard_shortcut", True))
    enable_live_search = bool(getattr(app.config, "spladex_enable_live_search", True))
    show_engine_toggle = bool(getattr(app.config, "spladex_show_engine_toggle", False))
    show_matched_terms = bool(getattr(app.config, "spladex_show_matched_terms", False))

    records: list[dict[str, Any]] = []
    if hasattr(app.env, "model_semantic_records"):
        for doc_records in app.env.model_semantic_records.values():
            records.extend(doc_records)

    if not records:
        logger.info("[SpladeX] No documentation records found to index.")
        return

    # Attempt to initialize SPLADE encoder
    encoder: SemanticSparseEncoder | None = None
    try:
        encoder = SemanticSparseEncoder(
            model_name=model_name,
            device=device,
            max_length=max_length,
            top_k=top_k_terms,
            min_weight=min_weight,
            backend=backend,
        )
    except Exception as err:
        logger.warning("[SpladeX] Neural SPLADE encoder initialization failed (%s). Indexing will fall back to BM25-only search.", err)

    documents: list[dict[str, Any]] = []
    encode_started = time.perf_counter()

    batch_size = int(getattr(app.config, "spladex_batch_size", 16) or 16)
    texts = [" ".join([record["title"], record["object_type"], record["text"]]) for record in records]
    vectors: list[dict[str, float]] = [{} for _ in records]
    with progress_message(f"[SpladeX] Encoding {len(records)} documentation records for search"):
        if encoder is not None:
            try:
                vectors = encoder.encode_batch(texts, batch_size=batch_size)
            except Exception as err:
                logger.warning("[SpladeX] Batched encoding failed (%s); encoding records one by one.", err)
                for position, (record, text) in enumerate(zip(records, texts)):
                    try:
                        vectors[position] = encoder.encode(text)
                    except Exception as record_err:
                        logger.warning("[SpladeX] Encoding failed for record '%s': %s", record.get("id"), record_err)

        for record, searchable_text, vector in zip(records, texts, vectors):
            documents.append(
                {
                    "id": record["id"],
                    "title": record["title"],
                    "url": record["url"],
                    "text": record["text"][:1000],
                    "granularity": record["granularity"],
                    "object_type": record["object_type"],
                    "searchable_text": searchable_text,
                    "vector": vector,
                }
            )

    encode_seconds = time.perf_counter() - encode_started
    logger.info("[SpladeX] Encoded %d records in %.1f s with %s", len(documents), encode_seconds, model_name)

    inverted_index = build_inverted_index(documents)
    bm25_index = build_bm25_index(documents, tokenizer=bm25_tokenizer)

    query_assets: dict[str, Any] = {}
    if encoder is not None:
        try:
            query_assets = build_static_query_assets(model_name, encoder.tokenizer)
        except Exception as err:
            logger.warning("[SpladeX] Failed to build static query assets: %s", err)

    index = {
        "version": "0.3",
        "kind": "hybrid-splade-bm25-inverted-index",
        "description": "SPLADE and BM25 index generated at build time.",
        "model": {
            "name": model_name,
            "backend": backend,
            "max_length": max_length,
            "top_k_terms": top_k_terms,
            "min_weight": min_weight,
        },
        "hybrid": {
            "fusion": "rrf",
            "semantic_weight": semantic_weight,
            "bm25_weight": bm25_weight,
            "rrf_k": rrf_k,
            "title_boost": title_boost,
            "bm25_prefix_match": bm25_prefix_match,
            "dedupe": dedupe,
        },
        "ui_config": {
            "results_per_page": results_per_page,
            "highlight_color": highlight_color,
            "accent_color": accent_color,
            "badge_style": badge_style,
            "enable_keyboard_shortcut": enable_keyboard_shortcut,
            "enable_live_search": enable_live_search,
            "show_engine_toggle": show_engine_toggle,
            "show_matched_terms": show_matched_terms,
        },
        "query_assets": "model_static_query_assets.json",
        "stats": {
            "encode_seconds": round(encode_seconds, 1),
            "num_documents": len(documents),
            "num_terms": len(inverted_index),
            "num_bm25_terms": len(bm25_index["postings"]),
        },
        "documents": {
            doc["id"]: {
                "title": doc["title"],
                "url": doc["url"],
                "text": doc["text"],
                "granularity": doc["granularity"],
                "object_type": doc["object_type"],
            }
            for doc in documents
        },
        "inverted_index": inverted_index,
        "bm25_index": bm25_index,
    }

    index_format = str(getattr(app.config, "spladex_index_format", "compact")).lower()
    if index_format == "compact":
        header = {key: value for key, value in index.items() if key not in ("documents", "inverted_index", "bm25_index")}
        index = build_compact_index(documents, inverted_index, bm25_index, header)
        query_assets = compact_query_assets(query_assets)
    elif index_format != "legacy":
        logger.warning("Unknown spladex_index_format '%s'. Using 'legacy'.", index_format)

    index_path = out_static / "model_semantic_index.json"
    index_path.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    (out_static / "model_static_query_assets.json").write_text(
        json.dumps(query_assets, separators=(",", ":")), encoding="utf-8"
    )
    # Small file read on every page (keyboard shortcut, theme), so pages other than search.html
    # never download the index.
    (out_static / "model_semantic_config.json").write_text(
        json.dumps({"ui_config": index.get("ui_config", {})}, separators=(",", ":")), encoding="utf-8"
    )
    logger.info("[SpladeX] Successfully built semantic index at %s", index_path)


def setup(app: Any) -> dict[str, Any]:
    # Modern Configs
    app.add_config_value("spladex_model_name", "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill", "html")
    app.add_config_value("spladex_device", "cpu", "html")
    app.add_config_value("spladex_max_length", 256, "html")
    app.add_config_value("spladex_top_k_terms", 96, "html")
    app.add_config_value("spladex_min_weight", 0.0, "html")
    app.add_config_value("spladex_encoder_backend", "auto", "html")
    app.add_config_value("spladex_semantic_weight", 0.6, "html")
    app.add_config_value("spladex_bm25_weight", 0.4, "html")
    app.add_config_value("spladex_rrf_k", 60, "html")
    app.add_config_value("spladex_index_format", "compact", "html")
    app.add_config_value("spladex_batch_size", 16, "html")
    app.add_config_value("spladex_bm25_tokenizer", "identifier", "html")
    app.add_config_value("spladex_title_boost", "off", "html")
    app.add_config_value("spladex_bm25_prefix_match", False, "html")
    app.add_config_value("spladex_dedupe", "exact", "html")

    # UI Theme & Customization Configs
    app.add_config_value("spladex_results_per_page", 10, "html")
    app.add_config_value("spladex_highlight_color", "#fff3a0", "html")
    app.add_config_value("spladex_accent_color", "#0066cc", "html")
    app.add_config_value("spladex_badge_style", "pill", "html")
    app.add_config_value("spladex_enable_keyboard_shortcut", True, "html")
    app.add_config_value("spladex_enable_live_search", True, "html")
    app.add_config_value("spladex_show_engine_toggle", False, "html")
    app.add_config_value("spladex_show_matched_terms", False, "html")

    # Backwards compatibility configs
    app.add_config_value("semantic_search_model_name", "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill", "html")
    app.add_config_value("semantic_search_device", "cpu", "html")
    app.add_config_value("semantic_search_max_length", 256, "html")
    app.add_config_value("semantic_search_top_k_terms", 96, "html")
    app.add_config_value("semantic_search_min_weight", 0.0, "html")
    app.add_config_value("semantic_search_encoder_backend", "auto", "html")

    app.connect("builder-inited", on_builder_inited)
    app.connect("env-purge-doc", on_env_purge_doc)
    app.connect("env-merge-info", on_env_merge_info)
    app.connect("doctree-read", on_doctree_read)
    app.connect("build-finished", on_build_finished)

    return {
        "version": "0.2.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
