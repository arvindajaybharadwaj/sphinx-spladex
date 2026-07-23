from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx import addnodes

from .encoder import (
    SemanticSparseEncoder,
    build_bm25_index,
    build_inverted_index,
    build_static_query_assets,
)

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
            "Arvind0101/static-query-splade-code-docs",
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
    semantic_weight = getattr(app.config, "spladex_semantic_weight", 0.6)
    bm25_weight = getattr(app.config, "spladex_bm25_weight", 0.4)
    rrf_k = getattr(app.config, "spladex_rrf_k", 60)
    if semantic_weight < 0 or bm25_weight < 0 or semantic_weight + bm25_weight == 0:
        raise ValueError("spladex semantic and BM25 weights must be non-negative and not both zero")
    if rrf_k <= 0:
        raise ValueError("spladex_rrf_k must be greater than zero")

    records = []
    if hasattr(app.env, "model_semantic_records"):
        for doc_records in app.env.model_semantic_records.values():
            records.extend(doc_records)

    if not records:
        return

    encoder = SemanticSparseEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        top_k=top_k_terms,
        min_weight=min_weight,
        backend=backend,
    )

    documents: list[dict[str, Any]] = []

    for record in records:
        searchable_text = " ".join([record["title"], record["object_type"], record["text"]])
        vector = encoder.encode(searchable_text)

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

    inverted_index = build_inverted_index(documents)
    bm25_index = build_bm25_index(documents)
    query_assets = build_static_query_assets(model_name, encoder.tokenizer)

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
        },
        "query_assets": "model_static_query_assets.json",
        "stats": {
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

    index_path = out_static / "model_semantic_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    (out_static / "model_static_query_assets.json").write_text(
        json.dumps(query_assets, separators=(",", ":")), encoding="utf-8"
    )


def setup(app: Any) -> dict[str, Any]:
    # Modern Configs
    app.add_config_value("spladex_model_name", "Arvind0101/static-query-splade-code-docs", "html")
    app.add_config_value("spladex_device", "cpu", "html")
    app.add_config_value("spladex_max_length", 256, "html")
    app.add_config_value("spladex_top_k_terms", 96, "html")
    app.add_config_value("spladex_min_weight", 0.0, "html")
    app.add_config_value("spladex_encoder_backend", "auto", "html")
    app.add_config_value("spladex_semantic_weight", 0.6, "html")
    app.add_config_value("spladex_bm25_weight", 0.4, "html")
    app.add_config_value("spladex_rrf_k", 60, "html")

    # Backwards compatibility configs
    app.add_config_value("semantic_search_model_name", "Arvind0101/static-query-splade-code-docs", "html")
    app.add_config_value("semantic_search_device", "cpu", "html")
    app.add_config_value("semantic_search_max_length", 256, "html")
    app.add_config_value("semantic_search_top_k_terms", 96, "html")
    app.add_config_value("semantic_search_min_weight", 0.0, "html")
    app.add_config_value("semantic_search_encoder_backend", "auto", "html")

    app.connect("builder-inited", on_builder_inited)
    app.connect("env-purge-doc", on_env_purge_doc)
    app.connect("doctree-read", on_doctree_read)
    app.connect("build-finished", on_build_finished)

    return {
        "version": "0.1.0",
        "parallel_read_safe": False,
        "parallel_write_safe": False,
    }
