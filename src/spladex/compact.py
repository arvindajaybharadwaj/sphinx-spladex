"""Compact on-disk encoding for the static search index.

The browser downloads the whole index before the first search, so its size is the main cost
SpladeX adds to a documentation site. The compact format stores:

* documents column-wise, so field names are not repeated per record;
* document references as integers, delta-encoded within each posting list;
* SPLADE weights quantised to integers in ``[1, 255]`` with a single global scale;
* BM25 term frequencies as plain integers.

The JavaScript client decodes posting lists lazily, one query term at a time.
"""

from __future__ import annotations

from typing import Any

COMPACT_KIND = "spladex-compact"
COMPACT_VERSION = "1"
SNIPPET_CHARS = 320


def _delta(values: list[int]) -> list[int]:
    out: list[int] = []
    previous = 0
    for value in values:
        out.append(value - previous)
        previous = value
    return out


def compact_postings(
    postings: dict[str, list[list[Any]]],
    doc_index: dict[str, int],
    scale: float | None = None,
) -> dict[str, list[list[int]]]:
    """Encode ``term -> [[doc_id, value], ...]`` as ``term -> [delta_doc_ints, values]``.

    With ``scale`` set, values are quantised to ``round(value / scale)`` clamped to ``[1, 255]``;
    otherwise they are stored as integers unchanged (BM25 term frequencies).
    """
    encoded: dict[str, list[list[int]]] = {}
    for term, pairs in postings.items():
        rows = sorted((doc_index[doc_id], value) for doc_id, value in pairs if doc_id in doc_index)
        if not rows:
            continue
        ids = [row[0] for row in rows]
        if scale:
            values = [min(255, max(1, round(row[1] / scale))) for row in rows]
        else:
            values = [int(row[1]) for row in rows]
        encoded[term] = [_delta(ids), values]
    return encoded


def build_compact_index(
    documents: list[dict[str, Any]],
    inverted_index: dict[str, list[list[Any]]],
    bm25_index: dict[str, Any],
    header: dict[str, Any],
) -> dict[str, Any]:
    """Return the compact index. ``header`` carries model/hybrid/ui metadata copied verbatim."""
    doc_ids = [doc["id"] for doc in documents]
    doc_index = {doc_id: position for position, doc_id in enumerate(doc_ids)}

    max_weight = max((weight for pairs in inverted_index.values() for _, weight in pairs), default=0.0)
    scale = max_weight / 255.0 if max_weight > 0 else 1.0

    lengths = bm25_index.get("document_lengths", {})
    return {
        **header,
        "kind": COMPACT_KIND,
        "format_version": COMPACT_VERSION,
        "docs": {
            "id": doc_ids,
            "title": [doc["title"] for doc in documents],
            "url": [doc["url"] for doc in documents],
            "text": [(doc.get("text") or "")[:SNIPPET_CHARS] for doc in documents],
            "granularity": [doc.get("granularity", "") for doc in documents],
            "object_type": [doc.get("object_type", "") for doc in documents],
        },
        "splade": {
            "scale": scale,
            "postings": compact_postings(inverted_index, doc_index, scale=scale),
        },
        "bm25": {
            "tokenizer": bm25_index.get("tokenizer", "simple"),
            "k1": bm25_index.get("k1", 1.2),
            "b": bm25_index.get("b", 0.75),
            "num_documents": bm25_index.get("num_documents", len(documents)),
            "avg_document_length": bm25_index.get("avg_document_length", 0.0),
            "document_lengths": [lengths.get(doc_id, 0) for doc_id in doc_ids],
            "postings": compact_postings(bm25_index.get("postings", {}), doc_index),
        },
    }


def compact_query_assets(query_assets: dict[str, Any]) -> dict[str, Any]:
    """Store the vocabulary as an id-ordered list and weights with 4 decimals."""
    if not query_assets:
        return query_assets
    tokenizer = dict(query_assets["tokenizer"])
    vocab = tokenizer.pop("vocab", {})
    vocab_list = [""] * (max(vocab.values()) + 1 if vocab else 0)
    for token, token_id in vocab.items():
        vocab_list[token_id] = token
    tokenizer["vocab_list"] = vocab_list
    return {
        **query_assets,
        "tokenizer": tokenizer,
        "weights": [round(float(weight), 4) for weight in query_assets.get("weights", [])],
    }
