from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import hf_hub_download

logger = logging.getLogger("spladex")


def resolve_device(device_str: str) -> torch.device:
    """Resolve target torch device with fallback to CPU if requested accelerator is unavailable."""
    target = (device_str or "cpu").strip().lower()
    if target.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA device '%s' requested but CUDA is not available. Falling back to CPU.", device_str)
            return torch.device("cpu")
    elif target.startswith("mps"):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS device requested but Apple Silicon MPS is not available. Falling back to CPU.")
            return torch.device("cpu")
    
    try:
        return torch.device(target)
    except Exception as err:
        logger.warning("Invalid device string '%s' (%s). Falling back to CPU.", device_str, err)
        return torch.device("cpu")


class SemanticSparseEncoder:
    """Encodes text using a SPLADE Masked Language Model."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 256,
        top_k: int = 96,
        min_weight: float = 0.0,
        backend: str = "auto",  # Ignored, kept for backwards compatibility
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_length = max(1, int(max_length)) if max_length else 256
        self.top_k = max(1, int(top_k)) if top_k else 96
        self.min_weight = float(min_weight) if min_weight is not None else 0.0

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as err:
            logger.error("Failed to load tokenizer for model '%s': %s", model_name, err)
            raise RuntimeError(f"Failed to load tokenizer for '{model_name}': {err}") from err

        try:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as err:
            logger.error("Failed to load model '%s' on device %s: %s", model_name, self.device, err)
            raise RuntimeError(f"Failed to load model '{model_name}': {err}") from err

    @torch.no_grad()
    def encode(self, text: str) -> dict[str, float]:
        if not text or not text.strip():
            return {}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        outputs = self.model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        # SPLADE log-ReLU max pooling
        weights = torch.log1p(torch.relu(outputs.logits)) * attention_mask
        scores = torch.max(weights, dim=1).values.squeeze(0)

        k = min(self.top_k, scores.shape[0])
        values, indices = torch.topk(scores, k=k)

        result: dict[str, float] = {}
        for idx, val in zip(indices.tolist(), values.tolist()):
            if val <= self.min_weight:
                continue

            raw_token = self.tokenizer.convert_ids_to_tokens(idx)
            if not raw_token or raw_token in self.tokenizer.all_special_tokens:
                continue
            # Keep the original WordPiece token. Query-time tokenization uses
            # the same vocabulary, so collapsing pieces (for example ##ing)
            # would make static query weights impossible to apply correctly.
            token = raw_token.lower()
            if token and re.search(r"[a-zA-Z0-9_]", token):
                result[token] = round(val, 6)

        return result


def build_inverted_index(documents: list[dict[str, Any]]) -> dict[str, list[list[Any]]]:
    """Converts documents to an inverted index sorted by term weights.

    Deduplicates postings by keeping the maximum weight for each (term, doc_id)
    pair to prevent score inflation when the same doc_id appears in multiple
    records (e.g. page record and object record with the same id).
    """
    # Use a dict to accumulate max weight per (term, doc_id)
    max_weights: dict[str, dict[str, float]] = {}
    for doc in documents:
        doc_id = doc["id"]
        vector = doc.get("vector") or {}
        for term, weight in vector.items():
            term_map = max_weights.setdefault(term, {})
            if weight > term_map.get(doc_id, 0.0):
                term_map[doc_id] = weight

    inverted: dict[str, list[list[Any]]] = {}
    for term, doc_weights in max_weights.items():
        postings = [[doc_id, weight] for doc_id, weight in doc_weights.items()]
        postings.sort(key=lambda pair: pair[1], reverse=True)
        inverted[term] = postings

    return inverted


def build_static_query_assets(model_name: str, tokenizer: Any) -> dict[str, Any]:
    """Load static query weights and the matching tokenizer vocabulary.

    The browser uses the generated JSON for local WordPiece tokenization and
    query weighting; it never downloads or executes the Transformer model.
    If static query weights are not found on HF Hub, returns fallback uniform weights.
    """
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.vocab_size

    try:
        weight_path = hf_hub_download(repo_id=model_name, filename="static_query_weights.pt")
        payload = torch.load(Path(weight_path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "query_weights" not in payload:
            raise ValueError("static_query_weights.pt must contain 'query_weights' key")

        weights_tensor = payload["query_weights"]
        if not isinstance(weights_tensor, torch.Tensor) or weights_tensor.ndim != 1:
            raise ValueError("query_weights must be a 1D tensor")
        if len(weights_tensor) != vocab_size:
            raise ValueError(f"query_weights size ({len(weights_tensor)}) != vocab size ({vocab_size})")

        weights = [round(float(w), 6) for w in weights_tensor.tolist()]
        special_token_ids = payload.get("special_token_ids", tokenizer.all_special_ids)
    except Exception as err:
        logger.warning(
            "Could not load static query weights for '%s' (%s). Falling back to default uniform query weights.",
            model_name,
            err,
        )
        weights = [1.0] * vocab_size
        special_token_ids = list(tokenizer.all_special_ids)

    return {
        "version": "1",
        "kind": "learned-static-query-weights",
        "tokenizer": {
            "type": "wordpiece",
            "do_lower_case": bool(getattr(tokenizer, "do_lower_case", True)),
            "vocab": vocab,
            "unknown_token": getattr(tokenizer, "unk_token", "[UNK]"),
        },
        "weights": weights,
        "special_token_ids": special_token_ids,
    }


def tokenize_for_bm25(text: str) -> list[str]:
    """Return a small, deterministic tokenizer suitable for a static index."""
    if not text:
        return []
    return re.findall(r"[a-z0-9_]+", text.lower())


def build_bm25_index(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact BM25 index consumed by the browser search client.

    Postings store a document id and term frequency. IDF and final scoring are
    intentionally calculated in JavaScript so an index can be tuned without a
    Python server at search time.
    """
    postings: dict[str, list[list[Any]]] = {}
    document_lengths: dict[str, int] = {}

    for doc in documents:
        searchable_text = doc.get("searchable_text") or ""
        tokens = tokenize_for_bm25(searchable_text)
        doc_id = doc["id"]
        document_lengths[doc_id] = len(tokens)
        for term, frequency in Counter(tokens).items():
            postings.setdefault(term, []).append([doc_id, frequency])

    num_documents = len(documents)
    avg_document_length = (
        sum(document_lengths.values()) / num_documents if num_documents > 0 else 0.0
    )
    return {
        "k1": 1.2,
        "b": 0.75,
        "num_documents": num_documents,
        "avg_document_length": round(avg_document_length, 6),
        "document_lengths": document_lengths,
        "postings": postings,
    }
