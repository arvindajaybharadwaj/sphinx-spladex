from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import hf_hub_download
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
        self.device = torch.device(device)
        self.max_length = max_length
        self.top_k = top_k
        self.min_weight = min_weight
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
    @torch.no_grad()
    def encode(self, text: str) -> dict[str, float]:
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
    """Converts documents to an inverted index sorted by term weights."""
    inverted: dict[str, list[list[Any]]] = {}
    for doc in documents:
        doc_id = doc["id"]
        for term, weight in doc["vector"].items():
            inverted.setdefault(term, []).append([doc_id, weight])
    for term in inverted:
        inverted[term].sort(key=lambda pair: pair[1], reverse=True)

    return inverted


def build_static_query_assets(model_name: str, tokenizer: Any) -> dict[str, Any]:
    """Load static query weights and the matching tokenizer vocabulary.

    The browser uses the generated JSON for local WordPiece tokenization and
    query weighting; it never downloads or executes the Transformer model.
    """
    weight_path = hf_hub_download(repo_id=model_name, filename="static_query_weights.pt")
    payload = torch.load(Path(weight_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "query_weights" not in payload:
        raise ValueError("static_query_weights.pt must contain query_weights")

    weights = payload["query_weights"]
    if not isinstance(weights, torch.Tensor) or weights.ndim != 1:
        raise ValueError("query_weights must be a one-dimensional tensor")
    if len(weights) != tokenizer.vocab_size:
        raise ValueError("static query weights and tokenizer vocabulary sizes differ")

    return {
        "version": "1",
        "kind": "learned-static-query-weights",
        "tokenizer": {
            "type": "wordpiece",
            "do_lower_case": bool(getattr(tokenizer, "do_lower_case", True)),
            "vocab": tokenizer.get_vocab(),
            "unknown_token": tokenizer.unk_token,
        },
        "weights": [round(float(weight), 6) for weight in weights.tolist()],
        "special_token_ids": payload.get("special_token_ids", tokenizer.all_special_ids),
    }


def tokenize_for_bm25(text: str) -> list[str]:
    """Return a small, deterministic tokenizer suitable for a static index."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def build_bm25_index(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact BM25 index consumed by the browser search client.

    Postings store a document id and term frequency.  IDF and final scoring are
    intentionally calculated in JavaScript so an index can be tuned without a
    Python server at search time.
    """
    postings: dict[str, list[list[Any]]] = {}
    document_lengths: dict[str, int] = {}

    for doc in documents:
        tokens = tokenize_for_bm25(doc["searchable_text"])
        document_lengths[doc["id"]] = len(tokens)
        for term, frequency in Counter(tokens).items():
            postings.setdefault(term, []).append([doc["id"], frequency])

    num_documents = len(documents)
    avg_document_length = (
        sum(document_lengths.values()) / num_documents if num_documents else 0.0
    )
    return {
        "k1": 1.2,
        "b": 0.75,
        "num_documents": num_documents,
        "avg_document_length": round(avg_document_length, 6),
        "document_lengths": document_lengths,
        "postings": postings,
    }
