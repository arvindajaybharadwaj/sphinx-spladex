from spladex.encoder import build_inverted_index


def test_build_inverted_index_sorts_by_descending_weight():
    documents = [
        {"id": "first", "vector": {"search": 1.2, "python": 0.2}},
        {"id": "second", "vector": {"search": 2.4}},
    ]

    index = build_inverted_index(documents)

    assert index["search"] == [["second", 2.4], ["first", 1.2]]
    assert index["python"] == [["first", 0.2]]

class _BatchTokenizer:
    """Tiny stand-in tokenizer: one id per character, used to check batching logic."""

    all_special_tokens = []

    def __call__(self, texts, return_tensors=None, truncation=True, padding=False, max_length=8):
        import torch

        single = isinstance(texts, str)
        batch = [texts] if single else texts
        ids = [[ord(c) % 50 + 1 for c in t][:max_length] for t in batch]
        if return_tensors is None:
            return {"input_ids": ids[0] if single else ids}
        width = max(len(x) for x in ids)
        input_ids = torch.tensor([x + [0] * (width - len(x)) for x in ids])
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])
        return _Batch({"input_ids": input_ids, "attention_mask": mask})

    def convert_ids_to_tokens(self, idx):
        return f"t{idx}"


class _Batch(dict):
    def to(self, device):
        return self


class _Model:
    def __call__(self, input_ids, attention_mask):
        import torch

        logits = torch.nn.functional.one_hot(input_ids, 64).float() * 2.0 - 0.5
        return type("Out", (), {"logits": logits})()


def test_encode_batch_matches_single_record_encoding():
    from spladex.encoder import SemanticSparseEncoder

    enc = SemanticSparseEncoder.__new__(SemanticSparseEncoder)
    enc.tokenizer, enc.model, enc.max_length, enc.top_k, enc.min_weight = _BatchTokenizer(), _Model(), 8, 5, 0.0
    enc.device = "cpu"
    texts = ["abc", "a much longer text", "", "zz", "hello world"]
    batched = enc.encode_batch(texts, batch_size=2)
    assert batched == [enc.encode(t) for t in texts]
    assert batched[2] == {}
