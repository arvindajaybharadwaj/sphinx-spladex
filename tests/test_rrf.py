def rrf_fuse(channels: list[tuple[dict[str, float], float]], rrf_k: int = 60):
    """Reference implementation of weighted Reciprocal Rank Fusion."""
    fused = {}

    for scores, channel_weight in channels:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for rank, (document_id, _) in enumerate(ranked, start=1):
            fused[document_id] = fused.get(document_id, 0.0) + (
                channel_weight / (rrf_k + rank)
            )

    return dict(sorted(fused.items(), key=lambda item: item[1], reverse=True))


def test_rrf_uses_rank_not_raw_score_scale():
    semantic = {"a": 1000.0, "b": 1.0}
    bm25 = {"b": 10.0, "a": 0.1}

    result = rrf_fuse([(semantic, 1.0), (bm25, 1.0)], rrf_k=60)

    assert result["a"] == result["b"]
    assert list(result) == ["a", "b"]


def test_weighted_rrf_favors_the_heavier_channel():
    semantic = {"semantic-first": 10.0, "bm25-first": 1.0}
    bm25 = {"bm25-first": 10.0, "semantic-first": 1.0}

    result = rrf_fuse([(semantic, 0.6), (bm25, 0.4)], rrf_k=60)

    assert list(result)[0] == "semantic-first"


def test_rrf_formula():
    result = rrf_fuse([({"document": 99.0}, 0.6)], rrf_k=60)

    assert result["document"] == 0.6 / 61