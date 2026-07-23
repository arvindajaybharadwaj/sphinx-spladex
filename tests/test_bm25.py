from spladex.encoder import build_bm25_index, tokenize_for_bm25


def test_tokenize_for_bm25():
    assert tokenize_for_bm25("SPLADE + BM25: Search_Engine!") == [
        "splade",
        "bm25",
        "search_engine",
    ]


def test_build_bm25_index():
    documents = [
        {"id": "intro", "searchable_text": "Python semantic search"},
        {"id": "bm25", "searchable_text": "BM25 ranking for search"},
    ]

    index = build_bm25_index(documents)

    assert index["num_documents"] == 2
    assert index["document_lengths"] == {"intro": 3, "bm25": 4}
    assert index["postings"]["search"] == [["intro", 1], ["bm25", 1]]
    assert index["k1"] == 1.2
    assert index["b"] == 0.75