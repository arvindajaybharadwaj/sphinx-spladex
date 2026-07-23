from spladex.encoder import build_inverted_index


def test_build_inverted_index_sorts_by_descending_weight():
    documents = [
        {"id": "first", "vector": {"search": 1.2, "python": 0.2}},
        {"id": "second", "vector": {"search": 2.4}},
    ]

    index = build_inverted_index(documents)

    assert index["search"] == [["second", 2.4], ["first", 1.2]]
    assert index["python"] == [["first", 0.2]]