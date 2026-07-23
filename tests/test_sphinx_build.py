import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_SPHINX_INTEGRATION") != "1",
    reason="Set RUN_SPHINX_INTEGRATION=1 to run the model-backed Sphinx build.",
)
def test_sphinx_build_generates_search_assets(tmp_path: Path):
    docs = tmp_path / "docs"
    output = tmp_path / "_build" / "html"
    docs.mkdir()

    (docs / "conf.py").write_text(
        """
extensions = ["spladex"]
project = "SpladeX integration test"
html_theme = "alabaster"
spladex_device = "cpu"
spladex_top_k_terms = 16
""".strip()
    )
    (docs / "index.rst").write_text(
        """
SpladeX Test
============

Python semantic search with BM25 and SPLADE.
""".strip()
    )

    subprocess.run(
        ["sphinx-build", "-b", "html", "-E", str(docs), str(output)],
        check=True,
    )

    static = output / "_static"
    index = json.loads((static / "model_semantic_index.json").read_text())
    assets = json.loads((static / "model_static_query_assets.json").read_text())

    assert (static / "model_semantic_search.js").exists()
    assert index["hybrid"]["fusion"] == "rrf"
    assert index["query_assets"] == "model_static_query_assets.json"
    assert assets["kind"] == "learned-static-query-weights"