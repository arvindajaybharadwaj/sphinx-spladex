(function () {
  let indexPromise = null;
  let queryAssetsPromise = null;

  function getQueryFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get("q") || "";
  }

  function findSearchInput() {
    return (
      document.querySelector("input[name='q']") ||
      document.querySelector("input[type='search']") ||
      document.querySelector("#searchbox input")
    );
  }

  function findResultsContainer() {
    return (
      document.querySelector("#search-results") ||
      document.querySelector(".body") ||
      document.querySelector("main") ||
      document.body
    );
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function makeSnippet(text) {
    const cleanText = String(text).replace(/\s+/g, " ").trim();
    if (cleanText.length <= 320) {
      return cleanText;
    }
    return cleanText.slice(0, 320) + "...";
  }

  function loadIndex() {
    if (indexPromise) {
      return indexPromise;
    }

    const urlRoot =
      (typeof DOCUMENTATION_OPTIONS !== "undefined" &&
        DOCUMENTATION_OPTIONS.URL_ROOT) ||
      "";
    const indexUrl = new URL(
      urlRoot + "_static/model_semantic_index.json",
      window.location.href,
    ).toString();

    indexPromise = fetch(indexUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load search index");
        }
        return response.json();
      })
      .then((index) => index);

    return indexPromise;
  }

  function loadQueryAssets(index) {
    if (queryAssetsPromise) return queryAssetsPromise;
    const urlRoot =
      (typeof DOCUMENTATION_OPTIONS !== "undefined" &&
        DOCUMENTATION_OPTIONS.URL_ROOT) ||
      "";
    const assetName = index.query_assets || "model_static_query_assets.json";
    const assetUrl = new URL(
      urlRoot + "_static/" + assetName,
      window.location.href,
    ).toString();
    queryAssetsPromise = fetch(assetUrl).then((response) => {
      if (!response.ok) throw new Error("Failed to load static query weights");
      return response.json();
    });
    return queryAssetsPromise;
  }

  function wordPieceTokens(query, queryAssets) {
    const tokenizer = queryAssets.tokenizer;
    const vocab = tokenizer.vocab || {};
    const normalised = tokenizer.do_lower_case ? query.toLowerCase() : query;
    const words = normalised.match(/[a-z0-9_]+|[^\s]/gi) || [];
    const tokens = [];
    for (const word of words) {
      let start = 0;
      const pieces = [];
      while (start < word.length) {
        let end = word.length;
        let piece = null;
        while (start < end) {
          const candidate = (start ? "##" : "") + word.slice(start, end);
          if (Object.prototype.hasOwnProperty.call(vocab, candidate)) {
            piece = candidate;
            break;
          }
          end -= 1;
        }
        if (!piece) {
          pieces.length = 0;
          pieces.push(tokenizer.unknown_token || "[UNK]");
          break;
        }
        pieces.push(piece);
        start = end;
      }
      tokens.push(...pieces);
    }
    return tokens;
  }

  function searchLocalIndex(index, queryAssets, query, topK = 10) {
    const terms = query
      .toLowerCase()
      .replace(/[^\w\s_]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 0);

    const semanticScores = {};
    const bm25Scores = {};
    const invertedIndex = index.inverted_index || {};
    const vocab = queryAssets.tokenizer.vocab || {};
    const weights = queryAssets.weights || [];
    const specialIds = new Set(queryAssets.special_token_ids || []);

    for (const term of wordPieceTokens(query, queryAssets)) {
      const tokenId = vocab[term];
      if (tokenId === undefined || specialIds.has(tokenId)) continue;
      const queryWeight = weights[tokenId] || 0;
      if (queryWeight <= 0) continue;
      const postings = invertedIndex[term] || [];
      for (const [docId, documentWeight] of postings) {
        semanticScores[docId] =
          (semanticScores[docId] || 0.0) + queryWeight * documentWeight;
      }
    }

    const bm25 = index.bm25_index || {};
    const postingsByTerm = bm25.postings || {};
    const documentLengths = bm25.document_lengths || {};
    const numDocuments = bm25.num_documents || 0;
    const averageLength = bm25.avg_document_length || 1;
    const k1 = bm25.k1 || 1.2;
    const b = bm25.b || 0.75;
    for (const term of terms) {
      const postings = postingsByTerm[term] || [];
      if (!postings.length) continue;
      const idf = Math.log(
        1 + (numDocuments - postings.length + 0.5) / (postings.length + 0.5),
      );
      for (const [docId, frequency] of postings) {
        const length = documentLengths[docId] || 0;
        const denominator =
          frequency + k1 * (1 - b + (b * length) / averageLength);
        bm25Scores[docId] =
          (bm25Scores[docId] || 0) +
          (idf * (frequency * (k1 + 1))) / denominator;
      }
    }

    const hybrid = index.hybrid || {};
    const semanticWeight = hybrid.semantic_weight ?? 0.6;
    const bm25Weight = hybrid.bm25_weight ?? 0.4;
    const rrfK = hybrid.rrf_k ?? 60;
    const scores = {};
    const addRrfScores = (channelScores, channelWeight) => {
      const ranked = Object.entries(channelScores).sort((a, b) => b[1] - a[1]);
      ranked.forEach(([docId], position) => {
        const rank = position + 1;
        scores[docId] = (scores[docId] || 0) + channelWeight / (rrfK + rank);
      });
    };
    addRrfScores(semanticScores, semanticWeight);
    addRrfScores(bm25Scores, bm25Weight);
    if (hybrid.fusion && hybrid.fusion !== "rrf") {
      console.warn("[SpladeX] Unknown fusion mode; using RRF", hybrid.fusion);
    }

    return Object.entries(scores)
      .sort(([, leftScore], [, rightScore]) => rightScore - leftScore)
      .slice(0, topK)
      .map(([docId]) => index.documents[docId])
      .filter(Boolean);
  }

  function renderLoading(query) {
    const container = findResultsContainer();
    container.innerHTML = `
      <h1>Search</h1>
      <p>Searching for <code>${escapeHtml(query)}</code>. Please wait...</p>
    `;
  }

  function renderError(query, error) {
    const container = findResultsContainer();
    container.innerHTML = `
      <h1>Search</h1>
      <p>Search is temporarily unavailable.</p>
    `;
    console.error("[SpladeX search error]", error);
  }

  function renderResults(query, results) {
    const container = findResultsContainer();

    container.innerHTML = `
      <h1>Search</h1>
      <p>Search results for <code>${escapeHtml(query)}</code></p>
    `;

    if (!query.trim()) {
      container.innerHTML += "<p>Type a query in the search box.</p>";
      return;
    }

    if (results.length === 0) {
      container.innerHTML += "<p>No results found.</p>";
      return;
    }

    const list = document.createElement("ul");
    list.className = "search";

    for (const result of results) {
      const item = document.createElement("li");

      item.innerHTML = `
        <a href="${escapeHtml(result.url)}">${escapeHtml(result.title)}</a>
        <div class="context">${escapeHtml(makeSnippet(result.text || ""))}</div>
      `;

      list.appendChild(item);
    }

    container.appendChild(list);
  }

  async function runSearch() {
    const query = getQueryFromUrl();

    const input = findSearchInput();
    if (input) {
      input.value = query;
    }

    if (!query.trim()) {
      renderResults(query, []);
      return;
    }

    renderLoading(query);

    try {
      const index = await loadIndex();
      const queryAssets = await loadQueryAssets(index);
      const results = searchLocalIndex(index, queryAssets, query, 10);
      renderResults(query, results);
    } catch (error) {
      renderError(query, error);
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    const path = window.location.pathname;
    // Match both html builder ("…/search.html") and dirhtml builder ("…/search/")
    if (path.endsWith("search.html") || path.endsWith("/search/") || path.endsWith("/search")) {
      runSearch().catch(console.error);
    }
  });
})();
