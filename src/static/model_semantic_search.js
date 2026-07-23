(function () {
  let cachedIndex = null;
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
      .then((index) => {
        cachedIndex = index;
        return index;
      });

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
    const matchedTerms = {};
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
        if (!matchedTerms[docId]) {
          matchedTerms[docId] = [];
        }
        matchedTerms[docId].push(term);
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
        if (!matchedTerms[docId]) matchedTerms[docId] = [];
        if (!matchedTerms[docId].includes(term)) matchedTerms[docId].push(term);
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

    const results = [];
    for (const [docId, score] of Object.entries(scores)) {
      const docInfo = index.documents[docId];
      if (docInfo) {
        results.push({
          id: docId,
          score: score,
          matched_terms: matchedTerms[docId],
          title: docInfo.title,
          url: docInfo.url,
          text: docInfo.text,
          granularity: docInfo.granularity,
          object_type: docInfo.object_type,
        });
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  function renderLoading(query) {
    const container = findResultsContainer();
    container.innerHTML = `
      <h1>Search</h1>
      <p>Searching for <code>${escapeHtml(query)}</code>...</p>
    `;
  }

  function injectToggle() {
    const container = findResultsContainer();
    if (!container) return;

    if (document.querySelector(".spladex-toggle-container")) return;

    const currentMode =
      localStorage.getItem("spladex_search_mode") || "semantic";

    const toggleDiv = document.createElement("div");
    toggleDiv.className = "spladex-toggle-container";
    toggleDiv.style.cssText =
      "margin-bottom: 25px; padding: 14px; background: #eef2f7; border: 1px solid #d0d7de; border-radius: 6px; display: inline-flex; gap: 10px; align-items: center; font-family: -apple-system, BlinkMacSystemFont, sans-serif;";

    toggleDiv.innerHTML = `
      <span style="font-weight: bold; font-size: 0.95em; color: #24292f; margin-right: 5px;">Search Engine:</span>
      <button class="spladex-btn semantic-btn" style="padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; font-size: 0.9em; font-weight: bold; transition: all 0.2s; ${currentMode === "semantic" ? "background: #0969da; color: white; border-color: #0969da;" : "background: white; color: #24292f;"}">
        SpladeX Semantic
      </button>
      <button class="spladex-btn normal-btn" style="padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; font-size: 0.9em; font-weight: bold; transition: all 0.2s; ${currentMode === "normal" ? "background: #0969da; color: white; border-color: #0969da;" : "background: white; color: #24292f;"}">
        Standard Sphinx
      </button>
    `;

    container.parentNode.insertBefore(toggleDiv, container);

    toggleDiv.querySelector(".semantic-btn").addEventListener("click", () => {
      localStorage.setItem("spladex_search_mode", "semantic");
      window.location.reload();
    });

    toggleDiv.querySelector(".normal-btn").addEventListener("click", () => {
      localStorage.setItem("spladex_search_mode", "normal");
      window.location.reload();
    });
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

    const list = document.createElement("ol");

    for (const result of results) {
      const item = document.createElement("li");

      item.innerHTML = `
        <p style="margin-bottom: 0.2em;">
          <a href="${escapeHtml(result.url)}">
            <strong>${escapeHtml(result.title)}</strong>
          </a>
          <span style="font-size: 0.8em; color: #0969da; margin-left: 10px; background: #ddf4ff; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-weight: bold;">
            Score: ${result.score.toFixed(4)}
          </span>
          <span style="font-size: 0.8em; color: #57606a; margin-left: 5px;">
            (matched: ${result.matched_terms.join(", ")})
          </span>
        </p>
        <p style="margin-top: 0.1em; color: #24292f; font-size: 0.95em; line-height: 1.5;">${escapeHtml(makeSnippet(result.text || ""))}</p>
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

    const currentMode =
      localStorage.getItem("spladex_search_mode") || "semantic";
    if (currentMode === "normal") {
      return;
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

  function interceptSearchForm() {
    const input = findSearchInput();
    if (!input) return;

    const form = input.closest("form");
    if (!form) return;

    form.addEventListener("submit", function (event) {
      event.preventDefault();

      const query = input.value || "";
      const url = new URL(window.location.href);

      url.pathname = url.pathname.replace(/[^/]*$/, "search.html");
      url.searchParams.set("q", query);

      window.location.href = url.toString();
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    interceptSearchForm();

    if (window.location.pathname.endsWith("search.html")) {
      injectToggle();
      runSearch().catch(console.error);

      setTimeout(() => {
        const currentMode =
          localStorage.getItem("spladex_search_mode") || "semantic";
        if (currentMode === "semantic") {
          runSearch().catch(console.error);
        }
      }, 300);
    }
  });
})();
