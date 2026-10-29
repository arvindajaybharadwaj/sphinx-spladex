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

  // Compact index ("spladex-compact"): posting lists are [deltaDocIds, values] and are
  // decoded on first use. Legacy indexes store [[docId, value], ...] directly.
  function lazyPostings(encoded, docIds, scale) {
    const decoded = Object.create(null);
    return {
      __lazy: true,
      keys: () => Object.keys(encoded),
      get(term) {
        if (term in decoded) return decoded[term];
        const entry = encoded[term];
        if (!entry) return [];
        const [deltas, values] = entry;
        const pairs = new Array(deltas.length);
        let docIndex = 0;
        for (let i = 0; i < deltas.length; i++) {
          docIndex += deltas[i];
          pairs[i] = [docIds[docIndex], scale ? values[i] * scale : values[i]];
        }
        decoded[term] = pairs;
        return pairs;
      },
    };
  }

  function getPostings(store, term) {
    if (!store) return [];
    if (store.__lazy) return store.get(term);
    return Object.prototype.hasOwnProperty.call(store, term) ? store[term] : [];
  }

  function getTerms(store) {
    if (!store) return [];
    return store.__lazy ? store.keys() : Object.keys(store);
  }

  function normaliseIndex(index) {
    if (index.kind !== "spladex-compact") return index;
    const docs = index.docs;
    const ids = docs.id;
    const documents = {};
    const lengths = {};
    for (let i = 0; i < ids.length; i++) {
      documents[ids[i]] = {
        title: docs.title[i],
        url: docs.url[i],
        text: docs.text[i],
        granularity: docs.granularity[i],
        object_type: docs.object_type[i],
      };
      lengths[ids[i]] = index.bm25.document_lengths[i];
    }
    index.documents = documents;
    index.inverted_index = lazyPostings(index.splade.postings, ids, index.splade.scale);
    index.bm25_index = {
      k1: index.bm25.k1,
      b: index.bm25.b,
      num_documents: index.bm25.num_documents,
      avg_document_length: index.bm25.avg_document_length,
      document_lengths: lengths,
      postings: lazyPostings(index.bm25.postings, ids, null),
    };
    return index;
  }

  function normaliseQueryAssets(assets) {
    const tokenizer = (assets && assets.tokenizer) || {};
    if (!tokenizer.vocab && Array.isArray(tokenizer.vocab_list)) {
      const vocab = Object.create(null);
      tokenizer.vocab_list.forEach((token, tokenId) => {
        if (token) vocab[token] = tokenId;
      });
      tokenizer.vocab = vocab;
    }
    return assets;
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
      .then(normaliseIndex);

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
    }).then(normaliseQueryAssets);
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

  // mode: "hybrid" (shipped RRF + title boost), "semantic" (SPLADE channel alone) or
  // "lexical" (BM25 channel alone). Each result carries the query terms it matched.
  function searchLocalIndex(index, queryAssets, query, topK = 100, mode = "hybrid") {
    const terms = query
      .toLowerCase()
      .replace(/[^\w\s_]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 0);

    const semanticScores = {};
    const bm25Scores = {};
    const semanticMatched = {};
    const bm25Matched = {};
    const addMatch = (store, docId, term) => {
      (store[docId] = store[docId] || new Set()).add(term);
    };
    const invertedIndex = index.inverted_index || {};
    const vocab = queryAssets.tokenizer.vocab || {};
    const weights = queryAssets.weights || [];
    const specialIds = new Set(queryAssets.special_token_ids || []);

    for (const term of wordPieceTokens(query, queryAssets)) {
      const tokenId = vocab[term];
      if (tokenId === undefined || specialIds.has(tokenId)) continue;
      const queryWeight = weights[tokenId] || 0;
      if (queryWeight <= 0) continue;
      const postings = getPostings(invertedIndex, term);
      for (const [docId, documentWeight] of postings) {
        semanticScores[docId] =
          (semanticScores[docId] || 0.0) + queryWeight * documentWeight;
        addMatch(semanticMatched, docId, term.replace(/^##/, ""));
      }
    }

    // Helper: get base stem of a word (e.g. 'changing' -> 'chang', 'repositioning' -> 'reposit')
    function getStem(w) {
      if (w.length <= 4) return w;
      return w.replace(/(ing|ed|es|s|tion|able|ive)$/, "");
    }

    const termStems = terms.map((t) => getStem(t)).filter((s) => s.length >= 3);

    const bm25 = index.bm25_index || {};
    const postingsByTerm = bm25.postings || {};
    const documentLengths = bm25.document_lengths || {};
    const numDocuments = bm25.num_documents || 0;
    const averageLength = bm25.avg_document_length || 1;
    const k1 = bm25.k1 || 1.2;
    const b = bm25.b || 0.75;

    // Collect all candidate index terms matching either exact term or subword stem
    const allIndexTerms = getTerms(postingsByTerm);

    for (let i = 0; i < terms.length; i++) {
      const term = terms[i];
      const stem = termStems[i] || term;

      // Find all index terms that match exact term or share the subword stem
      const matchingIndexTerms = allIndexTerms.filter((idxTerm) => {
        if (idxTerm === term) return true;
        if (stem.length >= 4 && idxTerm.startsWith(stem)) return true;
        return false;
      });

      for (const matchedTerm of matchingIndexTerms) {
        const postings = getPostings(postingsByTerm, matchedTerm);
        if (!postings.length) continue;

        // Discount partial stem matches slightly (0.75x) vs exact term matches (1.0x)
        const matchWeight = matchedTerm === term ? 1.0 : 0.75;

        const idf = Math.log(
          1 + (numDocuments - postings.length + 0.5) / (postings.length + 0.5),
        );

        for (const [docId, frequency] of postings) {
          const length = documentLengths[docId] || 0;
          const denominator =
            frequency + k1 * (1 - b + (b * length) / averageLength);
          bm25Scores[docId] =
            (bm25Scores[docId] || 0) +
            matchWeight * ((idf * (frequency * (k1 + 1))) / denominator);
          addMatch(bm25Matched, docId, term);
        }
      }
    }

    const hybrid = index.hybrid || {};
    const semanticWeight = hybrid.semantic_weight ?? 0.6;
    const bm25Weight = hybrid.bm25_weight ?? 0.4;
    const rrfK = hybrid.rrf_k ?? 60;
    let scores = {};
    const addRrfScores = (channelScores, channelWeight) => {
      const ranked = Object.entries(channelScores).sort((a, b) => b[1] - a[1]);
      ranked.forEach(([docId], position) => {
        const rank = position + 1;
        scores[docId] = (scores[docId] || 0) + channelWeight / (rrfK + rank);
      });
    };
    if (mode === "semantic") {
      scores = semanticScores;
    } else if (mode === "lexical") {
      scores = bm25Scores;
    } else {
      addRrfScores(semanticScores, semanticWeight);
      addRrfScores(bm25Scores, bm25Weight);
    }

    // Universal exact symbol title boost: if query contains exact symbol name (e.g. 'reshape'), boost primary function
    for (const [docId, doc] of mode === "hybrid" ? Object.entries(index.documents) : []) {
      if (scores[docId] && doc) {
        const titleLower = (doc.title || "").toLowerCase();
        const parts = titleLower.split(".");
        const shortName = parts[parts.length - 1];

        for (const term of terms) {
          if (shortName === term) {
            // Give 1.5x boost if title shortName matches exact query term
            scores[docId] *= 1.5;
            // Additional boost for top-level functions over subclass methods
            if (doc.object_type === "function") {
              scores[docId] *= 1.25;
            }
          }
        }
      }
    }

    const sortedDocs = Object.entries(scores)
      .sort(([, leftScore], [, rightScore]) => rightScore - leftScore)
      .slice(0, 50) // Cap at top 50 candidate documents
      .map(([docId]) => {
        const doc = index.documents[docId];
        if (!doc) return null;
        const matched = new Set([
          ...(mode !== "lexical" ? semanticMatched[docId] || [] : []),
          ...(mode !== "semantic" ? bm25Matched[docId] || [] : []),
        ]);
        return Object.assign({}, doc, { matched: Array.from(matched) });
      })
      .filter(Boolean);

    // Universal Deduplication across any Sphinx documentation site
    const deduplicated = [];
    const seenBaseUrls = new Set();
    const seenTitles = new Set();
    const seenShortNames = new Set();

    for (const doc of sortedDocs) {
      const url = doc.url || "";
      const baseUrl = url.split("#")[0];
      const title = doc.title || "";
      const titleParts = title.split(".");
      const shortName = titleParts[titleParts.length - 1];

      // 1. Skip exact duplicate titles (e.g. page 'numpy.isfortran' vs object 'numpy.isfortran')
      if (seenTitles.has(title)) {
        continue;
      }

      // 2. Skip redundant page records when an object anchor record on the same base page is already present
      if (seenBaseUrls.has(baseUrl) && doc.granularity === "page") {
        continue;
      }

      // 3. Skip duplicate subclass method clones sharing the exact same method shortname
      if (doc.granularity === "object" && shortName && seenShortNames.has(shortName)) {
        const count = Array.from(seenShortNames.values()).filter((n) => n === shortName).length;
        if (count >= 1) continue;
      }

      seenTitles.add(title);
      seenBaseUrls.add(baseUrl);
      if (doc.granularity === "object" && shortName) {
        seenShortNames.add(shortName);
      }

      deduplicated.push(doc);
    }

    return deduplicated;
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

  function applyThemeStyles(uiConfig) {
    if (document.getElementById("spladex-theme-styles")) return;
    const accent = (uiConfig && uiConfig.accent_color) || "#0066cc";
    const highlight = (uiConfig && uiConfig.highlight_color) || "#fff3a0";
    const badgeStyle = (uiConfig && uiConfig.badge_style) || "pill";

    const style = document.createElement("style");
    style.id = "spladex-theme-styles";
    style.textContent = `
      :root {
        --spladex-accent: ${accent};
        --spladex-highlight-bg: ${highlight};
      }
      .spladex-badge {
        display: ${badgeStyle === "none" ? "none" : "inline-block"};
        font-size: 0.72em;
        font-weight: 600;
        padding: 2px 7px;
        margin-left: 8px;
        border-radius: ${badgeStyle === "pill" ? "12px" : "3px"};
        background-color: var(--spladex-accent);
        color: #ffffff;
        vertical-align: middle;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        line-height: 1.2;
      }
      .spladex-highlight {
        background-color: var(--spladex-highlight-bg);
        padding: 1px 3px;
        border-radius: 2px;
      }
      .spladex-pagination a {
        color: var(--spladex-accent);
      }
      .spladex-matched {
        font-size: 0.8em;
        opacity: 0.7;
        margin-top: 2px;
      }
    `;
    document.head.appendChild(style);
  }

  function initKeyboardShortcuts(uiConfig) {
    if (!uiConfig || uiConfig.enable_keyboard_shortcut === false) return;
    document.addEventListener("keydown", function (e) {
      const activeElement = document.activeElement;
      const isInput =
        activeElement &&
        (activeElement.tagName === "INPUT" ||
          activeElement.tagName === "TEXTAREA" ||
          activeElement.isContentEditable);

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        const input = findSearchInput();
        if (input) input.focus();
      } else if (e.key === "/" && !isInput) {
        e.preventDefault();
        const input = findSearchInput();
        if (input) input.focus();
      }
    });
  }

  let currentUiConfig = {
    results_per_page: 10,
    highlight_color: "#fff3a0",
    accent_color: "#0066cc",
    badge_style: "pill",
    enable_keyboard_shortcut: true,
    enable_live_search: true,
  };

  let currentPage = 1;

  const ENGINE_MODES = [
    { id: "hybrid", label: "Hybrid (SpladeX + BM25)" },
    { id: "semantic", label: "SpladeX only" },
    { id: "lexical", label: "BM25 only" },
  ];
  let currentMode = "hybrid";
  try {
    currentMode = window.localStorage.getItem("spladex_engine_mode") || "hybrid";
  } catch (e) {
    currentMode = "hybrid";
  }

  // Optional engine switch (spladex_show_engine_toggle) for demonstrations and comparisons.
  function renderEngineToggle(container) {
    if (!currentUiConfig.show_engine_toggle) return;
    const bar = document.createElement("div");
    bar.className = "spladex-engine-toggle";
    bar.style.cssText = "display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:0 0 16px 0;";
    bar.innerHTML = '<span style="opacity:0.7; font-size:0.85em;">Engine:</span>';
    for (const mode of ENGINE_MODES) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = mode.label;
      const active = mode.id === currentMode;
      button.style.cssText =
        "padding:4px 12px; border-radius:999px; cursor:pointer; font-size:0.85em; border:1px solid var(--spladex-accent);" +
        (active ? "background:var(--spladex-accent); color:#fff;" : "background:transparent; color:inherit;");
      button.addEventListener("click", () => {
        if (mode.id === currentMode) return;
        currentMode = mode.id;
        try {
          window.localStorage.setItem("spladex_engine_mode", mode.id);
        } catch (e) {
          /* storage unavailable: keep the choice for this page only */
        }
        runSearch().catch(console.error);
      });
      bar.appendChild(button);
    }
    container.appendChild(bar);
  }

  function renderResults(query, results, page = 1) {
    currentPage = page;
    const container = findResultsContainer();
    const resultsPerPage = (currentUiConfig && currentUiConfig.results_per_page) || 10;

    container.innerHTML = `
      <h1>Search</h1>
      <p>Search results for <code>${escapeHtml(query)}</code></p>
    `;
    renderEngineToggle(container);

    if (!query.trim()) {
      container.innerHTML += "<p>Type a query in the search box.</p>";
      return;
    }

    if (results.length === 0) {
      container.innerHTML += "<p>No results found.</p>";
      return;
    }

    const totalPages = Math.ceil(results.length / resultsPerPage);
    const startIdx = (page - 1) * resultsPerPage;
    const pageResults = results.slice(startIdx, startIdx + resultsPerPage);

    const list = document.createElement("ul");
    list.className = "search";

    for (const result of pageResults) {
      const item = document.createElement("li");
      const badgeHtml = result.object_type
        ? `<span class="spladex-badge">${escapeHtml(result.object_type)}</span>`
        : "";

      const matchedHtml =
        currentUiConfig.show_matched_terms && result.matched && result.matched.length
          ? `<div class="spladex-matched">matched: ${escapeHtml(result.matched.join(", "))}</div>`
          : "";
      item.innerHTML = `
        <a href="${escapeHtml(result.url)}">${escapeHtml(result.title)}</a>${badgeHtml}
        <div class="context">${escapeHtml(makeSnippet(result.text || ""))}</div>
        ${matchedHtml}
      `;

      list.appendChild(item);
    }

    container.appendChild(list);

    // Normal, clean Sphinx-compatible pagination UI
    if (totalPages > 1) {
      const nav = document.createElement("div");
      nav.className = "spladex-pagination";
      nav.style.cssText = "margin-top: 20px; margin-bottom: 20px; font-size: 0.95em;";

      let navHtml = '<p style="display: flex; gap: 15px; align-items: center; margin: 0;">';

      if (page > 1) {
        navHtml += `<a href="#" id="spladex-prev" style="text-decoration: underline;">&laquo; Previous</a>`;
      } else {
        navHtml += `<span style="opacity: 0.5; cursor: not-allowed; user-select: none;">&laquo; Previous</span>`;
      }

      navHtml += `<span>Page ${page} of ${totalPages} (${results.length} results)</span>`;

      if (page < totalPages) {
        navHtml += `<a href="#" id="spladex-next" style="text-decoration: underline;">Next &raquo;</a>`;
      } else {
        navHtml += `<span style="opacity: 0.5; cursor: not-allowed; user-select: none;">Next &raquo;</span>`;
      }

      navHtml += "</p>";
      nav.innerHTML = navHtml;
      container.appendChild(nav);

      const prevLink = nav.querySelector("#spladex-prev");
      if (prevLink) {
        prevLink.onclick = (e) => {
          e.preventDefault();
          renderResults(query, results, page - 1);
          window.scrollTo({ top: 0, behavior: "smooth" });
        };
      }

      const nextLink = nav.querySelector("#spladex-next");
      if (nextLink) {
        nextLink.onclick = (e) => {
          e.preventDefault();
          renderResults(query, results, page + 1);
          window.scrollTo({ top: 0, behavior: "smooth" });
        };
      }
    }
  }

  async function runSearch() {
    const query = getQueryFromUrl();

    const input = findSearchInput();
    if (input) {
      input.value = query;
    }

    if (!query.trim()) {
      renderResults(query, [], 1);
      return;
    }

    renderLoading(query);

    try {
      const index = await loadIndex();
      if (index.ui_config) {
        currentUiConfig = Object.assign({}, currentUiConfig, index.ui_config);
      }
      applyThemeStyles(currentUiConfig);
      initKeyboardShortcuts(currentUiConfig);

      const queryAssets = await loadQueryAssets(index);
      const results = searchLocalIndex(index, queryAssets, query, 100, currentMode);
      renderResults(query, results, 1);
    } catch (error) {
      renderError(query, error);
    }
  }

  // Allow the ranking code to be exercised from Node (tests and offline evaluation).
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { normaliseIndex, normaliseQueryAssets, searchLocalIndex, wordPieceTokens };
  }
  if (typeof document === "undefined") return;

  document.addEventListener("DOMContentLoaded", function () {
    const path = window.location.pathname;

    // Load theme styles and keyboard shortcuts across all documentation pages if index is available
    loadIndex()
      .then((index) => {
        if (index && index.ui_config) {
          currentUiConfig = Object.assign({}, currentUiConfig, index.ui_config);
        }
        applyThemeStyles(currentUiConfig);
        initKeyboardShortcuts(currentUiConfig);
      })
      .catch(() => {});

    // Match both html builder ("…/search.html") and dirhtml builder ("…/search/")
    if (path.endsWith("search.html") || path.endsWith("/search/") || path.endsWith("/search")) {
      runSearch().catch(console.error);
    }
  });
})();
