// Suppress mkdocs-jupyter's MathJax v2 config (it injects MathJax.Hub.Config
// which doesn't exist in MathJax v3, causing silent failures)
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams"
  },
  options: {
    // Process math in all elements, not just arithmatex spans
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "tex2jax_process|arithmatex|jp-RenderedHTMLCommon|jp-RenderedMarkdown|rendered_html|text_cell_render|markdown|cell_output"
  },
  startup: {
    ready: function () {
      MathJax.startup.defaultReady();
      // Typeset the whole document after load
      MathJax.startup.promise.then(function () {
        MathJax.typesetPromise();
      });
    }
  }
};

// Stub out MathJax.Hub to prevent v2 config errors from mkdocs-jupyter
window.MathJax.Hub = {
  Config: function() {},
  Queue: function() {},
  Typeset: function() {},
  Register: { StartupHook: function() {} }
};

// Re-typeset on navigation (Material theme instant loading)
if (typeof document$ !== 'undefined') {
  document$.subscribe(function () {
    if (window.MathJax && window.MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
  });
}
