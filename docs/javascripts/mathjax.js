window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex|jp-RenderedHTMLCommon|jp-RenderedMarkdown|cell_output|rendered_html|text_cell_render|markdown"
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      // Re-typeset after page load for dynamically rendered notebook content
      setTimeout(() => MathJax.typesetPromise(), 500);
    }
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
