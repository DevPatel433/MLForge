window.MathJax = {
  tex: { inlineMath: [["\\(", "\\)"], ["$", "$"]], displayMath: [["\\[", "\\]"], ["$$", "$$"]], processEscapes: true },
  options: { ignoreHtmlClass: "no-mathjax|^((?!arithmatex).)*$", processHtmlClass: "arithmatex" }
};
document$.subscribe(() => { MathJax.startup.output.clearCache(); MathJax.typesetClear(); MathJax.texReset(); MathJax.typesetPromise(); });
