/* ============================================================
   HierAD · 共享脚本：深浅色切换
   在 <head> 里同步加载（不要 defer），否则首屏会闪一下错误配色。
   index.html 与 landing.html 共用；主题键名 hierad-theme 必须一致。
   ============================================================ */
(() => {
  const KEY = "hierad-theme";
  const root = document.documentElement;

  // 尽早应用已保存的选择
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (_) {}
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("themeBtn");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const cur = root.getAttribute("data-theme")
        || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = cur === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem(KEY, next); } catch (_) {}
    });
  });
})();
