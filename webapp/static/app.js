/* ============================================================
   HierAD · 共享脚本：深浅色切换 + 顶栏用户槽
   在 <head> 里同步加载（不要 defer），否则首屏会闪一下错误配色。
   landing.html / features.html / index.html / login.html 共用；
   主题键名 hierad-theme 必须一致。

   登录页给 <body> 加 data-no-user-slot，跳过用户槽注入。
   ============================================================ */
(() => {
  const KEY = "hierad-theme";
  const root = document.documentElement;

  // 尽早应用已保存的选择
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (_) {}
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);

  function setupTheme() {
    const btn = document.getElementById("themeBtn");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const cur = root.getAttribute("data-theme")
        || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = cur === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem(KEY, next); } catch (_) {}
    });
  }

  function logout() {
    fetch("/api/auth/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: "{}"
    }).catch(() => {}).finally(() => {
      location.href = "/login";
    });
  }

  function renderUser(user) {
    const nav = document.querySelector(".topbar-right");
    if (!nav) return;
    const slot = document.createElement("span");
    slot.className = "user-slot";

    const name = document.createElement("span");
    name.className = "user-chip";
    name.textContent = user.username;
    name.title = user.is_admin ? "管理员" : "普通用户";
    slot.appendChild(name);

    if (user.is_admin) {
      const badge = document.createElement("span");
      badge.className = "user-badge";
      badge.textContent = "管理员";
      slot.appendChild(badge);
    }

    const out = document.createElement("button");
    out.type = "button";
    out.className = "navlink user-logout";
    out.textContent = "退出";
    out.addEventListener("click", logout);
    slot.appendChild(out);

    const themeBtn = document.getElementById("themeBtn");
    if (themeBtn && themeBtn.parentNode === nav) nav.insertBefore(slot, themeBtn);
    else nav.appendChild(slot);
  }

  function renderLoginLink() {
    const nav = document.querySelector(".topbar-right");
    if (!nav) return;
    const link = document.createElement("a");
    link.className = "navlink";
    link.href = "/login";
    link.textContent = "登录";
    const themeBtn = document.getElementById("themeBtn");
    if (themeBtn && themeBtn.parentNode === nav) nav.insertBefore(link, themeBtn);
    else nav.appendChild(link);
  }

  function setupUserSlot() {
    if (document.body.hasAttribute("data-no-user-slot")) return;
    if (!document.querySelector(".topbar-right")) return;
    fetch("/api/me", { credentials: "same-origin" })
      .then((r) => (r.ok ? r.json() : null))
      .then((user) => {
        if (user && user.username) renderUser(user);
        else renderLoginLink();
      })
      .catch(() => renderLoginLink());
  }

  document.addEventListener("DOMContentLoaded", () => {
    setupTheme();
    setupUserSlot();
  });
})();
