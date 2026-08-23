// Hover popover for ARC game ids: shows the game's initial-frame thumbnail and
// links to the interactive play page on arcprize.org.
//
// Usage: mark any element with data-game-peek="<game id>" (full or base id).
// Call window.installGamePeek() once; dynamically rendered elements are
// picked up automatically because the handlers are delegated from <body>.
(function () {
  const PLAY_BASE_URL = "https://arcprize.org/tasks/";
  // Resolve the thumbnail API against the viewer that served this script so the
  // popover keeps working when the page is opened from another origin.
  const SCRIPT_ORIGIN = (() => {
    try { return new URL(document.currentScript.src, window.location.href).origin; } catch (_) { return window.location.origin; }
  })();
  const SHOW_DELAY_MS = 120;
  const HIDE_DELAY_MS = 180;
  const CSS = `
    .game-peek { position: fixed; z-index: 9999; width: 200px; padding: 10px; border-radius: 12px;
      background: #fffaf1; color: #18242d; border: 1px solid rgba(24, 36, 45, 0.14);
      box-shadow: 0 12px 30px rgba(24, 36, 45, 0.22); font: 500 0.74rem "IBM Plex Sans", system-ui, sans-serif;
      display: grid; gap: 7px; pointer-events: auto; }
    .game-peek[hidden] { display: none; }
    .game-peek img { display: block; width: 100%; aspect-ratio: 1 / 1; border-radius: 8px;
      image-rendering: pixelated; image-rendering: crisp-edges; background: #1f2327; }
    .game-peek .game-peek-missing { display: grid; place-items: center; width: 100%; aspect-ratio: 1 / 1;
      border-radius: 8px; background: rgba(24, 36, 45, 0.06); color: #6b7680; font-size: 0.7rem; text-align: center; padding: 8px; }
    .game-peek .game-peek-id { font: 700 0.8rem "IBM Plex Mono", "SFMono-Regular", monospace; }
    .game-peek .game-peek-link { color: #c55d1f; font-weight: 700; text-decoration: none; }
    .game-peek .game-peek-link:hover { text-decoration: underline; }
    [data-game-peek] { cursor: help; }
    a[data-game-peek] { cursor: pointer; }
  `;

  function baseId(gameId) {
    const base = String(gameId || "").trim().toLowerCase().split("-")[0];
    return /^[a-z0-9]{4}$/.test(base) ? base : null;
  }

  function playUrl(gameId) {
    const base = baseId(gameId);
    return PLAY_BASE_URL + encodeURIComponent(base || String(gameId || ""));
  }

  function thumbnailUrl(gameId) {
    return SCRIPT_ORIGIN + "/api/thumbnail?game=" + encodeURIComponent(baseId(gameId) || String(gameId || ""));
  }

  function install() {
    if (document.getElementById("game-peek")) return;
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const popover = document.createElement("div");
    popover.id = "game-peek";
    popover.className = "game-peek";
    popover.hidden = true;
    popover.setAttribute("role", "tooltip");
    document.body.appendChild(popover);

    let anchor = null;
    let showTimer = null;
    let hideTimer = null;
    let currentGame = null;

    function render(gameId) {
      const id = String(gameId);
      const link = document.createElement("a");
      link.className = "game-peek-link";
      link.href = playUrl(id);
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "Play on arcprize.org ↗";
      const label = document.createElement("div");
      label.className = "game-peek-id";
      label.textContent = id;
      const image = document.createElement("img");
      image.alt = "Initial frame of " + id;
      image.src = thumbnailUrl(id);
      image.addEventListener("error", () => {
        const missing = document.createElement("div");
        missing.className = "game-peek-missing";
        missing.textContent = "No thumbnail (offline env files unavailable)";
        image.replaceWith(missing);
      });
      popover.replaceChildren(image, label, link);
    }

    function position() {
      if (!anchor) return;
      const rect = anchor.getBoundingClientRect();
      const width = popover.offsetWidth || 220;
      const height = popover.offsetHeight || 260;
      let left = rect.right + 10;
      if (left + width > window.innerWidth - 8) left = Math.max(8, rect.left - width - 10);
      let top = rect.top - 6;
      if (top + height > window.innerHeight - 8) top = Math.max(8, window.innerHeight - height - 8);
      popover.style.left = left + "px";
      popover.style.top = top + "px";
    }

    function show(target) {
      const gameId = target.dataset.gamePeek;
      if (!gameId) return;
      anchor = target;
      if (currentGame !== gameId) {
        currentGame = gameId;
        render(gameId);
      }
      popover.hidden = false;
      position();
    }

    function hide() {
      popover.hidden = true;
      anchor = null;
    }

    function scheduleShow(target) {
      clearTimeout(hideTimer);
      clearTimeout(showTimer);
      showTimer = setTimeout(() => show(target), SHOW_DELAY_MS);
    }

    function scheduleHide() {
      clearTimeout(showTimer);
      clearTimeout(hideTimer);
      hideTimer = setTimeout(hide, HIDE_DELAY_MS);
    }

    document.body.addEventListener("mouseover", (event) => {
      if (popover.contains(event.target)) { clearTimeout(hideTimer); return; }
      const target = event.target.closest("[data-game-peek]");
      if (target) scheduleShow(target);
    });
    document.body.addEventListener("mouseout", (event) => {
      const related = event.relatedTarget;
      if (related && (popover.contains(related) || (anchor && anchor.contains(related)))) return;
      const target = event.target.closest("[data-game-peek]");
      if (target || popover.contains(event.target)) scheduleHide();
    });
    document.body.addEventListener("focusin", (event) => {
      const target = event.target.closest("[data-game-peek]");
      if (target) scheduleShow(target);
    });
    document.body.addEventListener("focusout", (event) => {
      const target = event.target.closest("[data-game-peek]");
      if (target) scheduleHide();
    });
    window.addEventListener("scroll", () => { if (!popover.hidden) position(); }, true);
    window.addEventListener("resize", () => { if (!popover.hidden) position(); });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") hide(); });
  }

  window.installGamePeek = install;
  window.gamePlayUrl = playUrl;
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install);
  } else {
    install();
  }
})();
