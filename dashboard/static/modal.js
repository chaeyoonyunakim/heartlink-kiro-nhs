/**
 * HeartLink — lightweight graph info modal.
 *
 * Each graph image wrapped in a `.hl-graph` container becomes clickable.
 * Clicking opens a modal card showing the "What does this mean?" text
 * stored in the sibling `.hl-graph__info` element.
 */
(function () {
  "use strict";

  // Build modal overlay once
  var overlay = document.createElement("div");
  overlay.className = "hl-modal-overlay";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  overlay.setAttribute("aria-label", "Graph explanation");
  overlay.innerHTML =
    '<div class="hl-modal">' +
    '  <button class="hl-modal__close" aria-label="Close">&times;</button>' +
    '  <div class="hl-modal__body"></div>' +
    "</div>";
  document.body.appendChild(overlay);

  var modal = overlay.querySelector(".hl-modal");
  var body = overlay.querySelector(".hl-modal__body");
  var closeBtn = overlay.querySelector(".hl-modal__close");

  function openModal(html) {
    body.innerHTML = html;
    overlay.classList.add("hl-modal-overlay--open");
    document.body.style.overflow = "hidden";
    closeBtn.focus();
  }

  function closeModal() {
    overlay.classList.remove("hl-modal-overlay--open");
    document.body.style.overflow = "";
  }

  closeBtn.addEventListener("click", closeModal);
  overlay.addEventListener("click", function (e) {
    if (e.target === overlay) closeModal();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeModal();
  });

  // Attach click handlers to all .hl-graph containers
  var graphs = document.querySelectorAll(".hl-graph");
  for (var i = 0; i < graphs.length; i++) {
    (function (graph) {
      var img = graph.querySelector("img");
      var info = graph.querySelector(".hl-graph__info");
      if (!img || !info) return;

      img.style.cursor = "pointer";
      img.setAttribute("tabindex", "0");
      img.setAttribute("role", "button");
      img.setAttribute("aria-label", "Click to learn what this graph means");

      function handleClick() {
        var imgSrc = img.getAttribute("src");
        var caption = graph.querySelector("figcaption");
        var title = caption ? caption.textContent : "Graph explanation";
        var html =
          '<h3 class="nhsuk-heading-m">' + title + "</h3>" +
          '<img src="' + imgSrc + '" alt="" class="hl-modal__img">' +
          '<div class="hl-modal__text">' + info.innerHTML + "</div>";
        openModal(html);
      }

      img.addEventListener("click", handleClick);
      img.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleClick();
        }
      });
    })(graphs[i]);
  }
})();
