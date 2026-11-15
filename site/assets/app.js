(function () {
  "use strict";

  var doc = document;
  var root = doc.documentElement;

  function ready(fn) {
    if (doc.readyState === "loading") {
      doc.addEventListener("DOMContentLoaded", fn, { once: true });
    } else {
      fn();
    }
  }

  function setupYear() {
    var year = doc.querySelector("[data-year]");
    if (year) {
      year.textContent = String(new Date().getFullYear());
    }
  }

  function setupHeader() {
    var header = doc.querySelector("[data-header]");
    var toggle = doc.querySelector("[data-nav-toggle]");
    var nav = doc.getElementById("site-nav");

    if (!header) {
      return;
    }

    var onScroll = function () {
      header.classList.toggle("is-scrolled", window.scrollY > 8);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    if (!toggle || !nav) {
      return;
    }

    var setOpen = function (open) {
      header.classList.toggle("nav-open", open);
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    };

    toggle.addEventListener("click", function () {
      setOpen(toggle.getAttribute("aria-expanded") !== "true");
    });

    nav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        setOpen(false);
      });
    });

    doc.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && header.classList.contains("nav-open")) {
        setOpen(false);
        toggle.focus();
      }
    });

    doc.addEventListener("click", function (event) {
      if (!header.classList.contains("nav-open")) {
        return;
      }
      if (!header.contains(event.target)) {
        setOpen(false);
      }
    });

    window.addEventListener("resize", function () {
      if (window.innerWidth > 900) {
        setOpen(false);
      }
    });
  }

  function setupCastFilter() {
    var buttons = Array.from(doc.querySelectorAll("[data-speaker-filter]"));
    var lines = Array.from(doc.querySelectorAll("[data-line-speaker]"));
    var status = doc.querySelector("[data-cast-status]");

    if (!buttons.length || !lines.length) {
      return;
    }

    var names = {
      all: "all five voices",
      narrator: "the narrator",
      mara: "Mara",
      elias: "Elias",
      mom: "Mom"
    };

    var apply = function (speaker) {
      buttons.forEach(function (button) {
        button.setAttribute(
          "aria-pressed",
          String(button.getAttribute("data-speaker-filter") === speaker)
        );
      });

      lines.forEach(function (line) {
        var match = speaker === "all" || line.getAttribute("data-line-speaker") === speaker;
        line.classList.toggle("is-dimmed", !match);
      });

      if (status) {
        status.textContent =
          speaker === "all"
            ? "Showing all five voices."
            : "Highlighting " + (names[speaker] || speaker) + ".";
      }
    };

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        apply(button.getAttribute("data-speaker-filter") || "all");
      });
    });
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }

    return new Promise(function (resolve, reject) {
      var area = doc.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.top = "-1000px";
      area.style.opacity = "0";
      doc.body.appendChild(area);
      area.select();

      try {
        var copied = doc.execCommand("copy");
        area.remove();
        copied ? resolve() : reject(new Error("copy failed"));
      } catch (error) {
        area.remove();
        reject(error);
      }
    });
  }

  function setupCopyButtons() {
    doc.querySelectorAll("[data-copy-target]").forEach(function (button) {
      var original = button.textContent;
      var timer = 0;

      button.addEventListener("click", function () {
        var target = doc.getElementById(button.getAttribute("data-copy-target"));
        if (!target) {
          return;
        }

        copyText(target.textContent.trim())
          .then(function () {
            window.clearTimeout(timer);
            button.textContent = "Copied";
            button.classList.add("is-copied");
            timer = window.setTimeout(function () {
              button.textContent = original;
              button.classList.remove("is-copied");
            }, 1800);
          })
          .catch(function () {
            window.clearTimeout(timer);
            button.textContent = "Select & copy";
            timer = window.setTimeout(function () {
              button.textContent = original;
            }, 1800);
          });
      });
    });
  }

  function setupRepoLinks() {
    var config = window.BOOKCAST_CONFIG || {};
    var url = typeof config.repoUrl === "string" ? config.repoUrl.trim() : "";

    doc.querySelectorAll("[data-repo-link]").forEach(function (link) {
      if (url) {
        link.href = url;
        link.hidden = false;
      } else {
        link.hidden = true;
      }
    });
  }

  function setupActiveNav() {
    if (!("IntersectionObserver" in window)) {
      return;
    }

    var links = Array.from(doc.querySelectorAll('.site-nav a[href^="#"]'));
    if (!links.length) {
      return;
    }

    var byId = new Map();
    links.forEach(function (link) {
      byId.set(link.getAttribute("href").slice(1), link);
    });

    var sections = Array.from(byId.keys())
      .map(function (id) {
        return doc.getElementById(id);
      })
      .filter(Boolean);

    if (!sections.length) {
      return;
    }

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) {
            return;
          }
          var link = byId.get(entry.target.id);
          if (!link) {
            return;
          }
          links.forEach(function (other) {
            other.removeAttribute("aria-current");
          });
          link.setAttribute("aria-current", "true");
        });
      },
      { rootMargin: "-35% 0px -60% 0px", threshold: 0 }
    );

    sections.forEach(function (section) {
      observer.observe(section);
    });
  }

  ready(function () {
    root.classList.add("is-ready");
    setupYear();
    setupHeader();
    setupCastFilter();
    setupCopyButtons();
    setupRepoLinks();
    setupActiveNav();
  });
})();
