/* Kaggle 案例拆解 · 共用脚本 CaseKit
 * 无依赖，file:// 直接打开可用。页面先加载 cases/nb/<slug>.nb.js（window.CASE_NB），再加载本文件。
 *
 * 常用 API（详见 tools/AUTHORING.md）：
 *   CaseKit.pipeline(el, steps)          步骤动画播放器
 *   CaseKit.lab(el, {controls, render, readout})   滑块交互实验
 *   CaseKit.cell(i, {highlight, collapse})          原 notebook 第 i 格（返回元素）
 *   CaseKit.notebook(el)                 原 notebook 全部单元格浏览
 *   CaseKit.svg(parent, tag, attrs)      建 SVG 元素；fill/stroke 可写 'var(--c1)'
 *   CaseKit.axes(svg, box, xdom, ydom, opt)  坐标轴，返回 {sx, sy}
 *   CaseKit.rng(seed) / CaseKit.randn(r)  可复现随机数
 *   CaseKit.tween(ms, fn, ctx)           动画插值，ctx 来自 pipeline 的 draw 参数
 */
(function () {
  "use strict";
  const K = (window.CaseKit = {});
  const NS = "http://www.w3.org/2000/svg";

  /* ---------- 基础工具 ---------- */
  K.esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  K.el = function (tag, attrs, html) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) { if (k === "class") e.className = attrs[k]; else e.setAttribute(k, attrs[k]); }
    if (html != null) e.innerHTML = html;
    return e;
  };
  K.rng = function (seed) {
    let a = (seed >>> 0) || 1;
    return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };
  K.randn = function (r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  K.clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  K.lerp = (a, b, t) => a + (b - a) * t;
  K.ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  K.fmt = (x, d = 2) => { const v = Math.abs(x) < 0.5 * Math.pow(10, -d) ? 0 : +x; return Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(d); };
  K.sigmoid = (z) => 1 / (1 + Math.exp(-z));
  K.css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  K.colors = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"];

  /* ---------- SVG ---------- */
  const STYLE_ATTRS = { fill: 1, stroke: 1, opacity: 1, "stroke-width": 1, "font-size": 1, "font-weight": 1, "text-anchor": 1, "dominant-baseline": 1, "stroke-dasharray": 1, "fill-opacity": 1, "stroke-opacity": 1, "font-family": 1, "font-style": 1 };
  K.svg = function (parent, tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      if (v == null) continue;
      if (k === "class") e.setAttribute("class", v);
      else if (STYLE_ATTRS[k]) e.style.setProperty(k, typeof v === "number" && k === "font-size" ? v + "px" : String(v));
      else e.setAttribute(k, v);
    }
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  };
  K.set = function (e, attrs) {
    for (const k in attrs) {
      const v = attrs[k];
      if (STYLE_ATTRS[k]) e.style.setProperty(k, typeof v === "number" && k === "font-size" ? v + "px" : String(v));
      else e.setAttribute(k, v);
    }
    return e;
  };
  K.stage = function (container, w, h) {
    container.innerHTML = "";
    return K.svg(container, "svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
  };
  K.scale = (d0, d1, r0, r1) => (x) => r0 + ((x - d0) / (d1 - d0 || 1)) * (r1 - r0);
  /* box = {x, y, w, h}; 返回 {sx, sy, g} */
  K.axes = function (svg, box, xdom, ydom, opt) {
    opt = opt || {};
    const g = K.svg(svg, "g", { class: "axis" });
    const sx = K.scale(xdom[0], xdom[1], box.x, box.x + box.w);
    const sy = K.scale(ydom[0], ydom[1], box.y + box.h, box.y);
    K.svg(g, "line", { x1: box.x, y1: box.y + box.h, x2: box.x + box.w, y2: box.y + box.h, stroke: "var(--ink-3)" });
    K.svg(g, "line", { x1: box.x, y1: box.y, x2: box.x, y2: box.y + box.h, stroke: "var(--ink-3)" });
    const nx = opt.xticks == null ? 5 : opt.xticks, ny = opt.yticks == null ? 4 : opt.yticks;
    const fx = opt.xfmt || ((v) => K.fmt(v, Math.abs(xdom[1] - xdom[0]) < 10 ? 1 : 0));
    const fy = opt.yfmt || ((v) => K.fmt(v, Math.abs(ydom[1] - ydom[0]) < 10 ? 1 : 0));
    for (let i = 0; i <= nx && nx > 0; i++) {
      const v = xdom[0] + ((xdom[1] - xdom[0]) * i) / nx;
      K.svg(g, "text", { x: sx(v), y: box.y + box.h + 15, "text-anchor": "middle" }, fx(v));
    }
    for (let i = 0; i <= ny && ny > 0; i++) {
      const v = ydom[0] + ((ydom[1] - ydom[0]) * i) / ny;
      K.svg(g, "line", { x1: box.x, y1: sy(v), x2: box.x + box.w, y2: sy(v), stroke: "var(--line)", "stroke-dasharray": "2 3" });
      K.svg(g, "text", { x: box.x - 6, y: sy(v) + 4, "text-anchor": "end" }, fy(v));
    }
    if (opt.xlabel) K.svg(g, "text", { x: box.x + box.w / 2, y: box.y + box.h + 32, "text-anchor": "middle", "font-size": 12 }, opt.xlabel);
    if (opt.ylabel) K.svg(g, "text", { x: box.x - 36, y: box.y + box.h / 2, "text-anchor": "middle", "font-size": 12, transform: `rotate(-90 ${box.x - 36} ${box.y + box.h / 2})` }, opt.ylabel);
    return { sx, sy, g };
  };
  /* 简单水平/竖直条形图：items=[{label, value, color}] */
  K.bars = function (svg, box, items, opt) {
    opt = opt || {};
    const max = opt.max || Math.max(...items.map((d) => d.value)) || 1;
    const g = K.svg(svg, "g");
    const n = items.length, gap = opt.gap == null ? 8 : opt.gap;
    const els = [];
    if (opt.horizontal) {
      const bh = (box.h - gap * (n - 1)) / n, lw = opt.labelWidth || 90;
      items.forEach((d, i) => {
        const y = box.y + i * (bh + gap);
        K.svg(g, "text", { x: box.x + lw - 8, y: y + bh / 2 + 4, "text-anchor": "end", "font-size": 12 }, d.label);
        const r = K.svg(g, "rect", { x: box.x + lw, y, width: ((box.w - lw - 50) * d.value) / max, height: bh, rx: 3, fill: d.color || "var(--c1)" });
        const t = K.svg(g, "text", { x: box.x + lw + ((box.w - lw - 50) * d.value) / max + 6, y: y + bh / 2 + 4, "font-size": 12, class: "muted" }, opt.fmt ? opt.fmt(d.value) : K.fmt(d.value));
        els.push({ rect: r, text: t });
      });
    } else {
      const bw = (box.w - gap * (n - 1)) / n;
      items.forEach((d, i) => {
        const x = box.x + i * (bw + gap), h = (box.h * d.value) / max;
        const r = K.svg(g, "rect", { x, y: box.y + box.h - h, width: bw, height: h, rx: 3, fill: d.color || "var(--c1)" });
        K.svg(g, "text", { x: x + bw / 2, y: box.y + box.h + 15, "text-anchor": "middle", "font-size": 12 }, d.label);
        const t = K.svg(g, "text", { x: x + bw / 2, y: box.y + box.h - h - 5, "text-anchor": "middle", "font-size": 12, class: "muted" }, opt.fmt ? opt.fmt(d.value) : K.fmt(d.value));
        els.push({ rect: r, text: t });
      });
    }
    return els;
  };
  /* 带圆角的方框+文字（流程图节点） */
  K.node = function (svg, x, y, w, h, label, opt) {
    opt = opt || {};
    const g = K.svg(svg, "g");
    K.svg(g, "rect", { x, y, width: w, height: h, rx: opt.rx == null ? 8 : opt.rx, fill: opt.fill || "var(--panel-2)", stroke: opt.stroke || "var(--line)", "stroke-width": opt.sw || 1.2 });
    const lines = String(label).split("\n");
    lines.forEach((ln, i) => K.svg(g, "text", { x: x + w / 2, y: y + h / 2 + (i - (lines.length - 1) / 2) * 15 + 4, "text-anchor": "middle", "font-size": opt.fs || 13, "font-weight": opt.bold ? 700 : 400, fill: opt.color || "var(--ink)" }, ln));
    return g;
  };
  K.arrow = function (svg, x1, y1, x2, y2, opt) {
    opt = opt || {};
    const color = opt.color || "var(--ink-3)";
    const g = K.svg(svg, "g");
    K.svg(g, "line", { x1, y1, x2, y2, stroke: color, "stroke-width": opt.sw || 1.5, "stroke-dasharray": opt.dash || null });
    const a = Math.atan2(y2 - y1, x2 - x1), s = 7;
    K.svg(g, "path", { d: `M${x2},${y2} L${x2 - s * Math.cos(a - 0.4)},${y2 - s * Math.sin(a - 0.4)} L${x2 - s * Math.cos(a + 0.4)},${y2 - s * Math.sin(a + 0.4)} Z`, fill: color });
    return g;
  };
  /* 热力图色：t∈[0,1]，diverging=true 时 t∈[-1,1] */
  K.heat = function (t, diverging) {
    if (diverging) {
      t = K.clamp(t, -1, 1);
      return t >= 0 ? `color-mix(in srgb, var(--c2) ${Math.round(t * 100)}%, var(--panel))` : `color-mix(in srgb, var(--c1) ${Math.round(-t * 100)}%, var(--panel))`;
    }
    return `color-mix(in srgb, var(--c1) ${Math.round(K.clamp(t, 0, 1) * 100)}%, var(--panel))`;
  };

  /* ---------- 动画 ---------- */
  /* ctx: {alive()}；返回 Promise，被打断时 resolve(false) */
  K.tween = function (ms, fn, ctx) {
    return new Promise((res) => {
      const t0 = performance.now();
      function tick(now) {
        if (ctx && !ctx.alive()) return res(false);
        const t = K.clamp((now - t0) / ms, 0, 1);
        fn(K.ease(t), t);
        if (t < 1) requestAnimationFrame(tick); else res(true);
      }
      requestAnimationFrame(tick);
    });
  };
  K.wait = (ms, ctx) => new Promise((res) => setTimeout(() => res(!ctx || ctx.alive()), ms));

  /* ---------- 代码高亮（Python） ---------- */
  const KW = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "));
  const BI = new Set("print len range enumerate zip list dict set tuple int float str sum min max abs sorted map filter open round isinstance super type self".split(" "));
  K.hl = function (src) {
    const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rRbBfFuU]{0,2}"(?:\\.|[^"\\\n])*"|[rRbBfFuU]{0,2}'(?:\\.|[^'\\\n])*')|(@[\w.]+)|(\b\d+(?:\.\d+)?(?:e[-+]?\d+)?\b)|([A-Za-z_]\w*)(?=\s*\()|([A-Za-z_]\w*)/g;
    let out = "", last = 0, m;
    while ((m = re.exec(src))) {
      out += K.esc(src.slice(last, m.index));
      const t = m[0];
      if (m[1]) out += `<span class="tk-com">${K.esc(t)}</span>`;
      else if (m[2]) out += `<span class="tk-str">${K.esc(t)}</span>`;
      else if (m[3]) out += `<span class="tk-dec">${K.esc(t)}</span>`;
      else if (m[4]) out += `<span class="tk-num">${K.esc(t)}</span>`;
      else if (m[5]) out += KW.has(t) ? `<span class="tk-kw">${t}</span>` : `<span class="tk-fn">${K.esc(t)}</span>`;
      else out += KW.has(t) ? `<span class="tk-kw">${t}</span>` : BI.has(t) ? `<span class="tk-fn">${t}</span>` : K.esc(t);
      last = re.lastIndex;
    }
    return out + K.esc(src.slice(last));
  };

  /* ---------- Markdown（安全的极简版：先去掉 HTML 标签再转义） ---------- */
  // 数学公式：$$…$$、\[…\]、\(…\)、$…$（开头和结尾紧挨着内容，避免把“$5 和 $10”当公式）。代码段里的不动。
  const MATH_RE = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$(?![\s$])([^$\n]+?)(?<![\s\\])\$|(\\begin\{(equation|align|gather|eqnarray|multline|split)\*?\}[\s\S]+?\\end\{\6\*?\})/g;
  function pullMath(src, store) {
    return String(src).split(/(```[\s\S]*?```|`[^`\n]*`)/).map((part, i) => i % 2 ? part : part.replace(MATH_RE, (m, a, b, c, d, e) => {
      const t = a || b || c || d || e;
      if (!/[\\^_={}]/.test(t) && /[A-Za-z]{3,}(\s+\S+){5,}/.test(t)) return m; // 一大段英文、没有任何数学符号：是正文，不是公式
      store.push({ tex: t, display: !!(a || b || e) });
      return `@@MATH${store.length - 1}@@`;
    })).join("");
  }
  // 把 &nbsp; &amp; &#10514; 这类实体还原成字符（textarea 只解析实体，不执行任何东西）
  const entBox = document.createElement("textarea");
  function decodeEntities(t) {
    if (!/&(#\d+|#x[0-9a-f]+|[a-z]+);/i.test(t)) return t;
    entBox.innerHTML = t.replace(/</g, "&lt;");
    return entBox.value.replace(/\u00a0/g, " ");
  }
  K.md = function (src) {
    const maths = [];
    src = pullMath(src, maths);
    let s = String(src)
      .replace(/<\s*h([1-6])[^>]*>([\s\S]*?)<\s*\/\s*h\1\s*>/gi, (m, n, t) => "\n" + "#".repeat(+n) + " " + t.replace(/\n/g, " ") + "\n")
      .replace(/<\s*(b|strong)\s*>([\s\S]*?)<\s*\/\s*\1\s*>/gi, "**$2**")
      .replace(/<\s*(i|em)\s*>([\s\S]*?)<\s*\/\s*\1\s*>/gi, "*$2*")
      .replace(/<\s*li[^>]*>/gi, "\n- ")
      .replace(/<\s*br\s*\/?>/gi, "\n")
      .replace(/<\s*\/?\s*p[^>]*>/gi, "\n")
      .replace(/<\s*img[^>]*>/gi, "［图片］")
      .replace(/<\s*a[^>]*>([\s\S]*?)<\s*\/\s*a\s*>/gi, "$1")
      .replace(/<\/?[a-zA-Z!][^<>]*>/g, "")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, "［图片］");
    s = K.esc(decodeEntities(s));
    const lines = s.split("\n");
    let html = "", inList = false, inCode = false, code = [];
    const inline = (t) => t
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      .replace(/\[([^\]]+)\]\((?!https?:)[^)\s]*\)/g, "$1");
    let tbl = [];
    const flushTbl = () => {
      if (!tbl.length) return;
      const rows = tbl.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r)).map((r) => r.trim().replace(/^\||\|$/g, "").split("|").map((x) => inline(x.trim())));
      html += '<div class="scroll-x"><table class="res">' + rows.map((r, i) => "<tr>" + r.map((x) => (i ? `<td>${x}</td>` : `<th>${x}</th>`)).join("") + "</tr>").join("") + "</table></div>";
      tbl = [];
    };
    for (const raw of lines) {
      const ln = raw.replace(/\s+$/, "");
      if (!inCode && /^\s*\|.*\|\s*$/.test(ln)) { if (inList) { html += "</ul>"; inList = false; } tbl.push(ln); continue; }
      flushTbl();
      const q = !inCode && ln.match(/^\s*&gt;\s?(.*)$/);
      if (q) { if (inList) { html += "</ul>"; inList = false; } html += `<blockquote>${inline(q[1])}</blockquote>`; continue; }
      if (/^```/.test(ln.trim())) {
        if (inCode) { html += `<pre>${code.join("\n")}</pre>`; code = []; inCode = false; } else { if (inList) { html += "</ul>"; inList = false; } inCode = true; }
        continue;
      }
      if (inCode) { code.push(ln); continue; }
      const h = ln.match(/^\s*(#{1,6})\s+(.*)$/);
      const li = ln.match(/^\s*[-*+]\s+(.*)$/) || ln.match(/^\s*\d+[.)]\s+(.*)$/);
      if (li) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${inline(li[1])}</li>`; continue; }
      if (inList && ln.trim() === "") continue;
      if (inList) { html += "</ul>"; inList = false; }
      if (h) { const n = Math.min(4, h[1].length); html += `<h${n}>${inline(h[2])}</h${n}>`; }
      else if (ln.trim()) html += `<p>${inline(ln)}</p>`;
    }
    flushTbl();
    if (inList) html += "</ul>";
    if (inCode) html += `<pre>${code.join("\n")}</pre>`;
    if (maths.length) html = html.replace(/@@MATH(\d+)@@/g, (m, i) => {
      const x = maths[+i];
      return x ? `<span class="kmath${x.display ? " kd" : ""}" data-tex="${K.esc(x.tex)}">${K.esc(x.display ? "$$" + x.tex + "$$" : "$" + x.tex + "$")}</span>` : m;
    });
    return html;
  };

  /* ---------- 公式排版：本地 KaTeX（assets/katex），按需加载；加载失败就保留原文 ---------- */
  const ASSET_BASE = (() => { try { return document.currentScript.src.replace(/case\.js(\?.*)?$/, ""); } catch (e) { return "../assets/"; } })();
  let katexState = 0; // 0 未加载 1 加载中 2 就绪 3 失败
  const mathPending = [];
  function renderMathNow(root) {
    root.querySelectorAll("span.kmath:not([data-done])").forEach((el) => {
      el.setAttribute("data-done", "1");
      try { window.katex.render(el.getAttribute("data-tex"), el, { displayMode: el.classList.contains("kd"), throwOnError: false, strict: "ignore" }); } catch (e) { /* 保留原文 */ }
    });
  }
  K.mathIn = function (root) {
    root = root || document.body;
    if (!root || !root.querySelector || !root.querySelector("span.kmath:not([data-done])")) return;
    if (katexState === 2) return renderMathNow(root);
    if (katexState === 3) return;
    mathPending.push(root);
    if (katexState === 1) return;
    katexState = 1;
    const css = document.createElement("link");
    css.rel = "stylesheet"; css.href = ASSET_BASE + "katex/katex.min.css";
    document.head.appendChild(css);
    const js = document.createElement("script");
    js.src = ASSET_BASE + "katex/katex.min.js";
    js.onload = () => { katexState = window.katex ? 2 : 3; if (katexState === 2) mathPending.splice(0).forEach(renderMathNow); };
    js.onerror = () => { katexState = 3; };
    document.head.appendChild(js);
  };

  /* ---------- 原 notebook 单元格 ---------- */
  const NB = () => window.CASE_NB || { cells: [] };
  K.cell = function (i, opt) {
    opt = opt || {};
    const c = NB().cells[i];
    const box = K.el("div", { class: "cell" + (c && c.type === "markdown" ? " md" : "") });
    if (!c) { box.innerHTML = `<div class="hd">找不到第 ${i} 格</div>`; return box; }
    const label = opt.label || (c.type === "code" ? "代码" : "说明");
    box.innerHTML = `<div class="hd"><span>${K.esc(label)}</span><span class="badge">原 notebook · 第 ${i} 格</span></div>`;
    if (c.type === "markdown") {
      const body = K.el("div", { class: "body" }, K.md(c.src));
      box.appendChild(body);
      K.mathIn(body);
    } else {
      let html = K.hl(c.src);
      (opt.highlight || []).forEach((word) => {
        const w = K.esc(word).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        html = html.replace(new RegExp(`(?![^<]*>)(${w})`, "g"), '<mark class="hl">$1</mark>');
      });
      const pre = K.el("pre", null, html);
      box.appendChild(pre);
      const nLines = c.src.split("\n").length;
      const collapse = opt.collapse == null ? nLines > 16 : opt.collapse;
      if (collapse) {
        pre.classList.add("collapsed");
        const b = K.el("button", { class: "more" }, `展开全部 ${nLines} 行`);
        b.onclick = () => { const col = pre.classList.toggle("collapsed"); b.textContent = col ? `展开全部 ${nLines} 行` : "收起"; };
        box.appendChild(b);
      }
      if ((c.out || (c.imgs && c.imgs.length)) && opt.output !== false) {
        const o = K.el("div", { class: "out" });
        o.appendChild(K.el("div", { class: "out-h" }, "运行结果（作者原 notebook）"));
        if (c.out) o.appendChild(K.el("pre", null, K.esc(c.out)));
        (c.imgs || []).forEach((src) => { if (/^data:image\/(png|jpeg|gif);base64,/.test(src)) { const im = K.el("img", { alt: "运行结果图", loading: "lazy" }); im.src = src; o.appendChild(im); } });
        box.appendChild(o);
      }
    }
    return box;
  };
  K.notebook = function (el) {
    const nb = NB();
    el.innerHTML = "";
    const tools = K.el("div", { class: "nbtools" });
    const q = K.el("input", { type: "search", placeholder: "在原 notebook 里搜索（如 fit、GridSearchCV、Dense）" });
    const bAll = K.el("button", { class: "btn", "aria-pressed": "true" }, "全部");
    const bCode = K.el("button", { class: "btn", "aria-pressed": "false" }, "只看代码");
    const stat = K.el("span", { class: "stat" });
    tools.append(q, bAll, bCode, stat);
    const list = K.el("div");
    el.append(tools, list);
    let onlyCode = false, shown = 0;
    const PAGE = 12;
    function filtered() {
      const kw = q.value.trim().toLowerCase();
      return nb.cells.map((c, i) => [c, i]).filter(([c]) => (!onlyCode || c.type === "code") && (!kw || c.src.toLowerCase().includes(kw)));
    }
    function render(reset) {
      const f = filtered();
      if (reset) { list.innerHTML = ""; shown = 0; }
      f.slice(shown, shown + PAGE).forEach(([c, i]) => list.appendChild(K.cell(i, { collapse: c.src.split("\n").length > 24 })));
      shown = Math.min(f.length, shown + PAGE);
      stat.textContent = `显示 ${shown} / ${f.length} 格（全本 ${nb.cells.length} 格）`;
      const old = el.querySelector(".nb-more"); if (old) old.remove();
      if (shown < f.length) {
        const m = K.el("button", { class: "btn nb-more", style: "margin-top:8px" }, `继续加载（还有 ${f.length - shown} 格）`);
        m.onclick = () => render(false);
        el.appendChild(m);
      }
    }
    q.addEventListener("input", () => render(true));
    bAll.onclick = () => { onlyCode = false; bAll.setAttribute("aria-pressed", "true"); bCode.setAttribute("aria-pressed", "false"); render(true); };
    bCode.onclick = () => { onlyCode = true; bAll.setAttribute("aria-pressed", "false"); bCode.setAttribute("aria-pressed", "true"); render(true); };
    render(true);
  };

  /* ---------- 逐章讲解（纵向长页，按原 notebook 顺序） ----------
   * steps: [{title, kicker, html, cells:[i...], highlight:[...], notes:{i: "这一格在做什么"}, at, draw(svg, ctx), width, height}]
   *   cells   本章的关键代码格（高亮显示）；章节从 min(cells) 这一格开始，到下一章开始前结束
   *   at      可选，手动指定章节从第几格开始（没有 cells 的纯动画章节用）
   *   notes   可选，给某些格子加一句中文旁注
   *   draw    动画，章节滚动到屏幕里时自动播放一次，可点“重播”
   * 原 notebook 的每一格都会按顺序出现，作者的运行结果（如有）显示在代码下方。
   * 页面里带 data-at="N" 的元素（如交互实验）会被移到第 N 格之后。
   */
  K.pipeline = function (el, steps, opt) {
    opt = Object.assign({ width: 720, height: 340 }, opt || {});
    const nb = NB(), N = nb.cells.length;
    el.className = "walk";
    el.innerHTML = "";
    // 每章的起点
    let last = 0;
    const order = steps.map((s, i) => {
      let a = s.at != null ? s.at : s.cells && s.cells.length ? Math.min.apply(null, s.cells) : null;
      if (a == null) a = last; else last = a;
      return { s, i, a: K.clamp(a, 0, Math.max(0, N - 1)) };
    }).sort((x, y) => x.a - y.a || x.i - y.i);
    const chapters = [];
    if (N && (!order.length || order[0].a > 0)) chapters.push({ intro: true, from: 0 });
    order.forEach((o) => chapters.push({ step: o.s, from: o.a }));
    chapters.forEach((c, k) => (c.to = k + 1 < chapters.length ? chapters[k + 1].from - 1 : N - 1));

    // 顶部：目录 + 说明
    const hasOut = nb.cells.some((c) => c.out || (c.imgs && c.imgs.length));
    const top = K.el("div", { class: "walk-top panel pad" });
    top.innerHTML = `<p style="margin:0 0 8px">按原 notebook 的顺序从上往下讲。每一章先看<b>动画和讲解</b>，再看作者的<b>代码</b>${hasOut ? "和<b>运行结果</b>" : ""}。共 ${N} 格，分 ${chapters.length} 章。</p>` +
      (hasOut ? "" : `<p class="note">这份 notebook 没有附带作者的运行结果（Kaggle 下载接口只提供源码）。作者的图表和分数可以在 Kaggle 原页面查看。</p>`) +
      `<ol class="toc"></ol><div class="walk-actions"><button class="btn ghost" data-a="fold">折叠全部代码</button></div>`;
    el.appendChild(top);
    const toc = top.querySelector(".toc");

    const runners = [];
    chapters.forEach((c, k) => {
      const s = c.step || { title: "开场：导入与读数据", kicker: "", html: "<p>作者在正式分析前做的准备：导入库、读入数据、看一眼数据的样子。</p>" };
      const id = "ch-" + (k + 1);
      const sec = K.el("section", { class: "chapter", id });
      const range = c.to >= c.from ? `第 ${c.from}–${c.to} 格` : "";
      sec.innerHTML = `<header class="ch-head"><span class="ch-n">${k + 1}</span><div><div class="kick">${K.esc(s.kicker || range)}</div><h3>${K.esc(s.title)}</h3></div></header>`;
      toc.appendChild(K.el("li", null, `<a href="#${id}">${K.esc(s.title)}</a>`));
      const body = K.el("div", { class: "ch-body" });
      if (s.draw) {
        const fig = K.el("figure", { class: "anim panel" });
        const stage = K.el("div", { class: "stagebox" });
        const cap = K.el("figcaption", null, `<span>动画 · ${K.esc(s.title)}</span><button class="btn ghost" title="重播动画">↻ 重播</button>`);
        fig.append(stage, cap);
        body.appendChild(fig);
        let token = 0;
        const run = () => {
          const my = ++token;
          const w = s.width || opt.width, h = s.height || opt.height;
          const svg = K.stage(stage, w, h);
          const ctx = { alive: () => my === token, svg, w, h };
          ctx.tween = (ms, fn) => K.tween(ms, fn, ctx);
          ctx.wait = (ms) => K.wait(ms, ctx);
          try { s.draw(svg, ctx); } catch (e) { console.error(e); K.svg(svg, "text", { x: 20, y: 30, fill: "var(--bad)" }, "动画出错：" + e.message); }
        };
        cap.querySelector("button").onclick = run;
        runners.push({ el: fig, run, done: false });
      }
      if (s.html) body.appendChild(K.el("div", { class: "ch-explain" }, s.html));
      sec.appendChild(body);
      const cellsBox = K.el("div", { class: "ch-cells" });
      const key = new Set(s.cells || []);
      for (let i = c.from; i <= c.to; i++) {
        if (s.notes && s.notes[i]) cellsBox.appendChild(K.el("div", { class: "cell-note" }, s.notes[i]));
        const ce = K.cell(i, { highlight: key.has(i) ? s.highlight : null, collapse: (nb.cells[i].src.split("\n").length > (key.has(i) ? 40 : 18)) });
        ce.id = "cell-" + i;
        if (key.has(i)) ce.classList.add("key");
        cellsBox.appendChild(ce);
      }
      if (c.to >= c.from) sec.appendChild(cellsBox);
      else if (s.cells && s.cells.length) {
        // 和下一章从同一格开始：这里先展示关键格（不带编号锚点），完整位置在下一章
        const ref = K.el("div", { class: "ch-cells" });
        ref.appendChild(K.el("div", { class: "cell-note" }, "本章讲的代码和下一章写在同一格里，这里先看关键部分："));
        s.cells.forEach((i) => { if (nb.cells[i]) { const ce = K.cell(i, { highlight: s.highlight, collapse: true }); ce.classList.add("key"); ref.appendChild(ce); } });
        sec.appendChild(ref);
      }
      el.appendChild(sec);
    });

    // 动画进入视野时自动播一次
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => es.forEach((e) => {
        const r = runners.find((x) => x.el === e.target);
        if (e.isIntersecting && r && !r.done) { r.done = true; r.run(); }
      }), { threshold: 0.35 });
      runners.forEach((r) => io.observe(r.el));
    } else runners.forEach((r) => r.run());
    // 首屏还没滚到时先画静态第一帧，避免空白
    runners.forEach((r) => setTimeout(() => { if (!r.done) { r.done = true; r.run(); } }, 2500));

    // 把 data-at 的元素（交互实验等）挪到对应格子后面
    const lastAt = {};
    document.querySelectorAll("[data-at]").forEach((x) => {
      const n = x.getAttribute("data-at"), t = lastAt[n] || document.getElementById("cell-" + n);
      if (t) { x.classList.add("inline-lab"); t.after(x); lastAt[n] = x; }  // 同一格后的多个实验保持书写顺序
    });
    // 旧版的“原 notebook 全部单元格”一节不再需要
    const old = document.getElementById("notebook");
    if (old) old.style.display = "none";
    document.querySelectorAll('a[href="#notebook"]').forEach((a) => (a.style.display = "none"));
    const lead = el.parentElement && el.parentElement.querySelector(".lead");
    if (lead) lead.textContent = "下面按原 notebook 的顺序逐章展开：每章有动画、中文讲解、作者的全部代码。重点代码格有蓝色边框。";

    const fb = top.querySelector('[data-a="fold"]');
    fb.onclick = () => { const f = el.classList.toggle("folded"); fb.textContent = f ? "展开全部代码" : "折叠全部代码"; };
    K.bindCells();
    return { chapters };
  };

  /* ---------- 滑块交互实验 ----------
   * cfg: {width, height, controls:[{id,label,min,max,step,value,fmt} | {id,label,options:[[v,text]...],value} | {id,label,button:true}], render(svg, vals, api), note}
   * render 返回 {readout:[{k, v}]} 可显示指标卡片
   */
  K.lab = function (el, cfg) {
    cfg = Object.assign({ width: 640, height: 380 }, cfg);
    el.classList.add("lab", "panel");
    el.innerHTML = `<div class="ctrl"></div><div class="view"><div class="svgbox"></div><div class="readout"></div></div>`;
    const ctrl = el.querySelector(".ctrl"), svgbox = el.querySelector(".svgbox"), readout = el.querySelector(".readout");
    const vals = {};
    let token = 0;
    function update() {
      const my = ++token;
      const svg = K.stage(svgbox, cfg.width, cfg.height);
      const api = { alive: () => my === token, tween: null, w: cfg.width, h: cfg.height };
      api.tween = (ms, fn) => K.tween(ms, fn, api);
      api.wait = (ms) => K.wait(ms, api);
      const r = cfg.render(svg, Object.assign({}, vals), api) || {};
      readout.innerHTML = (r.readout || []).map((d) => `<div class="stat-tile"><div class="k">${K.esc(d.k)}</div><div class="v" style="${d.color ? "color:" + d.color : ""}">${K.esc(d.v)}</div></div>`).join("");
    }
    (cfg.controls || []).forEach((c) => {
      const wrap = K.el("div");
      if (c.options) {
        wrap.innerHTML = `<label>${K.esc(c.label)}</label>`;
        const s = K.el("select");
        c.options.forEach(([v, t]) => s.appendChild(K.el("option", { value: v }, K.esc(t))));
        s.value = c.value;
        vals[c.id] = c.value;
        s.onchange = () => { vals[c.id] = s.value; update(); };
        wrap.appendChild(s);
      } else if (c.button) {
        const b = K.el("button", { class: "btn" }, K.esc(c.label));
        vals[c.id] = 0;
        b.onclick = () => { vals[c.id]++; update(); };
        wrap.appendChild(b);
      } else {
        const f = c.fmt || ((v) => v);
        wrap.innerHTML = `<label>${K.esc(c.label)} <span class="v">${K.esc(f(c.value))}</span></label>`;
        const r = K.el("input", { type: "range", min: c.min, max: c.max, step: c.step || 1, value: c.value });
        vals[c.id] = +c.value;
        r.oninput = () => { vals[c.id] = +r.value; wrap.querySelector(".v").textContent = f(+r.value); update(); };
        wrap.appendChild(r);
      }
      ctrl.appendChild(wrap);
    });
    if (cfg.note) ctrl.appendChild(K.el("div", { class: "note" }, cfg.note));
    update();
    return { update, vals };
  };

  /* ---------- 页面初始化 ---------- */
  K.initPage = function () {
    const tb = document.querySelector("[data-theme-toggle]");
    if (tb) {
      const root = document.documentElement;
      try { const t = localStorage.getItem("kc-theme"); if (t) root.setAttribute("data-theme", t); } catch (e) {}
      tb.onclick = () => {
        const dark = root.getAttribute("data-theme") ? root.getAttribute("data-theme") === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
        const t = dark ? "light" : "dark";
        root.setAttribute("data-theme", t);
        try { localStorage.setItem("kc-theme", t); } catch (e) {}
      };
    }
    const nbEl = document.getElementById("nb-all");
    if (nbEl) K.notebook(nbEl);
    K.bindCells();
  };
  /* 把 data-cell="N" 变成可点开原代码格的出处徽章（逐章讲解生成后会再调用一次） */
  K.bindCells = function () {
    document.querySelectorAll("[data-cell]:not(.src-badge)").forEach((e) => {
      e.classList.add("src-badge");
      e.title = "点击查看原 notebook 这一格";
      e.onclick = () => {
        const i = +e.getAttribute("data-cell");
        let pop = e.nextElementSibling;
        if (pop && pop.classList.contains("cell")) { pop.remove(); return; }
        e.after(K.cell(i, { collapse: true }));
      };
    });
  };
  try {
    const t = localStorage.getItem("kc-theme");
    if (t) document.documentElement.setAttribute("data-theme", t);
  } catch (e) {}
  /* 脚本出错时在页面顶部显示红条，截图自查时一眼可见 */
  window.addEventListener("error", function (ev) {
    const show = () => {
      const b = document.createElement("div");
      b.className = "kc-error";
      b.style.cssText = "position:fixed;left:0;right:0;top:0;z-index:999;background:#d1435b;color:#fff;padding:8px 16px;font:14px/1.5 monospace";
      b.textContent = "脚本出错：" + (ev.message || "") + " @ " + (ev.filename || "").split("/").pop() + ":" + (ev.lineno || "");
      document.body.appendChild(b);
    };
    document.body ? show() : addEventListener("DOMContentLoaded", show);
  });
})();
