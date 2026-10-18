/* app.js: draws what web_dashboard.py reports.                              owner: Gio
 *
 * Presentation only. Every reading, the stale rule, the valve-open rule, the class name and
 * the latency p99 come from the server, which reads packets exactly like dashboard.py.
 * Packet text is only ever inserted as text, never as HTML.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const TELEMETRY_HZ = 10;          // the supervisor sends telemetry 10 times a second
  const LOST_AFTER_MS = 1500;       // no snapshot from the console for this long: it's gone
  const DOWN_AFTER_S = 10;          // "interrupted" reads as "no telemetry" after this long
  const MINUS = "−";
  const isLocalPage = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);

  let snap = null;                  // latest snapshot from the console
  let lastMsgAt = 0;
  let consoleUp = false;
  let paused = false;
  let frozen = null;                // snapshot the image and history show while paused
  let view = "chart";
  let renderQueued = false;
  let lastAnnounced = "";
  let cursorFromEnd = null;         // hovered or keyboard-selected history point, counted from the newest
  let thumbShown = 0, thumbWanted = 0, thumbBusy = false;

  // ------------------------------------------------------------------ small helpers
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const fixed = (v, digits) => (isNum(v) ? v.toFixed(digits) : "—");
  const grouped = new Intl.NumberFormat("en-US");
  const count = (v) => (isNum(v) ? grouped.format(v) : "—");
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  function setText(el, text) {
    if (el.textContent !== text) el.textContent = text;
  }

  function setValue(el, text, unit) {
    const key = `${text}|${unit || ""}`;
    if (el.dataset.v === key) return;
    el.dataset.v = key;
    const nodes = [document.createTextNode(text)];
    if (unit && text !== "—") {
      const u = document.createElement("small");
      u.textContent = unit;
      nodes.push(u);
    }
    el.replaceChildren(...nodes);
  }

  function setIcon(svg, id) {
    const use = svg.querySelector("use");
    if (use.getAttribute("href") !== `#${id}`) use.setAttribute("href", `#${id}`);
  }

  function duration(s) {
    if (!isNum(s)) return "—";
    if (s < 60) return `${s.toFixed(1)} s`;
    const m = Math.floor(s / 60);
    return `${m} min ${String(Math.floor(s % 60)).padStart(2, "0")} s`;
  }

  function richText(el, parts) {          // parts: strings, or {code: "..."}; all inserted as text
    const key = JSON.stringify(parts);
    if (el.dataset.v === key) return;
    el.dataset.v = key;
    el.replaceChildren(...parts.map((p) => {
      if (typeof p === "string") return document.createTextNode(p);
      const c = document.createElement("code");
      c.textContent = p.code;
      return c;
    }));
  }

  // ------------------------------------------------------------------ data in
  function connect() {
    const es = new EventSource("/api/stream");
    es.onmessage = (e) => {
      let next;
      try { next = JSON.parse(e.data); } catch { return; }
      snap = next;
      lastMsgAt = performance.now();
      consoleUp = true;
      queueRender();
    };
    // EventSource reconnects by itself; the watchdog below decides when to say the console is gone.
  }

  setInterval(() => {
    if (consoleUp && performance.now() - lastMsgAt > LOST_AFTER_MS) {
      consoleUp = false;
      queueRender();
    }
  }, 250);

  function queueRender() {
    if (renderQueued) return;
    renderQueued = true;
    if (document.hidden) setTimeout(render, 0);     // keep the tab title current in the background
    else requestAnimationFrame(render);
  }

  // ------------------------------------------------------------------ state, as words
  function linkState() {
    if (!consoleUp) {
      return snap ? { key: "lost", label: "Console not responding", meta: "is web_dashboard.py still running?" }
                  : { key: "connecting", label: "Connecting to the console", meta: "" };
    }
    const t = snap.telemetry;
    if (!t.received) return { key: "waiting", label: "Waiting for telemetry", meta: `nothing on UDP ${snap.ports.telemetry} yet` };
    if (t.stale) {
      const down = t.age_s >= DOWN_AFTER_S;
      return { key: down ? "down" : "stale", label: down ? "No telemetry" : "Telemetry interrupted",
               meta: `last packet ${duration(t.age_s)} ago` };
    }
    const from = t.simulated ? "simulator on this Mac" : `from ${t.latest_source}`;
    return { key: "live", label: "Live", meta: `${fixed(t.rate_hz, 1)} updates/s · ${from}` };
  }

  function reportedValve(st, c) {
    const d2t = isNum(st.detect_to_trip_ms) && st.detect_to_trip_ms > 0
      ? ` First gas frame to valve: ${st.detect_to_trip_ms.toFixed(0)} ms.` : "";
    switch (st.reason) {
      case "leak":
        return { tone: "alarm", icon: "i-alarm", word: "Closed", reason: "Gas leak", closed: true,
                 line: `${c.trip_run} gas verdicts in a row tripped the valve.${d2t}` };
      case "deadline":
        return { tone: "serious", icon: "i-warn", word: "Closed", reason: "Verdict deadline missed", closed: true,
                 line: `No verdict for more than ${c.deadline_ms} ms, so the supervisor failed closed. Vision stalled or stopped.` };
      case "boot":
        return { tone: "hold", icon: "i-lock", word: "Closed", reason: "Start-up", closed: true,
                 line: "The supervisor always starts closed. A person reopens it once verdicts are clear." };
      default:
        return { tone: "serious", icon: "i-warn", word: "Closed", reason: st.reason == null ? "Not reported" : String(st.reason),
                 closed: true, line: "Closed by the supervisor." };
    }
  }

  function valveState(link) {
    const c = snap && snap.constants;
    if (link.key === "connecting") return { tone: "none", icon: "i-none", word: "—", reason: "—", line: "Waiting for the first update from web_dashboard.py." };
    if (link.key === "lost") return { tone: "unknown", icon: "i-unknown", word: "Unknown", reason: "—", tag: "Not current",
                                      line: "The console stopped sending updates, so nothing on this page is current." };
    const t = snap.telemetry, st = t.state;
    if (!t.received) {
      return { tone: "none", icon: "i-none", word: "No data", reason: "—",
               line: `Nothing has arrived on UDP ${snap.ports.telemetry}. Start the supervisor with this Mac’s address, or run the simulator.` };
    }
    if (t.stale) {
      const last = st.valve === "open" ? "open" : st.valve === "closed" ? `closed (${st.reason ?? "no reason given"})` : "not reported";
      const reason = st.valve === "closed" ? reportedValve(st, c).reason : st.valve === "open" ? "None" : st.reason == null ? "—" : String(st.reason);
      return { tone: "unknown", icon: "i-unknown", word: "Unknown", tag: "Last reported", reason,
               closed: st.valve === "closed",
               line: `No telemetry for ${duration(t.age_s)}, so the current state isn’t known. Last report: ${last}.` };
    }
    if (t.valve_open) {
      return { tone: "ok", icon: "i-ok", word: "Open", reason: "None",
               line: `Not tripped. Verdicts are arriving within the ${c.deadline_ms} ms deadline.` };
    }
    if (st.valve !== "closed") {
      return { tone: "unknown", icon: "i-unknown", word: "Not reported", reason: st.reason == null ? "—" : String(st.reason),
               line: "The latest packet had no valve state." };
    }
    return reportedValve(st, c);
  }

  // ------------------------------------------------------------------ top bar, banners, footer
  function renderLink(link) {
    document.body.dataset.link = link.key;
    document.body.dataset.lost = link.key === "lost" ? "1" : "0";
    setText($("link-label"), link.label);
    setText($("link-meta"), link.meta);
  }

  function renderSimButton(link) {
    const btn = $("sim-btn");
    const usable = isLocalPage && consoleUp && snap;
    btn.hidden = !usable;
    if (!usable) return;
    const running = snap.sim.running;
    setIcon(btn.querySelector("svg"), running ? "i-stop" : "i-play");
    setText(btn.querySelector("span"), running ? "Stop simulator" : "Run simulator");
    const blocked = !running && snap.sim.blocked;
    if (blocked) {
      btn.setAttribute("aria-disabled", "true");
      btn.title = blocked;
    } else {
      btn.removeAttribute("aria-disabled");
      btn.title = running ? "Stop fake_pi.py" : "Start fake_pi.py, sending to this Mac";
    }
  }

  function renderBanners(link) {
    const t = snap && snap.telemetry;
    const live = consoleUp && t && t.received;
    const simulated = live && t.simulated;
    $("banner-sim").hidden = !simulated;
    if (simulated) {
      let text;
      if (t.stale) text = "The last telemetry came from this Mac, not the Pi.";
      else if (snap.sim.running) text = "fake_pi.py, started from this page, is sending to this Mac. Nothing here comes from the Pi. Stop it from the top bar.";
      else text = "Telemetry is coming from this Mac, not the Pi: a simulator such as fake_pi.py is running here.";
      setText($("banner-sim-text"), text);
    }
    const multi = live && t.multi_source;
    $("banner-multi").hidden = !multi;
    if (multi) {
      const names = t.sources.filter((s) => s.active).map((s) => `${s.addr}:${s.port}${s.local ? " (this Mac)" : ""}`);
      const n = names.length;
      const who = n === 2 ? `${names[0]} and ${names[1]} are both` : `${names.slice(0, -1).join(", ")} and ${names[n - 1]} are all`;
      setText($("banner-multi-title"), `${n === 2 ? "Two" : n === 3 ? "Three" : n} senders`);
      setText($("banner-multi-text"), `${who} sending to UDP ${snap.ports.telemetry}. Their packets interleave, so the readings jump between them. Keep only one.`);
    }
    $("banner-lost").hidden = !(link.key === "lost");
  }

  function renderFooter() {
    if (!snap) return;
    const t = snap.telemetry, img = snap.image, c = snap.constants;
    richText($("foot-host"), ["This Mac ", { code: snap.host_ips[0] || "offline" }]);
    setText($("foot-ports"), `UDP ${snap.ports.telemetry} telemetry · ${snap.ports.image} images, owned by this console`);
    const rejected = img.rejected ? ` · ${count(img.rejected)} unreadable images` : "";
    setText($("foot-packets"), `${count(t.packets)} packets · ${count(t.malformed)} malformed · ${count(img.seq)} images${rejected}`);
    const rules = `Rules from ${c.source}: trip after ${c.trip_run} gas verdicts · deadline ${c.deadline_ms} ms · reset after ${c.reset_clear} clear`;
    setText($("foot-rules"), rules);
    $("foot-rules").title = rules;
  }

  function renderConnection(link) {
    if (!snap) return;
    const t = snap.telemetry, img = snap.image;
    setText($("lp-meta"), `UDP ${snap.ports.telemetry} and ${snap.ports.image}`);
    if (!t.received) setText($("lp-rate"), "nothing yet");
    else if (link.key === "live") setText($("lp-rate"), `${fixed(t.rate_hz, 1)} updates/s`);
    else setText($("lp-rate"), `last packet ${duration(t.age_s)} ago`);
    const active = t.sources.filter((s) => s.active);
    const list = $("lp-senders");
    const key = JSON.stringify(active.map((s) => [s.addr, s.port, s.local, s.packets]));
    if (list.dataset.v !== key) {
      list.dataset.v = key;
      if (!active.length) {
        list.replaceChildren(document.createTextNode(t.received ? "none right now" : "—"));
      } else {
        const box = document.createElement("span");
        box.className = "senders";
        for (const s of active) {
          const row = document.createElement("span");
          const note = document.createElement("small");
          note.textContent = ` ${s.local ? "this Mac · " : ""}${count(s.packets)} packets`;
          row.append(document.createTextNode(s.addr), note);
          box.append(row);
        }
        list.replaceChildren(box);
      }
    }
    const parts = [`${count(img.seq)} received`];
    if (img.rejected) parts.push(`${count(img.rejected)} unreadable`);
    if (isNum(img.age_s) && img.age_s > snap.constants.stale_s) parts.push(`last ${duration(img.age_s)} ago`);
    setText($("lp-images"), parts.join(" · "));
  }

  const TONE_COLORS = { ok: "#45c07c", alarm: "#f3585b", serious: "#f08a4b", hold: "#e3a33b" };

  function renderTitle(v, link) {
    let title;
    if (link.key === "live") title = v.tone === "ok" ? "Open · Mirage" : `Closed: ${v.reason} · Mirage`;
    else if (link.key === "lost") title = "Console not responding · Mirage";
    else if (link.key === "stale" || link.key === "down") title = "No telemetry · Mirage";
    else title = "Waiting · Mirage";
    if (document.title !== title) document.title = title;
    const color = (link.key === "live" && TONE_COLORS[v.tone]) || "#8c939c";
    const href = `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" fill="${color}"/></svg>`)}`;
    const fav = $("favicon");
    if (fav.getAttribute("href") !== href) fav.setAttribute("href", href);
  }

  function announce(v, link) {
    let msg;
    if (link.key === "connecting") return;
    if (link.key === "live") msg = v.tone === "ok" ? "Valve open." : `Valve closed: ${v.reason}.`;
    else msg = `${link.label}. Valve state unknown.`;
    if (msg !== lastAnnounced) {
      lastAnnounced = msg;
      $("announcer").textContent = msg;
    }
  }

  // ------------------------------------------------------------------ valve
  function renderValve(v, link) {
    const panel = $("valve");
    panel.dataset.tone = v.tone;
    document.body.dataset.tone = link.key === "live" ? v.tone : "none";
    setIcon($("valve-icon"), v.icon);
    setText($("valve-word"), v.word);
    setText($("valve-line"), v.line);
    setText($("valve-reason"), v.reason);
    $("valve-tag").hidden = !v.tag;
    if (v.tag) setText($("valve-tag"), v.tag);
    const showReset = Boolean(v.closed) && link.key !== "lost";
    $("reset-row").hidden = !showReset;
    $("reset-hint").hidden = !showReset;
    if (showReset) {
      setText($("reset-hint"), `Or press the reset button, if one is wired. The supervisor refuses a reset until it has seen ${snap.constants.reset_clear} clear verdicts in a row.`);
    }
  }

  // ------------------------------------------------------------------ verdict
  const CLASS_ICONS = { clear: "c-clear", gas: "c-gas", heat: "c-heat", motion: "c-motion" };

  function renderVerdict(link) {
    const t = snap && snap.telemetry;
    const has = Boolean(t && t.received) && link.key !== "connecting";
    const st = has ? t.state : {};
    const live = link.key === "live";
    const noVerdicts = has && (st.verdicts === 0 || (isNum(st.since_verdict_ms) && st.since_verdict_ms < 0));
    const deadline = snap ? snap.constants.deadline_ms : 250;
    const overdue = has && live && !noVerdicts && isNum(st.since_verdict_ms) && st.since_verdict_ms > deadline;
    $("verdict").classList.toggle("is-stale", (has && !live) || overdue);
    $("verdict-tag").hidden = !((has && !live) || overdue);
    setText($("verdict-tag"), overdue ? `${st.since_verdict_ms.toFixed(0)} ms old` : "Last reported");
    const icon = $("class-icon");
    const trip = snap ? snap.constants.p_gas_trip : 0.6;
    $("pgas-mark").style.left = `${clamp01(trip) * 100}%`;
    setText($("pgas-caption"), `Probability reported by vision; counts as gas at ${trip.toFixed(2)} or above. Not a gas concentration.`);

    if (!has || noVerdicts) {
      setIcon(icon, "c-unknown");
      icon.dataset.class = "none";
      setText($("class-name"), has ? "No verdicts yet" : "—");
      setValue($("pgas-value"), "—");
      setValue($("peak-value"), "—");
      $("pgas-fill").style.width = "0";
      setText($("verdict-meta"), has && isNum(st.verdicts) ? "0 verdicts" : "");
      return;
    }
    const label = t.class_label;
    setIcon(icon, CLASS_ICONS[label] || "c-unknown");
    icon.dataset.class = label;
    setText($("class-name"), label === "?" ? `Unknown class (${st.cls === undefined ? "missing" : String(st.cls)})`
                                           : label.charAt(0).toUpperCase() + label.slice(1));
    const p = st.p_gas;
    setValue($("pgas-value"), fixed(p, 2));
    const fill = $("pgas-fill");
    fill.style.width = isNum(p) ? `${clamp01(p) * 100}%` : "0";
    fill.classList.toggle("is-gas", isNum(p) && p >= trip);
    setValue($("peak-value"), fixed(st.peak_px, 2), "px");
    const parts = [];
    if (isNum(st.seq)) parts.push(`seq ${count(st.seq)}`);
    if (isNum(st.verdicts)) parts.push(`${count(st.verdicts)} verdicts`);
    setText($("verdict-meta"), parts.join(" · "));
  }

  // ------------------------------------------------------------------ timing
  function renderTiming(link) {
    const t = snap && snap.telemetry;
    const has = Boolean(t && t.received) && link.key !== "connecting";
    const st = has ? t.state : {};
    const c = snap ? snap.constants : { deadline_ms: 250 };
    $("timing").classList.toggle("is-stale", has && link.key !== "live");
    $("timing-tag").hidden = !(has && link.key !== "live");

    const since = st.since_verdict_ms, sinceFill = $("since-fill");
    if (!has || !isNum(since) || since < 0) {
      setValue($("since-value"), "—");
      setText($("since-sub"), !has ? "" : isNum(since) ? "no verdict yet" : "not reported");
      sinceFill.style.width = "0";
    } else {
      setValue($("since-value"), since.toFixed(0), "ms");
      setText($("since-sub"), `deadline ${c.deadline_ms} ms`);
      sinceFill.style.width = `${clamp01(since / c.deadline_ms) * 100}%`;
      sinceFill.classList.toggle("is-over", since > c.deadline_ms);
    }

    const lat = st.lat_ms, latMax = st.lat_max_ms;
    if (has && isNum(lat) && lat > 0) {
      setValue($("lat-value"), lat.toFixed(1), "ms");
      setText($("lat-sub"), isNum(latMax) && latMax > 0 ? `max ${latMax.toFixed(1)} ms since start` : "");
    } else {
      setValue($("lat-value"), "—");
      setText($("lat-sub"), has ? "not measured yet" : "");
    }

    if (has && isNum(t.lat_p99)) {
      setValue($("p99-value"), t.lat_p99.toFixed(1), "ms");
      setText($("p99-sub"), `over the last ${count(t.lat_count)} updates`);
    } else {
      setValue($("p99-value"), "—");
      setText($("p99-sub"), has ? "no latency reported yet" : "");
    }

    const d2t = st.detect_to_trip_ms;
    if (has && isNum(d2t) && d2t > 0) {
      setValue($("d2t-value"), d2t.toFixed(0), "ms");
      setText($("d2t-sub"), "most recent leak trip");
    } else {
      setValue($("d2t-value"), "—");
      setText($("d2t-sub"), has ? "no leak trip since start" : "");
    }
  }

  // ------------------------------------------------------------------ shift map
  function requestThumb(seq) {
    thumbWanted = seq;
    if (thumbBusy) return;
    thumbBusy = true;
    const want = seq;
    const im = new Image();
    im.onload = () => {
      $("thumb").src = im.src;
      thumbShown = want;
      thumbBusy = false;
      queueRender();
      if (thumbWanted !== want) requestThumb(thumbWanted);
    };
    im.onerror = () => { thumbBusy = false; };
    im.src = `/api/thumb.jpg?seq=${want}`;
  }

  function renderImage(s, link) {
    const img = s && s.image;
    const has = Boolean(img && img.seq > 0);
    if (has && img.seq !== thumbWanted && !paused) requestThumb(img.seq);
    const visible = has && thumbShown > 0;
    $("thumb").hidden = !visible;
    $("stage-empty").hidden = visible;
    if (!visible) {
      const ip = (s && s.host_ips[0]) || "<this Mac’s IP>";
      if (!s) {
        setText($("stage-empty-title"), "Connecting to the console");
        richText($("stage-empty-body"), []);
      } else if (s.telemetry.received) {
        setText($("stage-empty-title"), "Telemetry is arriving, but no shift-map image has");
        richText($("stage-empty-body"), ["Images come from ", { code: "mirage_vision" }, " when it runs with ",
          { code: `-i ${ip}` }, ". mock_vision doesn’t send any."]);
      } else {
        setText($("stage-empty-title"), "No shift-map image yet");
        richText($("stage-empty-body"), ["The vision process sends one with every verdict to UDP ",
          String(s.ports.image), " when it runs with ", { code: `-i ${ip}` }, "."]);
      }
    }
    setText($("img-meta"), has ? `${img.width} × ${img.height} px · from ${img.simulated ? "this Mac" : img.source}` : "");
    $("tag-sim").hidden = !(visible && img.simulated);
    const staleImg = visible && (link.key === "lost" || (isNum(img.age_s) && img.age_s > s.constants.stale_s));
    $("tag-stale").hidden = !staleImg;
    if (staleImg) setText($("tag-stale"), link.key === "lost" ? "Not current" : `Image ${duration(img.age_s)} old`);
    $("tag-paused").hidden = !paused;
  }

  // ------------------------------------------------------------------ history
  function historyAxis(hist) {
    const times = hist.map((h) => h[2]);
    const nominal = (snap ? snap.constants.history_len : 120) / TELEMETRY_HZ;
    if (!hist.length) return { timeMode: true, xs: [], x0: -nominal };
    const timeMode = times.every(isNum) && times.every((t, i) => i === 0 || t >= times[i - 1]);
    if (timeMode) {
      const newest = times[times.length - 1];
      const xs = times.map((t) => (t - newest) / 1000);
      return { timeMode, xs, x0: Math.min(xs[0], -nominal) };
    }
    const n = hist.length;
    return { timeMode, xs: hist.map((_, i) => i - (n - 1)), x0: -((snap ? snap.constants.history_len : 120) - 1) };
  }

  function axisLabel(x, timeMode, step) {
    if (timeMode) return x === 0 ? "0 s" : `${MINUS}${Math.abs(x).toFixed(step < 1 ? 1 : 0)} s`;
    return x === 0 ? "latest" : `${MINUS}${Math.abs(x)}`;
  }

  const chart = { built: false };

  function svgEl(tag, attrs, parent) {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    parent.appendChild(el);
    return el;
  }

  function buildChart() {
    const svg = $("chart");
    chart.svg = svg;
    chart.grid = svgEl("g", { class: "grid" }, svg);
    chart.axis = svgEl("g", { class: "axis" }, svg);
    chart.area = svgEl("path", { class: "series-area" }, svg);
    chart.thresh = svgEl("line", { class: "thresh" }, svg);
    chart.threshLabel = svgEl("text", { class: "thresh-label", "text-anchor": "start" }, svg);
    chart.line = svgEl("path", { class: "series-line" }, svg);
    chart.strip = svgEl("g", {}, svg);
    chart.cross = svgEl("line", { class: "crosshair", visibility: "hidden" }, svg);
    chart.dot = svgEl("circle", { class: "cursor-dot", r: 4, visibility: "hidden" }, svg);
    chart.built = true;

    const pick = (clientX) => {
      if (!chart.layout || !chart.layout.n) return;
      const { X, xs } = chart.layout;
      const x = clientX - svg.getBoundingClientRect().left;
      let best = 0;
      for (let i = 1; i < xs.length; i++) if (Math.abs(X(xs[i]) - x) < Math.abs(X(xs[best]) - x)) best = i;
      cursorFromEnd = xs.length - 1 - best;
      queueRender();
    };
    svg.addEventListener("pointermove", (e) => pick(e.clientX));
    svg.addEventListener("pointerleave", () => { cursorFromEnd = null; queueRender(); });
    svg.addEventListener("blur", () => { cursorFromEnd = null; queueRender(); });
    svg.addEventListener("keydown", (e) => {
      const n = chart.layout ? chart.layout.n : 0;
      if (!n) return;
      const cur = cursorFromEnd ?? 0;
      if (e.key === "ArrowLeft") cursorFromEnd = Math.min(n - 1, cur + (cursorFromEnd === null ? 0 : 1));
      else if (e.key === "ArrowRight") cursorFromEnd = Math.max(0, cur - 1);
      else if (e.key === "Home") cursorFromEnd = n - 1;
      else if (e.key === "End") cursorFromEnd = 0;
      else if (e.key === "Escape") cursorFromEnd = null;
      else return;
      e.preventDefault();
      queueRender();
    });
  }

  function niceStep(raw, steps) {
    return steps.find((s) => s >= raw) || steps[steps.length - 1];
  }

  function renderChart(s) {
    if (!chart.built) buildChart();
    const svg = chart.svg;
    const hist = (s && s.telemetry.history) || [];
    const box = svg.getBoundingClientRect();
    const W = box.width, H = box.height;
    if (!W || !H) return;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    $("chart-empty").hidden = hist.length > 0;
    setText($("chart-empty"), s ? "No telemetry yet" : "Connecting to the console");

    const m = { l: 34, r: 14, t: 10 };
    const tickBand = 20, stripH = 7, stripGap = 7;
    const plotB = H - tickBand - stripH - stripGap;
    const { timeMode, xs, x0 } = historyAxis(hist);
    const X = (v) => m.l + ((v - x0) / (0 - x0)) * (W - m.l - m.r);
    const Y = (p) => m.t + (1 - clamp01(p)) * (plotB - m.t);
    chart.layout = { X, xs, n: hist.length };

    // grid and axes
    chart.grid.replaceChildren();
    chart.axis.replaceChildren();
    for (const p of [0, 0.5, 1]) {
      svgEl("line", { x1: m.l, x2: W - m.r, y1: Math.round(Y(p)) + 0.5, y2: Math.round(Y(p)) + 0.5 }, chart.grid);
      const label = svgEl("text", { x: m.l - 8, y: Y(p) + 4, "text-anchor": "end" }, chart.axis);
      label.textContent = p === 0 ? "0" : p === 1 ? "1.0" : "0.5";
    }
    const span = -x0;
    const step = timeMode ? niceStep(span / 6, [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600])
                          : niceStep(span / 6, [5, 10, 20, 25, 50, 100]);
    for (let v = 0; v >= x0 - 1e-9; v -= step) {
      const x = X(v);
      const label = svgEl("text", { x, y: H - 5, "text-anchor": v === 0 ? "end" : "middle" }, chart.axis);
      label.textContent = axisLabel(Number(v.toFixed(3)), timeMode, step);
    }

    // threshold where a verdict counts as gas
    const trip = s ? s.constants.p_gas_trip : 0.6;
    const ty = Math.round(Y(trip)) + 0.5;
    chart.thresh.setAttribute("x1", m.l);
    chart.thresh.setAttribute("x2", W - m.r);
    chart.thresh.setAttribute("y1", ty);
    chart.thresh.setAttribute("y2", ty);
    chart.threshLabel.setAttribute("x", m.l + 6);
    chart.threshLabel.setAttribute("y", ty - 5);
    chart.threshLabel.textContent = trip.toFixed(2);

    // p(gas): held between updates, so drawn as steps
    let d = "";
    hist.forEach((h, i) => {
      const x = X(xs[i]).toFixed(1), y = Y(isNum(h[0]) ? h[0] : 0).toFixed(1);
      d += i === 0 ? `M${x},${y}` : `H${x}V${y}`;
    });
    chart.line.setAttribute("d", d);
    chart.area.setAttribute("d", hist.length ? `${d}V${plotB}H${X(xs[0]).toFixed(1)}Z` : "");

    // valve state per update, as runs: closed is a thick bar, open a thin line, so it reads without color too
    chart.strip.replaceChildren();
    const sy = plotB + stripGap;
    let start = 0;
    for (let i = 1; i <= hist.length; i++) {
      if (i < hist.length && Boolean(hist[i][1]) === Boolean(hist[start][1])) continue;
      const x1 = X(xs[start]), x2 = i < hist.length ? X(xs[i]) : X(0);
      const closed = Boolean(hist[start][1]);
      svgEl("rect", { x: x1, y: closed ? sy : sy + 2, width: Math.max(2, x2 - x1), height: closed ? stripH : 3, rx: 1.5,
                      class: closed ? "strip-closed" : "strip-open" }, chart.strip);
      start = i;
    }

    // cursor and tooltip
    const tip = $("tooltip");
    if (cursorFromEnd !== null && hist.length) {
      const i = Math.max(0, hist.length - 1 - Math.min(cursorFromEnd, hist.length - 1));
      const h = hist[i], x = X(xs[i]), y = Y(isNum(h[0]) ? h[0] : 0);
      for (const [k, v] of [["x1", x], ["x2", x], ["y1", m.t], ["y2", plotB]]) chart.cross.setAttribute(k, v);
      chart.dot.setAttribute("cx", x);
      chart.dot.setAttribute("cy", y);
      chart.cross.setAttribute("visibility", "visible");
      chart.dot.setAttribute("visibility", "visible");
      const when = timeMode ? (xs[i] === 0 ? "Latest update" : `${Math.abs(xs[i]).toFixed(1)} s before latest`)
                            : (xs[i] === 0 ? "Latest update" : `${Math.abs(xs[i])} updates before latest`);
      tip.replaceChildren();
      const title = document.createElement("div");
      title.className = "tooltip-title";
      title.textContent = when;
      tip.append(title, tooltipRow("p(gas)", fixed(h[0], 3), "key-line"),
                 tooltipRow("Valve", h[1] ? "closed" : "open", h[1] ? "key-closed" : "key-open"));
      tip.hidden = false;
      const tw = tip.offsetWidth;
      tip.style.left = `${x + 14 + tw > W ? x - 14 - tw : x + 14}px`;
    } else {
      chart.cross.setAttribute("visibility", "hidden");
      chart.dot.setAttribute("visibility", "hidden");
      tip.hidden = true;
    }

    // a sentence for screen readers
    if (hist.length) {
      const ps = hist.map((h) => h[0]).filter(isNum);
      const last = hist[hist.length - 1];
      svg.setAttribute("aria-label", `p(gas) for the last ${hist.length} updates. Latest ${fixed(last[0], 2)}, highest ${fixed(Math.max(...ps), 2)}. Valve ${last[1] ? "closed" : "open"} at the latest update. Use the arrow keys to read single updates.`);
    } else {
      svg.setAttribute("aria-label", "No telemetry yet");
    }
  }

  function tooltipRow(name, value, keyClass) {
    const row = document.createElement("div");
    row.className = "tooltip-row";
    const left = document.createElement("span");
    const key = document.createElement("span");
    key.className = `key ${keyClass}`;
    left.append(key, document.createTextNode(name));
    const right = document.createElement("b");
    right.textContent = value;
    row.append(left, right);
    return row;
  }

  function renderTable(s) {
    const hist = (s && s.telemetry.history) || [];
    const { timeMode, xs } = historyAxis(hist);
    setText($("table-time"), timeMode ? "Seconds before latest" : "Updates before latest");
    const frag = document.createDocumentFragment();
    for (let i = hist.length - 1; i >= 0; i--) {
      const tr = document.createElement("tr");
      const t = document.createElement("td");
      t.textContent = xs[i] === 0 ? "latest" : timeMode ? `${MINUS}${Math.abs(xs[i]).toFixed(1)}` : `${MINUS}${Math.abs(xs[i])}`;
      const p = document.createElement("td");
      p.textContent = fixed(hist[i][0], 3);
      const v = document.createElement("td");
      const key = document.createElement("span");
      key.className = `key ${hist[i][1] ? "key-closed" : "key-open"}`;
      v.append(key, document.createTextNode(hist[i][1] ? "closed" : "open"));
      tr.append(t, p, v);
      frag.append(tr);
    }
    $("table-body").replaceChildren(frag);
  }

  function renderHistory(s) {
    const hist = (s && s.telemetry.history) || [];
    const { timeMode } = historyAxis(hist);
    const n = s ? s.constants.history_len : 120;
    let meta = timeMode ? `last ${n} updates · seconds on the supervisor’s clock` : `last ${n} updates`;
    if (paused) meta += " · paused, status above stays live";
    setText($("history-meta"), meta);
    $("legend").hidden = view !== "chart";
    $("chart-wrap").hidden = view !== "chart";
    $("table-wrap").hidden = view !== "table";
    if (view === "chart") renderChart(s);
    else renderTable(s);
  }

  // ------------------------------------------------------------------ render
  function render() {
    renderQueued = false;
    const link = linkState();
    renderLink(link);
    const v = valveState(link);
    renderValve(v, link);
    renderVerdict(link);
    renderTiming(link);
    renderBanners(link);
    renderSimButton(link);
    renderFooter();
    renderConnection(link);
    renderTitle(v, link);
    announce(v, link);
    const shown = paused ? frozen : snap;
    renderImage(shown, link);
    renderHistory(shown);
  }

  // ------------------------------------------------------------------ controls
  function setPaused(on) {
    paused = on;
    frozen = on ? snap : null;
    const btn = $("pause-btn");
    btn.setAttribute("aria-pressed", String(on));
    setIcon(btn.querySelector("svg"), on ? "i-play" : "i-pause");
    btn.querySelector("svg").classList.toggle("icon-fill", on);
    setText(btn.querySelector("span"), on ? "Resume" : "Pause");
    queueRender();
  }

  function setView(next) {
    view = next;
    for (const b of document.querySelectorAll(".segmented button")) b.setAttribute("aria-pressed", String(b.dataset.view === view));
    queueRender();
  }

  function note(text) {
    const el = $("sim-note");
    el.textContent = text;
    clearTimeout(note.timer);
    note.timer = setTimeout(() => { el.textContent = ""; }, 6000);
  }

  async function simulator(action) {
    try {
      const r = await fetch(`/api/sim/${action}`, { method: "POST", headers: { "X-Mirage-Control": "1" } });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) note(body.error || `The console refused (${r.status}).`);
      else note(action === "start" ? "Simulator running: fake_pi.py is sending to this Mac." : "Simulator stopped.");
    } catch {
      note("The console didn’t answer.");
    }
  }

  $("sim-btn").addEventListener("click", async () => {
    const btn = $("sim-btn");
    if (!snap) return;
    if (btn.getAttribute("aria-disabled") === "true") {
      note(snap.sim.blocked || "The simulator can’t start right now.");
      return;
    }
    btn.setAttribute("aria-busy", "true");
    await simulator(snap.sim.running ? "stop" : "start");
    btn.removeAttribute("aria-busy");
  });
  $("pause-btn").addEventListener("click", () => setPaused(!paused));
  for (const b of document.querySelectorAll(".segmented button")) b.addEventListener("click", () => setView(b.dataset.view));

  $("copy-btn").addEventListener("click", async () => {
    const label = $("copy-btn").querySelector("span");
    const text = $("reset-cmd").textContent;
    try {
      await navigator.clipboard.writeText(text);
      label.textContent = "Copied";
    } catch {
      const range = document.createRange();
      range.selectNodeContents($("reset-cmd"));
      getSelection().removeAllRanges();
      getSelection().addRange(range);
      label.textContent = "Press ⌘C";
    }
    setTimeout(() => { label.textContent = "Copy"; }, 1600);
  });

  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
    const tag = e.target && e.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || (e.target && e.target.isContentEditable)) return;
    if (e.key === "p" || e.key === "P") { setPaused(!paused); e.preventDefault(); }
    else if (e.key === "t" || e.key === "T") { setView(view === "chart" ? "table" : "chart"); e.preventDefault(); }
  });

  new ResizeObserver(queueRender).observe($("chart-wrap"));
  document.addEventListener("visibilitychange", queueRender);
  connect();
  render();
})();
