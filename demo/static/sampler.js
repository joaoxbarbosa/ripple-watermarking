/* Sampler visualizer: watch the keyed exponential race, token by token.
 *
 *   SamplerViz.mount(containerEl, { watermarked: trace, plain: trace })
 *
 * A trace is { steps: [...], context_width } as recorded by the server. Each step
 * has the top candidate tokens with their probability p and keyed uniform u,
 * the index of the chosen candidate, the sampler mode, entropy and the
 * detector's score for the chosen token.
 *
 * The race is the actual sampling rule: every candidate's arrival time is
 * E / p with E = -ln(u), and the first to arrive is emitted. With the key, u
 * comes from hashing the key with the previous tokens. Without it (or in the
 * warm-up / repeated-context steps) the real draw is fresh randomness that we
 * can't replay, so we show fresh dice in which the recorded token wins.
 *
 * Styling reads the host page's CSS variables (--ink, --bg, --line, --hair,
 * --muted, --faint, --field, --heat, --mono, --sans).
 */
(function () {
  const CSS = `
  .sv { --lane-h: 34px; font: 14px/1.4 var(--sans); color: var(--ink); }
  .sv * { box-sizing: border-box; }
  .sv-bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 18px; }
  .sv-bar button { font: 500 12px var(--mono); text-transform: uppercase; letter-spacing: .05em; padding: 7px 11px; min-width: 36px; }
  .sv-bar button[aria-pressed="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink); }
  .sv-seg { display: inline-flex; }
  .sv-seg button + button { border-left: 0; }
  .sv-spacer { flex: 1; }
  .sv-scrub { flex: 1 1 220px; min-width: 160px; }
  .sv-head { display: grid; grid-template-columns: auto 1fr auto; gap: 16px; align-items: center; border: 1px solid var(--line); border-bottom: 0; padding: 12px 16px; }
  .sv-step { font: 500 12px var(--mono); color: var(--muted); white-space: nowrap; }
  .sv-mode { display: flex; align-items: center; gap: 12px; min-width: 0; }
  .sv-badge { font: 600 11px var(--mono); letter-spacing: .08em; text-transform: uppercase; padding: 4px 8px; border: 1px solid var(--line); white-space: nowrap; }
  .sv-badge.keyed { background: var(--ink); color: var(--bg); border-color: var(--ink); }
  .sv-modetxt { font-size: 12px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sv-ent { font: 500 12px var(--mono); color: var(--muted); white-space: nowrap; }
  .sv-ctx { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; border: 1px solid var(--line); border-bottom: 0; padding: 10px 16px; font: 12px var(--mono); }
  .sv-ctx .k { padding: 3px 7px; background: var(--ink); color: var(--bg); }
  .sv-ctx .t { padding: 3px 7px; border: 1px solid var(--hair); background: var(--field); white-space: pre; }
  .sv-ctx .op { color: var(--muted); }
  .sv-ctx .hash { padding: 3px 7px; border: 1px dashed var(--line); transition: background .2s, color .2s; }
  .sv-ctx .hash.on { background: var(--ink); color: var(--bg); }
  .sv-ctx.off .k, .sv-ctx.off .hash { opacity: .3; }
  .sv-stage { border: 1px solid var(--line); position: relative; }
  .sv-cols { display: grid; grid-template-columns: minmax(90px, 1.1fr) minmax(110px, 1.4fr) minmax(160px, 3fr); font: 600 10px var(--sans); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--hair); }
  .sv-cols div { padding: 8px 12px; }
  .sv-cols div:last-child { display: flex; justify-content: space-between; }
  .sv-lane { display: grid; grid-template-columns: minmax(90px, 1.1fr) minmax(110px, 1.4fr) minmax(160px, 3fr); height: var(--lane-h); align-items: center; border-bottom: 1px solid var(--hair); transition: background .25s, color .25s; }
  .sv-lane:last-child { border-bottom: 0; }
  .sv-lane.win { background: var(--ink); color: var(--bg); }
  .sv-lane.win .sv-prob i { background: var(--bg); }
  .sv-lane.win .sv-track::before { background: var(--bg); opacity: .35; }
  .sv-lane.win .sv-runner { background: var(--bg); }
  .sv-lane.win .sv-u, .sv-lane.win .sv-pv { color: var(--bg); }
  .sv-lane.dim { opacity: .38; }
  .sv-tok { padding: 0 12px; font: 500 13px var(--mono); white-space: pre; overflow: hidden; text-overflow: ellipsis; }
  .sv-prob { padding: 0 12px; display: flex; align-items: center; gap: 8px; min-width: 0; }
  .sv-prob i { display: block; height: 8px; background: var(--ink); flex: none; transition: width .35s cubic-bezier(.2,.8,.2,1); }
  .sv-pv { font: 11px var(--mono); color: var(--muted); white-space: nowrap; }
  .sv-track { position: relative; height: 100%; margin-right: 12px; }
  .sv-track::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: var(--hair); }
  .sv-track::after { content: ""; position: absolute; right: 54px; top: 6px; bottom: 6px; width: 2px;
    background: repeating-linear-gradient(var(--ink) 0 3px, transparent 3px 6px); }
  .sv-runner { position: absolute; left: 0; top: 50%; width: 12px; height: 12px; margin-top: -6px; background: var(--ink); will-change: transform; }
  .sv-u { position: absolute; right: 0; top: 50%; transform: translateY(-50%); width: 48px; text-align: right; font: 11px var(--mono); color: var(--muted); }
  .sv-u.roll { color: var(--faint); }
  .sv-out { border: 1px solid var(--line); border-top: 0; display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); }
  .sv-out > div { padding: 16px; min-width: 0; }
  .sv-out > div + div { border-left: 1px solid var(--line); }
  .sv-lbl { font: 600 10px var(--sans); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; display: flex; justify-content: space-between; gap: 8px; }
  .sv-lbl b { font: 500 11px var(--mono); color: var(--ink); letter-spacing: 0; text-transform: none; }
  .sv-text { white-space: pre-wrap; word-break: break-word; line-height: 1.85; font-size: 14px; max-height: 200px; overflow: auto; }
  .sv-text span { padding: 1px 0; transition: background .4s; }
  .sv-text span.warm { color: var(--muted); }
  .sv-text span.rep { color: var(--faint); text-decoration: underline dotted var(--faint); text-underline-offset: 3px; }
  .sv-text span.cur { outline: 1px solid var(--ink); outline-offset: 1px; }
  .sv-text span.ghost { visibility: hidden; }
  .sv-chart svg { width: 100%; height: 150px; display: block; overflow: visible; }
  .sv-read { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; font: 12px var(--mono); }
  .sv-read span { color: var(--muted); display: block; font: 600 10px var(--sans); letter-spacing: .08em; text-transform: uppercase; }
  .sv-flyer { position: fixed; z-index: 50; pointer-events: none; font: 600 13px var(--mono); white-space: pre; padding: 2px 4px; background: var(--ink); color: var(--bg); }
  .sv-note { margin-top: 14px; font-size: 13px; color: var(--muted); max-width: 760px; }
  .sv-note b { color: var(--ink); font-weight: 600; }
  @media (max-width: 760px) {
    .sv-head { grid-template-columns: 1fr auto; row-gap: 8px; }
    .sv-mode { grid-column: 1 / -1; order: 3; align-items: flex-start; }
    .sv-modetxt { white-space: normal; }
    .sv-out { grid-template-columns: 1fr; }
    .sv-out > div + div { border-left: 0; border-top: 1px solid var(--line); }
    .sv-cols, .sv-lane { grid-template-columns: minmax(70px, 1fr) minmax(70px, 1fr) minmax(120px, 2fr); }
    .sv-pv { display: none; }
  }`;

  const MODES = {
    keyed: ["Keyed draw", "u = SHAKE-256(key, last tokens): fair to outsiders, replayable by the detector"],
    warmup: ["Warm-up", "true randomness until the reply has 8 bits of entropy, so repeats of a prompt still differ"],
    repeat: ["Repeated context", "this context appeared before: true randomness, and the detector skips it"],
    plain: ["No key", "fresh random draws: an ordinary sample"],
    start: ["Start", "not enough context yet: true randomness"],
  };

  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const show = (t) => (t === "" ? "∅" : t.replace(/\n/g, "↵").replace(/\t/g, "⇥"));
  const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Tail probability of Gamma(n,1) >= S (Wilson–Hilferty), as log10, for the readout.
  function log10pGamma(S, n) {
    if (n <= 0) return 0;
    const z = (Math.cbrt(S / n) - (1 - 1 / (9 * n))) / Math.sqrt(1 / (9 * n));
    if (z < 3) { // normal tail via erfc approximation
      const t = 1 / (1 + 0.5 * Math.abs(z)), y = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
      const q = z >= 0 ? y / 2 : 1 - y / 2;
      return Math.log10(Math.max(q, 1e-300));
    }
    return (-z * z / 2 - Math.log(z * Math.sqrt(2 * Math.PI))) / Math.LN10;
  }
  const fmtP = (l) => (l > -2 ? (10 ** l).toFixed(2) : l > -6 ? `1 in ${Math.round(10 ** -l).toLocaleString()}` : `1 in 10^${Math.floor(-l)}`);

  function mount(root, traces) {
    if (!document.getElementById("sv-style")) { const s = el("style"); s.id = "sv-style"; s.textContent = CSS; document.head.appendChild(s); }
    root.classList.add("sv");
    root.replaceChildren();

    // ---------- controls
    const bar = el("div", "sv-bar");
    const play = el("button", null, "Play");
    const prev = el("button", null, "‹"), next = el("button", null, "›"), restart = el("button", null, "⟲");
    prev.title = "Previous step"; next.title = "Next step"; restart.title = "Restart";
    const keySeg = el("div", "sv-seg");
    const withKey = el("button", null, "With key"), noKey = el("button", null, "Without key");
    keySeg.append(withKey, noKey);
    const speedSeg = el("div", "sv-seg");
    const speeds = [1, 2, 4, 8].map((s) => { const b = el("button", null, `${s}×`); b.dataset.s = s; speedSeg.appendChild(b); return b; });
    const scrub = el("input", "sv-scrub"); scrub.type = "range"; scrub.min = 0; scrub.value = 0; scrub.setAttribute("aria-label", "Step");
    bar.append(play, prev, next, restart, keySeg, speedSeg, scrub);

    // ---------- header + context
    const head = el("div", "sv-head");
    const stepTxt = el("div", "sv-step");
    const modeBox = el("div", "sv-mode"); const badge = el("span", "sv-badge"); const modeTxt = el("span", "sv-modetxt");
    modeBox.append(badge, modeTxt);
    const ent = el("div", "sv-ent");
    head.append(stepTxt, modeBox, ent);
    const ctx = el("div", "sv-ctx");

    // ---------- lanes
    const stage = el("div", "sv-stage");
    const cols = el("div", "sv-cols");
    cols.innerHTML = "<div>Candidate</div><div>Model probability</div><div><span>The race: speed = p ÷ −ln u</span><span>u</span></div>";
    stage.appendChild(cols);
    const lanes = [];
    for (let i = 0; i < 8; i++) {
      const lane = el("div", "sv-lane");
      const tok = el("div", "sv-tok");
      const prob = el("div", "sv-prob"); const pbar = el("i"); const pv = el("span", "sv-pv"); prob.append(pbar, pv);
      const track = el("div", "sv-track"); const runner = el("div", "sv-runner"); const u = el("span", "sv-u");
      track.append(runner, u);
      lane.append(tok, prob, track);
      stage.appendChild(lane);
      lanes.push({ lane, tok, pbar, pv, track, runner, u });
    }

    // ---------- output: text + evidence chart
    const out = el("div", "sv-out");
    const textBox = el("div"); const textLbl = el("div", "sv-lbl"); textLbl.innerHTML = "<span>Generated text</span><b></b>";
    const text = el("div", "sv-text"); textBox.append(textLbl, text);
    const chartBox = el("div", "sv-chart"); const chartLbl = el("div", "sv-lbl"); chartLbl.innerHTML = "<span>Detector evidence</span><b></b>";
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "0 0 300 150"); svg.setAttribute("preserveAspectRatio", "none");
    const band = document.createElementNS(NS, "path"); band.setAttribute("fill", "rgba(var(--heat),0.10)");
    const zero = document.createElementNS(NS, "line"); zero.setAttribute("stroke", "var(--hair)"); zero.setAttribute("stroke-width", "1"); zero.setAttribute("vector-effect", "non-scaling-stroke");
    const line = document.createElementNS(NS, "path"); line.setAttribute("fill", "none"); line.setAttribute("stroke", "var(--ink)"); line.setAttribute("stroke-width", "2"); line.setAttribute("vector-effect", "non-scaling-stroke");
    const dot = document.createElementNS(NS, "rect"); dot.setAttribute("width", "6"); dot.setAttribute("height", "6"); dot.setAttribute("fill", "var(--ink)");
    svg.append(band, zero, line, dot);
    const read = el("div", "sv-read");
    read.innerHTML = `<div><span>Excess score</span><b data-r="ex">—</b></div><div><span>Chance of a fluke</span><b data-r="p">—</b></div>`;
    chartBox.append(chartLbl, svg, read);
    out.append(textBox, chartBox);

    const note = el("p", "sv-note");
    note.innerHTML = "Every candidate runs a race and the first to the line is the next token. A runner's speed is its probability divided by a random draw <b>−ln u</b>, so the winner is always an exact sample from the model. <b>With the key</b>, the draws come from hashing the secret key with the last few tokens. To anyone without the key that's indistinguishable from fresh dice, but the detector can recompute it and sees that the winners kept drawing <b>u close to 1</b>. The line climbs out of the band that ordinary text stays in.";

    root.append(bar, head, ctx, stage, out, note);

    // ---------- state
    let which = traces.watermarked ? "watermarked" : "plain";
    let trace = traces[which];
    let i = 0, playing = false, speed = 2, timer = null, raf = null, token = 0, live = false;
    let cum = []; // cumulative evidence per step (null for skipped)

    function rebuildCum() {
      cum = []; let s = 0, n = 0;
      for (const st of trace.steps) {
        if (st.score != null) { s += st.score; n += 1; }
        cum.push({ excess: s - n, S: s, n });
      }
    }

    // Draws used by the race. Keyed steps use the recorded keyed u; other steps
    // get fresh dice, rerolled until the recorded winner wins.
    function drawsFor(st) {
      const c = st.cands;
      if (st.mode === "keyed") return c.map((x) => Math.max(1e-12, -Math.log(x.u)));
      for (let tries = 0; tries < 200; tries++) {
        const E = c.map(() => -Math.log(Math.random()));
        const T = c.map((x, j) => E[j] / Math.max(x.p, 1e-12));
        if (T.indexOf(Math.min(...T)) === st.chosen) return E;
      }
      const E = c.map(() => 1 + Math.random()); // fall back: give the winner a tiny draw
      E[st.chosen] = 1e-3; return E;
    }

    function setButtons() {
      withKey.setAttribute("aria-pressed", which === "watermarked");
      noKey.setAttribute("aria-pressed", which === "plain");
      withKey.disabled = !traces.watermarked; noKey.disabled = !traces.plain;
      speeds.forEach((b) => b.setAttribute("aria-pressed", +b.dataset.s === speed));
      play.textContent = live ? "Live" : playing ? "Pause" : i >= trace.steps.length - 1 ? "Replay" : "Play";
      [play, prev, next, restart, scrub].forEach((c) => (c.disabled = live));
      scrub.max = Math.max(0, trace.steps.length - 1); scrub.value = i;
    }

    function renderText(upto, ghostLast) {
      text.replaceChildren();
      for (let k = 0; k <= upto; k++) {
        const st = trace.steps[k];
        const s = el("span", null, st.cands[st.chosen].t);
        if (st.mode === "repeat") s.className = "rep";
        else if (st.mode !== "keyed" && st.mode !== "plain") s.className = "warm";
        if (st.score != null) s.style.background = `rgba(var(--heat),${(0.03 + 0.42 * Math.max(0, Math.min(1, (st.score - 1) / 4))).toFixed(3)})`;
        if (k === upto) { s.classList.add("cur"); if (ghostLast) s.classList.add("ghost"); }
        text.appendChild(s);
      }
      text.scrollTop = text.scrollHeight;
      textLbl.querySelector("b").textContent = `${upto + 1} tokens`;
      return text.lastChild;
    }

    // One vertical scale for both traces, so "with key" and "without key" compare directly.
    function chartScale() {
      let m = 8;
      for (const t of Object.values(traces)) {
        if (!t) continue;
        let s = 0, n = 0;
        for (const st of t.steps) { if (st.score != null) { s += st.score; n += 1; } m = Math.max(m, Math.abs(s - n), 2.5 * Math.sqrt(n)); }
      }
      return m;
    }

    function renderChart(upto) {
      const N = trace.steps.length, W = 300, H = 150;
      const maxAbs = chartScale();
      const y = (v) => H / 2 - (v / maxAbs) * (H / 2 - 6);
      const x = (k) => (k / Math.max(1, N - 1)) * W;
      // ±2σ band for ordinary text: excess ~ N(0, n)
      let top = "", bot = "";
      for (let k = 0; k < N; k++) { const sd = 2 * Math.sqrt(Math.max(cum[k].n, 0)); top += `${k ? "L" : "M"}${x(k)},${y(sd)}`; bot = `L${x(k)},${y(-sd)}` + bot; }
      band.setAttribute("d", top + bot + "Z");
      zero.setAttribute("x1", 0); zero.setAttribute("x2", W); zero.setAttribute("y1", y(0)); zero.setAttribute("y2", y(0));
      let d = "";
      for (let k = 0; k <= upto; k++) d += `${k ? "L" : "M"}${x(k)},${y(cum[k].excess)}`;
      line.setAttribute("d", d);
      dot.setAttribute("x", x(upto) - 3); dot.setAttribute("y", y(cum[upto].excess) - 3);
      const c = cum[upto];
      read.querySelector('[data-r="ex"]').textContent = `${c.excess >= 0 ? "+" : ""}${c.excess.toFixed(1)}  (${c.n} scored)`;
      read.querySelector('[data-r="p"]').textContent = fmtP(log10pGamma(c.S, c.n));
      chartLbl.querySelector("b").textContent = c.n ? `mean ${(c.S / c.n).toFixed(2)} / token (ordinary: 1.00)` : "";
    }

    function renderHeader(st) {
      stepTxt.textContent = `Step ${i + 1} / ${trace.steps.length}`;
      const [b, t] = MODES[st.mode] || [st.mode, ""];
      badge.textContent = b; badge.className = "sv-badge" + (st.mode === "keyed" ? " keyed" : "");
      modeTxt.textContent = t;
      ent.textContent = `${st.entropy.toFixed(2)} bits`;
      // context window = previous chosen tokens
      ctx.replaceChildren();
      ctx.classList.toggle("off", st.mode !== "keyed");
      ctx.append(el("span", "k", "KEY"), el("span", "op", "+"));
      const w = trace.context_width || 3;
      for (let k = i - w; k < i; k++) ctx.append(el("span", "t", k < 0 ? "…prompt" : show(trace.steps[k].cands[trace.steps[k].chosen].t)));
      ctx.append(el("span", "op", "→"), el("span", "hash", "SHAKE-256 → u"));
    }

    function renderLanes(st, E, frac, settled) {
      const maxp = Math.max(...st.cands.map((c) => c.p));
      st.cands.forEach((c, j) => {
        const L = lanes[j];
        L.tok.textContent = show(c.t);
        L.pbar.style.width = `${Math.max(1, (c.p / maxp) * 70)}%`;
        L.pv.textContent = c.p >= 0.001 ? `${(c.p * 100).toFixed(1)}%` : c.p.toExponential(0);
        const tw = L.track.clientWidth - 54 - 12;
        L.runner.style.transform = `translateX(${Math.max(0, Math.min(1, frac[j])) * tw}px)`;
        L.u.textContent = st.mode === "keyed" ? c.u.toFixed(3) : (Math.exp(-E[j])).toFixed(3);
        L.lane.classList.toggle("win", settled && j === st.chosen);
        L.lane.classList.toggle("dim", settled && j !== st.chosen);
      });
      for (let j = st.cands.length; j < lanes.length; j++) lanes[j].lane.style.visibility = "hidden";
      for (let j = 0; j < st.cands.length; j++) lanes[j].lane.style.visibility = "";
    }

    function positions(st, E, t) { // t in [0,1] = fraction of the winner's arrival time
      const T = st.cands.map((c, j) => E[j] / Math.max(c.p, 1e-12));
      const Tw = T[st.chosen];
      return T.map((Tj) => (t * Tw) / Tj);
    }

    function empty(msg) {
      stepTxt.textContent = "Step 0"; badge.textContent = live ? "Live" : "—"; badge.className = "sv-badge" + (live ? " keyed" : "");
      modeTxt.textContent = msg; ent.textContent = ""; ctx.replaceChildren();
      lanes.forEach((L) => (L.lane.style.visibility = "hidden"));
      text.replaceChildren(el("span", "warm", msg));
      line.setAttribute("d", ""); band.setAttribute("d", "");
      setButtons();
    }

    // Show step i fully settled (no animation).
    function snap() {
      if (!trace.steps.length) return empty(live ? "Waiting for the first token…" : "No steps.");
      const st = trace.steps[i];
      const E = drawsFor(st);
      renderHeader(st);
      renderLanes(st, E, positions(st, E, 1), true);
      renderText(i, false);
      renderChart(i);
      setButtons();
    }

    // Animate step i: roll the dice, run the race, fly the winner into the text.
    function animate(done) {
      const my = ++token;
      const st = trace.steps[i];
      const E = drawsFor(st);
      const k = 1 / speed;
      renderHeader(st);
      renderText(Math.max(0, i - 1), false);
      if (i === 0) text.replaceChildren();
      renderChart(Math.max(0, i - 1));
      const zeros = st.cands.map(() => 0);
      renderLanes(st, E, zeros, false);
      if (reduced() || speed >= 8) { snap(); return setTimeout(done, 120 * k); }

      const hash = ctx.querySelector(".hash");
      const rollMs = st.mode === "keyed" ? 380 * k : 220 * k, raceMs = 1000 * k;
      const t0 = performance.now();
      const frame = (now) => {
        if (my !== token) return;
        const t = now - t0;
        if (t < rollMs) { // dice roll: scramble the u readouts
          hash && hash.classList.toggle("on", Math.floor(t / 70) % 2 === 0);
          lanes.forEach((L, j) => { if (j < st.cands.length) { L.u.textContent = Math.random().toFixed(3); L.u.classList.add("roll"); } });
          raf = requestAnimationFrame(frame); return;
        }
        hash && hash.classList.add("on");
        lanes.forEach((L) => L.u.classList.remove("roll"));
        const f = Math.min(1, (t - rollMs) / raceMs);
        const ease = f; // constant speed: positions are linear in time, like a real race
        renderLanes(st, E, positions(st, E, ease), false);
        if (f < 1) { raf = requestAnimationFrame(frame); return; }
        renderLanes(st, E, positions(st, E, 1), true);
        fly(st, my, () => { renderChart(i); setButtons(); done(); });
      };
      raf = requestAnimationFrame(frame);
    }

    function fly(st, my, done) {
      const target = renderText(i, true);
      const from = lanes[st.chosen].tok.getBoundingClientRect();
      const to = target.getBoundingClientRect();
      const flyer = el("div", "sv-flyer", show(st.cands[st.chosen].t));
      document.body.appendChild(flyer);
      flyer.style.left = `${from.left + 8}px`; flyer.style.top = `${from.top + from.height / 2 - 10}px`;
      const dx = to.left - from.left - 8, dy = to.top - (from.top + from.height / 2 - 10);
      const anim = flyer.animate(
        [{ transform: "translate(0,0) scale(1.15)", opacity: 1 }, { transform: `translate(${dx}px,${dy}px) scale(1)`, opacity: 0.9 }],
        { duration: 520 / speed, easing: "cubic-bezier(.5,0,.2,1)" });
      anim.onfinish = () => { flyer.remove(); if (my === token) { target.classList.remove("ghost"); done(); } };
    }

    function stop() { playing = false; token++; cancelAnimationFrame(raf); clearTimeout(timer); document.querySelectorAll(".sv-flyer").forEach((f) => f.remove()); setButtons(); }
    function loop() {
      if (!playing) return;
      animate(() => {
        if (!playing) return;
        if (i >= trace.steps.length - 1) { playing = false; setButtons(); return; }
        timer = setTimeout(() => { i += 1; loop(); }, 260 / speed);
      });
      setButtons();
    }
    function go(k) { stop(); i = Math.max(0, Math.min(trace.steps.length - 1, k)); snap(); }

    play.onclick = () => {
      if (playing) return stop();
      if (i >= trace.steps.length - 1) i = 0;
      playing = true; loop();
    };
    prev.onclick = () => go(i - 1);
    next.onclick = () => { stop(); i = Math.min(trace.steps.length - 1, i + 1); animate(() => setButtons()); };
    restart.onclick = () => { go(0); };
    scrub.oninput = () => go(+scrub.value);
    speeds.forEach((b) => (b.onclick = () => { speed = +b.dataset.s; setButtons(); }));
    const switchTo = (w) => { if (!traces[w]) return; const was = playing; stop(); which = w; trace = traces[w]; rebuildCum(); i = Math.min(i, trace.steps.length - 1); snap(); if (was) { playing = true; loop(); } };
    withKey.onclick = () => switchTo("watermarked");
    noKey.onclick = () => switchTo("plain");
    addEventListener("resize", () => { if (!playing) snap(); });

    // Live: each streamed step runs a quick race as it arrives (the model is faster
    // than the full animation); a new token cancels the previous race.
    function quick() {
      const my = ++token;
      const st = trace.steps[i];
      const E = drawsFor(st);
      renderHeader(st);
      renderText(i, false);
      renderChart(i);
      setButtons();
      if (reduced()) return renderLanes(st, E, positions(st, E, 1), true);
      const t0 = performance.now(), ms = 110;
      const frame = (now) => {
        if (my !== token) return;
        const f = Math.min(1, (now - t0) / ms);
        renderLanes(st, E, positions(st, E, f), f >= 1);
        if (f < 1) raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);
    }

    rebuildCum(); snap();
    return {
      load(newTraces) { stop(); live = false; traces = newTraces; which = traces.watermarked ? "watermarked" : "plain"; trace = traces[which]; i = 0; rebuildCum(); snap(); },
      play() { if (!playing) play.onclick(); },
      /** Start a live stream. ``watermark`` picks which side the steps belong to. */
      startLive(contextWidth, watermark = true) {
        stop(); live = true;
        which = watermark ? "watermarked" : "plain";
        trace = { steps: [], context_width: contextWidth || 3 };
        traces = { [which]: trace };
        i = 0; rebuildCum(); snap();
      },
      push(step) {
        if (!live || !step) return;
        trace.steps.push(step); rebuildCum();
        i = trace.steps.length - 1; quick();
      },
      /** Stream finished: hand control back for slow replays. */
      endLive() { if (!live) return; live = false; token++; i = Math.max(0, trace.steps.length - 1); snap(); },
    };
  }

  window.SamplerViz = { mount };
})();
