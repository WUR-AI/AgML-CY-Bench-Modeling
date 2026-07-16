// Embedded in insights pages (sample-size scatter +/or country bootstrap table).
(function initFamilyRq4Panel() {
  const FAMILY = DATA.family_dashboard;
  const scatterPanel = document.getElementById("rq4-scatter-panel");
  const scatterNote = document.getElementById("rq4-scatter-note");
  const horizonSelect = document.getElementById("rq4-horizon")
    || document.getElementById("family-map-horizon")
    || document.getElementById("family-metrics-horizon");
  const cropSelect = document.getElementById("rq4-crop")
    || document.getElementById("family-map-crop")
    || document.getElementById("family-metrics-crop");
  const rq4Card = document.getElementById("rq4-card");
  const bootstrapNote = document.getElementById("bootstrap-note");
  const bootstrapTable = document.getElementById("bootstrap-table");
  const bootstrapExportLatexBtn = document.getElementById("bootstrap-export-latex");
  const bootstrapExportLatexCopyBtn = document.getElementById("bootstrap-export-latex-copy");
  const countryBootstrapSection = document.getElementById("country-bootstrap-section");

  if (!FAMILY || (!scatterPanel && !bootstrapTable) || !horizonSelect || !cropSelect) return;

  const horizonLabels = FAMILY.horizon_labels || DATA.horizon_labels || {};
  const scatterMetric = FAMILY.sample_scatter_metric || DATA.sample_scatter_metric || {
    key: "relative_nrmse",
    label: "NRMSE / average yield",
    reference: 1.0,
    lower_is_better: true,
  };

  const scatterHorizons = Object.keys(FAMILY.sample_scatter || {});
  const bootstrapHorizons = Object.keys(
    (DATA.country_bootstrap || FAMILY.country_bootstrap || {}).by_horizon || {},
  );
  const horizonOrder = ["eos", "mid", "early", "qtr"].filter(
    hz => scatterHorizons.includes(hz) || bootstrapHorizons.includes(hz),
  );
  if (!horizonOrder.length) {
    if (rq4Card) rq4Card.style.display = "none";
    return;
  }

  let currentBootstrapRows = [];

  function fillHorizonSelect() {
    horizonSelect.innerHTML = horizonOrder.map(hz => {
      const label = horizonLabels[hz] || hz;
      return `<option value="${hz}">${label}</option>`;
    }).join("");
    horizonSelect.value = horizonOrder.includes("eos") ? "eos" : horizonOrder[0];
  }

  function cropsForHorizon(hz) {
    const scatterCrops = Object.keys((FAMILY.sample_scatter || {})[hz] || {});
    const bootstrapCrops = Object.keys(
      ((DATA.country_bootstrap || FAMILY.country_bootstrap || {}).by_horizon || {})[hz] || {},
    );
    const keys = new Set([...scatterCrops, ...bootstrapCrops]);
    const order = ["all", ...(FAMILY.crops || DATA.crops || []).filter(c => keys.has(c))];
    return order.filter(c => c === "all" || keys.has(c));
  }

  function fillCropSelect() {
    const hz = horizonSelect.value;
    const crops = cropsForHorizon(hz);
    cropSelect.innerHTML = crops.map(crop => {
      const label = crop === "all" ? "All crops" : crop.charAt(0).toUpperCase() + crop.slice(1);
      return `<option value="${crop}">${label}</option>`;
    }).join("");
    if (!crops.includes(cropSelect.value)) {
      cropSelect.value = crops.includes("all") ? "all" : crops[0];
    }
  }

  function currentScatterSlice() {
    const hz = horizonSelect.value;
    const crop = cropSelect.value;
    const raw = ((FAMILY.sample_scatter || {})[hz] || {})[crop] || [];
    if (Array.isArray(raw)) {
      return { families: raw, summary: {} };
    }
    return { families: raw.families || [], summary: raw.summary || {} };
  }

  function percentile(sorted, p) {
    if (!sorted.length) return 0;
    const idx = (sorted.length - 1) * p;
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }

  function spearmanRho(xs, ys) {
    if (xs.length < 5) return null;
    const rank = arr => {
      const order = arr.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
      const out = new Array(arr.length);
      order.forEach(([_, i], r) => { out[i] = r + 1; });
      return out;
    };
    const rx = rank(xs);
    const ry = rank(ys);
    const mx = rx.reduce((a, b) => a + b, 0) / rx.length;
    const my = ry.reduce((a, b) => a + b, 0) / ry.length;
    let num = 0; let dx = 0; let dy = 0;
    for (let i = 0; i < rx.length; i++) {
      const vx = rx[i] - mx;
      const vy = ry[i] - my;
      num += vx * vy;
      dx += vx * vx;
      dy += vy * vy;
    }
    return dx && dy ? num / Math.sqrt(dx * dy) : null;
  }

  function renderScatter() {
    if (!scatterPanel) return;
    const metricKey = scatterMetric.key || "relative_nrmse";
    const refY = scatterMetric.reference ?? 1.0;
    const { families, summary } = currentScatterSlice();
    const series = families
      .map(fam => ({
        ...fam,
        points: (fam.points || []).filter(
          p => p[metricKey] !== null && p[metricKey] !== undefined && p.n_train > 0,
        ),
      }))
      .filter(fam => fam.points.length);

    if (!series.length) {
      scatterPanel.innerHTML =
        '<p class="muted">No scatter data for this selection (re-collect summaries with average_yield baseline and n_train).</p>';
      if (scatterNote) {
        scatterNote.textContent =
          "Each point is one crop×country dataset for the family representative.";
      }
      return;
    }

    const allPoints = series.flatMap(fam =>
      fam.points.map(p => ({ ...p, family: fam.family, color: fam.color, rep: fam.display_name })),
    );
    const nTrainSorted = allPoints.map(p => p.n_train).sort((a, b) => a - b);
    const xP05 = summary.x_p05 ?? Math.round(percentile(nTrainSorted, 0.05));
    const xP95 = summary.x_p95 ?? Math.round(percentile(nTrainSorted, 0.95));
    const xP50 = summary.x_p50 ?? Math.round(percentile(nTrainSorted, 0.5));
    const log = v => Math.log10(Math.max(v, 1));
    const logP05 = log(xP05);
    const logP95 = log(xP95);
    const logPad = 0.04 * (logP95 - logP05 || 1);

    const corePts = allPoints.filter(p => p.n_train >= xP05 && p.n_train <= xP95);
    const rho = summary.spearman_rho_core ?? spearmanRho(
      corePts.map(p => p.n_train),
      corePts.map(p => p[metricKey]),
    );
    const nOut = summary.n_outliers_x ?? allPoints.length - corePts.length;

    const width = 720;
    const height = 380;
    const pad = { l: 58, r: 24, t: 20, b: 52 };
    const innerW = width - pad.l - pad.r;
    const innerH = height - pad.t - pad.b;

    const ys = allPoints.map(p => p[metricKey]);
    const yMin = Math.max(0, Math.min(...ys, refY) - 0.08);
    const yMax = Math.max(...ys, refY) + 0.08;
    const xScale = v => pad.l + ((log(v) - logP05 + logPad) / (logP95 - logP05 + 2 * logPad || 1)) * innerW;
    const yScale = v => pad.t + innerH - ((v - yMin) / (yMax - yMin || 1)) * innerH;

    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Performance vs training size">`;
    svg += `<rect x="${pad.l}" y="${pad.t}" width="${innerW}" height="${innerH}" fill="#fff" stroke="#e6eaef"/>`;

    const yRef = yScale(refY);
    svg += `<line x1="${pad.l}" y1="${yRef}" x2="${pad.l + innerW}" y2="${yRef}" stroke="#bbb" stroke-dasharray="5 4"/>`;
    svg += `<text x="${pad.l + innerW + 4}" y="${yRef + 3}" font-size="10" fill="#656d76">1.0</text>`;

    const tickVals = [xP05, xP50, xP95].filter((v, i, a) => a.indexOf(v) === i);
    tickVals.forEach(tx => {
      svg += `<text x="${xScale(tx)}" y="${height - 12}" text-anchor="middle" font-size="10" fill="#656d76">${tx >= 1000 ? `${Math.round(tx / 1000)}k` : tx}</text>`;
    });
    for (let i = 0; i <= 4; i++) {
      const ty = yMin + (i / 4) * (yMax - yMin);
      svg += `<text x="10" y="${yScale(ty) + 3}" text-anchor="start" font-size="10" fill="#656d76">${ty.toFixed(2)}</text>`;
    }
    svg += `<text x="${pad.l + innerW / 2}" y="${height - 2}" text-anchor="middle" font-size="11" fill="#1f2328">Training rows (mean, log scale; ${xP05.toLocaleString()}–${xP95.toLocaleString()})</text>`;
    svg += `<text x="16" y="${pad.t + innerH / 2}" text-anchor="middle" font-size="11" fill="#1f2328" transform="rotate(-90 16 ${pad.t + innerH / 2})">${scatterMetric.label || metricKey}</text>`;

    allPoints.forEach(p => {
      const isOut = p.n_train < xP05 || p.n_train > xP95;
      const cx = xScale(Math.min(Math.max(p.n_train, xP05), xP95));
      const cy = yScale(p[metricKey]);
      const r = isOut ? 5 : 6;
      const stroke = isOut ? p.color : "#fff";
      const sw = isOut ? 2 : 1.2;
      const fill = isOut ? "#fff" : p.color;
      const outLabel = isOut ? " [outside range]" : "";
      svg += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" fill-opacity="${isOut ? 1 : 0.82}" stroke="${stroke}" stroke-width="${sw}">
            <title>${p.family} · ${p.display_name} · ${p.dataset}${outLabel}
Train=${p.n_train.toLocaleString()}, rel NRMSE=${p[metricKey]}
(model=${p.nrmse}, baseline=${p.baseline_nrmse})</title>
          </circle>`;
    });
    svg += "</svg>";

    const legend = series.map(fam =>
      `<span style="color:${fam.color}"><strong>${fam.family}</strong> · ${fam.display_name}</span>`,
    ).join("");

    let statLine = "";
    if (rho !== null && rho !== undefined) {
      const strength = Math.abs(rho) < 0.15 ? "no clear" : Math.abs(rho) < 0.35 ? "weak" : "moderate";
      const dir = rho < -0.05 ? "more data → lower relative NRMSE" : rho > 0.05 ? "more data → higher relative NRMSE" : "flat trend";
      statLine = `Spearman ρ = ${Number(rho).toFixed(2)} across datasets in range (${strength} trend; ${dir}). `;
    }
    if (scatterNote) {
      scatterNote.textContent =
        `Y-axis: model NRMSE ÷ average yield NRMSE (1.0 = baseline). X-axis uses log₁₀ scale over the 5th–95th percentile of training size (median ${xP50.toLocaleString()} rows). `
        + `${nOut} point${nOut === 1 ? "" : "s"} outside that range shown as open circles at the axis edge. `
        + statLine;
    }

    scatterPanel.innerHTML = `${svg}<div class="scatter-legend">${legend}</div>`;
  }

  function bootstrapPayloadRoot() {
    return DATA.country_bootstrap || FAMILY.country_bootstrap || {};
  }

  function bootstrapCropsForSelect() {
    const crop = cropSelect.value;
    if (crop !== "all") return [crop];
    const hz = horizonSelect.value;
    const byHz = (bootstrapPayloadRoot().by_horizon || {})[hz] || {};
    const keys = Object.keys(byHz);
    if (keys.length) return keys.sort();
    return (FAMILY.crops || DATA.crops || []).slice().sort();
  }

  function formatBootstrapNrmsePct(value) {
    if (value == null || !Number.isFinite(value)) return "—";
    return (100 * Number(value)).toFixed(1);
  }

  function formatBootstrapNum(value, digits = 1) {
    if (value == null || !Number.isFinite(value)) return "—";
    return Number(value).toFixed(digits);
  }

  function formatBootstrapCi(lo, hi, { digits = 1, percent = false } = {}) {
    if (lo == null || hi == null || !Number.isFinite(lo) || !Number.isFinite(hi)) return "—";
    if (percent) return `[${Math.round(100 * lo)}, ${Math.round(100 * hi)}]`;
    return `[${Number(lo).toFixed(digits)}, ${Number(hi).toFixed(digits)}]`;
  }

  function currentBootstrapRowsForView() {
    const hz = horizonSelect.value;
    const byHz = (bootstrapPayloadRoot().by_horizon || {})[hz] || {};
    return bootstrapCropsForSelect()
      .filter(crop => byHz[crop])
      .map(crop => ({ crop, ...byHz[crop] }));
  }

  function renderBootstrapTable() {
    const root = bootstrapPayloadRoot();
    const nBoot = root.n_bootstrap;
    if (bootstrapNote) {
      const note = root.note || (
        "Best data-driven vs best traditional baseline per country; seed-averaged NRMSE."
      );
      bootstrapNote.textContent = nBoot
        ? `${note} Bootstrap replicates: ${nBoot.toLocaleString()}.`
        : note;
    }
    currentBootstrapRows = currentBootstrapRowsForView();
    const hasBootstrap = Boolean(Object.keys(root.by_horizon || {}).length);
    if (countryBootstrapSection) {
      countryBootstrapSection.style.display = hasBootstrap ? "" : "none";
    }
    if (!bootstrapTable) return;
    const tbody = bootstrapTable.querySelector("tbody");
    if (!currentBootstrapRows.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="muted">No bootstrap summary for this horizon/crop.</td></tr>`;
      return;
    }
    tbody.innerHTML = currentBootstrapRows.map(row => {
      const wr = row.win_rate;
      return `<tr>
          <td>${row.crop.charAt(0).toUpperCase() + row.crop.slice(1)}</td>
          <td>${row.n_ai_wins ?? "—"}/${row.n_countries ?? "—"}</td>
          <td>${formatBootstrapNrmsePct(row.median_nrmse_trad)}</td>
          <td>${formatBootstrapNrmsePct(row.median_nrmse_ai)}</td>
          <td>${formatBootstrapNum(row.median_delta_pct)}</td>
          <td>${formatBootstrapCi(row.delta_pct_ci_lo, row.delta_pct_ci_hi)}</td>
          <td>${wr == null ? "—" : `${Math.round(100 * wr)}%`}</td>
          <td>${formatBootstrapCi(row.win_rate_ci_lo, row.win_rate_ci_hi, { percent: true })}</td>
        </tr>`;
    }).join("");
  }

  function escapeLatex(text) {
    return String(text)
      .replace(/\\/g, "\\textbackslash{}")
      .replace(/[&%$#_{}]/g, ch => `\\${ch}`)
      .replace(/~/g, "\\textasciitilde{}")
      .replace(/\^/g, "\\textasciicircum{}");
  }

  function bootstrapExportMeta() {
    const hz = horizonSelect.value;
    const crop = cropSelect.value;
    const hzLabel = horizonLabels[hz] || hz;
    const cropLabel = crop === "all" ? "all crops" : crop;
    return { hz, crop, hzLabel, cropLabel };
  }

  function buildBootstrapTableLatex(rows) {
    if (!rows.length) return "";
    const { hzLabel, cropLabel } = bootstrapExportMeta();
    const nBoot = (bootstrapPayloadRoot().n_bootstrap || 10000).toLocaleString();
    const bodyRows = rows.map(row => {
      const trad = formatBootstrapNrmsePct(row.median_nrmse_trad);
      const ai = formatBootstrapNrmsePct(row.median_nrmse_ai);
      const wr = row.win_rate == null ? "---" : `${Math.round(100 * row.win_rate)}\\%`;
      const wrCi = formatBootstrapCi(row.win_rate_ci_lo, row.win_rate_ci_hi, { percent: true });
      const pctCi = formatBootstrapCi(row.delta_pct_ci_lo, row.delta_pct_ci_hi);
      return `${row.crop.charAt(0).toUpperCase() + row.crop.slice(1)} & ${trad} & ${ai} & `
        + `${formatBootstrapNum(row.median_delta_pct)} & ${pctCi} & ${wr} & ${wrCi} \\\\`;
    }).join("\n");
    return `% CY-Bench country-level AI bootstrap table (generated from dashboard)
% Requires: \\usepackage{booktabs}
\\begin{table}[t]
\\centering
\\caption{Country-level bootstrap summary of AI advantage over traditional baselines. Traditional and data-driven NRMSE are medians across countries of the best traditional (Average, Trend, LPJmL) and best data-driven model per country. Median improvement (\\%) is the median of $100\\times(\\mathrm{NRMSE}_{\\mathrm{trad}}-\\mathrm{NRMSE}_{\\mathrm{AI}})/\\mathrm{NRMSE}_{\\mathrm{trad}}$ per country. Horizon: ${escapeLatex(hzLabel)}; crops: ${escapeLatex(cropLabel)}. Bootstrap resamples countries ($B=${nBoot}$).}
\\label{tab:ai_country_bootstrap}
\\begin{tabular}{lcccccc}
\\toprule
Crop & Trad.\\ NRMSE (\\%) & AI NRMSE (\\%) & Median impr. (\\%) & 95\\% CI & AI win rate & 95\\% CI \\\\
\\midrule
${bodyRows}
\\bottomrule
\\end{tabular}
\\end{table}
`;
  }

  function downloadBootstrapTableLatex() {
    const latex = buildBootstrapTableLatex(currentBootstrapRows);
    if (!latex) return;
    const { hz, crop } = bootstrapExportMeta();
    const blob = new Blob([latex], { type: "application/x-tex;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ai_country_bootstrap_${hz}_${crop}.tex`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  async function copyBootstrapTableLatex() {
    const latex = buildBootstrapTableLatex(currentBootstrapRows);
    if (!latex || !bootstrapExportLatexCopyBtn) return;
    const original = bootstrapExportLatexCopyBtn.textContent;
    try {
      await navigator.clipboard.writeText(latex);
      bootstrapExportLatexCopyBtn.textContent = "Copied!";
      setTimeout(() => { bootstrapExportLatexCopyBtn.textContent = original; }, 1600);
    } catch (_err) {
      downloadBootstrapTableLatex();
      bootstrapExportLatexCopyBtn.textContent = "Downloaded instead";
      setTimeout(() => { bootstrapExportLatexCopyBtn.textContent = original; }, 1600);
    }
  }

  function render() {
    if (scatterPanel) renderScatter();
    if (bootstrapTable || countryBootstrapSection) renderBootstrapTable();
  }

  const ownsHorizonCrop = Boolean(document.getElementById("rq4-horizon"));
  if (ownsHorizonCrop) {
    fillHorizonSelect();
    fillCropSelect();
    horizonSelect.addEventListener("change", () => {
      fillCropSelect();
      render();
    });
    cropSelect.addEventListener("change", render);
  } else {
    // Share family-map / family-metrics controls on the performance page.
    horizonSelect.addEventListener("change", render);
    cropSelect.addEventListener("change", render);
  }
  if (bootstrapExportLatexBtn) bootstrapExportLatexBtn.addEventListener("click", downloadBootstrapTableLatex);
  if (bootstrapExportLatexCopyBtn) bootstrapExportLatexCopyBtn.addEventListener("click", copyBootstrapTableLatex);
  render();
})();
