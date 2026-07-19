// Embedded in insights.html (RQ1 family table + world map). Expects global DATA with family_dashboard.
(function initFamilyRq1Panel() {
  const FAMILY = DATA.family_dashboard;
  if (!FAMILY || !FAMILY.by_horizon) return;

  const metricsWrap = document.getElementById("family-metrics-table-wrap");
  const metricsHorizon = document.getElementById("family-metrics-horizon");
  const metricsCrop = document.getElementById("family-metrics-crop");
  const mapHorizon = document.getElementById("family-map-horizon");
  const mapCrop = document.getElementById("family-map-crop");
  const mapWrap = document.getElementById("family-winner-map-wrap");
  const benefitMapWrap = document.getElementById("family-benefit-map-wrap");
  const mapSummary = document.getElementById("family-winner-summary");
  const benefitSummary = document.getElementById("family-benefit-summary");
  const mapModeNote = document.getElementById("family-map-note");
  const aspectLabel = document.getElementById("family-winner-aspect-label");
  const aspectSelect = document.getElementById("family-winner-aspect");
  const aspectToolbar = document.getElementById("family-winner-aspect-toolbar");
  const winnerLegend = document.getElementById("family-winner-legend");
  const benefitLegend = document.getElementById("family-benefit-legend");
  const benefitLegendLabel = document.getElementById("family-benefit-legend-label");
  const metricsNote = document.getElementById("family-metrics-note");
  const mapExportSvgBtn = document.getElementById("family-map-export-svg");
  const benefitMapExportSvgBtn = document.getElementById("family-benefit-map-export-svg");
  if (!metricsWrap || !metricsHorizon) return;

  const horizonLabels = FAMILY.horizon_labels || {};
  const horizonOrder = ["eos", "mid", "early", "qtr"].filter(hz => FAMILY.by_horizon[hz]);
  let famHorizon = horizonOrder.includes("eos") ? "eos" : horizonOrder[0];
  let famCrop = "all";
  let winnerAspect = (FAMILY.views || [])[0]?.label || "Overall";

  const MAP_FILL_OUTSIDE = "#e2e6eb";
  const benchmarkIsos = new Set(DATA.benchmark_map_isos || []);

  function isColoredBenchmarkIso(iso) {
    return benchmarkIsos.has(iso) && iso !== "XX";
  }

  function mapDataStroke(hasData) {
    return hasData
      ? { color: "#6d7d8f", width: 0.55 }
      : { color: "none", width: 0 };
  }

  function fillHorizonSelect(sel) {
    sel.innerHTML = horizonOrder.map(hz => {
      const label = horizonLabels[hz] || hz;
      return `<option value="${hz}">${label}</option>`;
    }).join("");
    sel.value = famHorizon;
  }

  function fillCropSelect(sel) {
    const crops = Object.keys(FAMILY.by_horizon[famHorizon] || {});
    const order = ["all", ...(FAMILY.crops || []).filter(c => crops.includes(c))];
    sel.innerHTML = order.filter(c => crops.includes(c)).map(crop => {
      const label = crop === "all" ? "All crops" : crop.charAt(0).toUpperCase() + crop.slice(1);
      return `<option value="${crop}">${label}</option>`;
    }).join("");
    if (!crops.includes(famCrop)) famCrop = crops.includes("all") ? "all" : crops[0];
    sel.value = famCrop;
  }

  function currentSlice() {
    return ((FAMILY.by_horizon[famHorizon] || {})[famCrop] || { families: [] });
  }

  function currentWinnerSlice() {
    return ((FAMILY.winner_maps || {})[famHorizon] || {})[famCrop]
      || ((FAMILY.winner_maps || {})[famHorizon] || {}).all
      || {};
  }

  function currentBenefitSlice() {
    return ((FAMILY.ai_benefit_maps || {})[famHorizon] || {})[famCrop]
      || ((FAMILY.ai_benefit_maps || {})[famHorizon] || {}).all
      || { countries: [] };
  }

  function tableColumns() {
    return FAMILY.table_columns || FAMILY.views || [];
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function metricLowerIsBetter(metric) {
    return metric === "nrmse";
  }

  function metricIsBetterValue(metric, candidate, incumbent) {
    if (candidate == null || Number.isNaN(candidate)) return false;
    if (incumbent == null || Number.isNaN(incumbent)) return true;
    return metricLowerIsBetter(metric) ? candidate < incumbent : candidate > incumbent;
  }

  function bestFamiliesPerMetric(families, cols) {
    const best = {};
    for (const col of cols) {
      const metric = col.metric;
      let bestFamily = null;
      let bestVal = null;
      for (const fam of families) {
        const v = fam.raw && fam.raw[metric];
        if (v == null || Number.isNaN(v)) continue;
        if (metricIsBetterValue(metric, v, bestVal)) {
          bestVal = v;
          bestFamily = fam.family;
        }
      }
      if (bestFamily) best[metric] = bestFamily;
    }
    return best;
  }

  function metricSigMarker(fam, metric) {
    if (fam.is_naive || fam.family === "Naive baselines") return "";
    const stats = (fam.vs_naive || {})[metric];
    if (stats && stats.significant_worse) return "†";
    if (stats && stats.significant) return "*";
    if ((fam.vs_naive_sig_worse || {})[metric]) return "†";
    if ((fam.vs_naive_sig || {})[metric]) return "*";
    return "";
  }

  function metricVsNaiveTooltip(fam, metric) {
    if (fam.is_naive || fam.family === "Naive baselines") {
      return "Naive baseline reference (no comparison).";
    }
    const stats = (fam.vs_naive || {})[metric];
    if (!stats) {
      return "No paired country data vs naive baseline.";
    }
    const n = stats.n_countries != null ? stats.n_countries : 0;
    if (n === 0) {
      return "No paired country data vs naive baseline for this metric.";
    }
    if (stats.median_delta == null) {
      return `Only ${n} countr${n === 1 ? "y" : "ies"} with paired data (bootstrap needs ≥2).`;
    }
    let text = `vs naive: median per-country Δ = ${Number(stats.median_delta).toFixed(3)}`;
    if (stats.table_median_gap != null) {
      text += `; table median gap = ${Number(stats.table_median_gap) >= 0 ? "+" : ""}${Number(stats.table_median_gap).toFixed(3)}`;
    }
    if (stats.ci_lo != null && stats.ci_hi != null) {
      text += `, 95% bootstrap CI [${Number(stats.ci_lo).toFixed(3)}, ${Number(stats.ci_hi).toFixed(3)}]`;
    } else if (n < 2) {
      text += " (bootstrap CI needs ≥2 countries)";
    }
    if (stats.p_one_sided != null) {
      text += `, p(better) = ${Number(stats.p_one_sided).toFixed(3)}`;
    }
    if (stats.p_one_sided_worse != null) {
      text += `, p(worse) = ${Number(stats.p_one_sided_worse).toFixed(3)}`;
    }
    if (stats.significant) text += " *";
    if (stats.significant_worse) text += " †";
    text += ` (${n} countries; positive Δ = family better)`;
    return text;
  }

  function formatMetricSigMarkup(marker) {
    if (!marker) return "";
    return `<sup class="sig-mark">${marker}</sup>`;
  }

  function formatMetricWithIqr(fam, metric) {
    const raw = fam.raw && fam.raw[metric];
    if (raw == null) return "—";
    const mark = formatMetricSigMarkup(metricSigMarker(fam, metric));
    const band = (fam.iqr || {})[metric];
    if (!band || band.q25 == null || band.q75 == null) {
      return `${Number(raw).toFixed(3)}${mark}`;
    }
    return `${Number(raw).toFixed(3)}${mark} [${Number(band.q25).toFixed(3)}, ${Number(band.q75).toFixed(3)}]`;
  }

  function renderFamilyTable() {
    const families = currentSlice().families || [];
    const cols = tableColumns();
    if (!families.length) {
      metricsWrap.innerHTML = '<p class="muted">No family data for this selection.</p>';
      return;
    }
    const sigNote = FAMILY.family_vs_naive_sig_note || "";
    if (metricsNote) {
      metricsNote.textContent = sigNote
        ? `Five paradigms — one best-NRMSE representative per family; median [IQR] across countries. ${sigNote}`
        : "Five paradigms — one best-NRMSE representative per family; median [IQR] across countries.";
    }
    const bestByMetric = bestFamiliesPerMetric(families, cols);
    const viewHeaders = cols.map(v => {
      const metric = v.display || v.metric;
      if (v.header) return v.header;
      return `${v.label} (${metric})`;
    });
    const headers = ["Family", "Representative", ...viewHeaders];
    let html = '<table id="family-metrics-table"><thead><tr>';
    html += headers.map(h => `<th>${escapeHtml(h)}</th>`).join("");
    html += "</tr></thead><tbody>";
    for (const fam of families) {
      html += `<tr><td>${escapeHtml(fam.family)}</td><td>${escapeHtml(fam.display_name)}</td>`;
      for (const col of cols) {
        const metric = col.metric;
        const isBest = bestByMetric[metric] === fam.family;
        const content = formatMetricWithIqr(fam, metric);
        const title = metricVsNaiveTooltip(fam, metric);
        const cls = isBest ? "metric-cell metric-best" : "metric-cell";
        html += `<td class="${cls}" title="${escapeHtml(title)}">${content}</td>`;
      }
      html += "</tr>";
    }
    html += "</tbody></table>";
    metricsWrap.innerHTML = html;
  }

  const MAP_WIDTH = 1280;
  const MAP_HEIGHT = 694;
  const MAP_MARGIN = 8;
  const MAP_EXPORT_SCALE = 4;
  const MAP_EXPORT_PAD_B = 115;
  const MAP_EXPORT_LEGEND_FS = 22;
  const MAP_EXPORT_LEGEND_BAR_W = 520;
  const MAP_EXPORT_LEGEND_BAR_H = 20;
  const MAP_EXPORT_FONT = "DejaVu Sans";
  const MAP_EXPORT_LABEL_GAP = 20;
  const MAP_EXPORT_LEGEND_TICK = 7;
  let worldFeatures = null;
  let mapProjection = d3.geoNaturalEarth1();
  let mapPath = d3.geoPath(mapProjection);
  const mapSvg = mapWrap
    ? d3.select("#family-winner-map-wrap").html("").append("svg")
        .attr("viewBox", `0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`)
        .attr("role", "img")
        .attr("aria-label", "Winning model family by country")
    : null;
  const benefitSvg = benefitMapWrap
    ? d3.select("#family-benefit-map-wrap").html("").append("svg")
        .attr("viewBox", `0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`)
        .attr("role", "img")
        .attr("aria-label", "AI error reduction by country")
    : null;

  function winnerColorMap() {
    const out = {};
    Object.entries(FAMILY.family_catalog || {}).forEach(([fam, obj]) => {
      out[fam] = obj.color || "#999";
    });
    return out;
  }

  function percentileSorted(sorted, p) {
    if (!sorted.length) return 0;
    const idx = (sorted.length - 1) * p;
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }

  /** Symmetric color scale around 0%; ignores the largest |value| outlier when n≥4. */
  function benefitExtent(vals) {
    const DEFAULT = 40;
    const MIN = 15;
    const MAX = 40;
    if (!vals.length) return DEFAULT;
    const absSorted = vals.map(v => Math.abs(v)).sort((a, b) => a - b);
    const trimmed = absSorted.length >= 4 ? absSorted.slice(0, -1) : absSorted;
    const p85 = percentileSorted(trimmed, 0.85);
    const rounded = Math.ceil(Math.max(p85, 10) / 5) * 5;
    return Math.min(MAX, Math.max(MIN, rounded));
  }

  const BENEFIT_COLOR_TRAD = "#d95f02";
  const BENEFIT_COLOR_ZERO = "#9eb4c8";
  const BENEFIT_COLOR_AI = "#2166ac";

  function benefitColor(pct, extent) {
    const clamped = Math.max(-extent, Math.min(extent, pct));
    const t = (clamped + extent) / (2 * extent);
    if (t <= 0.5) {
      return d3.interpolateRgb(BENEFIT_COLOR_TRAD, BENEFIT_COLOR_ZERO)(t * 2);
    }
    return d3.interpolateRgb(BENEFIT_COLOR_ZERO, BENEFIT_COLOR_AI)((t - 0.5) * 2);
  }

  function benefitOffScale(pct, extent) {
    return pct != null && Math.abs(pct) > extent + 1e-9;
  }

  function updateMapModeUi() {
    if (winnerLegend) winnerLegend.style.display = "";
    if (benefitLegend) benefitLegend.style.display = "flex";
    if (mapModeNote) {
      mapModeNote.textContent =
        "Left: winning modeling paradigm per country. Right: AI error reduction vs traditional approaches. "
        + (FAMILY.ai_benefit_note || "");
    }
  }

  function renderWinnerMap() {
    if (!worldFeatures || !mapSvg) return;
    const slice = currentWinnerSlice();
    const byAspect = slice[winnerAspect] || { countries: [] };
    const winners = new Map((byAspect.countries || []).map(r => [r.map_cc || r.country, r]));
    const colors = winnerColorMap();

    const countriesSel = mapSvg.selectAll("path.country")
      .data(worldFeatures)
      .join("path")
      .attr("class", "country")
      .attr("shape-rendering", "geometricPrecision")
      .attr("d", mapPath)
      .attr("fill", d => {
        const iso = d.properties.ISO_A2;
        if (!isColoredBenchmarkIso(iso)) return MAP_FILL_OUTSIDE;
        const rec = winners.get(iso);
        if (!rec) return MAP_FILL_OUTSIDE;
        return colors[rec.winner_family] || "#999";
      })
      .attr("stroke", d => {
        const iso = d.properties.ISO_A2;
        const hasData = isColoredBenchmarkIso(iso) && winners.has(iso);
        return mapDataStroke(hasData).color;
      })
      .attr("stroke-width", d => {
        const iso = d.properties.ISO_A2;
        const hasData = isColoredBenchmarkIso(iso) && winners.has(iso);
        return mapDataStroke(hasData).width;
      });

    countriesSel.selectAll("title")
      .data(d => [d])
      .join("title")
      .text(d => {
        const iso = d.properties.ISO_A2;
        if (!isColoredBenchmarkIso(iso)) {
          return `${d.properties.NAME || iso}: not in CY-Bench`;
        }
        const rec = winners.get(iso);
        if (!rec) return `${d.properties.NAME || iso}: no data for this horizon/crop`;
        return `${rec.country}: ${rec.winner_family} (${rec.winner_model}), ${byAspect.metric}=${rec.value.toFixed(3)}`;
      });

    if (winnerLegend) {
      const familiesPresent = Array.from(
        new Set((byAspect.countries || []).map(r => r.winner_family))
      );
      winnerLegend.innerHTML = familiesPresent.map(fam => {
        const color = colors[fam] || "#999";
        return `<span><i class="swatch" style="background:${color}"></i>${fam}</span>`;
      }).join("");
    }
    if (mapSummary) {
      mapSummary.textContent = `${winnerAspect}: ${byAspect.countries?.length || 0} countries with a winning family.`;
    }
  }

  function renderBenefitMap() {
    if (!worldFeatures || !benefitSvg) return;
    const slice = currentBenefitSlice();
    const rows = slice.countries || [];
    const byCountry = new Map(rows.map(r => [r.map_cc || r.country, r]));
    const vals = rows.map(r => r.benefit_pct).filter(v => v != null);
    const extent = benefitExtent(vals);
    const nOffScale = rows.filter(r => benefitOffScale(r.benefit_pct, extent)).length;
    if (benefitLegendLabel) {
      benefitLegendLabel.textContent = `−${extent}% · 0% · +${extent}%`;
    }

    const countriesSel = benefitSvg.selectAll("path.country")
      .data(worldFeatures)
      .join("path")
      .attr("class", "country")
      .attr("shape-rendering", "geometricPrecision")
      .attr("d", mapPath)
      .attr("fill", d => {
        const iso = d.properties.ISO_A2;
        if (!isColoredBenchmarkIso(iso)) return MAP_FILL_OUTSIDE;
        const rec = byCountry.get(iso);
        if (!rec || rec.benefit_pct == null) return MAP_FILL_OUTSIDE;
        return benefitColor(rec.benefit_pct, extent);
      })
      .attr("stroke", d => {
        const iso = d.properties.ISO_A2;
        const rec = byCountry.get(iso);
        const hasData = isColoredBenchmarkIso(iso) && rec && rec.benefit_pct != null;
        return mapDataStroke(hasData).color;
      })
      .attr("stroke-width", d => {
        const iso = d.properties.ISO_A2;
        const rec = byCountry.get(iso);
        const hasData = isColoredBenchmarkIso(iso) && rec && rec.benefit_pct != null;
        return mapDataStroke(hasData).width;
      });

    countriesSel.selectAll("title")
      .data(d => [d])
      .join("title")
      .text(d => {
        const iso = d.properties.ISO_A2;
        if (!isColoredBenchmarkIso(iso)) {
          return `${d.properties.NAME || iso}: not in CY-Bench`;
        }
        const rec = byCountry.get(iso);
        if (!rec) return `${d.properties.NAME || iso}: no data for this horizon/crop`;
        const offScale = benefitOffScale(rec.benefit_pct, extent)
          ? ` (color saturated; scale ±${extent}%)`
          : "";
        return `${rec.country}: ${rec.benefit_pct.toFixed(1)}% error reduction${offScale}\n`
          + `Traditional (${rec.traditional_family} · ${rec.traditional_model}) NRMSE=${rec.nrmse_traditional.toFixed(3)}\n`
          + `AI (${rec.ai_family} · ${rec.ai_model}) NRMSE=${rec.nrmse_ai.toFixed(3)}`;
      });

    if (!mapSummary) return;
    if (!rows.length) {
      (benefitSummary || mapSummary).textContent = "No AI vs traditional comparison for this horizon/crop.";
      return;
    }
    const medianBenefit = d3.median(vals);
    const nPositive = rows.filter(r => r.benefit_pct > 0).length;
    let summary =
      `Median error reduction ${medianBenefit == null ? "—" : medianBenefit.toFixed(1)}% across `
      + `${rows.length} countries (${nPositive} with AI better than traditional). `
      + `Color scale ±${extent}% (85th percentile, largest outlier excluded).`;
    if (nOffScale) {
      summary += ` ${nOffScale} countr${nOffScale === 1 ? "y" : "ies"} beyond scale (color clamped; hover for exact value).`;
    }
    (benefitSummary || mapSummary).textContent = summary;
  }

  function renderFamilyMap() {
    if (!worldFeatures) return;
    renderWinnerMap();
    renderBenefitMap();
  }

  function setMapExportEnabled(enabled) {
    if (mapExportSvgBtn) mapExportSvgBtn.disabled = !enabled;
    if (benefitMapExportSvgBtn) benefitMapExportSvgBtn.disabled = !enabled;
  }

  function escapeXml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function mapExportMeta(kind) {
    const hzLabel = horizonLabels[famHorizon] || famHorizon;
    const cropLabel = famCrop === "all" ? "all crops" : famCrop;
    const modeLabel = kind === "benefit"
      ? "AI error reduction"
      : `Winning family · ${winnerAspect}`;
    return { hz: famHorizon, crop: famCrop, hzLabel, cropLabel, modeLabel, kind };
  }

  function mapExportFilename(ext, kind = "winner") {
    const { hz, crop } = mapExportMeta(kind);
    const modeSlug = kind === "benefit"
      ? "ai-error-reduction"
      : `winner-${winnerAspect.toLowerCase().replace(/\s+/g, "-")}`;
    return `cybench-map_${hz}_${crop}_${modeSlug}.${ext}`;
  }

  function exportLegendBaselineY(yBase) {
    return yBase + MAP_EXPORT_LEGEND_BAR_H + MAP_EXPORT_LABEL_GAP + MAP_EXPORT_LEGEND_FS;
  }

  let exportMeasureSvg = null;
  let exportFont = null;
  let exportFontLoading = null;

  function exportMeasureLayer() {
    if (exportMeasureSvg) return exportMeasureSvg;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", "0");
    svg.setAttribute("height", "0");
    svg.style.cssText = "position:fixed;left:-10000px;top:0;visibility:hidden;pointer-events:none";
    document.body.appendChild(svg);
    exportMeasureSvg = svg;
    return svg;
  }

  function measureExportTextWidth(text) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = exportMeasureLayer();
    const t = document.createElementNS(ns, "text");
    t.setAttribute("font-size", String(MAP_EXPORT_LEGEND_FS));
    t.setAttribute("font-family", MAP_EXPORT_FONT);
    t.textContent = String(text);
    svg.appendChild(t);
    const w = t.getBBox().width;
    svg.removeChild(t);
    return w;
  }

  function exportLegendLabel(refX, yBaseline, text, anchor) {
    const w = measureExportTextWidth(text);
    let textX = 0;
    if (anchor === "middle") textX = -w / 2;
    else if (anchor === "end") textX = -w;
    return `<g transform="translate(${refX} ${yBaseline})"><text x="${textX}" y="0" font-size="${MAP_EXPORT_LEGEND_FS}" font-family="${MAP_EXPORT_FONT}" fill="#656d76">${escapeXml(text)}</text></g>`;
  }

  function exportLegendMarkup(refX, yBaseline, text, anchor) {
    if (exportFont) {
      const fontSize = MAP_EXPORT_LEGEND_FS;
      let x = refX;
      const w = exportFont.getAdvanceWidth(text, fontSize);
      if (anchor === "middle") x = refX - w / 2;
      else if (anchor === "end") x = refX - w;
      const d = exportFont.getPath(text, x, yBaseline, fontSize).toPathData(2);
      return `<path d="${d}" fill="#656d76"/>`;
    }
    return exportLegendLabel(refX, yBaseline, text, anchor);
  }

  function ensureExportFont() {
    if (exportFont !== null) return Promise.resolve(exportFont || null);
    if (!exportFontLoading) {
      if (typeof opentype === "undefined") {
        exportFont = false;
        return Promise.resolve(null);
      }
      exportFontLoading = fetch("https://cdn.jsdelivr.net/npm/dejavu-fonts-ttf@2.37.3/ttf/DejaVuSans.ttf")
        .then(r => r.arrayBuffer())
        .then(buf => {
          exportFont = opentype.parse(buf);
          return exportFont;
        })
        .catch(() => {
          exportFont = false;
          return null;
        });
    }
    return exportFontLoading;
  }

  function buildWinnerLegendSvg(yBase) {
    if (!winnerLegend) return "";
    const items = winnerLegend.querySelectorAll("span");
    if (!items.length) return "";
    const fs = MAP_EXPORT_LEGEND_FS;
    const sw = MAP_EXPORT_LEGEND_FS;
    let x = 24;
    let out = "";
    items.forEach(span => {
      const swatch = span.querySelector(".swatch");
      const color = swatch ? (swatch.style.background || "#999") : "#999";
      const text = span.textContent.trim();
      out += `<rect x="${x}" y="${yBase}" width="${sw}" height="${sw}" fill="${color}" rx="1.5"/>`;
      out += `<g transform="translate(${x + sw + 4} ${yBase + 1})"><text y="0" font-size="${fs}" font-family="${MAP_EXPORT_FONT}" fill="#656d76">${escapeXml(text)}</text></g>`;
      x += Math.min(160, Math.max(56, text.length * (fs * 0.52) + sw + 10));
    });
    return out;
  }

  function buildBenefitLegendSvg(yBase) {
    const slice = currentBenefitSlice();
    const vals = (slice.countries || []).map(r => r.benefit_pct).filter(v => v != null);
    const extent = benefitExtent(vals);
    const barX = 24;
    const barW = MAP_EXPORT_LEGEND_BAR_W;
    const barH = MAP_EXPORT_LEGEND_BAR_H;
    const labelY = exportLegendBaselineY(yBase);
    return `
      <defs>
        <linearGradient id="export-benefit-grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#d95f02"/>
          <stop offset="25%" stop-color="#c4a574"/>
          <stop offset="50%" stop-color="#9eb4c8"/>
          <stop offset="75%" stop-color="#92c5de"/>
          <stop offset="100%" stop-color="#2166ac"/>
        </linearGradient>
      </defs>
      <rect x="${barX}" y="${yBase}" width="${barW}" height="${barH}" fill="url(#export-benefit-grad)" stroke="#c8d0d8" stroke-width="0.75"/>
      <line x1="${barX}" y1="${yBase + barH}" x2="${barX}" y2="${yBase + barH + MAP_EXPORT_LEGEND_TICK}" stroke="#656d76" stroke-width="0.75"/>
      <line x1="${barX + barW / 2}" y1="${yBase + barH}" x2="${barX + barW / 2}" y2="${yBase + barH + MAP_EXPORT_LEGEND_TICK}" stroke="#656d76" stroke-width="0.75"/>
      <line x1="${barX + barW}" y1="${yBase + barH}" x2="${barX + barW}" y2="${yBase + barH + MAP_EXPORT_LEGEND_TICK}" stroke="#656d76" stroke-width="0.75"/>
      ${exportLegendMarkup(barX, labelY, `-${extent}%`, "start")}
      ${exportLegendMarkup(barX + barW / 2, labelY, "0%", "middle")}
      ${exportLegendMarkup(barX + barW, labelY, `+${extent}%`, "end")}`;
  }

  function buildMapExportSvgString(kind = "winner", exportScale = MAP_EXPORT_SCALE) {
    const wrap = kind === "benefit" ? benefitMapWrap : mapWrap;
    const mapNode = wrap && wrap.querySelector("svg");
    if (!mapNode || !worldFeatures) return null;
    const padT = 12;
    const padB = MAP_EXPORT_PAD_B;
    const totalH = MAP_HEIGHT + padT + padB;
    const legendY = padT + MAP_HEIGHT + 6;
    const legendSvg = kind === "benefit"
      ? buildBenefitLegendSvg(legendY)
      : buildWinnerLegendSvg(legendY);
    const mapContent = mapNode.innerHTML;
    const exportW = MAP_WIDTH * exportScale;
    const exportH = totalH * exportScale;
    return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${MAP_WIDTH} ${totalH}" width="${exportW}" height="${exportH}" font-family="${MAP_EXPORT_FONT}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <svg x="0" y="${padT}" width="${MAP_WIDTH}" height="${MAP_HEIGHT}" viewBox="0 0 ${MAP_WIDTH} ${MAP_HEIGHT}">
    ${mapContent}
  </svg>
  ${legendSvg}
</svg>`;
  }

  function triggerBlobDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function downloadFamilyMapSvg(kind = "winner") {
    ensureExportFont().then(() => {
      const svg = buildMapExportSvgString(kind);
      if (!svg) return;
      triggerBlobDownload(
        new Blob([svg], { type: "image/svg+xml;charset=utf-8" }),
        mapExportFilename("svg", kind),
      );
    });
  }

  if (mapExportSvgBtn) {
    mapExportSvgBtn.addEventListener("click", () => downloadFamilyMapSvg("winner"));
  }
  if (benefitMapExportSvgBtn) {
    benefitMapExportSvgBtn.addEventListener("click", () => downloadFamilyMapSvg("benefit"));
  }
  ensureExportFont();

  function loadMap() {
    if (!DATA.geojson_href) {
      setMapExportEnabled(false);
      mapWrap.innerHTML = '<p class="muted">Map geometry not available.</p>';
      return;
    }
    d3.json(DATA.geojson_href).then(geo => {
      worldFeatures = (geo.features || []).filter(f => f.properties.ISO_A2 !== "AQ");
      mapProjection = d3.geoNaturalEarth1().fitExtent(
        [[MAP_MARGIN, MAP_MARGIN], [MAP_WIDTH - MAP_MARGIN, MAP_HEIGHT - MAP_MARGIN]],
        { type: "FeatureCollection", features: worldFeatures },
      );
      mapPath = d3.geoPath(mapProjection);
      setMapExportEnabled(true);
      renderFamilyMap();
    }).catch(() => {
      setMapExportEnabled(false);
      mapWrap.innerHTML = '<p class="muted">Could not load map geometry.</p>';
    });
  }

  if (aspectSelect) {
    aspectSelect.innerHTML = (FAMILY.views || []).map(v =>
      `<option value="${v.label}">${v.label}</option>`
    ).join("");
    aspectSelect.value = winnerAspect;
    aspectSelect.addEventListener("change", () => {
      winnerAspect = aspectSelect.value || winnerAspect;
      renderFamilyMap();
    });
  } else if (aspectToolbar) {
    aspectToolbar.innerHTML = (FAMILY.views || []).map((v, idx) =>
      `<button type="button" data-aspect="${v.label}"${idx === 0 ? ' class="active"' : ""}>${v.label}</button>`
    ).join("");
    aspectToolbar.addEventListener("click", ev => {
      const btn = ev.target.closest("button[data-aspect]");
      if (!btn) return;
      winnerAspect = btn.dataset.aspect;
      aspectToolbar.querySelectorAll("button").forEach(b => b.classList.toggle("active", b === btn));
      renderFamilyMap();
    });
  }
  updateMapModeUi();

  function syncSelects() {
    fillHorizonSelect(metricsHorizon);
    if (mapHorizon) fillHorizonSelect(mapHorizon);
    fillCropSelect(metricsCrop);
    if (mapCrop) fillCropSelect(mapCrop);
  }

  function renderAll() {
    renderFamilyTable();
    renderFamilyMap();
  }

  metricsHorizon.addEventListener("change", () => {
    famHorizon = metricsHorizon.value;
    if (mapHorizon) mapHorizon.value = famHorizon;
    syncSelects();
    renderAll();
  });
  if (mapHorizon) {
    mapHorizon.addEventListener("change", () => {
      famHorizon = mapHorizon.value;
      metricsHorizon.value = famHorizon;
      syncSelects();
      renderAll();
    });
  }
  metricsCrop.addEventListener("change", () => {
    famCrop = metricsCrop.value;
    if (mapCrop) mapCrop.value = famCrop;
    renderAll();
  });
  if (mapCrop) {
    mapCrop.addEventListener("change", () => {
      famCrop = mapCrop.value;
      metricsCrop.value = famCrop;
      renderAll();
    });
  }

  syncSelects();
  loadMap();
  renderFamilyTable();
})();
