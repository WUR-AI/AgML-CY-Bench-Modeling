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
  const mapSummary = document.getElementById("family-winner-summary");
  const mapModeNote = document.getElementById("family-map-note");
  const mapModeToolbar = document.getElementById("family-map-mode-toolbar");
  const aspectLabel = document.getElementById("family-winner-aspect-label");
  const aspectToolbar = document.getElementById("family-winner-aspect-toolbar");
  const winnerLegend = document.getElementById("family-winner-legend");
  const benefitLegend = document.getElementById("family-benefit-legend");
  const benefitLegendLabel = document.getElementById("family-benefit-legend-label");
  if (!metricsWrap || !metricsHorizon) return;

  const horizonLabels = FAMILY.horizon_labels || {};
  const horizonOrder = ["eos", "early", "mid", "qtr"].filter(hz => FAMILY.by_horizon[hz]);
  let famHorizon = horizonOrder.includes("eos") ? "eos" : horizonOrder[0];
  let famCrop = "all";
  let mapMode = "winner";
  let winnerAspect = (FAMILY.views || [])[0]?.label || "Overall";

  const MAP_FILL_OUTSIDE = "#e2e6eb";
  const MAP_FILL_BENCHMARK = "#9eb4c8";
  const benchmarkIsos = new Set(DATA.benchmark_map_isos || []);

  function isColoredBenchmarkIso(iso) {
    return benchmarkIsos.has(iso) && iso !== "XX";
  }

  function mapCountryStroke(iso) {
    return isColoredBenchmarkIso(iso)
      ? { color: "#6d7d8f", width: 0.55 }
      : { color: "#c8d0d8", width: 0.22 };
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

  function renderFamilyTable() {
    const families = currentSlice().families || [];
    const cols = tableColumns();
    if (!families.length) {
      metricsWrap.innerHTML = '<p class="muted">No family data for this selection.</p>';
      return;
    }
    const headers = ["Family", "Representative", ...cols.map(v => v.header || v.label)];
    let html = "<table><thead><tr>";
    html += headers.map(h => `<th>${h}</th>`).join("");
    html += "</tr></thead><tbody>";
    for (const fam of families) {
      html += `<tr><td>${fam.family}</td><td>${fam.display_name}</td>`;
      for (const col of cols) {
        const metric = col.metric;
        const raw = fam.raw && fam.raw[metric];
        const band = fam.iqr && fam.iqr[metric];
        let cell = raw == null ? "—" : Number(raw).toFixed(3);
        if (band && band.q25 != null && band.q75 != null) {
          cell += ` [${Number(band.q25).toFixed(3)}, ${Number(band.q75).toFixed(3)}]`;
        }
        html += `<td>${cell}</td>`;
      }
      html += "</tr>";
    }
    html += "</tbody></table>";
    metricsWrap.innerHTML = html;
  }

  const MAP_WIDTH = 960;
  const MAP_HEIGHT = 520;
  const MAP_MARGIN = 8;
  let worldFeatures = null;
  let mapProjection = d3.geoNaturalEarth1();
  let mapPath = d3.geoPath(mapProjection);
  const mapSvg = d3.select("#family-winner-map-wrap").html("").append("svg")
    .attr("viewBox", `0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`)
    .attr("role", "img");

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
    const isBenefit = mapMode === "benefit";
    if (aspectLabel) aspectLabel.style.display = isBenefit ? "none" : "";
    if (aspectToolbar) aspectToolbar.style.display = isBenefit ? "none" : "";
    if (winnerLegend) winnerLegend.style.display = isBenefit ? "none" : "";
    if (benefitLegend) benefitLegend.style.display = isBenefit ? "flex" : "none";
    if (mapModeNote) {
      mapModeNote.textContent = isBenefit
        ? (FAMILY.ai_benefit_note || "100 × (1 − NRMSE_AI / NRMSE_Traditional). Positive ⇒ AI reduced error.")
        : "Which modeling paradigm wins per country for the selected evaluation aspect.";
    }
  }

  function renderWinnerMap() {
    if (!worldFeatures) return;
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
        if (!rec) return MAP_FILL_BENCHMARK;
        return colors[rec.winner_family] || "#999";
      })
      .attr("stroke", d => mapCountryStroke(d.properties.ISO_A2).color)
      .attr("stroke-width", d => mapCountryStroke(d.properties.ISO_A2).width);

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
    if (!worldFeatures) return;
    const slice = currentBenefitSlice();
    const rows = slice.countries || [];
    const byCountry = new Map(rows.map(r => [r.map_cc || r.country, r]));
    const vals = rows.map(r => r.benefit_pct).filter(v => v != null);
    const extent = benefitExtent(vals);
    const nOffScale = rows.filter(r => benefitOffScale(r.benefit_pct, extent)).length;
    if (benefitLegendLabel) {
      benefitLegendLabel.textContent = `−${extent}% · 0% · +${extent}%`;
    }

    const countriesSel = mapSvg.selectAll("path.country")
      .data(worldFeatures)
      .join("path")
      .attr("class", "country")
      .attr("shape-rendering", "geometricPrecision")
      .attr("d", mapPath)
      .attr("fill", d => {
        const iso = d.properties.ISO_A2;
        if (!isColoredBenchmarkIso(iso)) return MAP_FILL_OUTSIDE;
        const rec = byCountry.get(iso);
        if (!rec || rec.benefit_pct == null) return MAP_FILL_BENCHMARK;
        return benefitColor(rec.benefit_pct, extent);
      })
      .attr("stroke", d => mapCountryStroke(d.properties.ISO_A2).color)
      .attr("stroke-width", d => mapCountryStroke(d.properties.ISO_A2).width);

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
      mapSummary.textContent = "No AI vs traditional comparison for this horizon/crop.";
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
    mapSummary.textContent = summary;
  }

  function renderFamilyMap() {
    if (!worldFeatures) return;
    if (mapMode === "benefit") {
      renderBenefitMap();
    } else {
      renderWinnerMap();
    }
  }

  function loadMap() {
    if (!DATA.geojson_href) {
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
      renderFamilyMap();
    });
  }

  if (aspectToolbar) {
    aspectToolbar.innerHTML = (FAMILY.views || []).map((v, idx) =>
      `<button type="button" data-aspect="${v.label}"${idx === 0 ? ' class="active"' : ""}>${v.label}</button>`
    ).join("");
    aspectToolbar.addEventListener("click", ev => {
      const btn = ev.target.closest("button[data-aspect]");
      if (!btn || mapMode === "benefit") return;
      winnerAspect = btn.dataset.aspect;
      aspectToolbar.querySelectorAll("button").forEach(b => b.classList.toggle("active", b === btn));
      renderFamilyMap();
    });
  }
  if (mapModeToolbar) {
    mapModeToolbar.addEventListener("click", ev => {
      const btn = ev.target.closest("button[data-mode]");
      if (!btn) return;
      mapMode = btn.dataset.mode;
      mapModeToolbar.querySelectorAll("button").forEach(b => b.classList.toggle("active", b === btn));
      updateMapModeUi();
      renderFamilyMap();
    });
    updateMapModeUi();
  }

  function syncSelects() {
    fillHorizonSelect(metricsHorizon);
    fillHorizonSelect(mapHorizon);
    fillCropSelect(metricsCrop);
    fillCropSelect(mapCrop);
  }

  function renderAll() {
    renderFamilyTable();
    renderFamilyMap();
  }

  metricsHorizon.addEventListener("change", () => {
    famHorizon = metricsHorizon.value;
    mapHorizon.value = famHorizon;
    syncSelects();
    renderAll();
  });
  mapHorizon.addEventListener("change", () => {
    famHorizon = mapHorizon.value;
    metricsHorizon.value = famHorizon;
    syncSelects();
    renderAll();
  });
  metricsCrop.addEventListener("change", () => {
    famCrop = metricsCrop.value;
    mapCrop.value = famCrop;
    renderAll();
  });
  mapCrop.addEventListener("change", () => {
    famCrop = mapCrop.value;
    metricsCrop.value = famCrop;
    renderAll();
  });

  syncSelects();
  loadMap();
  renderFamilyTable();
})();
