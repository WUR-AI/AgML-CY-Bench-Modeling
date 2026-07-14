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
  const mapModeToolbar = document.getElementById("family-map-mode-toolbar");
  const aspectToolbar = document.getElementById("family-winner-aspect-toolbar");
  if (!metricsWrap || !metricsHorizon) return;

  const horizonLabels = FAMILY.horizon_labels || {};
  const horizonOrder = ["eos", "early", "mid", "qtr"].filter(hz => FAMILY.by_horizon[hz]);
  let famHorizon = horizonOrder.includes("eos") ? "eos" : horizonOrder[0];
  let famCrop = "all";
  let mapMode = "winner";
  let winnerAspect = (FAMILY.views || [])[0]?.label || "Overall";

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

  function renderFamilyMap() {
    if (!worldFeatures) return;
    if (mapMode === "benefit") {
      const slice = ((FAMILY.ai_benefit_maps || {})[famHorizon] || {})[famCrop]
        || ((FAMILY.ai_benefit_maps || {})[famHorizon] || {}).all
        || { countries: [] };
      const byCountry = new Map((slice.countries || []).map(r => [r.map_cc || r.country, r]));
      mapSvg.selectAll("path.country").data(worldFeatures).join("path")
        .attr("class", "country")
        .attr("d", mapPath)
        .attr("fill", d => {
          const iso = d.properties.ISO_A2;
          const rec = byCountry.get(iso);
          if (!rec || rec.benefit_pct == null) return "#9eb4c8";
          const t = Math.max(-40, Math.min(40, rec.benefit_pct));
          return d3.interpolateRgb("#d95f02", "#2166ac")((t + 40) / 80);
        });
      mapSummary.textContent = FAMILY.ai_benefit_note || "";
      return;
    }
    const slice = ((FAMILY.winner_maps || {})[famHorizon] || {})[famCrop]
      || ((FAMILY.winner_maps || {})[famHorizon] || {}).all
      || {};
    const byAspect = slice[winnerAspect] || { countries: [] };
    const winners = new Map((byAspect.countries || []).map(r => [r.map_cc || r.country, r]));
    const colors = winnerColorMap();
    mapSvg.selectAll("path.country").data(worldFeatures).join("path")
      .attr("class", "country")
      .attr("d", mapPath)
      .attr("fill", d => {
        const iso = d.properties.ISO_A2;
        const rec = winners.get(iso);
        if (!rec) return "#9eb4c8";
        return colors[rec.winner_family] || "#999";
      });
    mapSummary.textContent = `${winnerAspect}: ${byAspect.countries?.length || 0} countries with a winning family.`;
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
      if (!btn) return;
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
      renderFamilyMap();
    });
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
