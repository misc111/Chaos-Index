(function () {
  const zone = "America/New_York";
  const ctZone = "America/Chicago";

  function byId(id) {
    return document.getElementById(id);
  }

  function fmtTime(value) {
    const date = new Date(String(value || ""));
    if (Number.isNaN(date.getTime())) return "TBD";
    return date.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: zone });
  }

  function fmtStamp(value) {
    const date = new Date(String(value || ""));
    if (Number.isNaN(date.getTime())) return "Snapshot unavailable";
    return `${date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: ctZone })} ${date.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: ctZone })} CT`;
  }

  function pct(value, digits) {
    const num = Number(value);
    return Number.isFinite(num) ? `${(num * 100).toFixed(digits)}%` : "N/A";
  }

  function teamMark(code) {
    const fileCode = String(code || "").toLowerCase() === "chw" ? "cws" : String(code || "").toLowerCase();
    return `<span class="teamMark"><img src="/team-icons/mlb/${fileCode}.png" alt="" /></span>`;
  }

  function americanOddsToProbability(value) {
    const price = Number(value);
    if (!Number.isFinite(price) || price === 0) return null;
    return price > 0 ? 100 / (price + 100) : Math.abs(price) / (Math.abs(price) + 100);
  }

  function moneylineOverlay(row) {
    const homeModel = Number(row.home_win_probability);
    if (!Number.isFinite(homeModel)) return { edge: 0, side: "No bet" };

    const homeMarketRaw = americanOddsToProbability(row.moneyline && row.moneyline.home_price);
    const awayMarketRaw = americanOddsToProbability(row.moneyline && row.moneyline.away_price);
    if (homeMarketRaw === null || awayMarketRaw === null) return { edge: 0, side: "No bet" };

    const marketTotal = homeMarketRaw + awayMarketRaw;
    if (!Number.isFinite(marketTotal) || marketTotal <= 0) return { edge: 0, side: "No bet" };

    const homeOverlay = homeModel - homeMarketRaw / marketTotal;
    const awayOverlay = (1 - homeModel) - awayMarketRaw / marketTotal;
    if (homeOverlay <= 0 && awayOverlay <= 0) return { edge: 0, side: "No bet" };
    return homeOverlay >= awayOverlay ? { edge: homeOverlay, side: row.home_team } : { edge: awayOverlay, side: row.away_team };
  }

  function chaosIndexFromOverlay(edge) {
    return Math.max(0, Math.min(100, Math.round(edge * 3000)));
  }

  function isPricedMoneyline(row) {
    return americanOddsToProbability(row.moneyline && row.moneyline.home_price) !== null && americanOddsToProbability(row.moneyline && row.moneyline.away_price) !== null;
  }

  function gameRow(row) {
    const overlay = moneylineOverlay(row);
    const chaos = chaosIndexFromOverlay(overlay.edge);
    const betTone = overlay.edge > 0 ? "tealBet" : "blueBet";
    return `<div class="gameRow" role="row"><span class="gameTime">${fmtTime(row.start_time_utc)}</span><span class="matchup">${teamMark(row.away_team)}<strong>${row.away_team}</strong><span>@</span>${teamMark(row.home_team)}<strong>${row.home_team}</strong></span><span class="marketCell"><strong>Market pending</strong><small>No spread</small></span><span class="marketCell"><strong>Market pending</strong><small>No total</small></span><span class="chaosCell" title="Largest positive vig-free gap between the ensemble probability and market implied probability."><strong>${chaos}</strong><span class="chaosTrack"><span style="width:${chaos}%"></span></span></span><strong class="edgeValue">${pct(overlay.edge, 1)}</strong><strong class="bestBet ${betTone}">${overlay.side}</strong><svg viewBox="0 0 16 24" class="chevron" aria-hidden="true"><path d="m3 3 9 9-9 9"></path></svg></div>`;
  }

  function familyRows(rows) {
    const source = (rows || []).slice(0, 3);
    if (!source.length) {
      return '<div class="familyRow"><span class="rank">-</span><span>No rows</span><strong>N/A</strong><span>research-only</span><span>N/A</span></div>';
    }
    return source.map((row, index) => `<div class="familyRow"><span class="rank">${index + 1}</span><span>${row.display_name || row.model_name || "Research row"}</span><strong>${pct(row.roc_auc, 1)}</strong><span>${row.champion_status || "research-only"}</span><span>${pct(row.accuracy, 1)}</span></div>`).join("");
  }

  function ensembleRows(rows) {
    return (rows || []).slice(0, 5).map((row) => `<div class="ensembleRow" role="row"><span>${row.model_name || "model"}</span><span>${Number(row.n_games || 0)} games</span><strong>${Number.isFinite(Number(row.avg_log_loss)) ? Number(row.avg_log_loss).toFixed(3) : "N/A"}</strong><strong>${pct(row.accuracy, 1)}</strong></div>`).join("");
  }

  async function load() {
    const [market, games, nested, performance] = await Promise.all([
      fetch("/staging-data/mlb/market-board.json").then((response) => response.json()),
      fetch("/staging-data/mlb/games-today.json").then((response) => response.json()),
      fetch("/staging-data/mlb/nested-tournament.json").then((response) => response.json()),
      fetch("/staging-data/mlb/performance.json").then((response) => response.json()),
    ]);
    const rows = market.rows || [];
    const overlays = rows.map((row) => ({ row, overlay: moneylineOverlay(row) }));
    const positiveOverlays = overlays.filter((item) => item.overlay.edge > 0);
    const topOverlay = positiveOverlays.reduce((best, item) => (!best || item.overlay.edge > best.overlay.edge ? item : best), null);
    const totalEdge = positiveOverlays.reduce((sum, item) => sum + item.overlay.edge, 0);
    const pricedSides = rows.filter(isPricedMoneyline).length * 2;
    byId("model-stamp").textContent = fmtStamp(games.as_of_utc || market.as_of_utc);
    byId("kpi-games").textContent = String(rows.length);
    byId("kpi-games-note").textContent = market.date_central || "MLB snapshot";
    const kpis = document.querySelectorAll(".kpi");
    const kpiHtml = [
      `<span>Games Today</span><strong id="kpi-games">${rows.length}</strong><small id="kpi-games-note" class="tealText">${market.date_central || "MLB snapshot"}</small>`,
      topOverlay
        ? `<span>Top Edge</span><strong>${pct(topOverlay.overlay.edge, 1)}</strong><small class="orangeText">${topOverlay.row.away_team} @ ${topOverlay.row.home_team}</small>`
        : '<span>Top Edge</span><strong>Market pending</strong><small class="orangeText">No sportsbook odds</small>',
      topOverlay
        ? `<span>Best Bet</span><strong>${topOverlay.overlay.side}</strong><small class="tealText">Edge ${pct(topOverlay.overlay.edge, 1)}</small>`
        : '<span>Best Bet</span><strong>No bet</strong><small class="tealText">Awaiting prices</small>',
      `<span>Positive EV</span><strong>${positiveOverlays.length}</strong><small class="tealText">of ${pricedSides} priced sides</small>`,
      `<span>Total Edge</span><strong>${pct(totalEdge, 2)}</strong><small class="tealText">market overlays</small>`,
    ];
    kpiHtml.forEach((html, index) => {
      if (kpis[index]) kpis[index].innerHTML = html;
    });
    byId("games-body").insertAdjacentHTML("beforeend", rows.slice(0, 8).map(gameRow).join(""));
    byId("starter-grid").innerHTML = rows.slice(8, 11).map((row) => `<article class="starterCard"><time>${fmtTime(row.start_time_utc)}</time>${teamMark(row.away_team)}<span>vs</span>${teamMark(row.home_team)}<a href="/games-today?league=MLB">More info →</a></article>`).join("");
    byId("intra-family-body").innerHTML = familyRows(nested.family_champions);
    byId("inter-family-body").innerHTML = familyRows(nested.inter_family_leaderboard);
    byId("ensemble-body").innerHTML = ensembleRows(performance.run_summaries);
  }

  load().catch(() => {
    byId("model-stamp").textContent = "Snapshot unavailable";
  });
})();
