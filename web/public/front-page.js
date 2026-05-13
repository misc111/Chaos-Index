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

  function modelDisagreement(row) {
    const probs = Object.values(row.model_win_probabilities || {}).filter((value) => Number.isFinite(Number(value))).map(Number);
    if (probs.length < 2) return 0;
    return Math.max(...probs) - Math.min(...probs);
  }

  function gameRow(row) {
    const homeProb = Number(row.home_win_probability);
    const favorite = Number.isFinite(homeProb) && homeProb >= 0.5 ? row.home_team : row.away_team;
    const edge = Number.isFinite(homeProb) ? pct(Math.max(homeProb, 1 - homeProb) - 0.5, 1) : "N/A";
    const chaos = Math.max(1, Math.min(100, Math.round(modelDisagreement(row) * 500)));
    const bestBet = row.moneyline && row.moneyline.books_count ? favorite : "No bet";
    const betTone = row.moneyline && row.moneyline.books_count ? "tealBet" : "blueBet";
    return `<div class="gameRow" role="row"><span class="gameTime">${fmtTime(row.start_time_utc)}</span><span class="matchup">${teamMark(row.away_team)}<strong>${row.away_team}</strong><span>@</span>${teamMark(row.home_team)}<strong>${row.home_team}</strong></span><span class="marketCell"><strong>Market pending</strong><small>No spread</small></span><span class="marketCell"><strong>Market pending</strong><small>No total</small></span><span class="chaosCell"><strong>${chaos}</strong><span class="chaosTrack"><span style="width:${chaos}%"></span></span></span><strong class="edgeValue">${edge}</strong><strong class="bestBet ${betTone}">${bestBet}</strong><svg viewBox="0 0 16 24" class="chevron" aria-hidden="true"><path d="m3 3 9 9-9 9"></path></svg></div>`;
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
    byId("model-stamp").textContent = fmtStamp(games.as_of_utc || market.as_of_utc);
    byId("kpi-games").textContent = String(rows.length);
    byId("kpi-games-note").textContent = market.date_central || "MLB snapshot";
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
