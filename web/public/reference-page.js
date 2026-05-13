(function () {
  const chaosTitle = "Largest positive vig-free gap between the ensemble probability and market implied probability.";

  const games = [
    ["1:05 PM", "LAD", "SF", "SF -1.5", "(-110)", "O 8.0", "(-110)", 72, "2.41%", "SF -1.5", "orangeBet"],
    ["1:10 PM", "BOS", "SEA", "SEA -1.5", "(+105)", "O 7.5", "(-105)", 68, "1.87%", "SEA -1.5", "tealBet"],
    ["1:20 PM", "CHC", "PIT", "CHC -1.5", "(-115)", "O 8.5", "(-110)", 66, "1.64%", "CHC -1.5", "tealBet"],
    ["2:10 PM", "COL", "ARI", "ARI -1.5", "(-120)", "O 9.0", "(-110)", 61, "1.21%", "ARI -1.5", "redBet"],
    ["2:20 PM", "BAL", "TOR", "TOR -1.5", "(-105)", "O 8.0", "(-115)", 59, "0.98%", "TOR -1.5", "blueBet"],
    ["3:10 PM", "MIN", "CWS", "MIN -1.5", "(-110)", "O 7.5", "(-105)", 57, "0.76%", "MIN -1.5", "redBet"],
    ["4:05 PM", "MIA", "NYM", "NYM -1.5", "(-120)", "O 8.5", "(-110)", 56, "0.68%", "NYM -1.5", "orangeBet"],
    ["4:10 PM", "PHI", "ATL", "ATL -1.5", "(-105)", "O 8.0", "(-115)", 54, "0.55%", "ATL -1.5", "blueBet"],
  ];

  const starters = [
    ["7:05 PM", "STL", "MIL"],
    ["7:10 PM", "SEA", "BAL"],
    ["8:40 PM", "COL", "TOR"],
  ];

  const intraFamilies = [
    ["1", "East Power", "184.7", "23-11", "67.6%"],
    ["2", "West Power", "173.2", "21-13", "61.8%"],
    ["3", "Central Core", "161.9", "20-14", "58.8%"],
  ];

  const interFamilies = [
    ["1", "East Power", "92.1", "14-6", "70.0%"],
    ["2", "West Power", "88.7", "13-7", "65.0%"],
    ["3", "Central Core", "76.4", "11-9", "55.0%"],
  ];

  const ensemble = [
    ["Chaos Index", "32%", "8.31%", "1.23%"],
    ["Power Model", "24%", "6.27%", "1.02%"],
    ["Pitching Model", "20%", "5.18%", "0.88%"],
    ["Market Model", "14%", "4.02%", "0.76%"],
    ["Batted Ball Model", "10%", "3.15%", "0.61%"],
  ];

  function byId(id) {
    return document.getElementById(id);
  }

  function teamMark(code) {
    const fileCode = String(code || "").toLowerCase() === "chw" ? "cws" : String(code || "").toLowerCase();
    return `<span class="teamMark"><img src="/team-icons/mlb/${fileCode}.png" alt="" /></span>`;
  }

  function chevron() {
    return '<svg viewBox="0 0 16 24" class="chevron" aria-hidden="true"><path d="m3 3 9 9-9 9"></path></svg>';
  }

  function gameRow([time, away, home, spread, spreadPrice, total, totalPrice, chaos, edge, bestBet, betTone]) {
    return `<div class="gameRow" role="row"><span class="gameTime">${time}</span><span class="matchup">${teamMark(away)}<strong>${away}</strong><span>@</span>${teamMark(home)}<strong>${home}</strong></span><span class="marketCell"><strong>${spread}</strong><small>${spreadPrice}</small></span><span class="marketCell"><strong>${total}</strong><small>${totalPrice}</small></span><span class="chaosCell" title="${chaosTitle}"><strong>${chaos}</strong><span class="chaosTrack"><span style="width:${chaos}%"></span></span></span><strong class="edgeValue">${edge}</strong><strong class="bestBet ${betTone}">${bestBet}</strong>${chevron()}</div>`;
  }

  function familyRow([rank, name, score, record, win]) {
    return `<div class="familyRow"><span class="rank">${rank}</span><span>${name}</span><strong>${score}</strong><span>${record}</span><span>${win}</span></div>`;
  }

  function ensembleRow([model, weight, roi, edge]) {
    return `<div class="ensembleRow" role="row"><span>${model}</span><span>${weight}</span><strong>${roi}</strong><strong>${edge}</strong></div>`;
  }

  function renderReferencePage() {
    const modelStamp = byId("model-stamp");
    if (modelStamp) modelStamp.textContent = "May 17, 2025 8:00 AM ET";

    const kpis = document.querySelectorAll(".kpi");
    const kpiHtml = [
      '<span>Games Today</span><strong>12</strong><small class="tealText">4 starting soon</small>',
      '<span>Top Edge</span><strong>2.41%</strong><small class="orangeText">LAD @ SF</small>',
      '<span>Best Bet</span><strong>SF -1.5</strong><small class="tealText">Edge 2.41%</small>',
      '<span>Positive EV</span><strong>6</strong><small class="tealText">of 24 sides</small>',
      '<span>Total Edge</span><strong>11.37%</strong><small class="tealText">All sides</small>',
    ];
    kpiHtml.forEach((html, index) => {
      if (kpis[index]) kpis[index].innerHTML = html;
    });

    const gamesBody = byId("games-body");
    if (gamesBody) gamesBody.innerHTML = games.map(gameRow).join("");

    const starterGrid = byId("starter-grid");
    if (starterGrid) {
      starterGrid.innerHTML = starters
        .map(([time, away, home]) => `<article class="starterCard"><time>${time}</time>${teamMark(away)}<span>vs</span>${teamMark(home)}<a href="/games-today?league=MLB">More info <span aria-hidden="true">→</span></a></article>`)
        .join("");
    }

    const intraBody = byId("intra-family-body");
    if (intraBody) intraBody.innerHTML = intraFamilies.map(familyRow).join("");

    const interBody = byId("inter-family-body");
    if (interBody) interBody.innerHTML = interFamilies.map(familyRow).join("");

    const ensembleHead = document.querySelector(".ensembleHead");
    if (ensembleHead) {
      ensembleHead.innerHTML = "<span>Model</span><span>Weight</span><span>ROI (30d)</span><span>Avg Edge</span>";
    }

    const ensembleBody = byId("ensemble-body");
    if (ensembleBody) ensembleBody.innerHTML = ensemble.map(ensembleRow).join("");
  }

  renderReferencePage();
  window.addEventListener("DOMContentLoaded", renderReferencePage);
  window.addEventListener("load", function () {
    renderReferencePage();
    window.setTimeout(renderReferencePage, 250);
    window.setTimeout(renderReferencePage, 750);
  });
})();
