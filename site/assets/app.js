(function () {
  "use strict";
  var state = { data: null, teams: { teams: {} }, threshold: 5 };
  var fmtDay = new Intl.DateTimeFormat("en-US", {
    weekday: "short", month: "short", day: "numeric", timeZone: "America/Mexico_City"
  });
  var fmtTime = new Intl.DateTimeFormat("en-US", {
    hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "America/Mexico_City"
  });

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }
  function missing(value) {
    return value == null || Number.isNaN(Number(value));
  }
  function pct(value) {
    if (missing(value)) return "n/a";
    return Math.round(Number(value) * 100) + "%";
  }
  function safeColor(color) {
    return /^#[0-9a-fA-F]{6}$/.test(color || "") ? color : "";
  }
  function teamRecord(abbr) {
    var catalog = state.teams.teams || {};
    var row = catalog[abbr] || {};
    return {
      abbr: abbr,
      name: row.name || abbr,
      team_color: row.team_color || "",
      logo: row.logo || ""
    };
  }
  function viewGame(game) {
    return {
      kickoff: game.kickoff,
      home: teamRecord(game.home),
      away: teamRecord(game.away),
      model_home: game.model_home_wp,
      market_home: game.market_home_wp,
      estimated_home_points: game.estimated_home_points,
      estimated_away_points: game.estimated_away_points
    };
  }
  function diffPp(game) {
    if (missing(game.model_home) || missing(game.market_home)) return null;
    return Math.round((Number(game.model_home) - Number(game.market_home)) * 100);
  }
  function homeWins(modelHome) {
    return Number(modelHome) >= 0.5;
  }

  function disc(team) {
    var chip = el("span", "disc");
    chip.appendChild(el("span", "abbr", team.abbr));
    if (!team.logo) return chip;
    var img = document.createElement("img");
    img.alt = "";
    img.decoding = "async";
    img.referrerPolicy = "no-referrer";
    img.className = "is-pending";
    img.onload = function () {
      img.classList.remove("is-pending");
      chip.classList.add("has-logo");
    };
    img.onerror = function () { img.remove(); };
    img.src = team.logo;
    chip.appendChild(img);
    return chip;
  }
  function teamRow(team, role) {
    var row = el("div", "team");
    var color = safeColor(team.team_color);
    if (color) row.style.setProperty("--team-accent", color);
    row.appendChild(disc(team));
    var name = el("span", "team-name", team.name);
    name.appendChild(el("small", null, role));
    row.appendChild(name);
    return row;
  }
  function pickBox(team, role, isWin) {
    var box = el("div", "pick-box" + (isWin ? " win" : ""));
    box.appendChild(teamRow(team, role));
    if (isWin) box.appendChild(el("span", "pick-label", "WINS"));
    return box;
  }
  function scoreLine(game) {
    var block = el("p", "est-score num");
    var model = (state.data && state.data.score_model) || {};
    if (missing(game.estimated_home_points) || missing(game.estimated_away_points)) {
      var reason = model.hidden_reason || "Estimate hidden because it did not pass its pre-registered test.";
      block.appendChild(el("span", "muted", reason));
      return block;
    }
    block.appendChild(document.createTextNode(game.away.abbr + " "));
    block.appendChild(el("b", null, String(Math.round(Number(game.estimated_away_points)))));
    block.appendChild(el("span", "sep", " \u00b7 "));
    block.appendChild(el("b", null, String(Math.round(Number(game.estimated_home_points)))));
    block.appendChild(document.createTextNode(" " + game.home.abbr));
    return block;
  }
  function matchup(game) {
    var homePick = homeWins(game.model_home);
    var block = el("div", "matchup");
    var winner = homePick ? game.home.name : game.away.name;
    block.appendChild(pickBox(game.away, "away", !homePick));
    block.appendChild(pickBox(game.home, "home", homePick));
    block.appendChild(scoreLine(game));
    block.setAttribute("aria-label", "Model pick: " + winner + " wins");
    return block;
  }
  function metricText(value) {
    if (missing(value)) return "n/a";
    return Number(value).toFixed(2);
  }
  function accuracyCard(title, row) {
    var card = el("article");
    card.appendChild(el("h3", null, title));
    var decision = (row && row.decision) || "not_run";
    card.appendChild(el("p", "decision", decision));
    [
      ["Model MAE", row && row.mae],
      [(row && row.baseline_name) || "Baseline", row && row.baseline_mae],
      ["Market MAE", row && row.market_mae]
    ].forEach(function (item) {
      var line = el("div", "metric-row");
      line.appendChild(el("span", null, item[0]));
      line.appendChild(el("b", "num", metricText(item[1])));
      card.appendChild(line);
    });
    return card;
  }
  function renderAccuracy() {
    var block = document.getElementById("accuracy-block");
    if (!block || !state.data) return;
    block.textContent = "";
    var model = state.data.score_model || {};
    block.appendChild(accuracyCard("Margin", model.margin || {}));
    block.appendChild(accuracyCard("Total", model.total || {}));
    var note = document.getElementById("accuracy-note");
    if (!note) return;
    if (model.hidden_reason) {
      note.textContent = "Open experiment. " + model.hidden_reason + " Not betting picks.";
    } else {
      note.textContent = "Open experiment. Estimated score from separate margin and total models. Scores are rounded for display. Not betting picks.";
    }
  }
  function bar(game) {
    var block = el("div", "pbar");
    block.setAttribute("role", "img");
    var marketLabel = missing(game.market_home) ? "market unavailable" : "market " + pct(game.market_home);
    block.setAttribute("aria-label", "Model " + pct(game.model_home) + ", " + marketLabel);
    [["t1", game.model_home], ["t2", game.market_home]].forEach(function (item) {
      var track = el("div", "track " + item[0]);
      var fill = el("div", "fill");
      var width = missing(item[1]) ? 0 : Number(item[1]) * 100;
      fill.style.width = width + "%";
      track.appendChild(fill);
      block.appendChild(track);
    });
    block.appendChild(el("div", "mid"));
    return block;
  }
  function diffNode(pp, showBadge) {
    var wrap = el("span", "diff");
    if (pp == null) {
      wrap.appendChild(el("span", "v num", "n/a"));
      return wrap;
    }
    if (showBadge) wrap.appendChild(el("span", "badge", "DISAGREES"));
    var sign = pp > 0 ? "+" : pp < 0 ? "\u2212" : "";
    var tone = pp > 0 ? "pos" : pp < 0 ? "neg" : "";
    wrap.appendChild(el("span", "v num " + tone, sign + Math.abs(pp) + " pp"));
    return wrap;
  }
  function kick(game) {
    var raw = String(game.kickoff);
    var hasTime = raw.indexOf("T") !== -1;
    var parsed = new Date(hasTime ? raw : raw + "T12:00:00-05:00");
    var block = el("div", "kick");
    block.appendChild(el("b", null, fmtDay.format(parsed)));
    block.appendChild(document.createTextNode(hasTime ? fmtTime.format(parsed) + " CDMX" : "Time TBD"));
    return block;
  }

  function render() {
    var data = state.data;
    var body = document.getElementById("games-body");
    var cards = document.getElementById("games-cards");
    body.textContent = "";
    cards.textContent = "";
    document.getElementById("thr-text").textContent = String(state.threshold);
    data.games.forEach(function (raw) {
      var game = viewGame(raw);
      var pp = diffPp(game);
      var disagrees = pp != null && Math.abs(pp) >= state.threshold;
      var row = el("tr", disagrees ? "dis" : "");
      var cell = el("td");
      cell.appendChild(matchup(game));
      row.appendChild(cell);
      cell = el("td");
      cell.appendChild(kick(game));
      row.appendChild(cell);
      row.appendChild(el("td", "r p-model", pct(game.model_home)));
      row.appendChild(el("td", "r", pct(game.market_home)));
      cell = el("td");
      cell.appendChild(bar(game));
      row.appendChild(cell);
      cell = el("td", "r");
      cell.appendChild(diffNode(pp, disagrees));
      row.appendChild(cell);
      body.appendChild(row);

      var card = el("li", "card" + (disagrees ? " dis" : ""));
      var top = el("div", "card-top");
      top.appendChild(kick(game));
      if (disagrees) top.appendChild(el("span", "badge", "DISAGREES"));
      card.appendChild(top);
      card.appendChild(matchup(game));
      card.appendChild(bar(game));
      var nums = el("div", "card-nums");
      [
        ["Model", pct(game.model_home), "p-model"],
        ["Market", pct(game.market_home), ""],
        ["Diff.", null, ""]
      ].forEach(function (item) {
        var column = el("div");
        var label = el("span", null, item[0]);
        if (item[0] === "Model") label.title = "Home team's win probability";
        column.appendChild(label);
        if (item[1]) column.appendChild(el("b", item[2], item[1]));
        else {
          var value = el("b");
          value.appendChild(diffNode(pp, false));
          column.appendChild(value);
        }
        nums.appendChild(column);
      });
      card.appendChild(nums);
      cards.appendChild(card);
    });
    renderAccuracy();
  }

  function isExample(data) {
    return data.sample === true || data.is_example === true;
  }
  function listedWeeks(data) {
    var weeks = (data.weeks_available || []).filter(function (week) {
      return week != null && String(week) !== "";
    });
    if (!weeks.length && data.week != null) weeks = [data.week];
    return weeks;
  }
  function fillWeekSelect(data) {
    var weekSelect = document.getElementById("week-select");
    if (isExample(data)) {
      weekSelect.textContent = "";
      var exampleOption = el("option", null, "Example slate");
      exampleOption.value = "example";
      exampleOption.selected = true;
      weekSelect.appendChild(exampleOption);
      return;
    }
    var weeks = listedWeeks(data).map(String);
    var existing = Array.prototype.map.call(weekSelect.options, function (option) {
      return option.value;
    });
    var keepExisting = existing.length > weeks.length && existing.indexOf("example") === -1;
    if (keepExisting) {
      Array.prototype.forEach.call(weekSelect.options, function (option) {
        option.selected = option.value === String(data.week);
      });
      return;
    }
    weekSelect.textContent = "";
    weeks.forEach(function (week) {
      var option = el("option", null, "Week " + week);
      option.value = week;
      if (Number(week) === Number(data.week)) option.selected = true;
      weekSelect.appendChild(option);
    });
  }
  function show(data) {
    state.data = data;
    state.threshold = data.threshold_pp || 5;
    var example = isExample(data);
    var kicker = document.getElementById("season-kicker");
    var title = document.getElementById("week-title");
    if (example) {
      kicker.textContent = "Example slate";
      title.textContent = "example";
    } else {
      kicker.textContent = "Season " + data.season;
      title.textContent = "week " + data.week;
    }
    fillWeekSelect(data);
    var threshold = document.getElementById("thr-select");
    threshold.value = String(state.threshold);
    var flag = document.getElementById("example-flag");
    if (example) {
      flag.textContent = data.label || "Example slate. These are not real predictions.";
      flag.hidden = false;
    } else {
      flag.textContent = "";
      flag.hidden = true;
    }
    render();
  }

  function loadJson(url) {
    return fetch(url, { cache: "no-store" }).then(function (response) {
      if (!response.ok) throw new Error(url);
      return response.json();
    });
  }

  document.getElementById("thr-select").addEventListener("change", function (event) {
    state.threshold = Number(event.target.value);
    if (state.data) render();
  });
  document.getElementById("week-select").addEventListener("change", function (event) {
    if (!state.data || isExample(state.data)) return;
    var week = event.target.value;
    if (Number(week) === Number(state.data.week)) return;
    var url = "data/predictions/" + state.data.season + "-w" + week + ".json";
    loadJson(url).then(show).catch(function () {
      var body = document.getElementById("games-body");
      var cards = document.getElementById("games-cards");
      body.textContent = "";
      cards.textContent = "";
      var row = el("tr");
      var cell = el("td", null, "That week is not in this build.");
      cell.colSpan = 6;
      row.appendChild(cell);
      body.appendChild(row);
    });
  });

  loadJson("data/latest.json").catch(function () {
    return loadJson("data/sample.json");
  }).then(function (data) {
    return loadJson("data/teams.json").catch(function () {
      return { teams: {} };
    }).then(function (teams) {
      state.teams = teams;
      show(data);
    });
  }).catch(function () {
    var body = document.getElementById("games-body");
    var row = el("tr");
    var cell = el("td", null, "Could not load predictions.");
    cell.colSpan = 6;
    row.appendChild(cell);
    body.appendChild(row);
  });
})();
