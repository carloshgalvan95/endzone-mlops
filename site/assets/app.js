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
      market_home: game.market_home_wp
    };
  }
  function diffPp(game) {
    if (missing(game.model_home) || missing(game.market_home)) return null;
    return Math.round((Number(game.model_home) - Number(game.market_home)) * 100);
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
  function matchup(game) {
    var block = el("div", "matchup");
    block.appendChild(teamRow(game.away, "away"));
    block.appendChild(teamRow(game.home, "home"));
    return block;
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
        column.appendChild(el("span", null, item[0]));
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
  }

  function show(data) {
    state.data = data;
    state.threshold = data.threshold_pp || 5;
    document.getElementById("season").textContent = data.season;
    document.getElementById("week-title").textContent = data.week;
    var weekSelect = document.getElementById("week-select");
    weekSelect.textContent = "";
    (data.weeks_available || [data.week]).forEach(function (week) {
      var option = el("option", null, "Week " + week);
      option.value = String(week);
      if (Number(week) === Number(data.week)) option.selected = true;
      weekSelect.appendChild(option);
    });
    var threshold = document.getElementById("thr-select");
    threshold.value = String(state.threshold);
    var flag = document.getElementById("example-flag");
    if (data.sample === true || data.is_example === true) {
      flag.textContent = data.label || "Example data for the layout. These are not real predictions.";
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
    if (!state.data) return;
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
