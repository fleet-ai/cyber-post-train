const data = window.REPORT_DATA;
let experimentMap;

function pct(value) {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setTab() {
  const requested = location.hash.slice(1);
  const tab = ["webexploitbench", "task-quality", "experiment-map"].includes(requested) ? requested : "webexploitbench";
  document.querySelectorAll("[data-tab-page]").forEach(page => { page.hidden = page.dataset.tabPage !== tab; });
  document.querySelectorAll("[data-tab-link]").forEach(link => {
    const active = link.dataset.tabLink === tab;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  const titles = {
    "webexploitbench": "WebExploitBench · Fleet Cyber",
    "task-quality": "Task quality · Fleet Cyber",
    "experiment-map": "Experiment plan · Fleet Cyber"
  };
  document.title = titles[tab];
  scrollTo({ top: 0, behavior: "instant" });
}

function renderResults() {
  const root = document.querySelector("#result-cards");
  root.innerHTML = data.runs.map(run => `
    <article class="result-card ${run.id} ${run.invalidScore ? "invalid-score" : ""}">
      <div class="card-top"><h3>${run.label}</h3><span>${run.status}</span></div>
      ${run.warning ? `<div class="score-warning">${run.warning}</div>` : ""}
      <div class="score-pair">
        <div><b>${pct(run.pass1)}</b><small>pass@1</small></div>
        <div><b>${pct(run.pass4)}</b><small>pass@4</small></div>
      </div>
      <dl><div><dt>${run.attemptLabel}</dt><dd>${run.attempts}</dd></div><div><dt>${run.scoreLabel}</dt><dd>${run.scored ?? "Not checked yet"}</dd></div><div><dt>${run.hitLabel}</dt><dd>${run.weaknessesFound}</dd></div></dl>
      <p>${run.note}</p>
    </article>`).join("");
  document.querySelector("#main-finding").textContent = "Qwen Code gives us a genuine partial result from 48 usable attempts, not the complete 15-website baseline we intended. OpenCode’s saved 2.4% four-attempt result is not a real model score: its AI report checker failed on all 206 submitted findings.";
}

function renderProtocol() {
  document.querySelector("#protocol-table").innerHTML = data.setup.map(row => `
    <tr><th>${row[0]}</th><td>${row[1]}</td><td>${row[2]}</td><td><span class="match ${row[3] ? "yes" : "no"}">${row[3] ? "Yes" : "No"}</span></td></tr>`).join("");
}

function renderTasks() {
  document.querySelector("#task-bars").innerHTML = data.tasks.map(([name, qwen, open]) => `
    <div class="task-row">
      <span class="task-name">${name}</span>
      <div class="bar-pair">
        <div class="bar-track"><span class="bar qwen" style="width:${qwen ?? 0}%"></span><em>${qwen == null ? "technical failure" : pct(qwen)}</em></div>
        <div class="bar-track"><span class="bar open" style="width:${open ?? 0}%"></span><em>${open == null ? "website did not start" : pct(open)}</em></div>
      </div>
    </div>`).join("");
}

function renderTraceFindings() {
  document.querySelector("#trace-findings").innerHTML = data.traces.map((item, i) => `
    <article><span>${String(i + 1).padStart(2, "0")}</span><div><h3>${item[0]}</h3><p>${item[1]}</p></div></article>`).join("");
  document.querySelector("#limitations").textContent = "We are rebuilding both tests on the Tensorlake cloud platform. Before the full run, one small test for each agent program must prove that the website, model, written-report checker, and final score all work together. The new pair will use the same model copies, website software, number of model turns, time limit, scoring code, and rules for technical failures.";
}

function renderFunnel() {
  const max = data.filtering.funnel[0][1];
  document.querySelector("#filter-funnel").innerHTML = data.filtering.funnel.map(([label, value], i) => `
    <div class="funnel-step" style="--width:${value / max * 100}%"><span>${String(i + 1).padStart(2, "0")}</span><div><b>${value}</b><small>${label}</small></div></div>`).join("");
}

function renderSimpleBars(selector, rows, total) {
  document.querySelector(selector).innerHTML = rows.map(([label, value]) => `
    <div class="mini-row"><span>${label}</span><div><i style="width:${value / total * 100}%"></i></div><b>${value}</b></div>`).join("");
}

function renderFiltering() {
  renderFunnel();
  renderSimpleBars("#exclusion-bars", data.filtering.exclusions, data.filtering.exclusionScale);
  renderSimpleBars("#app-mix", data.filtering.apps, data.filtering.appScale);
  renderSimpleBars("#difficulty-mix", data.filtering.difficulty, data.filtering.difficultyScale);
  document.querySelector("#quality-methods").innerHTML = data.filtering.methods.map((item, i) => `
    <article><span>${String(i + 1).padStart(2, "0")}</span><div><h3>${item[0]}</h3><p>${item[1]}</p></div></article>`).join("");
}

function tableColumns(table) {
  return table.groups.flatMap(group => group.columns.map(([key, label, meaning]) => ({ key, label, meaning })));
}

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function resolvedValue(table, run, key) {
  if (hasOwn(run, key)) return run[key];
  if (run.changes && hasOwn(run.changes, key)) return run.changes[key];
  return table.defaults[key];
}

function displayValue(value) {
  if (Array.isArray(value)) return value.join(" · ");
  if (value === true) return "Yes";
  if (value === false) return "No";
  return value ?? "—";
}

function renderExperimentTable(kind) {
  const table = experimentMap.tables[kind];
  const columns = tableColumns(table);
  const root = document.querySelector(`#${kind}-plan-table`);
  const stickyClass = index => index < 3 ? ` sticky-column sticky-column-${index + 1}` : "";

  root.innerHTML = `
    <table class="experiment-table">
      <caption>${escapeHtml(table.title)}. Rows are ordered from highest to lowest research priority.</caption>
      <thead>
        <tr class="column-groups">${table.groups.map(group => `<th colspan="${group.columns.length}" scope="colgroup">${escapeHtml(group.label)}</th>`).join("")}</tr>
        <tr>${columns.map((column, index) => `
          <th scope="col" class="${stickyClass(index)}">
            <button type="button" class="column-button" data-table-kind="${kind}" data-column-key="${escapeHtml(column.key)}">${escapeHtml(column.label)}</button>
          </th>`).join("")}</tr>
      </thead>
      <tbody>${table.runs.map(run => `
        <tr>${columns.map((column, index) => {
          const value = displayValue(resolvedValue(table, run, column.key));
          const changed = run.changes && hasOwn(run.changes, column.key);
          const tag = index === 0 ? "th" : "td";
          const scope = index === 0 ? ' scope="row"' : "";
          return `<${tag}${scope} class="${stickyClass(index)}${changed ? " changed-setting" : ""}">${escapeHtml(value)}</${tag}>`;
        }).join("")}</tr>`).join("")}</tbody>
    </table>`;

  document.querySelector(`#${kind}-run-count`).textContent = `${table.runs.length} ranked runs · ${columns.length} settings shown for every run`;
  root.querySelectorAll(".column-button").forEach(button => {
    button.addEventListener("click", () => {
      const column = columns.find(item => item.key === button.dataset.columnKey);
      document.querySelector(`#${kind}-column-help`).innerHTML = `<strong>${escapeHtml(column.label)}:</strong> ${escapeHtml(column.meaning)}`;
    });
  });
}

async function renderExperimentMap() {
  try {
    const response = await fetch("training-decision-space.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    experimentMap = await response.json();
    renderExperimentTable("sft");
    renderExperimentTable("rl");
    document.querySelector("#experiment-scope").textContent = experimentMap.scope;
  } catch (error) {
    document.querySelector("#sft-run-count").textContent = "The SFT plan could not be loaded.";
    document.querySelector("#rl-run-count").textContent = "The RL plan could not be loaded.";
  }
}

renderResults();
renderProtocol();
renderTasks();
renderTraceFindings();
renderFiltering();
renderExperimentMap();
addEventListener("hashchange", setTab);
setTab();
