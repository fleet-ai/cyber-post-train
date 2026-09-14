const data = window.REPORT_DATA;

function pct(value) {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

function setTab() {
  const tab = location.hash.slice(1) === "task-curation" ? "task-curation" : "webexploitbench";
  document.querySelectorAll("[data-tab-page]").forEach(page => { page.hidden = page.dataset.tabPage !== tab; });
  document.querySelectorAll("[data-tab-link]").forEach(link => {
    const active = link.dataset.tabLink === tab;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  document.title = tab === "task-curation" ? "Task curation · Fleet Cyber" : "WebExploitBench · Fleet Cyber";
  scrollTo({ top: 0, behavior: "instant" });
}

function renderResults() {
  const root = document.querySelector("#result-cards");
  root.innerHTML = data.runs.map(run => `
    <article class="result-card ${run.id} ${run.invalidScore ? "invalid-score" : ""}">
      <div class="card-top"><h3>${run.label}</h3><span>${run.status}</span></div>
      ${run.invalidScore ? '<div class="score-warning">Invalid performance estimate</div>' : ""}
      <div class="score-pair">
        <div><b>${pct(run.pass1)}</b><small>pass@1</small></div>
        <div><b>${pct(run.pass4)}</b><small>pass@4</small></div>
      </div>
      <dl><div><dt>Attempts</dt><dd>${run.attempts}</dd></div><div><dt>Scored</dt><dd>${run.scored ?? "Pending audit"}</dd></div><div><dt>Objectives found</dt><dd>${run.objectiveHits}</dd></div></dl>
      <p>${run.note}</p>
    </article>`).join("");
  document.querySelector("#main-finding").textContent = "OpenCode’s recorded 2.4% pass@4 is not a model result. Its judge rejected every submitted finding because the configured model ID did not exist. Independent target checks show that the traces contain real successes, but they cannot reconstruct the missing judge decisions.";
}

function renderProtocol() {
  document.querySelector("#protocol-table").innerHTML = data.protocol.map(row => `
    <tr><th>${row[0]}</th><td>${row[1]}</td><td>${row[2]}</td><td><span class="match ${row[3] ? "yes" : "no"}">${row[3] ? "Yes" : "No"}</span></td></tr>`).join("");
}

function renderTasks() {
  document.querySelector("#task-bars").innerHTML = data.tasks.map(([name, qwen, open]) => `
    <div class="task-row">
      <span class="task-name">${name}</span>
      <div class="bar-pair">
        <div class="bar-track"><span class="bar qwen" style="width:${qwen ?? 0}%"></span><em>${qwen == null ? "invalid" : pct(qwen)}</em></div>
        <div class="bar-track"><span class="bar open" style="width:${open ?? 0}%"></span><em>${open == null ? "not runnable" : pct(open)}</em></div>
      </div>
    </div>`).join("");
}

function renderTraceFindings() {
  document.querySelector("#trace-findings").innerHTML = data.traces.map((item, i) => `
    <article><span>${String(i + 1).padStart(2, "0")}</span><div><h3>${item[0]}</h3><p>${item[1]}</p></div></article>`).join("");
  document.querySelector("#limitations").textContent = "Repair the judge route, prove it with a scoring canary, then run a new versioned evaluation. A real harness comparison must use the same platform, model and judge endpoints, target images, action budget, time limit, grader revision, and invalid-attempt policy. The existing OpenCode attempts should remain frozen as incident evidence, not be silently rescored or overwritten.";
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
  renderSimpleBars("#exclusion-bars", data.filtering.exclusions, 71);
  renderSimpleBars("#app-mix", data.filtering.apps, 24);
  renderSimpleBars("#difficulty-mix", data.filtering.difficulty, 82);
  document.querySelector("#curation-methods").innerHTML = data.filtering.methods.map((item, i) => `
    <article><span>${String(i + 1).padStart(2, "0")}</span><div><h3>${item[0]}</h3><p>${item[1]}</p></div></article>`).join("");
}

renderResults();
renderProtocol();
renderTasks();
renderTraceFindings();
renderFiltering();
addEventListener("hashchange", setTab);
setTab();
