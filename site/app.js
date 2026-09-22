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

function pendingValue(detail = "Pending") {
  return `<span class="metric-pending">${escapeHtml(detail)}</span>`;
}

function paperPercent(value) {
  return value == null ? pendingValue() : escapeHtml(pct(value));
}

function paperCount(value, suffix = "") {
  return value == null ? pendingValue() : `${escapeHtml(value)}${escapeHtml(suffix)}`;
}

function paperTelemetry(value) {
  return value == null ? pendingValue("Missing — not published yet") : escapeHtml(value);
}

function paperCost(value, trialCount, missingCount, formatter) {
  if (value == null) {
    if (missingCount > 0) return pendingValue(`Missing for ${missingCount} accepted attempt${missingCount === 1 ? "" : "s"}`);
    return pendingValue();
  }
  const coverage = trialCount == null || missingCount == null
    ? "coverage not published"
    : `${trialCount} accepted attempts included · ${missingCount} missing`;
  return `${escapeHtml(formatter(value))}<small>${escapeHtml(coverage)}</small>`;
}

function renderPaperAligned() {
  const root = document.querySelector("#paper-headline");
  const paper = data.paperAligned;
  const suppliedReport = paper.report;
  const reportMatchesContract = suppliedReport == null || (
    suppliedReport.schema_version === paper.reportSchema
    && suppliedReport.fixed_denominator === paper.protocol.vulnerabilities
    && suppliedReport.pass_at_1?.predeclared_attempt_index === 0
    && suppliedReport.coverage?.expected_targets === paper.protocol.apps
    && suppliedReport.coverage?.expected_trials === paper.protocol.apps * paper.protocol.repeats
  );
  const report = reportMatchesContract ? suppliedReport : null;
  const expectedTrials = report?.coverage?.expected_trials ?? paper.protocol.apps * paper.protocol.repeats;
  const reportState = !reportMatchesContract
    ? "Unavailable — report does not match the fixed protocol"
    : report == null
      ? paper.status
      : report.complete
        ? "Complete matched result"
        : "Incomplete — coverage gaps remain";
  const range = report?.attempt_rate_range;
  const completeRange = range?.best_percent != null
    && range?.worst_percent != null
    && range?.range_percentage_points != null;
  const tokenCost = report?.cost;
  const coverage = report?.coverage;

  root.innerHTML = `
    <div class="paper-report-heading">
      <div>
        <p class="paper-kicker">Three-repeat paper view</p>
        <h3>${escapeHtml(paper.title)}</h3>
      </div>
      <span class="paper-status ${report?.complete ? "complete" : "pending"}">${escapeHtml(reportState)}</span>
    </div>
    <p class="paper-denominator"><strong>Fixed denominator:</strong> ${paper.protocol.apps} apps · ${paper.protocol.vulnerabilities} known vulnerabilities · ${paper.protocol.repeats} attempts per app. Missing or technically invalid attempts stay visible and never become zero scores.</p>
    <div class="paper-metric-grid">
      <div><b>${paperPercent(report?.pass_at_1?.rate_percent)}</b><span>Pass@1 · attempt 0</span></div>
      <div><b>${paperPercent(report?.pass_at_3?.avg_percent)}</b><span>Pass@3 Avg</span></div>
      <div><b>${paperPercent(report?.pass_at_3?.max_percent)}</b><span>Pass@3 Max</span></div>
    </div>
    <div class="paper-band">
      <h4>Best-to-worst three-repeat band</h4>
      ${completeRange ? `
        <p><strong>${escapeHtml(pct(range.worst_percent))}</strong> worst · <strong>${escapeHtml(pct(range.best_percent))}</strong> best · <strong>${escapeHtml(range.range_percentage_points.toFixed(1))} percentage points</strong> wide</p>
      ` : `<p>${pendingValue("Pending — all three attempts must be complete")}</p>`}
    </div>
    <div class="paper-details">
      <section aria-labelledby="paper-coverage-title">
        <h4 id="paper-coverage-title">Coverage</h4>
        <dl>
          <div><dt>Accepted attempts</dt><dd>${paperCount(coverage?.valid_model_outcome_trials, ` / ${expectedTrials}`)}</dd></div>
          <div><dt>Technically invalid</dt><dd>${paperCount(coverage?.infrastructure_invalid?.trial_count)}</dd></div>
          <div><dt>Missing</dt><dd>${paperCount(coverage?.missing?.trial_count)}</dd></div>
        </dl>
      </section>
      <section aria-labelledby="paper-telemetry-title">
        <h4 id="paper-telemetry-title">Telemetry and evidence</h4>
        <dl>
          <div><dt>Mean model tokens</dt><dd>${paperCost(tokenCost?.mean_token_cost_millions, tokenCost?.token_cost_trial_count, tokenCost?.token_cost_missing_valid_trial_count, value => `${value.toFixed(3)} million`)}</dd></div>
          <div><dt>Mean wall time</dt><dd>${paperCost(tokenCost?.mean_wall_time_seconds, tokenCost?.wall_time_trial_count, tokenCost?.wall_time_missing_valid_trial_count, value => `${value.toFixed(1)} seconds`)}</dd></div>
          <div><dt>Model requests</dt><dd>${paperTelemetry(paper.telemetry.modelRequests)}</dd></div>
          <div><dt>Provider cost</dt><dd>${paperTelemetry(paper.telemetry.providerCost)}</dd></div>
          <div><dt>Judge calls</dt><dd>${paperTelemetry(paper.telemetry.judgeCalls)}</dd></div>
          <div><dt>Collection receipts</dt><dd>${paperTelemetry(paper.telemetry.collectionReceiptDigests)}</dd></div>
          <div><dt>Scoring receipts</dt><dd>${paperTelemetry(paper.telemetry.scoringReceiptDigests)}</dd></div>
        </dl>
      </section>
    </div>
    <div class="paper-deviations">
      <h4>What matches the paper — and what does not</h4>
      <div>${paper.deviations.map(item => `
        <article>
          <div><strong>${escapeHtml(item.label)}</strong><span class="deviation-status">${escapeHtml(item.status)}</span></div>
          <p>${escapeHtml(item.detail)}</p>
        </article>`).join("")}</div>
    </div>`;
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
        <div><b>${pct(run.pass4)}</b><small>${run.pass4Label ?? "pass@4"}</small></div>
      </div>
      <dl><div><dt>${run.attemptLabel}</dt><dd>${run.attempts}</dd></div><div><dt>${run.scoreLabel}</dt><dd>${run.scored ?? "Not checked yet"}</dd></div><div><dt>${run.hitLabel}</dt><dd>${run.weaknessesFound}</dd></div></dl>
      <p>${run.note}${run.source ? ` <a href="${run.source}">Read the aggregate evidence.</a>` : ""}</p>
    </article>`).join("");
  document.querySelector("#main-finding").textContent = "The new one-attempt OpenCode run is still partial: 14 of 15 collections are valid, all 14 valid saved attempts have accepted scores under one policy, and 11 of 98 scored weakness checks received credit. The one technical exclusion is not a zero. This is not a complete baseline and does not show whether training helped.";
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
  document.querySelector("#limitations").textContent = "The fresh run gives one accepted score for each of the 14 valid collections. A complete baseline still requires valid evidence for the collection that failed technically. Any future replacement rule must be written before another run. A training claim also needs a separately collected trained-model result under the same rules.";
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

renderPaperAligned();
renderResults();
renderProtocol();
renderTasks();
renderTraceFindings();
renderFiltering();
renderExperimentMap();
addEventListener("hashchange", setTab);
setTab();
