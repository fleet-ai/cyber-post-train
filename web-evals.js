/* Public, aggregate-only campaign history. No credentials or private trace URLs. */
const webStatuses = {
  complete: "Complete result", partial: "Partial result", invalid: "Invalid scoring",
  incomplete: "Incomplete / needs checking", test: "Small system test", planned: "Not launched"
};

function webMetric(metric) {
  if (metric.percent != null) return `${metric.percent.toFixed(2)}% · ${metric.label}`;
  return `${(100 * metric.numerator / metric.denominator).toFixed(2)}% (${metric.numerator}/${metric.denominator}) · ${metric.label}`;
}

function renderWebRows(runs) {
  const search = document.querySelector("#web-search").value.toLowerCase();
  const status = document.querySelector("#web-status").value;
  const selected = runs.filter(run => (!status || run.status === status) &&
    [run.name, run.id, run.model, run.harness, run.judge].join(" ").toLowerCase().includes(search));
  document.querySelector("#web-count").textContent = `${selected.length} of ${runs.length} campaign records or repair groups`;
  document.querySelector("#web-campaigns").innerHTML = selected.map(run => `
    <article class="web-campaign" id="web-${escapeHtml(run.id)}">
      <div class="web-campaign-heading"><div><p class="eyebrow">${escapeHtml(run.date)}</p><h3>${escapeHtml(run.name)}</h3></div>
        <span class="web-status ${escapeHtml(run.status)}">${escapeHtml(webStatuses[run.status])}</span></div>
      <dl class="web-config"><div><dt>Model</dt><dd>${escapeHtml(run.model)}</dd></div><div><dt>Agent program</dt><dd>${escapeHtml(run.harness)}</dd></div><div><dt>Report-checking AI</dt><dd>${escapeHtml(run.judge)}</dd></div><div><dt>Websites covered</dt><dd>${escapeHtml(run.targets)}</dd></div></dl>
      <div class="web-stages"><p><strong>Collection</strong> ${escapeHtml(run.collection)}</p><p><strong>Scoring</strong> ${escapeHtml(run.scoring)}</p></div>
      <div class="web-metrics">${run.metrics.length ? run.metrics.map(m => `<p>${escapeHtml(webMetric(m))}</p>`).join("") : '<p class="web-no-score">No benchmark-wide performance number</p>'}</div>
      <p>${escapeHtml(run.note)}</p>
      <details><summary>Settings and evidence</summary><dl class="web-detail">
        ${Object.entries({"Campaign ID":run.id,"Checkpoint":run.checkpoint,"Execution platform":run.platform,"Planned attempts":run.planned ?? "Not established","Usable attempts":run.usable ?? "Not established","Planned attempts per website":run.attempts_per_target ?? "Not frozen / varies",...run.config}).map(([k,v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(Array.isArray(v) ? v.join("; ") : v)}</dd></div>`).join("")}
      </dl>${run.source ? `<p><a href="${escapeHtml(run.source)}">Read the supporting record</a> · Repository access may be required.</p>` : "<p>No public supporting record yet; no score is claimed.</p>"}</details>
    </article>`).join("") || "<p>No matching campaigns.</p>";
}

async function renderWebEvaluations() {
  try {
    const response = await fetch("web-evals.json", {cache:"no-store"});
    if (!response.ok) throw new Error("Could not load evaluation history");
    const history = await response.json();
    document.querySelector("#web-updated").textContent = `Last evidence review: ${history.updated_at}`;
    document.querySelector("#web-scope").textContent = history.scope;
    for (const id of ["web-search", "web-status"]) document.querySelector(`#${id}`).addEventListener("input", () => renderWebRows(history.runs));
    renderWebRows(history.runs);
  } catch {
    document.querySelector("#web-count").textContent = "Evaluation history could not load. Please reload; no results have been inferred.";
  }
}
renderWebEvaluations();
