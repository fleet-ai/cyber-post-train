/* Public, aggregate-only campaign history. No credentials or private trace URLs. */
const paperMetricLabels = {
  pass_at_1: "Pass@1",
  pass_at_3_avg: "Pass@3 (Avg.)",
  pass_at_3_max: "Pass@3 (Max)"
};

function standardizedMetric(key, metric) {
  return `<div><dt>${paperMetricLabels[key]}</dt><dd>${Number(metric.percent).toFixed(2)}%</dd><small>${escapeHtml(metric.coverage)} weaknesses · ${escapeHtml(metric.qualification)}</small></div>`;
}

function renderPaperReference(paper) {
  document.querySelector("#paper-reference-body").innerHTML = paper.results.map(result => `<tr>
    <th scope="row">${escapeHtml(result.model)}</th><td>${escapeHtml(result.agent)}</td>
    <td>${Number(result.pass_at_1).toFixed(2)}%</td><td>${Number(result.pass_at_3_avg).toFixed(2)}%</td><td>${Number(result.pass_at_3_max).toFixed(2)}%</td>
  </tr>`).join("");
  document.querySelector("#paper-reference-note").textContent = `${paper.level} · ${paper.coverage} · ${paper.attempts} independent attempts. ${paper.note}`;
  document.querySelector("#paper-reference-link").href = paper.url;
}

function renderWebRows(runs) {
  document.querySelector("#web-count").textContent = `${runs.length} checked results`;
  document.querySelector("#web-campaigns").innerHTML = runs.map(run => `
    <article class="web-campaign" id="web-${escapeHtml(run.id)}">
      <div class="web-campaign-heading"><div><p class="eyebrow">${escapeHtml(run.date)} · partial coverage</p><h3>${escapeHtml(run.name)}</h3></div></div>
      <dl class="web-config"><div><dt>Model</dt><dd>${escapeHtml(run.model)}</dd></div><div><dt>Agent program</dt><dd>${escapeHtml(run.harness)}</dd></div><div><dt>Report-checking AI</dt><dd>${escapeHtml(run.judge)}</dd></div><div><dt>Websites covered</dt><dd>${escapeHtml(run.targets)}</dd></div></dl>
      <div class="web-stages"><p><strong>Collection</strong> ${escapeHtml(run.collection)}</p><p><strong>Scoring</strong> ${escapeHtml(run.scoring)}</p></div>
      <dl class="web-metrics">${Object.entries(paperMetricLabels).map(([key]) => standardizedMetric(key, run.paper_metrics[key])).join("")}</dl>
      <p>${escapeHtml(run.note)}</p>
      <details><summary>Settings and evidence</summary><dl class="web-detail">
        ${Object.entries({"Campaign ID":run.id,"Checkpoint":run.checkpoint,"Execution platform":run.platform,"Planned attempts":run.planned ?? "Not established","Usable attempts":run.usable ?? "Not established","Planned attempts per website":run.attempts_per_target ?? "Not frozen / varies",...run.config}).map(([k,v]) => `<div><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(Array.isArray(v) ? v.join("; ") : v)}</dd></div>`).join("")}
      </dl>${run.source ? `<p><a href="${escapeHtml(run.source)}">Read the supporting record</a> · Repository access may be required.</p>` : "<p>No public supporting record yet; no score is claimed.</p>"}</details>
    </article>`).join("");
}

function renderResultsAnalysis(analysis) {
  const verdicts = analysis.verdicts.map(item => `<article>
    <p class="analysis-label">${escapeHtml(item.verdict)}</p>
    <h4>${escapeHtml(item.claim)}</h4>
    <p>${escapeHtml(item.reason)}</p>
  </article>`).join("");
  const observed = analysis.trace_review.observed.map(item => `<tr>
    <th scope="row">${escapeHtml(item.measure)}</th><td>${escapeHtml(item.smoke)}</td><td>${escapeHtml(item.baseline)}</td>
  </tr>`).join("");
  const list = items => `<ul>${items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;

  document.querySelector("#web-results-analysis").innerHTML = `<details class="results-analysis">
    <summary><span>Results analysis</span><small>${escapeHtml(analysis.summary)}</small></summary>
    <div class="analysis-body">
      <p class="analysis-lead">${escapeHtml(analysis.lead)}</p>
      <h3>What the evidence supports</h3>
      <div class="analysis-verdicts">${verdicts}</div>
      <h3>Why the headline scores are not a fair model comparison</h3>
      ${list(analysis.comparison_limits)}
      <h3>What the saved attempts show</h3>
      <p>${escapeHtml(analysis.trace_review.scope)}</p>
      <div class="analysis-observed"><table aria-label="Behavior seen in the saved attempts"><thead><tr><th>Measure</th><th>Smoke test 1</th><th>Available baseline sample</th></tr></thead><tbody>${observed}</tbody></table></div>
      <p class="analysis-caution">${escapeHtml(analysis.trace_review.caution)}</p>
      <h3>Best explanation at present</h3>
      ${list(analysis.explanations)}
      <h3>What was ruled out</h3>
      ${list(analysis.ruled_out)}
      <h3>The test that will settle it</h3>
      ${list(analysis.next_test)}
      <p class="analysis-conclusion"><strong>Current conclusion:</strong> ${escapeHtml(analysis.conclusion)}</p>
    </div>
  </details>`;
}

async function renderWebEvaluations() {
  try {
    const response = await fetch("web-evals.json", {cache:"no-store"});
    if (!response.ok) throw new Error("Could not load evaluation history");
    const history = await response.json();
    document.querySelector("#web-updated").textContent = `Last evidence review: ${history.updated_at}`;
    document.querySelector("#web-scope").textContent = history.scope;
    renderPaperReference(history.paper_reference);
    renderWebRows(history.runs);
    renderResultsAnalysis(history.results_analysis);
  } catch {
    document.querySelector("#web-count").textContent = "Evaluation history could not load. Please reload; no results have been inferred.";
  }
}
renderWebEvaluations();
