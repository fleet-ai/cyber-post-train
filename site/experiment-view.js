/* The planner page has no access to benchmark result data. */
async function renderExperimentMap() {
  const root = document.querySelector("#experiment-content");
  try {
    const response = await fetch("training-decision-space.json");
    if (!response.ok) throw new Error("Could not load research plan");
    const plan = await response.json();
    const e = escapeHtml;
    const stageNames = Object.fromEntries(plan.stages);
    const sourceLinks = ids => ids.map(id => {
      const source = plan.sources.find(item => item.id === id);
      return `<a href="${e(source.url)}" target="_blank" rel="noopener">${e(source.title)}</a>`;
    }).join(" · ");
    const number = value => typeof value === "number" ? value.toLocaleString("en-US") : value;
    const rate = value => Number(value).toExponential().replace("e-0", "e-");
    document.querySelector("#plan-controls-notes").innerHTML = plan.controls.map(item => `<details><summary>${e(item.title)}</summary><p>${e(item.text)}</p></details>`).join("");
    document.querySelector("#plan-sources").innerHTML = plan.sources.map(source => `<article><h3><a href="${e(source.url)}" target="_blank" rel="noopener">${e(source.title)}</a></h3><small>${e(source.section)}</small><p>${e(source.observed)}</p><p><strong>Our decision:</strong> ${e(source.use)}</p><p class="muted"><strong>Limit:</strong> ${e(source.limit)}</p></article>`).join("");
    document.querySelector("#plan-runtime-gap").textContent = plan.defaults.proposed_not_qualified;
    document.querySelector("#experiment-scope").textContent = plan.scope;
    const winner = document.querySelector("#plan-winner");
    const finalist = document.querySelector("#plan-finalist");
    finalist.innerHTML += plan.tables.sft.runs.filter(row => row.stage !== "confirm").map(row => `<option value="${row.id}">${row.id} · ${e(row.name)}</option>`).join("");
    function table(kind, rows) {
      const columns = plan.tables[kind].columns;
      document.querySelector(`#${kind}-plan-table`).innerHTML = `<table class="research-table"><caption>${e(plan.tables[kind].title)}. Proposed runs in priority order.</caption><thead><tr>${columns.map(([key,label,meaning]) => `<th scope="col"><button class="column-button" data-key="${key}" title="${e(meaning)}">${e(label)}</button></th>`).join("")}</tr></thead><tbody>${rows.map(row => {
        const settings = row.settings || row;
        const value = key => {
          if (key === "stage") return stageNames[row.stage];
          if (row.inherit && !row.resolved && ["data","method","lr","batch","epochs","context"].includes(key) && !Object.hasOwn(row.changes,key)) return "Choose recipe";
          if (key === "lr" && !row.resolved && row.changes?.lr_multiplier) return "2 × chosen LR";
          return key === "lr" ? rate(settings[key]) : number(settings[key]);
        };
        const details = `<details><summary><b>${e(row.id)}</b> ${e(row.name)}</summary><p><strong>Question:</strong> ${e(row.question)}</p><p>${e(row.why)}</p><p><strong>Compare:</strong> ${e(row.compare)}</p>${plan.datasets[settings.data] ? `<p><strong>Data:</strong> ${e(plan.datasets[settings.data])}</p>` : ""}<p><strong>Before running:</strong> ${e(row.gate)}</p><p><strong>Training seed:</strong> ${e(settings.seed)}. ${row.inherited_from ? `Settings taken from ${e(row.inherited_from)}.` : ""}</p><p class="paper-links">${sourceLinks(row.sources)}</p></details>`;
        return `<tr>${columns.map(([key]) => key === "id" ? `<th scope="row">${details}</th>` : `<td>${e(value(key))}</td>`).join("")}</tr>`;
      }).join("")}</tbody></table>`;
      document.querySelector(`#${kind}-plan-table`).querySelectorAll(".column-button").forEach(button => button.addEventListener("click", () => {
        const column = columns.find(([key]) => key === button.dataset.key);
        document.querySelector(`#${kind}-column-help`).textContent = `${column[1]}: ${column[2]}`;
      }));
    }
    let current;
    function update() {
      current = ExperimentPlan.build(plan, { stage: document.querySelector("#plan-stage").value, limit: Number(document.querySelector("#plan-limit").value), winner: winner.value || undefined, finalist: finalist.value || undefined });
      table("sft", current.candidates);
      document.querySelector("#sft-run-count").textContent = `${current.candidates.length} proposed runs · ${current.candidates.filter(row=>!row.resolved).length} waiting for a recipe choice · none submitted`;
      document.querySelector("#plan-budget-note").textContent = `This limits the number of proposals, not GPU-hours. ${current.omitted_by_limit ? `${current.omitted_by_limit} more rows are hidden by the run limit. ` : ""}No compute total is available until the exact corpus and measured training/evaluation costs are supplied to the command-line generator.`;
    }
    root.querySelectorAll("select").forEach(select => select.addEventListener("change",update));
    for (const format of ["json","csv"]) document.querySelector(`#download-${format}`).addEventListener("click",()=>{
      const body = format === "csv" ? ExperimentPlan.csv(current) : JSON.stringify(current,null,2);
      const url = URL.createObjectURL(new Blob([body], {type: format === "csv" ? "text/csv" : "application/json"}));
      const link = document.createElement("a"); link.href=url; link.download=`qwen-sft-plan.${format}`; document.body.append(link); link.click(); link.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000);
    });
    table("rl", plan.tables.rl.runs);
    update();
  } catch (error) {
    document.querySelector("#sft-run-count").textContent = "The plan could not load. Reload this page or use the versioned plan in the repository.";
  }
}
