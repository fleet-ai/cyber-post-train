/* Shared, dependency-free planner. This never submits or contacts a service. */
(function (root) {
  "use strict";
  function build(plan, options = {}) {
    const stages = plan.stages.map(([id]) => id);
    const stage = options.stage || "first";
    if (stage !== "all" && !stages.includes(stage)) throw new Error("Unknown stage");
    const limit = options.limit ?? 100;
    if (!Number.isSafeInteger(limit) || limit < 1) throw new Error("Run limit must be positive");
    const rows = plan.tables.sft.runs;
    const screen = rows.find(row => row.id === options.winner && row.stage === "first");
    if (options.winner && !screen) throw new Error("First-stage choice must be S01, S02 or S03");
    const finalist = rows.find(row => row.id === options.finalist && row.stage !== "confirm");
    if (options.finalist && !finalist) throw new Error("Unknown completed finalist recipe");
    function resolve(row) {
      let parent = null;
      if (row.inherit) parent = row.stage === "confirm" ? finalist : screen;
      let resolved = !row.inherit || Boolean(parent);
      let settings = { ...plan.defaults };
      if (parent) {
        const inherited = resolve(parent);
        settings = { ...inherited.settings };
        resolved = inherited.resolved;
      }
      settings = { ...settings, ...row.changes };
      if (row.changes.lr_multiplier) {
        settings.lr *= row.changes.lr_multiplier;
        delete settings.lr_multiplier;
      }
      return { settings, resolved };
    }
    const candidates = rows.filter(row => stage === "all" || row.stage === stage).map(row => {
      const { settings, resolved } = resolve(row);
      const inventoryKey = settings.context === plan.defaults.context ? settings.data : `${settings.data} / context ${settings.context}`;
      const inventory = options.inventory?.[inventoryKey];
      if (inventory) {
        for (const key of ["examples", "supervised_tokens"]) {
          if (!Number.isSafeInteger(inventory[key]) || inventory[key] < 1)
            throw new Error(`Invalid ${key} for ${settings.data}`);
        }
        if (!/^[a-f0-9]{64}$/.test(inventory.manifest_sha256 || ""))
          throw new Error("Inventory needs the exact corpus manifest SHA-256");
        if (settings.data.startsWith("Matched ") && !/^[a-f0-9]{64}$/.test(inventory.family_budget_sha256 || ""))
          throw new Error("Matched sources need an identical per-family token-budget manifest");
      }
      const steps = resolved && inventory ? settings.epochs * Math.ceil(inventory.examples / settings.batch) : null;
      const tokens = resolved && inventory ? settings.epochs * inventory.supervised_tokens : null;
      if ((steps !== null && !Number.isSafeInteger(steps)) || (tokens !== null && !Number.isSafeInteger(tokens)))
        throw new Error("Training count exceeds safe integer precision");
      return {
        ...row, settings, resolved,
        status: resolved ? "Proposed; not launch-qualified" : "Conditional; choose a completed Fleet-dev result",
        corpus: inventory || null, inventory_key: inventoryKey, optimizer_steps: steps, supervised_tokens_total: tokens,
        inherited_from: row.inherit ? (row.stage === "confirm" ? options.finalist : options.winner) || null : null
      };
    });
    const matched = candidates.filter(row => row.settings.data.startsWith("Matched ") && row.corpus);
    if (new Set(matched.map(row => row.corpus.family_budget_sha256)).size > 1 || new Set(matched.map(row => row.corpus.supervised_tokens)).size > 1)
      throw new Error("Matched source arms have different family budgets or supervised-token totals");
    return { schema: plan.schema, model: plan.model, stage, candidates: candidates.slice(0, limit),
      omitted_by_limit: Math.max(0, candidates.length - limit), submits_jobs: false };
  }
  function csv(result) {
    const headers = ["id", "name", "stage", "status", "data", "method", "lr", "batch", "epochs", "context", "seed", "optimizer_steps", "supervised_tokens_total", "spec_sha256", "estimated_gpu_hours", "question", "gate"];
    const quote = value => {
      let text = value == null ? "" : String(value);
      if (/^[=+@\-\t\r]/.test(text)) text = "'" + text;
      return '"' + text.replaceAll('"', '""') + '"';
    };
    return [headers, ...result.candidates.map(row => headers.map(key => {
      if (Object.hasOwn(row, key)) return row[key];
      if (!row.resolved) return null;
      return row.settings[key];
    }))].map(row => row.map(quote).join(",")).join("\n") + "\n";
  }
  const api = { build, csv };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ExperimentPlan = api;
})(typeof window !== "undefined" ? window : globalThis);
