#!/usr/bin/env node
// Read-only research specification generator. No job API or credentials.
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const planner = require("../site/experiment-plan.js");
const allowed = new Set(["stage", "limit", "winner", "finalist", "inventory", "costs", "budget-gpu-hours", "format"]);
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  return value;
}
try {
  const args = {};
  for (let i = 2; i < process.argv.length; i += 2) {
    const key = process.argv[i].slice(2);
    if (process.argv[i] === "--help") {
      console.log("Usage: node scripts/plan-experiments.cjs [--stage first|data|efficient|confirm|all] [--limit N] [--winner S01|S02|S03] [--finalist Sxx] [--inventory FILE] [--costs FILE --budget-gpu-hours H] [--format json|csv]\nOutputs proposed SFT specifications only. No submission. See docs/TRAINING_DECISION_SPACE.md.");
      process.exit(0);
    }
    if (!process.argv[i].startsWith("--") || !allowed.has(key) || !process.argv[i + 1] || Object.hasOwn(args, key)) throw new Error("Unknown, repeated or incomplete option");
    args[key] = process.argv[i + 1];
  }
  const format = args.format || "json";
  if (!["json", "csv"].includes(format)) throw new Error("Format must be json or csv");
  const plan = JSON.parse(fs.readFileSync(path.join(__dirname, "../site/training-decision-space.json")));
  const read = filename => JSON.parse(fs.readFileSync(filename));
  const result = planner.build(plan, { stage: args.stage, limit: args.limit ? Number(args.limit) : undefined, winner: args.winner, finalist: args.finalist, inventory: args.inventory ? read(args.inventory) : undefined });
  for (const row of result.candidates) {
    row.spec_sha256 = row.resolved ? crypto.createHash("sha256").update(JSON.stringify(canonical({ schema: plan.schema, settings: row.settings, corpus: row.corpus }))).digest("hex") : null;
  }
  const budget = args["budget-gpu-hours"] === undefined ? null : Number(args["budget-gpu-hours"]);
  if (budget !== null && (!Number.isFinite(budget) || budget <= 0 || !args.costs)) throw new Error("Budget must be positive and needs measured per-run costs");
  if (args.costs && budget === null) throw new Error("Costs require an explicit GPU-hour budget");
  if (budget !== null) {
    const costs = read(args.costs);
    if (costs.includes !== "training_export_and_fleet_dev") throw new Error("Cost estimate must include training, export and Fleet dev evaluation");
    let spent = 0;
    const selected = [], deferred = [];
    for (const row of result.candidates) {
      const cost = costs.runs?.[row.id];
      let reason;
      if (!row.resolved) reason = "No completed parent recipe selected";
      else if (!row.corpus || !cost) reason = "Missing exact corpus or measured cost";
      else if (cost.spec_sha256 !== row.spec_sha256) reason = "Cost belongs to a different recipe/corpus";
      else if (!Number.isFinite(cost.gpu_hours) || cost.gpu_hours <= 0 || typeof cost.evidence !== "string" || !cost.evidence.trim()) reason = "Invalid cost or missing measurement reference";
      else if (spent + cost.gpu_hours > budget) reason = "Outside remaining estimated budget";
      if (reason) deferred.push({ id: row.id, reason });
      else { spent += cost.gpu_hours; selected.push({ ...row, estimated_gpu_hours: cost.gpu_hours, cost_evidence: cost.evidence }); }
    }
    result.candidates = selected;
    result.deferred = deferred;
    result.budget = { gpu_hours: budget, estimated_used: spent, estimated_remaining: budget - spent, includes: costs.includes, note: "Planning estimate, not a hard runtime cap. Reserve failure/retry overhead separately. WEB transfer evaluation needs its own budget." };
    if (format === "csv" && deferred.length) console.error(JSON.stringify({ deferred }));
  }
  process.stdout.write(format === "csv" ? planner.csv(result) : JSON.stringify(result, null, 2) + "\n");
} catch (error) {
  console.error(`Plan not generated: ${error.message}`);
  process.exitCode = 1;
}
