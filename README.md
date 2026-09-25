# Cyber post-training

Compact Qwen cyber-training and evaluation research code. The repository was
reset on 2026-09-24; the [reset record](https://github.com/fleet-ai/cyber-post-train/blob/f6fb3b88/docs/RESET.md) remains in Git history.
Read [AGENTS.md](AGENTS.md) before any cluster or paid operation.

The current study and its unpassed launch gates are in
[QWEN38_SFT_STUDY.md](docs/QWEN38_SFT_STUDY.md). Task-family roles, source
quality, and held-out limitations are in [HOLDOUT.md](docs/HOLDOUT.md).
The files in [configs/data/](configs/data/) and [docs/evidence/](docs/evidence/)
are dated observations, not current proof that a task works or a model is
available. Keep training and evaluation inputs separate by task family; a
checkpoint or falling training loss alone does not establish capability lift.

`training/` contains the bounded data, preflight, and SFT tools; `evals/`
contains matched Fleet evaluation tools. A passing local test does not
authorize a job: use the exact live identity, resource, preview, and failure-
alert checks in [AGENTS.md](AGENTS.md). No job starts by reading this repo.

Run the checks with Python 3:

```sh
python3 -m unittest discover -s tests
python3 scripts/check_size.py
```

Keep fewer than 10,000 tracked text lines. Do not commit credentials, raw
prompts or traces, flags, answers, weights, or generated runtime folders.
