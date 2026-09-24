# Repository reset — 2026-09-24

The previous `main` at
[`c908d3a`](https://github.com/fleet-ai/cyber-post-train/tree/c908d3a828d070c6b27611fc388b1e7e3b4049dd)
had 1,607 tracked files and about 642,000 physical lines. It accumulated
training and evaluation launchers, dashboard code, generated configurations,
tests for retired routes, and many temporary copies. The new `main` keeps a
small research record and tested split logic. The old source remains readable
at that exact Git commit; it is not an active interface.

Before local cleanup, the inventory found **594 registered worktrees** and
**2,515 exact project path targets**. The non-overlapping targets occupied
about **333 GiB allocated on disk**. This measurement differs from an earlier
rough 328 GiB estimate because it was taken at a later time and used an exact
path inventory. The large local WebExploitBench materializations, project temp
directories, extra worktrees, and unused primary-checkout files were removed.
No cluster or cloud artifacts were deleted as part of this local reset.

A second scan found **12 independent project clones** and one Codex cache
clone, with their own linked checkouts; these were removed too. Three old
project folders already in Trash were permanently deleted. The final
checkout has **18 tracked files and fewer than 10,000 text lines**, one local
`main` branch, and one remote `main` branch. Its tests and GitHub size check
passed.
Free disk space was approximately **75 GiB before** and **223 GiB after** the
cleanup. This is an observed change in available space, not a sum of directory
sizes: filesystem sharing and other activity affect it.

The deletion inventory, per-path journal, Git refs, and an all-refs history
bundle are in the small local archive at
`/Users/christan/Desktop/cyber-post-train-consolidation.thpSFi`.
The bundle and current `main` retain earlier Git history. Large ignored local
outputs and environments were deleted without separate copies, as requested.

The remaining checks are documented in the [README](../README.md). The new
repository does not claim to run the old launchers or the old website.
