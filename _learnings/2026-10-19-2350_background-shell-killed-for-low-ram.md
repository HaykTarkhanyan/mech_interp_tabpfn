# Claude Code kills idle background commands when RAM is low, whatever their own footprint

**Symptom.** A background run of `lit_search.py` (a few HTTP requests, a few MB of RAM) was stopped with
"stopped because the system is running low on memory". Nothing from that run was saved, because the
script checkpointed only after all 16 queries.

**Cause.** System RAM, not the script:

```
1.3 GB free of 15.6 GB RAM
top users: VS Code ~920 MB (2 processes), four Claude Code sessions ~400 MB each, Defender 400 MB, node 380 MB
```

The harness reaps background shells under RAM pressure while the session is idle. The notice also says
not to restart the job without the user's say-so. Disable with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`
in the environment Claude Code starts in; setting it from inside a session has no effect.

**Consequences.**
- Long background scripts should checkpoint per unit of work (per query, per seed, per paper), not only at
  the end. The download and chaining scripts in `robotics_foundation_models_explainability/` do.
- When several Claude sessions and VS Code are open, expect this. Closing an idle session frees about 400 MB.
