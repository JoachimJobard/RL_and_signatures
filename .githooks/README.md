# Versioned git hooks

These hooks travel with the repository. Git does not use them automatically — point git at this
directory once per clone:

```bash
git config core.hooksPath .githooks
```

## `pre-commit`

Keeps the experiment-registry Markdown index consistent with its source. `experiment_registry.md`
is a derived artefact rendered from `experiment_registry.jsonl` by
`documents/methodology/render_registry_index.py`. If a commit touches the ledger, the generator, or
the index and the checked-in Markdown does not match a fresh render, the hook regenerates it and
fails the commit; stage the regenerated file (`git add documents/methodology/experiment_registry.md`)
and re-commit.

To bypass in an emergency: `git commit --no-verify`.
