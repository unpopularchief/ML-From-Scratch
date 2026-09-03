## What

<!-- One algorithm / layer / feature per PR. What does this add? -->

## Checklist

- [ ] Derivation was walked through and confirmed before this code was written (see plan.md §0)
- [ ] `docs/derivations/<name>.md` written or updated
- [ ] Unit tests added (analytic checks; gradient check too, if this has a `backward()`)
- [ ] Example script added/updated in `examples/`, and it runs end-to-end
- [ ] `README.md` algorithm table and `ROADMAP.md` updated in this same PR
- [ ] `uv run ruff check . && uv run ruff format --check .` passes
- [ ] `uv run pytest -q` passes
