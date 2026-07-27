# Contributing

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). Nothing
else — it provisions the Python interpreter, the virtualenv and every tool.

```bash
git clone https://github.com/Elouen-ginat/occam-tokenizer
cd occam-tokenizer
make setup
```

That runs `uv sync` (creating `.venv` from the committed `uv.lock`) and installs
the git hooks.

## Everyday commands

| Command        | What it does                                       |
| -------------- | -------------------------------------------------- |
| `make check`   | Lint, type check and test — exactly what CI runs    |
| `make test`    | Test suite only                                     |
| `make cov`     | Test suite with a coverage report                   |
| `make fmt`     | Format and apply lint autofixes                     |
| `make type`    | mypy                                                |
| `make example` | Train a tokenizer and print what it learned         |
| `make build`   | Build the sdist and wheel                           |

Anything not in the Makefile runs through uv directly:

```bash
uv run pytest tests/test_tokenizer.py::test_roundtrip -v
uv run python -c "import occam_tokenizer; print(occam_tokenizer.__version__)"
```

## Dependencies

Runtime dependencies go in `[project.dependencies]`. Everything else goes in a
[PEP 735](https://peps.python.org/pep-0735/) group under `[dependency-groups]`,
which keeps tooling out of the published wheel.

```bash
uv add numpy                    # runtime
uv add --group test pytest-xdist
uv add --group lint types-foo
```

`uv add` updates `uv.lock` for you. **Commit the lockfile with the change** —
CI runs with `UV_FROZEN=1` and fails on a stale lock. A pre-commit hook checks
this too, so you will normally hear about it before pushing.

## Hooks

`make setup` installs two stages:

- **pre-commit** — file hygiene, ruff lint + format, `uv.lock` freshness, mypy.
- **pre-push** — the test suite, which is too slow to want on every commit.

Bypass with `--no-verify` when you need to; CI is the real gate.

## Conventions

- **Formatting is not a matter of taste.** ruff decides; don't hand-tune around
  it. Markdown is excluded so documentation tables and aligned comments survive.
- **The numerical core is deliberately dense.** `tokenizer.py` is vectorised
  numpy where the comments carry the reasoning that the code cannot. If you
  change the criterion or the merge loop, update the surrounding comment in the
  same commit — a stale explanation is worse than none.
- **Typing.** The public surface is annotated and the package ships `py.typed`.
  Internals are not fully annotated, and `disallow_untyped_defs` is off for now;
  turning it on module by module as internals get typed is welcome.
- **Tests.** Warnings are errors and markers are strict. New behaviour needs a
  test; changed behaviour needs the existing test updated rather than deleted.

## Releasing

1. Bump `__version__` in `src/occam_tokenizer/__about__.py` — hatchling reads
   the version from there, so that file is the only place it lives.
2. Move the `Unreleased` entries in `CHANGELOG.md` under the new version.
3. Tag `vX.Y.Z` and push the tag.
