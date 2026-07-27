# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Renamed from `modl-tokenizer` to `occam-tokenizer`; `MODLTokenizer` is now
  `OccamTokenizer` and the import is `occam_tokenizer`. No behavioural change.
- Moved to a `src/` layout, with tests and examples out of the project root.
- Build backend is now hatchling; the version is single-sourced from
  `src/occam_tokenizer/__about__.py`.
- `requirements.txt` is gone. Runtime dependencies live in
  `[project.dependencies]`, tooling in PEP 735 dependency groups, and `uv.lock`
  is committed.

### Added

- MIT license — previously unchosen.
- `py.typed`, so downstream type checkers see the annotations.
- uv-managed development environment with ruff, mypy, pytest and pre-commit.
- GitHub Actions CI: lint, tests on Python 3.10–3.13 plus macOS and Windows,
  and a build job that checks the distribution metadata.

### Fixed

- `__call__` passed a `str | Sequence[str]` into `encode()` on the single-string
  path. Correct at runtime, but the types did not prove it.

## [0.1.0]

Initial implementation of the parameter-free MDL tokenizer.
