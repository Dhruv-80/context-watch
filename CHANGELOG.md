# Changelog

All notable changes to ContextWatch will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).
Versioning: each phase increments the minor version (Phase 1 → 0.1.0, Phase 2 → 0.2.0, etc.).

<!--
HOW TO UPDATE:
1. Copy the template below and paste it above the latest entry.
2. Fill in the date and describe changes under Added / Changed / Fixed.
3. Update the version in contextwatch/__init__.py and pyproject.toml to match.
4. Commit and tag:  git tag v0.X.0 && git push --tags

## [0.X.0] - YYYY-MM-DD
### Added
- ...
### Changed
- ...
### Fixed
- ...
### Notes
- ...
-->

## [0.2.0] - 2026-02-15

### Added
- `contextwatch/monitor/` package with `ContextTracker`, `ContextSnapshot`, and `ContextSummary`
- Max context window detection from model config (`max_position_embeddings` / `n_positions` / `n_ctx`)
- Per-step context usage tracking: % used, remaining tokens
- Configurable warning threshold via `--warn-threshold` CLI flag (default 75%)
- Context exhaustion early stopping — generation stops if context window is full
- Context summary printed in CLI output (`Context: X% (used/max)`, `Remaining tokens: N`)
- `context_summary` field on `InferenceResult` for programmatic access
- Context tracking assertions in `examples/validate.py`

### Notes
- Phase 2 complete — context tracking is now live
- Warning is emitted to stderr so it does not pollute piped stdout

## [0.1.0] - 2026-02-14

### Added
- CLI command: `contextwatch run --model <name> --prompt "<text>" --max-tokens <int>`
- Manual stepwise inference loop (no `model.generate()`)
- KV-cache (`past_key_values`) management across steps
- EOS-token early stopping
- Token accounting: prompt, generated, and total token counts
- Validation script (`examples/validate.py`) using `distilgpt2`

### Notes
- Phase 0 + Phase 1 combined into initial release
- Greedy decoding only (argmax); sampling strategies deferred to future phases
