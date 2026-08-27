# Changelog

## 1.0 - 2026-08-26

Initial release.

### Added
- Core pipeline: pull the selected Shot's trimmed frame range from
  SCRATCH, stabilize it with `stabilo` (lock-to-one-frame alignment,
  choice of first/middle/last reference), optional auto-crop of the
  wobble borders, and import the result back as a new version in the
  same Slot.
- Bit-depth-aware rendering through SCRATCH's real render queue and a
  configurable Output preset (`--output-preset`, default `Stabilizer
  16-bit TIFF`) instead of being limited to the always-8-bit
  `ImageSnapshot` fallback. Motion is tracked on an 8-bit proxy (a
  detector requirement), but the actual warp writes full-precision
  pixels throughout.
- Automatic detection and repair of a confirmed, permanent SCRATCH
  render-queue limitation: the true last frame of a copied render-node
  input is never produced, and a run of frames just before it can come
  back frozen/duplicated on longer renders. Every affected frame is
  individually re-fetched via a direct snapshot call and color-depth
  matched. Verified reliable across 5 shots, both QuickTime and DPX
  source, ranges from 15 to 100 frames. Full writeup in
  `docs-render-queue-bug-report.md`.
- Local-vs-absolute frame renumbering fix for render queue results
  (SCRATCH reports frame numbers local to the copied input, not the
  original shot's absolute numbering).
- Fixed the import step's file-extension filter, which only accepted
  `.png`, to also accept `.tif`/`.tiff`/`.jpg`/`.jpeg`/`.exr`/`.dpx`.
- `stabilizer_launcher.bat` / `Stabilize_Shot.acc` for one-click
  installation as a SCRATCH Custom Command, launched via `uv run` (no
  manual dependency install).
