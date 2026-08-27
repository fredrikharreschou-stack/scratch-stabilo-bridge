# SCRATCH Stabilizer Bridge

A Custom Command for Assimilate SCRATCH that stabilizes the selected Shot
using the [stabilo](https://github.com/rfonod/stabilo) library and
imports the result back into the timeline as a new version.

## What it does

1. Asks SCRATCH (via its REST API) which Shot is selected in the open
   timeline.
2. Renders that Shot's actual trimmed frame range to a temp folder —
   preferably through SCRATCH's real render queue at full bit depth
   (see **Bit depth**, below), falling back to plain 8-bit snapshots if
   no matching Output preset is found.
3. Runs `stabilo`'s stabilizer on the sequence: every frame is aligned
   to a single reference frame (first / middle / last — your choice).
   This locks the shot to one moment, like a tripod aimed at that
   frame — it's not a smoothed-camera-path stabilizer. Great for
   handheld/shaky-but-basically-static shots; fights an intentional
   pan, tilt, or dolly rather than smoothing it, so pick shots (or a
   reference frame mid-move) accordingly.
4. Optionally auto-crops the wobble borders stabilization leaves
   behind, then scales back up to the original frame size.
5. Creates a new Shot from the result and adds it as a new version in
   the same Slot the original came from — A/B it with one click.

## Bit depth

`stabilo`'s feature detectors only work on 8-bit images, so motion is
*tracked* on an 8-bit proxy of each frame — but the actual warp is
always applied to the real full-precision frame, so output keeps its
full bit depth throughout. To get this, create an Output preset in
SCRATCH named **`Stabilizer 16-bit TIFF`** (or point `--output-preset`
at a different name) with whatever bit depth/colorspace/file type you
want; the script uses it automatically. No matching preset -> falls
back to SCRATCH's plain `ImageSnapshot` call, which has no bit-depth
control (always effectively 8-bit).

One confirmed, permanent SCRATCH platform quirk this works around
automatically: the render queue can never produce the true last frame
of a copied render-node input, and a handful of frames just before it
can come back frozen/duplicated on longer renders. This script detects
both and patches them via a direct snapshot call — see
[`docs-render-queue-bug-report.md`](docs-render-queue-bug-report.md)
for the full writeup (reproduction steps, evidence, and the questions
sent to Assimilate about it).

## Requirements

- Windows 10/11.
- SCRATCH running locally with the REST API enabled (System Settings)
  and a project/timeline open with a Shot selected in the player.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) —
  the script is a [PEP 723](https://peps.python.org/pep-0723/)
  single-file script; `uv run` installs Python and every dependency
  (`stabilo`, `opencv-python-headless`, `numpy`, `tqdm`,
  `assimilate_client`) automatically on first run. No manual `pip
  install` needed.
- No GPU required — the default detector (ORB) runs fine on CPU.

## Running it

Normally SCRATCH launches this itself as a Custom Command (see
**Installing as a Custom Command**, below). To run it by hand:

```
uv run scratch_stabilizer_bridge.py --ref-frame-choice middle --auto-crop y
```

Full flag list: `uv run scratch_stabilizer_bridge.py --help`. Notable
ones:

- `--ref-frame-choice {first,middle,last}` — which frame every other
  frame gets aligned to.
- `--auto-crop {y,n}` — crop off the wobble borders and scale back up.
- `--detector {orb,sift}` — ORB (fast) or SIFT (slower, more accurate)
  feature detector.
- `--output-preset NAME` — the Output preset to render through for full
  bit depth (default: `Stabilizer 16-bit TIFF`).
- `--source-format {tif,png,jpg}` — format for the 8-bit snapshot
  fallback path (default: `tif`).

## Installing as a Custom Command

1. Put `scratch_stabilizer_bridge.py` and `stabilizer_launcher.bat` in
   the same folder.
2. Open `Stabilize_Shot.acc` in a text editor and change the
   `<cmdline>` path to wherever you put `stabilizer_launcher.bat`.
3. In SCRATCH: System Settings -> Custom Commands -> Import, and pick
   the edited `.acc` file.
4. Select a Shot in the player and run "Stabilize Shot (stabilo)" from
   the Player menu.

`stabilizer_launcher.bat` locates its own folder automatically and
finds `uv` (checking PATH, then the default install locations `uv`'s
own installer uses on Windows) — no other setup needed.

## License

MIT — see [`LICENSE`](LICENSE).
