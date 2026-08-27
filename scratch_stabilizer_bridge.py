#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "stabilo",
#     "opencv-python-headless",
#     "numpy",
#     "tqdm",
#     "assimilate_client @ git+https://github.com/Assimilate-Inc/Assimilate-REST.git",
# ]
# ///
"""
scratch_stabilizer_bridge.py -- Stabilize the selected Shot in Assimilate SCRATCH
using the 'stabilo' library, and import the result back as a new version.

WHAT THIS DOES
    1. Asks SCRATCH (via its REST API) which Shot is currently selected in the
       open timeline.
    2. Renders that Shot's frames (its actual trimmed in/out range, not the
       whole source file) to a temporary folder. Two ways this can happen:
         a) Preferred: through SCRATCH's real render queue, reusing an
            Output preset you configure once in SCRATCH's own Outputs panel
            (System Settings or the Outputs list) -- whatever bit depth,
            colorspace, and file type that preset is set to (e.g. 16-bit
            TIFF, Rec709) is exactly what gets rendered and carried all the
            way through to the final stabilized result. By default this
            looks for a preset named "Stabilizer 16-bit TIFF" (see
            DEFAULT_OUTPUT_PRESET_NAME / --output-preset) -- create one with
            that exact name and it's used automatically, no extra clicks
            needed per run.
         b) Fallback: SCRATCH's per-frame ImageSnapshot call (TIFF by
            default -- see --source-format), used automatically if no
            matching Output preset is found. Simple and robust, but this
            particular API call has no bit-depth control at all -- it's
            always effectively 8-bit no matter which file format you pick.
    3. Runs stabilo's Stabilizer on that frame sequence: every frame is
       aligned to a single reference frame (first / middle / last, your
       choice). This is a "lock to one frame" stabilizer, not a
       smoothed-camera-path one -- see the IMPORTANT NOTE below.
    4. Optionally auto-crops the sequence to the largest rectangle that stays
       inside every warped frame (removing the black wobble-borders that
       stabilization leaves behind), then scales back up to the original
       frame size so the result drops into the timeline at the same
       resolution.
    5. Creates a new Shot from the stabilized sequence and adds it as a new
       version in the same Slot the original Shot came from, so you can
       A/B it against the original with one click in SCRATCH's player.

WHERE THE FILES GO
    The rendered SOURCE frames (step 2, raw pulls from SCRATCH used only as
    input to the stabilizer) go in a temp folder and get deleted afterwards.
    The STABILIZED output (step 4) is different: SCRATCH's new Shot points
    at those files by path and reads pixels from disk on demand -- it does
    NOT copy them in -- so they are written to a permanent
    "StabilizedShots" folder and are never deleted automatically. That
    folder is created under the current project's own configured render
    path (the same place Assimilate's own example scripts write generated
    media), or next to this script if the project's render path can't be
    read. Don't remove that folder yourself while the stabilized version
    is still in your project; if you need to reclaim the space, delete the
    shot from SCRATCH's project first.

IMPORTANT NOTE ON WHAT "STABILIZE" MEANS HERE
    stabilo anchors every single frame to ONE reference frame (like locking
    a shot to a tripod aimed at that one moment). That's excellent for
    handheld/shaky shots that are basically static (a locked-off shot held
    by hand, a shaky close-up, a drone hover). It is NOT a "smooth the
    camera move" tool -- if the shot has an intentional pan, tilt, or
    dolly, anchoring the whole clip to one frame will fight that move and
    can look wrong. Pick shots accordingly, or pick a reference frame in
    the middle of a move as a compromise.

NOTE ON HIGHER-BIT-DEPTH RENDERS (--output-preset)
    stabilo's feature detectors (ORB/SIFT) only work on standard 8-bit
    images -- they error out on 16-bit input. So when rendering through a
    16-bit (or higher) Output preset, motion is tracked on an 8-bit proxy
    of each frame (just for finding the alignment), but the actual warp
    that produces the output is always applied to the real full-precision
    frame -- the written pixels keep their full bit depth throughout. The
    render-queue mechanism this uses is built on the same pattern as
    Assimilate's own examples/RenderSelectedShot.py, and has been verified
    end-to-end against a live 16-bit TIFF Output preset.

    One confirmed, permanent quirk of that render-queue path: SCRATCH is
    never able to render the true LAST frame of a copied render-node input,
    no matter what frame range is requested -- verified repeatedly across
    different shots and range formulations. This script expects that and
    works around it automatically: it detects the one-frame gap and fills
    it in via the plain 8-bit snapshot method, upconverted to match the
    other frames' bit depth. You'll see a "Filling in the one frame..."
    line on every run through this path -- that's expected, not an error.
    Only that single tail frame lacks true extra precision; everything
    else renders at the preset's real bit depth.

REQUIREMENTS
    - Windows 10/11 (this launches the same way as the Ai-Matte tool: via
      'uv run', which auto-installs Python + all dependencies on first run).
    - SCRATCH running locally with the REST API turned on
      (System Settings -> REST API) and a project/timeline open with a
      Shot selected in the player.
    - A GPU is NOT required. stabilo's default detector (ORB) runs fine on
      CPU. If you have an NVIDIA GPU and want a speed boost you'd need a
      CUDA-enabled OpenCV build, which is an advanced/optional step covered
      in stabilo's own docs -- most people can ignore this entirely.

RUNNING IT
    Normally this is launched by SCRATCH itself as a Custom Command (see
    Stabilize_Shot.acc / stabilizer_launcher.bat). To run it by hand for
    testing:

        uv run scratch_stabilizer_bridge.py --ref-frame-choice middle --auto-crop y

    Full flag list: run with --help.

BUILDING THE .acc CUSTOM COMMAND
    See stabilizer_launcher.bat and Stabilize_Shot.acc in this same folder.
    Import Stabilize_Shot.acc via SCRATCH's System Settings -> Custom
    Commands -> Import, after editing the <cmdline> path inside it to point
    at wherever you put stabilizer_launcher.bat.
"""

import argparse
import os
import sys
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# SCRATCH's TIFF snapshots carry a private metadata tag OpenCV's TIFF reader
# doesn't recognize (harmless -- it just skips it and reads the image fine),
# but it prints a "TIFFReadDirectory: Unknown field" warning on every single
# frame otherwise. Silence warnings; real errors still show.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - PEP 723 guarantees this, but be safe
    def tqdm(iterable=None, total=None, desc="", unit=""):
        if iterable is not None:
            for item in iterable:
                yield item
        return type("FakeBar", (), {"update": lambda self, n=1: None, "close": lambda self: None})()

import assimilate_client
from assimilate_client.rest import ApiException
from stabilo import Stabilizer


# The Construct Output preset name this script looks for by default, so the
# Custom Command (.acc) doesn't need a 4th input field just to type this in
# every time. Create an Output in SCRATCH's own Outputs panel with exactly
# this name (case-insensitive) set to whatever bit depth / colorspace / file
# type you want the stabilized result in (e.g. 16-bit TIFF, Rec709) and it
# will be picked up automatically. Override per-run with --output-preset, or
# pass --output-preset "" to force the plain 8-bit snapshot path.
DEFAULT_OUTPUT_PRESET_NAME = "Stabilizer 16-bit TIFF"


# --------------------------------------------------------------------------
# Context passed between pipeline stages
# --------------------------------------------------------------------------

@dataclass
class PipelineContext:
    # --- connection / user options (set from CLI args) ---
    host: str = "http://127.0.0.1:8080/APIV2"
    access_key: str = ""
    ref_frame_choice: str = "first"          # first | middle | last
    auto_crop: bool = True
    detector_name: str = "orb"               # orb | sift
    source_format: str = "tif"               # tif | png | jpg -- format for the disposable render pulls (snapshot fallback only)
    output_preset_name: str = DEFAULT_OUTPUT_PRESET_NAME  # name of a Construct Output to render through (for higher bit depth / DPX etc.)
    keep_temp: bool = False
    force_keep_temp: bool = False            # set by RenderStage if something looks wrong
    work_dir: str = ""                       # if empty, a temp dir is created (source frames only)
    output_root: str = ""                    # permanent home for stabilized output (never deleted)

    # --- resolved at runtime ---
    shot: Optional[object] = None
    slot_idx: int = -1
    construct_uuid: str = ""
    frame_in: int = 0
    frame_out: int = 0
    frame_count: int = 0

    render_dir: str = ""                     # only set/used by the snapshot fallback path
    stab_dir: str = ""
    frame_numbers: list = field(default_factory=list)  # absolute source frame numbers, in order
    rendered_frames: list = field(default_factory=list)  # [(frame_number, file_path), ...] in order -- what StabilizeStage actually reads

    new_shot_uuid: str = ""

    client: Optional[object] = None
    app_api: Optional[object] = None
    proj_api: Optional[object] = None


class StageError(Exception):
    """Raised by a stage to abort the pipeline with a clear message."""


# --------------------------------------------------------------------------
# Stage 1 - Metadata: find the selected shot and its frame range
# --------------------------------------------------------------------------

class MetadataStage:
    name = "Metadata"

    def run(self, ctx: PipelineContext):
        try:
            sel = ctx.proj_api.get_construct_current_selected_shots(level="ALL")
        except Exception as e:
            raise StageError(
                f"Couldn't reach SCRATCH at {ctx.host}. Is SCRATCH running with a "
                f"project/timeline open and the REST API turned on? Details: {e}"
            )

        selection = getattr(sel, "selection", None) or []
        if not selection:
            raise StageError(
                "No Shot is selected in SCRATCH's player. Click a shot in the "
                "timeline to select it, then run this again."
            )

        picked = selection[0]
        shot = picked.shot
        if shot is None or not getattr(shot, "uuid", None):
            raise StageError("SCRATCH returned a selection with no Shot data attached.")

        ctx.shot = shot
        ctx.slot_idx = picked.slot_idx
        ctx.construct_uuid = sel.construct_uuid

        # NOTE on frame counting: handles.frame_in/frame_out (both
        # INCLUSIVE -- confirmed live) are the TRIMMED in/out points of
        # THIS specific shot instance -- e.g. frame_in=5, frame_out=32 for
        # a shot that's been trimmed to start 5 frames into its source and
        # end at frame 32. shot.length, it turns out, does NOT track that
        # trim -- on a shot with custom in/out points set, shot.length was
        # confirmed live to report a much larger number (the underlying
        # source media's full length) while handles correctly reported the
        # actual trimmed range. An earlier version of this script preferred
        # shot.length, which worked by coincidence on the first two test
        # shots (both untrimmed, using their full source length, so the
        # two numbers happened to match) and only broke once a genuinely
        # trimmed shot was tested. handles is now the primary source, with
        # shot.length only as a last-resort fallback if handles is missing
        # entirely (meaning "use the whole clip"). ctx.frame_out is kept as
        # an EXCLUSIVE value internally (frame_in + frame_count) for
        # consistency with Python range()/len() math used elsewhere in
        # this script -- see render_via_output_preset for where it gets
        # converted back to the render queue's own INCLUSIVE convention.
        handles = getattr(shot, "handles", None)
        frame_in = getattr(handles, "frame_in", None) if handles else None
        handles_frame_out = getattr(handles, "frame_out", None) if handles else None
        if frame_in is None:
            frame_in = 0

        if handles_frame_out is not None:
            frame_count = handles_frame_out - frame_in + 1
        elif shot.length:
            frame_count = shot.length
        else:
            frame_count = 1

        ctx.frame_in = frame_in
        ctx.frame_out = frame_in + frame_count  # exclusive, kept internally consistent
        ctx.frame_count = frame_count

        if ctx.frame_count <= 1:
            raise StageError(
                f"Shot '{shot.name}' only has {ctx.frame_count} frame(s) in its "
                "trimmed range -- nothing to stabilize."
            )

        print(f"Selected shot: {shot.name}  (frames {frame_in}-{ctx.frame_out - 1}, "
              f"{ctx.frame_count} frames, shot.length={shot.length}, "
              f"handles.frame_out={handles_frame_out})")


# --------------------------------------------------------------------------
# Stage 2 - Render: pull every frame of the shot to disk
#
# Two paths, in order of preference:
#
#   A) Render-queue path (find_output_preset / render_via_output_preset):
#      reuses a real Output/render preset the user configured in SCRATCH's
#      own Outputs panel (whatever bit depth / colorspace / file type it was
#      set to -- 16-bit TIFF, DPX, float EXR, etc.) via SCRATCH's actual
#      render queue. This is the same mechanism Assimilate's own
#      examples/RenderSelectedShot.py uses: build a temporary "render node"
#      Shot that copies the preset's output/color_format, point it at the
#      selected Shot as input 0, submit it to the render queue, poll, collect
#      the results, then delete the temporary render node. Both helper
#      functions below are written to be reusable outside this script for
#      any other tool that needs "render a shot through an existing Output
#      preset via the API."
#
#   B) Snapshot fallback (used if no matching preset is found, or the queue
#      path errors out): the original per-frame ImageSnapshot loop. Simple
#      and robust, but this specific API call has no bit-depth control at
#      all -- it's effectively always 8-bit regardless of file extension.
# --------------------------------------------------------------------------

# A generic "image sequence" render node type -- reused verbatim from
# Assimilate's own examples/RenderSelectedShot.py (labelled "CC Render Tiff"
# there). The actual file type / bit depth / colorspace of what gets
# rendered comes entirely from whatever ShotDataOutput / ShotDataColorFormat
# we attach to the render node -- this type_uuid just selects "image
# sequence output" as opposed to e.g. a movie-container render node. Not
# independently confirmed against SCRATCH's own docs beyond that one example,
# so if render_via_output_preset ever fails outright, this is the first
# thing worth double-checking.
IMAGE_SEQUENCE_RENDER_NODE_TYPE_UUID = "00000000-0000-0000-0000-000000000004"


def find_output_preset(proj_api, name):
    """Look up a Construct Output (as configured in SCRATCH's own Outputs
    panel) by name, case-insensitively. Returns the full ShotData (with
    .output / .color_format populated) or None if no Output with that name
    exists in the current Construct, or the lookup itself fails for any
    reason (e.g. REST API down) -- callers should treat None as "fall back
    to something simpler," not as a hard error.
    """
    try:
        outputs = proj_api.get_construct_current_outputs(level="ALL")
    except Exception:
        return None
    candidates = getattr(outputs, "shots", None) or []
    target = name.strip().lower()
    for o in candidates:
        if (getattr(o, "name", "") or "").strip().lower() == target:
            try:
                # Re-fetch by uuid at level=ALL to be sure .output/.color_format
                # are fully populated (the list call above may return a
                # trimmed-down representation even with level="ALL").
                return proj_api.get_construct_current_output(o.uuid, level="ALL")
            except Exception:
                return o
    return None


def _run_render_queue_job(app_api, render_node_uuid, range_in, range_out, expected, log):
    """Submit one render-queue job for render_node_uuid over [range_in,
    range_out) (range_out exclusive -- confirmed empirically), poll it to
    completion, and return the sorted list of (frame_number, file_path)
    tuples from its results. Raises RuntimeError if the job's terminal
    status is 'error' (one of only 5 documented statuses -- a genuine hard
    failure, not just "fewer frames than hoped").
    """
    rqs = assimilate_client.RenderQueueSettings(
        output_uuid=render_node_uuid,
        range_in=range_in,
        range_out=range_out,
        delete_existing_media=True,
    )
    queue_item = app_api.new_application_render_queue_item_start(body=rqs)
    pbar = tqdm(total=expected, desc="Render", unit="frame")
    last_done = 0
    # Matches the exact poll loop from Assimilate's own
    # RenderSelectedShot.py -- these are the only "still going" statuses
    # confirmed from that example; anything else means it stopped.
    while True:
        queue_item = app_api.get_application_render_queue_item(queue_item.uuid)
        done = queue_item.frames_done or 0
        if done > last_done:
            pbar.update(done - last_done)
            last_done = done
        if queue_item.status in ("Idle", "waiting", "processing"):
            time.sleep(1)
        else:
            break
    pbar.close()
    log(f"Render queue item finished with status: {queue_item.status}")
    if queue_item.status == "error":
        raise RuntimeError(
            f"SCRATCH's render queue reported status 'error' for this job "
            f"(queue item {queue_item.uuid}, range {range_in}-{range_out})."
        )
    # HYPOTHESIS being tested: the "frozen tail frames" pattern (confirmed
    # live across several shots, always affecting roughly the LAST 15-25%
    # of whatever range was requested, scaling with total frame count
    # rather than sitting at a fixed frame number) looks like it could be
    # a write-lag race -- the queue status/frames_done may report
    # "finished" slightly before the actual file writes for the last
    # several frames land on disk (especially over a network render path),
    # so reading them immediately catches stale/incomplete content. A
    # short settle delay here is cheap to test; if it doesn't help, the
    # per-frame patch-via-snapshot mechanism still catches and fixes
    # whatever's left regardless of the root cause.
    time.sleep(1.5)
    results = app_api.get_application_render_queue_item_results(queue_item.uuid)
    files = sorted(
        (f for f in (results.files or []) if f.file),
        key=lambda f: f.frame if f.frame is not None else 0,
    )
    return [(f.frame, f.file) for f in files]


def _find_duplicate_run_frame_numbers(files):
    """Given a sorted [(frame_number, path), ...] list, returns the set of
    frame numbers that are pixel-identical to the frame immediately before
    them -- i.e. frames that look "stuck" repeating the prior frame's
    content instead of showing genuinely new footage. Used to catch
    SCRATCH's render queue failing to seek correctly for several frames
    near a range boundary (confirmed live, and cross-checked directly
    against ImageSnapshot to rule out the source media itself being
    static). The FIRST frame in a duplicate run is assumed to be the last
    one that genuinely advanced; every frame after it in that run is
    flagged as needing a patch.
    """
    dupes = set()
    prev_img = None
    for num, path in files:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            prev_img = None
            continue
        if prev_img is not None and img.shape == prev_img.shape:
            if cv2.absdiff(img, prev_img).mean() < 0.5:
                dupes.add(num)
        prev_img = img
    return dupes


def _patch_one_frame_via_snapshot(app_api, source_shot_uuid, frame_number, sibling_file_path,
                                   target_dtype, log, preceding_file_path=None):
    """Last-resort patch for the one frame SCRATCH's render queue won't
    produce through a copied-input render node (see render_via_output_preset
    for what's been ruled out). Renders that single frame the plain 8-bit
    ImageSnapshot way, then upconverts it to match the bit depth of the
    frames that DID come through the preset, so the sequence stays
    format-consistent for the rest of the pipeline. Returns the file path
    written, or None if even this fails.

    If preceding_file_path is given (the real rendered frame immediately
    before this gap), this also diffs the patched frame against it and
    logs whether they're suspiciously near-identical -- if SCRATCH can't
    produce this frame through EITHER the render queue OR a plain
    snapshot, and the snapshot comes back pixel-identical to the frame
    before it, that points to the underlying media genuinely not having
    distinct content at this frame number (e.g. the shot's out point sits
    one frame past the last real frame of source data) rather than a
    render-queue-specific bug.
    """
    ext = os.path.splitext(sibling_file_path)[1] or ".tif"
    patch_path = os.path.join(
        os.path.dirname(sibling_file_path), f"_stabilizer_patch_{frame_number:07d}{ext}"
    )
    snapshot = assimilate_client.ImageSnapshot(
        uuid=source_shot_uuid, frame=frame_number, proxy=False, file=patch_path,
    )
    try:
        app_api.do_application_render_snapshot(snapshot)
    except ApiException as e:
        log(f"  ! Couldn't patch in the missing frame {frame_number} either: {e}")
        return None
    # Small settle delay -- the same reasoning as the snapshot fallback
    # loop elsewhere in this script: checking the file immediately after
    # the API call returns can race a not-yet-flushed write, especially
    # over a network render path.
    time.sleep(0.1)
    if not os.path.exists(patch_path):
        log(f"  ! Patch snapshot for frame {frame_number} reported success "
            f"but no file showed up at {patch_path}.")
        return None
    img = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        log(f"  ! Patch file for frame {frame_number} exists but couldn't "
            f"be read as an image: {patch_path}")
        return None

    if preceding_file_path and os.path.exists(preceding_file_path):
        prev = cv2.imread(preceding_file_path, cv2.IMREAD_UNCHANGED)
        if prev is not None and prev.shape == img.shape:
            # Compare on a common 8-bit footing regardless of either
            # frame's actual dtype (the patch may still be raw 8-bit here,
            # upconversion happens below).
            prev8 = prev if prev.dtype == np.uint8 else (prev >> 8).astype(np.uint8) if prev.dtype == np.uint16 else None
            img8 = img if img.dtype == np.uint8 else (img >> 8).astype(np.uint8) if img.dtype == np.uint16 else None
            if prev8 is not None and img8 is not None:
                diff = cv2.absdiff(img8, prev8).mean()
                if diff < 0.5:
                    log(f"  Note: the patched frame {frame_number} is pixel-identical "
                        "to the frame right before it -- this looks like the "
                        "underlying media genuinely has no distinct content at "
                        "this frame number (e.g. the shot's out point sits one "
                        "frame past the real end of the source), not a "
                        "render-queue-specific bug. If you're seeing what look "
                        "like duplicate frames at the tail of the stabilized "
                        "result, this is almost certainly it.")
                else:
                    log(f"  Patched frame {frame_number} differs from its "
                        f"neighbor (avg pixel diff {diff:.1f}) -- looks like "
                        "genuinely distinct content, not a duplicate.")

    if target_dtype is not None and img.dtype != target_dtype:
        if img.dtype == np.uint8 and target_dtype == np.uint16:
            # Replicate the 8-bit value into both bytes (value * 257) rather
            # than a plain <<8 shift, so it spans the full 16-bit range
            # instead of always landing on a multiple of 256.
            img = (img.astype(np.uint16) * 257)
            cv2.imwrite(patch_path, img)
        else:
            log(f"  ! Patched frame {frame_number} is {img.dtype}, expected "
                f"{target_dtype} -- leaving as-is, this one frame may look "
                "slightly different from its neighbors.")
    return patch_path


def render_via_output_preset(app_api, proj_api, source_shot_uuid, preset, frame_in, frame_out, log=print):
    """Render frame_in..frame_out (frame_out exclusive) of source_shot_uuid
    through SCRATCH's real render queue, using preset's exact output
    format/color settings (as found by find_output_preset). Returns a list
    of (frame_number, file_path) tuples, sorted by frame number. Raises on
    total failure -- callers should catch and fall back to a simpler render
    path if desired.
    """
    render_node_data = assimilate_client.ShotData(
        type_uuid=IMAGE_SEQUENCE_RENDER_NODE_TYPE_UUID,
        name=f"_stabilizer_render_{int(time.time() * 1000) % 1000000}",
        output=preset.output,
        color_format=preset.color_format,
    )
    log(f"Creating a temporary render node from Output preset '{preset.name}'...")
    render_node = proj_api.add_shot(body=render_node_data)
    try:
        # Explicitly pass length here -- across multiple rounds of live
        # testing, leaving it unset (create_copy=True + input_uuid only)
        # consistently produced a copied input one frame shorter than the
        # source shot's real length. InputData.length is documented as
        # "Length of the Input Shot in frames," so pass the real count
        # directly instead of letting the copy infer it.
        inp = assimilate_client.InputData(
            create_copy=True,
            input_uuid=source_shot_uuid,
            length=frame_out - frame_in,
        )
        proj_api.set_shot_input(inp, render_node.uuid, 0)

        # RenderQueueSettings.range_out is INCLUSIVE (same convention as
        # shot.handles.frame_out) -- confirmed by reading SCRATCH's own
        # Render Queue panel directly. frame_out here is this script's
        # internal EXCLUSIVE value (frame_in + frame_count), so frame_out-1
        # is the real inclusive last-frame index to request.
        #
        # KNOWN, PERMANENT PLATFORM BEHAVIOR (not a bug in this script):
        # even with that exact correct in-bounds range, SCRATCH's render
        # queue reliably fails to produce the true last frame of a copied
        # render-node input -- confirmed across many rounds of live testing
        # on two different shots, with every reasonable range formulation
        # (inclusive, exclusive, one past the end, no range at all). It
        # always comes up exactly one frame short, at the tail, with a
        # clean "finished" status (no error) -- so this isn't something to
        # keep chasing with different parameters. The single-frame gap it
        # always leaves gets detected and patched in below every time; that
        # "Filling in the one frame..." log line is expected on every run
        # through this path, not a warning sign.
        expected = frame_out - frame_in
        try:
            files = _run_render_queue_job(
                app_api, render_node.uuid, frame_in, frame_out - 1, expected, log
            )
        except RuntimeError:
            # Extra defensive layer in case a particular shot/preset combo
            # errors outright instead of the usual clean "finished, one
            # frame short" -- retry one frame shorter rather than losing
            # the whole render. Any single-frame gap this leaves also gets
            # patched in below.
            log("  ! Render errored even with the expected-good range -- "
                "retrying one frame shorter as a fallback.")
            files = _run_render_queue_job(
                app_api, render_node.uuid, frame_in, frame_out - 2, expected - 1, log
            )

        # The render queue's results report frame numbers LOCAL to the
        # copied render-node input (starting near 0), NOT the original
        # shot's absolute frame numbers -- confirmed live on a shot trimmed
        # to frames 5-32: the results came back numbered from 0, not 5,
        # which made every real frame look like it was at the wrong number
        # and triggered a false "6 frames missing" (really just the usual
        # single dropped tail frame, miscounted because of this). Re-base
        # to this script's absolute numbering so gap detection, the patch
        # step, and downstream file naming all agree with the rest of the
        # pipeline (which uses absolute frame numbers throughout, e.g. the
        # 8-bit snapshot fallback via ImageSnapshot's own frame= param).
        reported = [f for f, _ in files if f is not None]
        if reported:
            offset = frame_in - min(reported)
            if offset != 0:
                files = [(f + offset if f is not None else f, p) for f, p in files]

        got_numbers = {f for f, _ in files}
        expected_numbers = set(range(frame_in, frame_out))
        missing = sorted(expected_numbers - got_numbers)

        # Beyond just dropping the true last frame, SCRATCH's render queue
        # can also fail to SEEK correctly for several frames right before
        # that boundary -- confirmed live (and cross-checked directly
        # against ImageSnapshot to rule out the source media itself):
        # several consecutive "rendered" frames near the tail came back
        # pixel-identical to each other even though the real footage has
        # genuine motion there. _find_duplicate_run_frame_numbers flags
        # every frame after the point a run gets "stuck" (the first frame
        # in a run is assumed to be the last one that genuinely advanced).
        duplicate_numbers = _find_duplicate_run_frame_numbers(files) if files else set()

        bad_frames = sorted(set(missing) | duplicate_numbers)
        if bad_frames and files:
            sample = cv2.imread(files[0][1], cv2.IMREAD_UNCHANGED)
            target_dtype = sample.dtype if sample is not None else None
            file_map = dict(files)
            log(f"  {len(bad_frames)} frame(s) need patching via the 8-bit "
                f"snapshot method -- missing: {missing or 'none'}, "
                f"frozen/duplicate: {sorted(duplicate_numbers) or 'none'}. "
                "Patching each individually...")
            patched = 0
            for bad in bad_frames:
                preceding_path = file_map.get(bad - 1)
                patch_path = _patch_one_frame_via_snapshot(
                    app_api, source_shot_uuid, bad, files[0][1], target_dtype, log,
                    preceding_file_path=preceding_path,
                )
                if patch_path:
                    file_map[bad] = patch_path  # replaces a frozen duplicate, or fills a gap
                    patched += 1
            files = sorted(file_map.items(), key=lambda t: t[0])
            log(f"  Patched {patched} of {len(bad_frames)} flagged frame(s) "
                "via the 8-bit path -- these don't carry the same real "
                "precision as the rest, but for a handful of frames within "
                "a longer sequence this is very unlikely to be visible.")
        elif bad_frames:
            log(f"  ! {len(bad_frames)} frame(s) flagged (missing or "
                f"frozen) but no frames came back at all to patch from: "
                f"{bad_frames}")

        return files
    finally:
        try:
            proj_api.delete_shot(render_node.uuid)
        except Exception:
            pass


class RenderStage:
    name = "Render"

    def run(self, ctx: PipelineContext):
        # frame_out is exclusive (see the note in MetadataStage.run()), so
        # this range already lands on exactly ctx.frame_count numbers.
        ctx.frame_numbers = list(range(ctx.frame_in, ctx.frame_out))

        if ctx.output_preset_name:
            preset = find_output_preset(ctx.proj_api, ctx.output_preset_name)
            if preset is None:
                print(f"  ! Output preset '{ctx.output_preset_name}' not found in the "
                      "current Construct -- falling back to the 8-bit snapshot render.")
            else:
                try:
                    ctx.rendered_frames = render_via_output_preset(
                        ctx.app_api, ctx.proj_api, ctx.shot.uuid, preset,
                        ctx.frame_in, ctx.frame_out, log=print,
                    )
                except Exception as e:
                    print(f"  ! Render-queue path failed ({e}) -- falling back "
                          "to the 8-bit snapshot render.")
                    ctx.rendered_frames = []

        if not ctx.rendered_frames:
            self._render_via_snapshots(ctx)

        if len(ctx.rendered_frames) < 2:
            raise StageError("Not enough frames were rendered to stabilize.")

        self._check_frame_count(ctx)
        self._sanity_check_distinct_frames(ctx)
        self._report_actual_bit_depth(ctx)

    def _check_frame_count(self, ctx: PipelineContext):
        # Catches range-off-by-one issues (or anything else that silently
        # drops frames) with a clear message instead of a shot that plays
        # back one or more frames shorter than the original.
        got = len(ctx.rendered_frames)
        if got == ctx.frame_count:
            return
        missing = ctx.frame_count - got
        rendered_numbers = {n for n, _ in ctx.rendered_frames}
        expected_numbers = set(ctx.frame_numbers)
        gaps = sorted(expected_numbers - rendered_numbers)
        gap_txt = f" (frame(s) {gaps})" if gaps and len(gaps) <= 10 else ""
        print(f"  ! WARNING: expected {ctx.frame_count} frame(s) but only got "
              f"{got} -- {missing} frame(s) missing{gap_txt}. The stabilized "
              "result will be that many frames shorter than the original.")
        ctx.force_keep_temp = True

    def _render_via_snapshots(self, ctx: PipelineContext):
        ctx.render_dir = os.path.join(ctx.work_dir, "source_frames")
        os.makedirs(ctx.render_dir, exist_ok=True)

        ext = ctx.source_format
        print(f"Rendering {ctx.frame_count} frame(s) from SCRATCH as {ext.upper()} "
              "(8-bit -- this API call has no bit-depth control)...")
        failures = 0
        for frame in tqdm(ctx.frame_numbers, desc="Render", unit="frame"):
            out_path = os.path.join(ctx.render_dir, f"src_{frame:07d}.{ext}")
            if not os.path.exists(out_path):
                snapshot = assimilate_client.ImageSnapshot(
                    uuid=ctx.shot.uuid,
                    frame=frame,
                    proxy=False,
                    file=out_path,
                )
                try:
                    ctx.app_api.do_application_render_snapshot(snapshot)
                except ApiException as e:
                    failures += 1
                    print(f"  ! frame {frame}: render failed -- {e}")
                # A small settle delay between snapshot requests. Without
                # this, rapid back-to-back calls against the same shot can
                # outrun SCRATCH's player seek, and every "different frame"
                # comes back showing the same content -- see the duplicate
                # check below.
                time.sleep(0.05)
            if os.path.exists(out_path):
                ctx.rendered_frames.append((frame, out_path))

        if failures:
            print(f"  ({failures} frame(s) failed to render and will be skipped)")

    def _sanity_check_distinct_frames(self, ctx: PipelineContext):
        # Check EVERY consecutive pair for near-duplicate content, not just
        # first-vs-middle -- a stuck/repeated seek can cluster anywhere in
        # the sequence (confirmed live: several frames right before the
        # tail of a render-queue result came back "frozen"), and a single
        # first/middle comparison would miss that entirely.
        duplicate_pairs = []
        prev_img = None
        prev_num = None
        for num, path in ctx.rendered_frames:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                prev_img, prev_num = None, num
                continue
            if prev_img is not None and img.shape == prev_img.shape:
                diff = cv2.absdiff(img, prev_img).mean()
                if diff < 0.5:
                    duplicate_pairs.append((prev_num, num))
            prev_img, prev_num = img, num

        if duplicate_pairs:
            pair_txt = ", ".join(f"{a}->{b}" for a, b in duplicate_pairs[:10])
            more = f" (+{len(duplicate_pairs) - 10} more)" if len(duplicate_pairs) > 10 else ""
            print(f"  ! WARNING: {len(duplicate_pairs)} pair(s) of consecutive "
                  f"rendered frames are pixel-identical: {pair_txt}{more}. "
                  "These will look frozen/stuck in the stabilized result. "
                  "This points to SCRATCH's render not actually "
                  "seeking/advancing between some of the requested frames "
                  "(seen before near a render-queue range boundary), "
                  "rather than anything in the stabilization step.")
            ctx.force_keep_temp = True
            self._cross_check_duplicate_via_snapshot(ctx, duplicate_pairs[0])

    def _cross_check_duplicate_via_snapshot(self, ctx: PipelineContext, pair):
        # Automatically answers the question "is the source media itself
        # static here, or is this specific to whatever render path
        # produced ctx.rendered_frames?" -- fetches both frame numbers of
        # one duplicate pair directly via ImageSnapshot (bypassing the
        # render queue / copied render-node input entirely) and compares
        # them. If they're ALSO identical, the source media genuinely has
        # no distinct content in this range. If they're different, the
        # render path that produced ctx.rendered_frames is failing to seek
        # correctly -- not a source-media problem.
        a, b = pair
        print(f"  Cross-checking frames {a} and {b} directly via "
              "ImageSnapshot (bypassing whichever render path produced "
              "the frames above) to tell source-media vs render-path...")
        tmp_dir = tempfile.mkdtemp(prefix="stabilizer_xcheck_")
        try:
            paths = {}
            for num in (a, b):
                p = os.path.join(tmp_dir, f"xcheck_{num:07d}.tif")
                snap = assimilate_client.ImageSnapshot(
                    uuid=ctx.shot.uuid, frame=num, proxy=False, file=p
                )
                try:
                    ctx.app_api.do_application_render_snapshot(snap)
                    time.sleep(0.1)
                    if os.path.exists(p):
                        paths[num] = p
                except ApiException as e:
                    print(f"  ! Cross-check snapshot for frame {num} failed: {e}")
            if a in paths and b in paths:
                img_a = cv2.imread(paths[a], cv2.IMREAD_UNCHANGED)
                img_b = cv2.imread(paths[b], cv2.IMREAD_UNCHANGED)
                if img_a is not None and img_b is not None and img_a.shape == img_b.shape:
                    diff = cv2.absdiff(img_a, img_b).mean()
                    if diff < 0.5:
                        print(f"  -> Frames {a} and {b} are ALSO identical via "
                              "direct ImageSnapshot -- the source media "
                              "itself genuinely has no distinct content in "
                              "this range (e.g. a held/frozen moment, or the "
                              "trim extending past real footage). Not "
                              "something this script can render around.")
                    else:
                        print(f"  -> Frames {a} and {b} are DIFFERENT via "
                              f"direct ImageSnapshot (avg diff {diff:.1f}) -- "
                              "the source media has real distinct content "
                              "here. SCRATCH's render queue is failing to "
                              "seek correctly in this range -- a "
                              "render-queue-specific issue, not your footage.")
                else:
                    print("  ! Cross-check images couldn't be compared "
                          "(missing or mismatched shape).")
            else:
                print("  ! Cross-check couldn't fetch both frames to compare.")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _report_actual_bit_depth(self, ctx: PipelineContext):
        sample = cv2.imread(ctx.rendered_frames[0][1], cv2.IMREAD_UNCHANGED)
        if sample is None:
            return
        depth_bits = sample.dtype.itemsize * 8
        print(f"Rendered frames are {depth_bits}-bit ({sample.dtype}). "
              f"{'Higher than 8-bit confirmed.' if depth_bits > 8 else ''}")


# --------------------------------------------------------------------------
# Stage 3 - Stabilize: run stabilo across the rendered sequence
# --------------------------------------------------------------------------

def _largest_inscribed_rect(mask: np.ndarray):
    """Largest all-255 axis-aligned rectangle inside a uint8 0/255 mask.
    Classic 'maximal rectangle in binary matrix' via per-row histograms.
    Returns (x, y, w, h) in pixel coordinates.
    """
    h, w = mask.shape
    binary = (mask > 0).astype(np.int32)
    heights = np.zeros(w, dtype=np.int32)
    best = (0, 0, 0, 0)
    best_area = 0

    for y in range(h):
        row = binary[y]
        heights = np.where(row > 0, heights + 1, 0)

        stack = []  # (start_x, height)
        x = 0
        while x <= w:
            cur_h = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur_h:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best_area:
                    best_area = area
                    best = (sx, y - sh + 1, x - sx, sh)
                start = sx
            stack.append((start, cur_h))
            x += 1

    return best


def _safe_folder_name(name):
    keep = "".join(c if (c.isalnum() or c in " ._-") else "_" for c in (name or "shot"))
    return keep.strip() or "shot"


def _tracking_proxy(frame):
    """Downconvert a frame to 8-bit for feature TRACKING only. stabilo's
    detectors (ORB/SIFT) hard-error on anything but CV_8U -- confirmed
    directly: both raise a cv2.error on 16-bit input instead of silently
    coping. The actual warp is always applied to the original full-precision
    frame via stabilizer.warp_frame(), never to this proxy, so this
    downconversion never touches the pixels that end up in the output.
    """
    if frame is None:
        return None
    if frame.dtype == np.uint8:
        return frame
    if frame.dtype == np.uint16:
        # Top 8 bits carry plenty of structure for feature matching --
        # matches standard "track on proxy, apply to full-res" practice.
        return (frame >> 8).astype(np.uint8)
    # Fallback for any other dtype (float32 EXR-style data, etc.) --
    # normalize to the full 0-255 range.
    f = frame.astype(np.float32)
    f -= f.min()
    maxval = f.max()
    if maxval > 0:
        f = f / maxval * 255.0
    return f.astype(np.uint8)


class StabilizeStage:
    name = "Stabilize"

    def run(self, ctx: PipelineContext):
        # IMPORTANT: this folder is NOT temp/scratch space. SCRATCH's new Shot
        # references these files by path and reads pixels from disk on demand
        # (playback, scrubbing, thumbnails) rather than copying them in --
        # deleting this folder after import breaks the shot. It lives under
        # output_root (next to this script, outside the temp work_dir) and is
        # never touched by CleanupStage.
        # ctx.frame_out is an exclusive bound internally -- use frame_out-1
        # for this label so the folder name reads as an inclusive range
        # (e.g. "0-27" for 28 frames), matching how a human would expect to
        # count it, instead of implying one frame more than it contains.
        folder_name = f"{_safe_folder_name(ctx.shot.name)}_{ctx.frame_in}-{ctx.frame_out - 1}"
        ctx.stab_dir = os.path.join(ctx.output_root, folder_name)
        # Start clean -- if this shot/range was stabilized before, don't let
        # leftover frames from that earlier attempt mix into this one.
        if os.path.isdir(ctx.stab_dir):
            shutil.rmtree(ctx.stab_dir, ignore_errors=True)
        os.makedirs(ctx.stab_dir, exist_ok=True)

        # ctx.rendered_frames was built by RenderStage -- re-verify each file
        # still actually exists now, in case something deleted them between
        # stages (seen before on Windows with Controlled Folder Access).
        frame_files = [(n, p) for n, p in ctx.rendered_frames if os.path.exists(p)]
        if len(frame_files) < len(ctx.rendered_frames):
            missing = len(ctx.rendered_frames) - len(frame_files)
            print(f"  ! WARNING: {missing} of {len(ctx.rendered_frames)} rendered "
                  "frame(s) went missing between the Render and Stabilize steps "
                  "-- something outside this script deleted them while it was "
                  "running. On Windows this is usually Controlled Folder Access "
                  "(ransomware protection) silently reverting new files written "
                  "by an unrecognized app. Check Windows Security -> Virus & "
                  "threat protection -> Manage ransomware protection, and either "
                  "add an exception for this project folder / python.exe / "
                  "uv.exe, or turn it off to test.")
            ctx.force_keep_temp = True
        if len(frame_files) < 2:
            raise StageError("Fewer than 2 frames survived rendering; can't stabilize.")

        if ctx.ref_frame_choice == "middle":
            ref_idx = len(frame_files) // 2
        elif ctx.ref_frame_choice == "last":
            ref_idx = len(frame_files) - 1
        else:
            ref_idx = 0

        ref_frame_full = cv2.imread(frame_files[ref_idx][1], cv2.IMREAD_UNCHANGED)
        if ref_frame_full is None:
            raise StageError(f"Couldn't read reference frame: {frame_files[ref_idx][1]}")
        h, w = ref_frame_full.shape[:2]
        out_ext = os.path.splitext(frame_files[ref_idx][1])[1] or ".tif"

        # We never pass exclusion boxes (no moving-object masking in this
        # tool), so turn masking off -- otherwise stabilo logs a "no bounding
        # boxes were provided" reminder on every single frame.
        stabilizer = Stabilizer(detector_name=ctx.detector_name, mask_use=False)
        # stabilo's feature detectors (ORB/SIFT) only accept 8-bit input --
        # confirmed directly, they raise a hard cv2 error on 16-bit data. So
        # motion is TRACKED on an 8-bit proxy of each frame, but WARPED is
        # always applied to the full-precision original via
        # stabilizer.warp_frame(...) rather than warp_cur_frame() (which
        # would only have the proxy). Standard practice: tracking doesn't
        # need full bit depth, only the final pixels do.
        stabilizer.set_ref_frame(_tracking_proxy(ref_frame_full))

        warped_paths = []
        valid_mask_accum = np.full((h, w), 255, dtype=np.uint8) if ctx.auto_crop else None
        white = np.full((h, w), 255, dtype=np.uint8)

        print(f"Stabilizing against the {ctx.ref_frame_choice} frame "
              f"({ctx.detector_name.upper()} detector)...")
        vanished_mid_loop = 0
        for i, (frame_num, path) in enumerate(tqdm(frame_files, desc="Stabilize", unit="frame")):
            frame_full = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if frame_full is None:
                vanished_mid_loop += 1
                continue
            if i == ref_idx:
                warped = frame_full
            else:
                stabilizer.stabilize(_tracking_proxy(frame_full))
                warped = stabilizer.warp_frame(frame_full)
                if warped is None:
                    warped = frame_full  # transform estimation failed; fall back to the raw frame

            # Output preserves whatever format/bit-depth the source frames
            # came in as (e.g. 16-bit TIFF stays 16-bit TIFF all the way
            # through) -- no forced conversion to PNG anymore.
            out_path = os.path.join(ctx.stab_dir, f"stab_{frame_num:07d}{out_ext}")
            cv2.imwrite(out_path, warped)
            warped_paths.append(out_path)

            if ctx.auto_crop and i != ref_idx:
                valid = stabilizer.warp_frame(white)
                if valid is not None:
                    valid_mask_accum = cv2.bitwise_and(valid_mask_accum, valid)

        if ctx.auto_crop:
            print("Computing crop to remove wobble borders...")
            x, y, cw, ch = _largest_inscribed_rect(valid_mask_accum)
            # Leave a hair of margin and guard against a degenerate crop.
            if cw > w * 0.3 and ch > h * 0.3:
                margin_x, margin_y = int(cw * 0.01), int(ch * 0.01)
                x = min(x + margin_x, w - 1)
                y = min(y + margin_y, h - 1)
                cw = max(cw - 2 * margin_x, 1)
                ch = max(ch - 2 * margin_y, 1)
                print(f"  Crop region: {cw}x{ch} at ({x},{y}) -- scaling back up to {w}x{h}")
                for p in tqdm(warped_paths, desc="Crop", unit="frame"):
                    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    cropped = img[y:y + ch, x:x + cw]
                    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(p, resized)
            else:
                print("  Skipped auto-crop -- the stabilized motion was too large to "
                      "find a sane common crop region. Output keeps the black borders.")

        if vanished_mid_loop:
            print(f"  ! WARNING: {vanished_mid_loop} source frame(s) couldn't be "
                  "read (they vanished between being written and being used, "
                  "mid-run). Same likely cause as above -- see the Controlled "
                  "Folder Access note.")
            ctx.force_keep_temp = True

        print(f"Wrote {len(warped_paths)} stabilized frame(s) to {ctx.stab_dir}")


# --------------------------------------------------------------------------
# Stage 4 - Import: create a new Shot from the stabilized sequence and
#           place it as a new version in the same Slot.
# --------------------------------------------------------------------------

class ImportStage:
    name = "Import"

    def run(self, ctx: PipelineContext):
        # Output extension now follows whatever the source frames were
        # (.tif, .png, .jpg, ...) instead of always being .png -- match
        # anything StabilizeStage could plausibly have written rather than
        # hardcoding one extension.
        image_exts = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".exr", ".dpx")
        stab_files = sorted(
            f for f in os.listdir(ctx.stab_dir)
            if f.lower().endswith(image_exts)
        )
        if not stab_files:
            raise StageError("No stabilized frames were produced.")
        if len(stab_files) == 1 and ctx.frame_count > 1:
            raise StageError(
                f"Only 1 of {ctx.frame_count} expected frame(s) survived to this "
                "point -- SCRATCH would import this as a single still image, not "
                "a moving sequence, so stopping here instead of creating a "
                "frozen shot. This means frames were lost earlier in the run "
                "(see the WARNING lines above) rather than a bug in this final "
                "step."
            )
        print(f"Handing {len(stab_files)} stabilized frame(s) to SCRATCH, "
              f"starting from: {stab_files[0]}")

        # SCRATCH auto-detects an image sequence from a single real frame's
        # path -- it scans the folder for sibling files matching that file's
        # own naming pattern. It does NOT accept a '#'-padded wildcard string
        # here (that's only valid inside render/naming *output* patterns, a
        # different part of the API) -- passing one gets rejected with a 409.
        # Confirmed against Assimilate's own examples/RenderSelectedShot.py,
        # which does the same thing with ShotData(file=<one rendered file>).
        first_file = os.path.join(ctx.stab_dir, stab_files[0])

        shot_data = assimilate_client.ShotData(
            file=first_file,
            name=f"{ctx.shot.name} (Stabilized)",
            fps=getattr(ctx.shot, "fps", None),
            frame_tc=getattr(ctx.shot, "frame_tc", None),
            notes=[assimilate_client.NoteData(
                note="Created by the stabilo Custom Command bridge."
            )],
        )

        print("Creating the stabilized Shot in SCRATCH...")
        try:
            new_shot = ctx.proj_api.add_shot(body=shot_data)
        except ApiException as e:
            raise StageError(f"SCRATCH rejected the new Shot: {e}")

        ctx.new_shot_uuid = new_shot.uuid

        try:
            existing_versions = ctx.proj_api.get_construct_current_slot_versions(ctx.slot_idx)
            next_version_idx = len(existing_versions.shots or [])
        except Exception:
            next_version_idx = 1  # best-effort fallback

        move_data = assimilate_client.MoveShotData(
            construct_uuid=ctx.construct_uuid,
            slot_idx=ctx.slot_idx,
            version_idx=next_version_idx,
            create_copy=False,
        )
        try:
            ctx.proj_api.move_shot(body=move_data, shot_uuid=new_shot.uuid)
        except ApiException as e:
            raise StageError(
                "The stabilized Shot was created but couldn't be placed into the "
                f"timeline slot automatically ({e}). It's sitting in SCRATCH's "
                "shot database -- drag it into the timeline by hand from there."
            )

        print(f"Done. '{shot_data.name}' added as version {next_version_idx} "
              f"in slot {ctx.slot_idx}.")


# --------------------------------------------------------------------------
# Stage 5 - Cleanup
# --------------------------------------------------------------------------

class CleanupStage:
    name = "Cleanup"

    def run(self, ctx: PipelineContext):
        # Only ever touches the disposable rendered-source-frames folder.
        # ctx.stab_dir (the stabilized output SCRATCH's new Shot points at)
        # lives under ctx.output_root, outside ctx.work_dir, and is never
        # deleted here.
        print(f"The stabilized output stays here permanently: {ctx.stab_dir}")
        if ctx.keep_temp or ctx.force_keep_temp:
            print(f"Keeping the source render temp files too, at: {ctx.work_dir}")
            return
        try:
            shutil.rmtree(ctx.work_dir, ignore_errors=True)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Pipeline orchestrator
# --------------------------------------------------------------------------

class Pipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, ctx: PipelineContext):
        for stage in self.stages:
            print(f"\n=== {stage.name} ===")
            t0 = time.time()
            stage.run(ctx)
            print(f"({stage.name} took {time.time() - t0:.1f}s)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Stabilize the selected SCRATCH shot with stabilo and "
                    "import the result as a new version."
    )
    p.add_argument("--scratch-host", default="http://127.0.0.1:8080/APIV2",
                    help="SCRATCH REST API base URL (default: %(default)s)")
    p.add_argument("--scratch-port", type=int, default=None,
                    help="Shortcut for --scratch-host http://127.0.0.1:PORT/APIV2")
    p.add_argument("--scratch-key", default="",
                    help="Optional REST API access key")

    # Custom Command style short flags (-P1/-P2/-P3). SCRATCH's Custom Command
    # dialog passes a dropdown's selected *index* (0, 1, 2, ...) as the value,
    # and a Yes/No checkbox as the literal text "yes"/"no" -- matching the
    # order of the <input> entries in Stabilize_Shot.acc:
    #   -P1: dropdown "First frame, Middle frame, Last frame"   -> 0/1/2
    #   -P2: checkbox "Auto-crop wobble borders"                -> yes/no
    #   -P3: dropdown "Fast (ORB), Accurate (SIFT, slower)"     -> 0/1
    # Plain word values (first/middle/last, y/n, orb/sift) also work, so the
    # tool is just as usable from a manual command line.
    p.add_argument("-P1", "--ref-frame-choice", dest="ref_frame_choice",
                    default="first",
                    help="Which frame to lock the shot to: 0/first, 1/middle, "
                        "2/last (default: %(default)s)")
    p.add_argument("-P2", "--auto-crop", dest="auto_crop",
                    default="yes",
                    help="Auto-crop the wobble borders and rescale to fit: "
                        "yes/no (default: %(default)s)")
    p.add_argument("-P3", "--detector", dest="detector",
                    default="orb",
                    help="Feature detector: 0/orb (fast), 1/sift (slower, "
                        "more precise) (default: %(default)s)")

    p.add_argument("--output-preset", dest="output_preset",
                    default=DEFAULT_OUTPUT_PRESET_NAME,
                    help="Name of a Construct Output preset (configured in "
                        "SCRATCH's own Outputs panel) to render through, "
                        "using SCRATCH's real render queue instead of the "
                        "8-bit ImageSnapshot call. This is how to get 16-bit "
                        "TIFF (or any other bit depth / colorspace / file "
                        "type the preset was set up with) all the way "
                        "through to the stabilized result -- the stabilized "
                        "output preserves whatever format the preset "
                        "rendered. Defaults to looking for a preset named "
                        "'%(default)s' -- create one with that exact name "
                        "(case-insensitive) in SCRATCH's Outputs panel and "
                        "it's picked up automatically, no flag needed. If "
                        "no preset with this name is found, or you pass "
                        "--output-preset \"\" explicitly, falls back to the "
                        "8-bit --source-format snapshot path below.")
    p.add_argument("--source-format", choices=["tif", "png", "jpg"], default="tif",
                    help="Format for the disposable rendered source frames "
                        "pulled via the 8-bit ImageSnapshot fallback (only "
                        "used when --output-preset is empty or not found). "
                        "The stabilized output preserves this same format. "
                        "'tif' is the default: lossless like png, but skips "
                        "png's slow DEFLATE compression, so it renders "
                        "noticeably faster. 'jpg' is fastest but lossy -- "
                        "its compression artifacts get baked into the final "
                        "result permanently. Note the ImageSnapshot API "
                        "call itself has no bit-depth control at all (no "
                        "field for it), so it's always 8-bit regardless of "
                        "which of these three you pick -- for real bit-depth "
                        "control, use --output-preset instead. "
                        "(default: %(default)s)")
    p.add_argument("--keep-temp", action="store_true",
                    help="Don't delete the rendered source frames afterwards "
                        "(the stabilized output is never deleted, regardless "
                        "of this flag)")
    p.add_argument("--work-dir", default="",
                    help="Where to render the disposable source frames "
                        "(default: a temp folder)")
    p.add_argument("--output-dir", default="",
                    help="Permanent folder for the stabilized output that "
                        "SCRATCH's new Shot will reference (default: a "
                        "'StabilizedShots' folder under the current "
                        "project's own configured render path, or next to "
                        "this script if that can't be read)")
    return p


def _normalize_ref_frame_choice(value):
    mapping = {"0": "first", "1": "middle", "2": "last"}
    value = str(value).strip().lower()
    value = mapping.get(value, value)
    if value not in ("first", "middle", "last"):
        raise SystemExit(f"--ref-frame-choice: unrecognized value '{value}'")
    return value


def _normalize_yes_no(value):
    value = str(value).strip().lower()
    if value in ("0", "1"):
        value = "no" if value == "0" else "yes"
    if value in ("yes", "y", "true"):
        return True
    if value in ("no", "n", "false"):
        return False
    raise SystemExit(f"expected yes/no, got '{value}'")


def _normalize_detector(value):
    mapping = {"0": "orb", "1": "sift"}
    value = str(value).strip().lower()
    value = mapping.get(value, value)
    if value not in ("orb", "sift"):
        raise SystemExit(f"--detector: unrecognized value '{value}'")
    return value


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    host = args.scratch_host
    if args.scratch_port:
        host = f"http://127.0.0.1:{args.scratch_port}/APIV2"

    configuration = assimilate_client.Configuration()
    configuration.host = host
    if args.scratch_key:
        configuration.api_key["Authorization"] = args.scratch_key
    client = assimilate_client.ApiClient(configuration)
    app_api = assimilate_client.ApplicationApi(client)
    proj_api = assimilate_client.ProjectsApi(client)

    # Prefer the project's own configured render path over an arbitrary
    # script-adjacent folder -- it's where SCRATCH already expects generated
    # media to live (this is the same $RENDER$ path Assimilate's own
    # examples/RenderSelectedShot.py writes into).
    render_path = args.output_dir
    if not render_path:
        try:
            proj = proj_api.get_projects_current()
            render_path = getattr(getattr(proj, "project_paths", None), "render_path", None) or ""
        except Exception as e:
            print(f"(Couldn't read the project's render path, using a local "
                  f"folder instead -- {e})")
            render_path = ""

    if render_path:
        output_root = os.path.join(render_path, "StabilizedShots")
    else:
        output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "StabilizedShots")

    ctx = PipelineContext(
        host=host,
        access_key=args.scratch_key,
        ref_frame_choice=_normalize_ref_frame_choice(args.ref_frame_choice),
        auto_crop=_normalize_yes_no(args.auto_crop),
        detector_name=_normalize_detector(args.detector),
        source_format=args.source_format,
        output_preset_name=args.output_preset,
        keep_temp=args.keep_temp,
        work_dir=args.work_dir or tempfile.mkdtemp(prefix="scratch_stabilizer_"),
        output_root=output_root,
        client=client,
        app_api=app_api,
        proj_api=proj_api,
    )
    os.makedirs(ctx.work_dir, exist_ok=True)
    os.makedirs(ctx.output_root, exist_ok=True)
    print(f"Stabilized shots will be written under: {ctx.output_root}")

    pipeline = Pipeline([
        MetadataStage(),
        RenderStage(),
        StabilizeStage(),
        ImportStage(),
        CleanupStage(),
    ])

    try:
        pipeline.run(ctx)
    except StageError as e:
        print(f"\nFAILED: {e}")
        return 1
    except Exception:
        print("\nUnexpected error:")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
