# Bug report: Render Queue drops/freezes trailing frames when rendering a copied render-node input

## Summary

When rendering a temporary "render node" Shot through the Application Render Queue — where the render node's input is a **copy** of another Shot (`InputData(create_copy=True, input_uuid=<source shot uuid>)`) — the render reliably fails to produce correct output for a cluster of frames at the **end** of the requested range:

- The single truly last frame of the requested range is **not produced at all** (missing from the results, even though the range passed to `RenderQueueSettings` is a valid, in-bounds, inclusive range).
- A further block of frames immediately before that (several to ~10 frames on longer renders) come back as files, but their pixel content is an exact duplicate of an earlier frame — i.e. the render appears to stop advancing/seeking but keeps writing files.

This happens consistently, is reproducible across different source shots, different trim ranges, and different source codecs (see below), and does **not** happen when the same frames are requested individually via the plain `ImageSnapshot` endpoint (`do_application_render_snapshot`) against the original Shot directly.

## Environment

- Assimilate SCRATCH, REST API v2 (`/APIV2`)
- Python SDK: `assimilate_client` (package `assimilate-client`, installed from `github.com/Assimilate-Inc/Assimilate-REST`), v1.1.0
- Workflow modeled directly on Assimilate's own official example script `examples/RenderSelectedShot.py`

## Reproduction steps

1. Select/identify a Shot in the current Construct (`source_shot_uuid`), with a known trimmed frame range (`shot.handles.frame_in` / `shot.handles.frame_out`, both inclusive — confirmed against `shot.length`).
2. Create a temporary "render node" Shot to hold the output configuration:
   ```python
   render_node_data = ShotData(
       type_uuid="00000000-0000-0000-0000-000000000004",  # image-sequence render node, per RenderSelectedShot.py
       name="<temp name>",
       output=<ShotDataOutput copied from a working Construct Output preset>,
       color_format=<ShotDataColorFormat copied from the same preset>,
   )
   render_node = projects_api.add_shot(body=render_node_data)
   ```
3. Point the render node's input at the source shot, as a copy, with the correct length explicitly supplied:
   ```python
   inp = InputData(create_copy=True, input_uuid=source_shot_uuid, length=frame_count)
   projects_api.set_shot_input(inp, render_node.uuid, 0)
   ```
4. Submit to the render queue for the shot's full trimmed range (inclusive `range_in`/`range_out`, confirmed via the Render Queue panel that these are taken literally/inclusively):
   ```python
   rqs = RenderQueueSettings(
       output_uuid=render_node.uuid,
       range_in=frame_in,
       range_out=frame_out,        # inclusive -- the true last frame index
       delete_existing_media=True,
   )
   queue_item = application_api.new_application_render_queue_item_start(body=rqs)
   ```
5. Poll `get_application_render_queue_item` until `status` leaves `Idle` / `waiting` / `processing`. It reliably reaches `finished` (not `error`) even when the output below is incomplete.
6. Fetch `get_application_render_queue_item_results(queue_item.uuid)` and inspect `.files` (each with a `frame` number and a `file` path).

## Observed behavior

- **The reported `frame` numbers in the results are local to the copied input** (starting near 0), not the original shot's absolute frame numbers. This is a separate, secondary point worth confirming/documenting — it is easy to misinterpret and mistake for a much larger set of "missing" frames than actually exist, since comparing local numbers against the original shot's absolute numbers makes almost the whole range look wrong.
- Rebasing those local numbers back to the shot's absolute numbering (offset = `frame_in - min(reported_frame_numbers)`), the results consistently show:
  - The single true last frame of the range (`frame_out`) is absent from the results entirely — no file, no error.
  - A block of frames immediately before that are present as files, but their image content is byte/pixel-identical to an earlier frame — confirmed via direct pixel-diff comparison (mean `absdiff` ~0, vs. mean diffs of 3–13 for genuinely distinct neighboring frames in the same shots).

## Confirming this is NOT a source-media or numbering issue

- **Numbering**: `shot.handles.frame_in`/`frame_out` are inclusive and self-consistent with `shot.length` (`frame_count = frame_out - frame_in + 1`, confirmed directly, e.g. `frame_in=0, frame_out=14, shot.length=15`). `RenderQueueSettings.range_out` is also inclusive (confirmed by reading the literal range shown in SCRATCH's own Render Queue panel after submission). Ruled out several rounds of off-by-one hypotheses on our own side before concluding the issue is server-side.
- **Source content, not missing/static footage**: for every "frozen" frame flagged, we independently re-fetched that exact frame number via a **direct** `ImageSnapshot` call against the *original* shot (bypassing the render node / copy / render queue entirely) and compared it against the adjacent frame. In every case, the directly-fetched frames were genuinely distinct (pixel diff 3–13), not duplicates — proving the source media has real, distinct frames there. This was also confirmed by the user manually scrubbing the original, untouched clip in the SCRATCH player through the affected frame range.
- **Not a codec/seek-difficulty issue**: reproduced identically on QuickTime-wrapped source media and on DPX (frame-based, uncompressed, trivially seekable) source media.
- **Not a short write-flush race**: adding a 1.5s settle delay between the queue reporting `finished` and reading `get_application_render_queue_item_results` did not reduce the affected-frame count on the larger test renders.

## Pattern across test runs

| Source shot | Requested range (inclusive) | Frames requested | Frame(s) missing entirely | Frame(s) present but pixel-duplicate | Total affected |
|---|---|---|---|---|---|
| A001C042_26050810 (full clip, untrimmed) | 0–14 | 15 | 14 | none observed | 1 |
| A001C065_260508FH (trimmed) | 5–32 | 28 | 32 | 28, 29, 30, 31 | 5 |
| A001C065_260508FH_ (same clip, trim shifted +2) | 7–34 | 28 | 34 | 28–33 | 7 |
| A001C064_260508HT_ (DPX source, trimmed) | 10–69 | 60 | 69 | 60–68 | 10 |
| A001C044_260508QL_ (trimmed) | 10–109 | 100 | 109 | 100–108 | 10 |

Notes on the pattern:
- The count of affected (missing + duplicate) frames does **not** scale as a fixed percentage of the requested range (18%, 25%, 17%, 10% across the above rows) — it looks closer to a **fixed cap around 10 frames** on longer renders, while shorter renders show fewer affected frames, possibly because the render simply hadn't reached that internal limit yet within a smaller total. This looks consistent with a fixed-size internal write buffer/queue that is not fully flushed to the output files by the time the render queue item reports `finished`, rather than a percentage-based or simple timing issue (a settle delay did not help).
- The only test that showed just 1 affected frame (the untrimmed, full-clip case) predates when we started checking for pixel-duplicate frames specifically (we were only checking for entirely-missing frames at that point) — so it's possible duplicate/frozen frames were also present there and simply not checked for. This should not be read as "untrimmed clips are unaffected."

## Workaround currently in use

Any frame number that is either missing from the render queue's results, or pixel-identical to the frame immediately before it, is re-rendered individually via a direct `do_application_render_snapshot` (`ImageSnapshot`) call against the *original* shot (not the copied render-node input), then color-depth-matched to the rest of the sequence. This has been 100% reliable in every test performed (5 separate shots, ranges from 15 to 100 frames), correctly recovering genuinely distinct content for every affected frame. This confirms the direct-snapshot code path does not share whatever limitation affects the render-queue-via-copied-input path.

## Suggested questions for Assimilate engineering

1. Does the render queue write frames to disk through an internal buffer/queue with a fixed depth (looks like it could be around 10 frames) that is not guaranteed to be flushed by the time the render queue item's `status` reports `finished`?
2. Is there a way to force/await a full flush before reading `get_application_render_queue_item_results` (e.g. an additional status value, a flush call, or a documented recommended delay/backoff)?
3. Is it expected that `get_application_render_queue_item_results`' `frame` values are local to the copied input rather than the original shot's absolute frame numbers? If so, is there a documented way to map between the two reliably (we currently infer the offset from `min(reported_frame_numbers)`, which works but isn't based on documented behavior)?
4. Is `create_copy=True` on `InputData` the recommended way to set up a render node's input from an existing Shot for this workflow, or is there a different/better-supported approach that avoids this issue?
