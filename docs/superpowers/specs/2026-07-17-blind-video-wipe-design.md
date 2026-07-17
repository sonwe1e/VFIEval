# Blind Evaluation Large-Format Video Comparison Design

Date: 2026-07-17
Status: Approved interaction design

## Purpose

The participant page currently renders GT, Candidate A, and Candidate B in three equal desktop columns. At the page's 1500 px maximum width, each video is roughly 480 px wide, which makes interpolation artifacts difficult to inspect. The fixed 16:9 media box also wastes space for portrait and non-16:9 material.

The approved design makes direct A/B inspection the default while retaining a simple full-view mode:

- GT is displayed at full width as a separate reference.
- Candidate A and Candidate B share one full-width draggable vertical wipe.
- A three-video vertical view remains available through an explicit view switch.

This is a participant-page presentation change. It does not alter Campaign publication, frozen media, task assignment, voting, anonymity, or objective metrics.

## Goals

- Make local spatial differences between Candidate A and Candidate B visible at the same screen coordinates.
- Give GT and both candidates substantially more display area on wide screens.
- Preserve the existing strict frozen-media identity and opaque participant URLs.
- Support both frozen MP4 and frame-sequence Campaign items.
- Keep one shared playback or frame control surface.
- Improve video synchronization enough that the wipe does not expose avoidable multi-frame drift.
- Support mouse, touch, and keyboard operation with a usable fallback view.

## Non-goals

- No three-way wipe or double-divider interaction.
- No GT-to-candidate pair selector in this iteration.
- No Canvas compositing or new precomposed comparison asset.
- No change to Campaign APIs, database schema, Alignment Plan, package manifest, or media files.
- No change to side randomization, vote semantics, or participant identity handling.
- No objective-metric repair in this change; Campaign-owned metrics are a separate workstream.

## Participant Experience

### Default wipe view

Each task initially opens in wipe view:

1. A full-width GT card appears first.
2. A full-width candidate comparison stage appears below it.
3. Candidate B fills the stage as the bottom layer.
4. Candidate A is the top layer and is clipped at the divider position.
5. The divider starts at 50%, with persistent anonymous labels for Candidate A and Candidate B.
6. The existing vote form remains below the media area.

The divider uses a native `input[type="range"]`. Its value controls a CSS custom property, and the upper candidate layer uses CSS clipping. The range remains keyboard-focusable, exposes an accessible name and value text, and accepts pointer and touch input. No pixels are copied through JavaScript or Canvas.

### Full vertical view

A two-option view switch above the media area exposes:

- `重叠对比`, the default wipe view.
- `完整视图`, a vertical GT, Candidate A, Candidate B stack.

Switching views changes only CSS classes and accessibility state. It reuses the same three media elements, so it does not reload sources, duplicate downloads or decoders, reset playback, or change the anonymous left/right mapping. A participant's choice persists for subsequent tasks in the current page session; a page reload returns to the default wipe view.

### Media sizing

The page no longer forces a 16:9 content box. After metadata loads, the layout derives a shared aspect ratio from `videoWidth/videoHeight` or `naturalWidth/naturalHeight`. It falls back to 16:9 only while dimensions are unavailable. All media continues to use `object-fit: contain` against a black background.

## Component Design

### `blind.html`

Add stable hooks for:

- the two-option view switch;
- wipe divider position;
- a short keyboard/touch instruction;
- a synchronization status message.

The existing playback controls, frame controls, vote form, boot-error handling, and isolated `/blind.js` startup remain intact.

### `blind.js`

Split media construction into small helpers with single responsibilities:

- create one anonymous media node for `reference`, `left`, or `right`;
- assemble the reference card and candidate comparison stage;
- apply wipe or full-view presentation without replacing media nodes;
- update the shared media aspect ratio after metadata loads;
- start and stop synchronization for the current task;
- update playback controls and divider accessibility text.

`blindState` records the current view mode and any active video-frame callback. Task replacement cancels the old callback, pauses the old media, clears the old task DOM, and then installs the new task exactly once.

The participant payload remains the only media source. Labels remain `参考 GT`, `候选 A`, and `候选 B`; no method, Run, model, checkpoint, asset, binding, or task identity is added to the DOM.

### `blind.css`

Replace the desktop three-column grid with a full-width media layout:

- wipe view uses two absolutely overlaid candidate layers and a compositor-friendly CSS clip;
- full view makes the same candidate layers participate in normal vertical flow;
- the divider and anonymous labels remain readable over bright and dark content;
- focus indication is visible;
- touch interaction permits vertical page scrolling while recognizing horizontal divider movement;
- narrow screens retain one-column controls and voting actions.

The existing light and dark color schemes remain supported.

## Playback and Synchronization

The reference video remains the authoritative clock. Native controls are not shown on the three individual videos; the existing shared controls are authoritative.

Synchronization behavior:

- Before play, all peers seek to the reference time and inherit the selected rate and loop setting.
- Play, pause, seek, rate, loop, and ended state are propagated to all three videos.
- While playing, browsers with `requestVideoFrameCallback` compare the reference's presented `mediaTime` with both candidates on each reference frame.
- The drift threshold is based on the task frame count and decoded duration, with a conservative one-frame fallback when FPS cannot be derived.
- A follower outside the threshold is corrected to the reference time. The existing 80 ms threshold is removed.
- Browsers without video-frame callbacks retain a `timeupdate` fallback with the same derived threshold.
- If any stream enters `waiting` during synchronized playback, all three pause and the page reports that playback stopped to preserve alignment. The participant resumes through the shared play button after buffering.
- Task replacement and page completion cancel synchronization callbacks and timers.

This improves practical alignment but does not claim browser-level sample-accurate lock between independent decoders. Paused frame inspection and frame-sequence tasks remain the strictest visual comparison modes.

## Frame-sequence Behavior

Frame-sequence items use the same wipe and full-view structures with three `<img>` elements. The existing frame slider updates all three sources to the same frame index. View switching never changes that index and never creates duplicate image elements.

Mixed video/frame tasks remain blocked with the existing explicit error because they cannot provide reliable synchronized interaction.

## Error Handling and Compatibility

- Missing or failed media keeps the existing participant error path and never exposes a storage path.
- A play rejection leaves all videos paused and reports the failure through the existing visible error panel.
- Invalid or unavailable natural dimensions use the 16:9 fallback instead of breaking layout.
- CSS clipping is the primary path. If clipping support is absent, the page automatically selects full vertical view and disables the wipe option.
- Side swapping continues to happen only in the server payload and media routes. View switching and clipping never reorder method identity.
- Existing HTTP Range streaming and frozen package immutability are unchanged.

## Testing

### Automated tests

- Verify every `byId` dependency introduced in `blind.js` exists in `blind.html`.
- Verify the default view is wipe and the explicit full-view control is present.
- Verify view switching reuses exactly three media nodes and does not rewrite their `src` values.
- Verify frame-sequence updates affect reference, left, and right at one shared index.
- Verify divider updates the clip position and accessible value text.
- Verify synchronization propagates play, pause, seek, rate, loop, and ended state.
- Verify `requestVideoFrameCallback` is feature-detected and `timeupdate` remains a fallback.
- Verify task replacement cancels an active frame callback and pauses old media.
- Verify mixed media kinds still hide voting and show the existing error.
- Verify participant code contains only opaque `/api/blind/` URLs and anonymous labels.
- Keep the existing participant HTTP isolation and byte-range tests passing.

### Browser checks

- Exercise wipe dragging by mouse, touch, left/right arrow keys, Home, and End.
- Switch between wipe and full view during play and while paused; confirm time and sources do not reset.
- Check wide, narrow, portrait-video, landscape-video, and unusual-aspect-ratio layouts.
- Check play, seek, rate, loop boundary, and an injected buffering stall.
- Check current Chrome/Edge and Firefox behavior; verify full-view fallback when clipping is disabled.
- Confirm the participant page never reveals method identity before voting.

### Regression suite

Run the focused Campaign participant/UI tests, then:

```text
python -m unittest discover -s tests
git diff --check
```

## Documentation

Update the Campaign participant documentation to describe the default A/B wipe, the full-view switch, unified controls, and the practical synchronization limitation of independent browser video decoders.

## Delivery Boundary

This specification is one implementation unit limited to `blind.html`, `blind.js`, `blind.css`, participant/UI tests, and directly related documentation. Backend or objective-metric changes require a separate approved specification.
