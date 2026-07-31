# A800 all-examples timing — 2026-08-01, binary 454fb59 (C6-l complete)

Method: every replay example plus towel, run PAIRED on the A800 (graph off,
then graph on = `STIFF_FRAME_GRAPH/FULL_GRAPH/C4/C6` knobs), merged mode,
sequential and exclusive on the card. 60-frame window where the example
accepts `CASE39*_FRAME_END`; towel runs its native 220+40 frames. "mean" is
each example's own per-step wall-clock report; fps = 1000/mean. The local
4090 was untouched throughout.

| example | off ms | off fps | on ms | on/off | on status |
|---|---|---|---|---|---|
| towel_scramble (961v) | 25.8 | 38.8 | ~25 | ~1.0x | C6-n: declined to release path (whole run 6s vs off 7s; was 2126.1ms/frame, 502s) |
| case39 | 113.2 | 8.8 | 203.9 | 1.80x | pass |
| case39_multienv | 114.5 | 8.7 | 155.3 | 1.36x | pass |
| cupshirt_finray | 124.8 | 8.0 | 240.4 | 1.93x | pass |
| case39_UMI_forcegrip | 138.0 | 7.2 | 379.7 | 2.75x | pass after C6-m (97% full-graph) |
| beaker_finray | 152.4 | 6.6 | 198.5 | 1.30x | pass |
| case39_UMI_beaker | 192.9 | 5.2 | 399.8 | 2.07x | pass after C6-m (95% full-graph) |
| foldshirt_finray | 319.1 | 3.1 | 475.1 | 1.49x | pass |
| beaker_finray_me | 326.2 | 3.1 | 460.8 | 1.41x | pass |
| cupshirt_finray_me | 362.4 | 2.8 | 702.5 | 1.94x | pass |
| foldshirt_stats | 391.1 | 2.6 | 538.8 | 1.38x | pass |
| foldshirt_multienv | 393.1 | 2.5 | 557.2 | 1.42x | pass |
| case39_UMI_sf_obb | 459.1 | 2.2 | 498.7 | **1.09x** | pass |
| case39_UMI_sf | 609.9 | 1.6 | 819.8 | 1.34x | pass |
| foldshirt_finray_me | 1167.2 | 0.9 | 1868.4 | 1.60x | pass |

## Readings

- **13 of 15 examples pass graph-on**, with the on/off ratio between 1.09x
  and 1.94x — consistently BETTER than the 4090's 1.6-3.3x band, matching
  the expectation that the A800's slow host link makes saved launch
  round-trips worth more. The heaviest-contact scenes amortize best
  (sf_obb 1.09x, beaker 1.30x); the lightest pay the most (cupshirt 1.9x).
- **towel-on anatomy — RESOLVED by C6-n**. The 82x outlier decomposed into
  three layers: 19 re-record-storm frames ate 219 of 468 s (a capture on
  the A800 costs ~20-30 s against ~1-3 s on the 4090); the remaining
  frames paid the fixed capacity-width launch cost; and the two-graph gate
  was no refuge (its inner graphs re-record on every capacity-generation
  bump — one hard crumple frame spent 15.7 s recording for the same 567
  Newton iterations the release solver finishes in 0.5 s). step() now
  declines ALL graph machinery below STIFF_FULL_GRAPH_MIN_VERTS (default
  1024) and runs the release frame (episodes exempt; gates pin 0).
  A800 after C6-n: the whole 220+40-frame towel run takes 6 s with the
  graph env vs 7 s off (was 502 s), footprint fingerprint unchanged
  (0.904). Measurement caveat recorded in the C6-n commit: per-frame
  [bench] timers understate the release path (async queue drains in the
  untimed get_vertices); total wall is the honest metric.
- **Two graph-on failures — RESOLVED by C6-m** (fallback runs the true
  graph-off frame: `s_layout_override_off` RAII forces `device_count_mode()`
  false during a capacity-fallback attempt). Root cause: the C6-i fallback
  still ran the tier-shaped partition + capacity guard, so a frame whose
  class counts exceeded the trained tiers re-flagged OVF_TRIPLETS on the
  fallback itself and burned the retry budget (`case39_UMI_beaker` frame
  20); the same capacity-shaped launches truncate/overrun in that state
  (`case39_UMI_forcegrip`'s sticky 700 surfacing at converter.cu:311).
  After C6-m both scenes pass on the A800: forcegrip 379.7 ms (97%
  full-graph), beaker 399.8 ms (95%). Both were re-verified locally too
  (G18/G19/D4/G17e PASS, towel det stack bitwise x2).
- Environment notes: the pod needed `polyscope` (now installed alongside
  trimesh). SOLVED: the missing `[graph-stats]` lines in the suite logs
  were a stale remote binary — its knob registry predated
  STIFF_GRAPH_STATS/STIFF_C6_ABD_STEP_GRAPH and the log headers show
  `[knob-registry][WARN] unknown STIFF_* knob` for both (silent no-op).
  Physics and timing were unaffected; the C6-m rerun on a fresh build
  prints coverage normally.
