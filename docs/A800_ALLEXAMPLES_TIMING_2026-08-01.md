# A800 all-examples timing — 2026-08-01, binary 454fb59 (C6-l complete)

Method: every replay example plus towel, run PAIRED on the A800 (graph off,
then graph on = `STIFF_FRAME_GRAPH/FULL_GRAPH/C4/C6` knobs), merged mode,
sequential and exclusive on the card. 60-frame window where the example
accepts `CASE39*_FRAME_END`; towel runs its native 220+40 frames. "mean" is
each example's own per-step wall-clock report; fps = 1000/mean. The local
4090 was untouched throughout.

| example | off ms | off fps | on ms | on/off | on status |
|---|---|---|---|---|---|
| towel_scramble (961v) | 25.8 | 38.8 | 2126.1 | 82.4x | pass, pathological (see below) |
| case39 | 113.2 | 8.8 | 203.9 | 1.80x | pass |
| case39_multienv | 114.5 | 8.7 | 155.3 | 1.36x | pass |
| cupshirt_finray | 124.8 | 8.0 | 240.4 | 1.93x | pass |
| case39_UMI_forcegrip | 138.0 | 7.2 | — | — | **CRASH: error 700 converter.cu:311** |
| beaker_finray | 152.4 | 6.6 | 198.5 | 1.30x | pass |
| case39_UMI_beaker | 192.9 | 5.2 | — | — | **FAIL: frame 20 OVF_TRIPLETS retry budget** |
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
- **towel-on anatomy** (the 82x outlier): 19 re-record-storm frames eat 219
  of 468 s (a capture on the A800 costs ~20-30 s against ~1-3 s on the
  4090), and the remaining frames still average 1239 ms vs 25.8 ms off —
  the fixed capacity-width launch cost dominates a 961-vertex scene. Same
  shape as the 4090 (11x there), amplified by the architecture. Whole-frame
  graphs are the wrong tool for tiny scenes; nothing here is new breakage.
- **Two graph-on failures, both scenes never before run graph-on**:
  `case39_UMI_forcegrip` crashes with an illegal access (700) at
  converter.cu:311 (the exact-count D2H region), and `case39_UMI_beaker`
  exhausts the capacity retry budget at frame 20 (invalid=0x40000,
  OVF_TRIPLETS ratcheting faster than boundary growth converges). Filed as
  follow-ups; both scenes pass graph-off.
- Environment notes: the pod needed `polyscope` (now installed alongside
  trimesh); `[graph-stats]` coverage lines did not appear in suite logs on
  the A800 (delivery of the knob is proven by the timing split; the missing
  print is unexplained and only affects telemetry, not physics or timing).
