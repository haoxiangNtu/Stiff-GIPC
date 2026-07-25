// ============================================================================
// contact/encoding.h — THE int4 contact-pair encoding contract (v0.8.6 3a).
//
// Doc-only header: no code. Single source of truth for how contact pairs are
// packed into int4. Emitters and decoders MUST agree with this table; if you
// change an encoding, update every site listed at the bottom AND this file in
// the same commit. Dead branches are documented as dead — per standing
// directive they stay in the code, FROZEN (do not activate, do not delete).
//
// ─── Buffers carrying this encoding ─────────────────────────────────────────
//   _collisonPairs        DCD pairs (barrier energy/gradient/Hessian)
//   _ccd_collisonPairs    swept pairs (CCD alpha only; SEPARATE simpler code)
//   _collisonPairs_lastH  lagged-friction snapshot of DCD pairs (same table)
//   (ground contact does NOT use this table: _environment_collisionPair is a
//    plain surface-vertex index list with its own counter _gpNum.)
//
// ─── DCD encoding (_collisonPairs, MMCVIDI in decoders) ─────────────────────
// Dispatch is on SIGNS. n(v) means -v-1 (non-negative id v stored negated).
//
//   .x    .y    .z    .w     family      vertices            arity  status
//   ---------------------------------------------------------------------
//   >=0   >=0   >=0   >=0    EE          x,y | z,w            4     LIVE
//   >=0   >=0   >=0   <0     EE-smooth   x,y | z,-w-1         4     DEAD (a)
//   <0    >=0   ==-1  any    PP          -x-1, y              2     LIVE (b)
//   <0    <0    <0    <0     PP-smooth   all -v-1             2*    DEAD (a)
//   <0    >=0   >=0   <0     PE          -x-1, y, z           3     LIVE (b)
//   <0    <0    >=0   <0     PE-smooth   -x-1,-y-1, z,-w-1    3*    DEAD (a)
//   <0    >=0   >=0   >=0    PT          -x-1, y, z, w        4     LIVE
//
//   (a) The "smooth" (mollified) family is emitted ONLY under
//       `bool smooth = false;` hardcoded in mlbvh_modules/03 (upstream
//       KemengHuang's own flip, a3479e7/7130bea — fork matches upstream).
//       Decoders keep full mollifier math (I1/eps_x, all-4-vertex coupling —
//       hence 12x12 blocks, arity slot 4 marked * above). FROZEN.
//   (b) PP/PE pairs born from near-parallel EE degeneracies carry
//       add_e = -other_edge_index - 2 in .w when the mollify CONDITION
//       (eeSqureNCross < eps_x) fires and g_ee_nomollify == 0 (device global,
//       mlbvh_modules/00, default 0). Plain PP/PE decoders IGNORE .w
//       entirely, so this marker is a dead channel today — written, never
//       read (the "445k requested / 0 executed" measurement). Do not repurpose
//       .w of PP/PE without claiming it here first.
//
// ─── CCD encoding (_ccd_collisonPairs) ──────────────────────────────────────
// Two families only; every DCD emission writes its swept twin at the SAME
// slot index (buffers are index-aligned mirrors — see contact/pair_buffers.cuh
// for the DCD<=CCD capacity invariant):
//
//   .x < 0   point-vs-triangle swept:  (n(p), t0, t1, t2)
//   .x >= 0  edge-vs-edge swept:       (a0, a1, b0, b1)
//
// PT-source emissions (any _dType_PT) always write the PT-swept form with the
// FULL triangle, regardless of which degenerate DCD family they produced;
// EE-source emissions always write the all-positive EE-swept form.
//
// ─── Counters and MatIndex (the block-placement contract) ───────────────────
//   _cpNum[0]  total DCD pair count == emit slot cursor (also the CCD mirror
//              count). Emission goes through _emit_slot(_cpNum, g_dcd_cp_cap)
//              — cap-clamped, overflow lands in the +1 trash slot
//              (contact/pair_buffers.cuh owns caps and publication).
//   _cpNum[1]  RESERVED — no live writer or reader (kept for layout compat;
//              h_cpNum/h_cpNum_last mirror all 5 slots).
//   _cpNum[2]  ordinal counter for arity-2 pairs (PP)  -> 6x6  Hessian blocks
//   _cpNum[3]  ordinal counter for arity-3 pairs (PE)  -> 9x9  Hessian blocks
//   _cpNum[4]  ordinal counter for arity-4 pairs (PT/EE/smooth*) -> 12x12
//
//   _MatIndex[i] = atomicAdd(_cpNum + arity_slot, 1) at emission: the pair's
//   ordinal WITHIN its arity class, i.e. its block position in the per-arity
//   Hessian block arrays. It is NOT a type tag — family always re-derives
//   from the int4 signs per the table above.
//
// ─── Site inventory (update together) ───────────────────────────────────────
//   emit    mlbvh_modules/03_pair_emission.inl  (_checkPTintersection,
//           _checkEEintersection; EE endpoint order canonicalized by position
//           first — [xenv pin/fix] — so identical envs classify identically)
//   decode  gipc_modules/01 (energy), 04 (fused barrier G/H), 05 (gradients),
//           02 (friction, via _collisonPairs_lastH), 10 (CCD alpha kernels,
//           buildFullCP; swept table only)
//   NOTE    inside DEAD smooth branches the _compute_epx rest-vertex argument
//           order differs per family — it is part of the frozen text, leave it.
// ============================================================================
#pragma once
