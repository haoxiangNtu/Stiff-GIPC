// [liquid] Implicit MPM Lite water as extra DOFs inside the IPC Newton solve.
//
// Grid nodes of the active voxels are FEM vertices of the engine (a point-only
// FEM body reserved at finalize, "node pool"); each step the liquid rewrites
// their lattice positions / masses / predictors (P2G), the voxel water energy
// is assembled as an 8-node element (36 upper 3x3 blocks) into the global
// triplets, and after the solve the node displacements are transferred back to
// the particles (G2P, APIC). Particles live in a second point-only FEM body
// (btype 2, rows dropped) so that later milestones can use them as IPC contact
// points. See MPM_IPC_MONOLITHIC_DESIGN.md.
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Core>

struct device_TetraData;
class GIPCTripletMatrix;
#define LIQUID_PARTICLE_BTYPE 4  // BoundaryType of liquid particles: rows dropped like pins, but still collidable

namespace gipc_liquid
{

struct LiquidParams
{
    double  dx         = 0.005;
    double3 origin     = make_double3(0, 0, 0);  // position of lattice node (0,0,0)
    int3    dims       = make_int3(1, 1, 1);     // number of CELLS per axis
    double  rho        = 1000.0;
    double  kappa      = 1e5;
    int     node_pool  = 0;      // engine vertex slots reserved for grid nodes (0 = auto)
    double  mass_floor = 0.05;   // node mass floor as a fraction of rho*dx^3 (free-surface conditioning)
    double  ghost_floor = 0.0;   // wall-ghost nodes (light nodes whose mass comes mostly from particles in solid contact) get this
                                 // inertia floor (fraction of rho*dx^3), weightless: stops cut cells from relieving their pressure by
                                 // letting the unconstrained nodes inside the wall drift into the solid (0 = off)
    int     cav_on = 0;          // TEST ONLY (LIQ_TEST_CAVITY): Dirichlet-fix grid nodes outside an analytic axisymmetric cavity
    double  cav[7] = {0, 0, 0, 0, 0, 0, 0};  // axis x, axis z, floor y, r0 (below yc0), yc0, r1 (at yc1), yc1
    double  ghost_frac  = 0.5;   // a node is a wall ghost when its contact-particle mass exceeds this fraction of its mass
    int     floor_gravity = 1;   // 1 = the floored mass is also weighed (containers carry ~2 % phantom weight at rest, m8),
                                 // 0 = the floor adds inertia only (gravity acts on the real P2G mass) -- but then small drops fall
                                 //     slowly (m11_freefall: 8 particles 78 % g, a single particle released at rest never falls)
    int     apic       = 1;      // 1 = APIC affine transfers, 0 = PIC
    double  apic_ratio = 1.0;    // APIC affine term scale carried to the next P2G (1 = APIC, 0 = PIC; blends damp the rest creep)
    double  flip       = 0.0;    // FLIP blend of the particle velocity update (0 = APIC/PIC resampling; 0.9-0.97 keeps sloshing alive)
    int     bdf2_wall_be = 1;    // LIQ_BDF2_WALL_BE (with LIQ_BDF2): grid nodes near a wall keep implicit Euler -- its damping of the stiff contact /
                                 //   shared-node (thin wall) modes stays, the bulk (where sloshing lives) is second order. 1 = the 27-node stencil of every
                                 //   wall-point particle (~1.5 cells: at dx 5 that is most of a beaker), 2 = only the corners of the cut cells (~0.5 cell),
                                 //   0 = BDF2 everywhere
    int     ug         = 0;      // LIQ_UG: particle-quadrature liquid of the recovered Unified_GIPC engine (liquid_ug.cuh): trilinear
                                 //   8-node stencil, per-particle kappa/2 (J-1)^2 + volume barrier, contact only through the particle
                                 //   IPC barrier; turns every wall / rest / regularisation option of this system off (see configure)
    double  ug_jact    = 0.0;    // LIQ_UG_JACT: volume-barrier activation ratio (0 = off; recovered default range (0, 1))
    double  ug_kb      = 1.0;    // LIQ_UG_KB: volume-barrier stiffness in units of kappa
    double  ug_mfloor  = 0.0;    // LIQ_UG_MFLOOR: node mass floor in UG mode (fraction of rho dx^3; the recovered engine has none)
    int     ug_bspline = 0;      // LIQ_UG_BSPLINE: 1 = quadratic B-spline 27-node stencil instead of the trilinear 8-node one (liquid_ug.cuh)
    double  ug_reanchor = 0.0;   // LIQ_UG_REANCHOR: beta of the density re-anchor of interior particles after each step (k_reanchor_J; 0 = off)
    double  ug_feasible = 0.9;   // LIQ_UG_FEASIBLE: safety factor of the volume feasible step (first alpha in (0,1] with det A = 0, x factor); 0 = off
    double  ug_hg       = 0.0;   // LIQ_UG_HG: hourglass viscosity (Pa s) on the trilinear stencil's zero-energy modes (sub-cell clustering control); 0 = off
    double  ug_rep      = 0.0;   // LIQ_UG_REP: sub-cell density control = pair repulsion energy 1/2 k (r0 - r)^2 between particles closer than
                                 //   r0 (k = ug_rep x kappa x h, the bulk stiffness of one particle); reuses the LIQ_STI pair pipeline; 0 = off
    double  ug_rep_r0   = 0.8;   // LIQ_UG_REP_R0: repulsion range in seed spacings (no pairs exist at rest: the seed spacing is 1.0)
    int     bdf2       = 0;      // LIQ_BDF2: second-order BDF2 (velocity form) for the grid nodes instead of implicit Euler:
                                 //   v1 = 4/3 v - 1/3 v_prev + 2/3 dt a(x1),  x1 = x + dt (19/18 v - 1/18 v_prev) + 4/9 dt^2 a(x1)
                                 //   = an incremental potential with node mass x 9/4 and that predictor (the solver is unchanged).
                                 //   Implicit Euler damps a mode of frequency w by zeta = w dt / 2 (sloshing at dt 5 ms: 0.06); this
                                 //   scheme by ~(w dt)^3 / 8 (0.0003), and stays stable for every w dt (stiff modes still decay).
    double  st_gamma   = 0.0;    // surface tension (Akinci et al. 2013 cohesion + curvature), explicit, applied before P2G; 0 = off
    double  st_c       = 1.0;    // cohesion support in units of dx (<= 2; the search spans ceil(st_c) cells). Akinci's cohesion
                                 // is neutral at 0.273 c: in a film too thin for grid pressure it packs particles to that distance
    double  st_thin    = 0.0;    // apply the surface tension only to particles whose SPH density is below st_thin * rho0 (thin layers,
                                 //   free surfaces, droplets); 0 = everywhere. Keeps the explicit cohesion out of the bulk, where it compresses the water.
    double  sti_gamma  = 0.0;    // LIQ_STI: implicit surface tension (N/m): a pair cohesion energy INSIDE the Newton solve between
                                 //   free-surface / thin-layer particles (bond energy 2 gamma h^2, no force at sti_r0 h, attraction up
                                 //   to sti_rc h); its gradient and Hessian reach the grid through the particle embedding; 0 = off
    double  sti_r0     = 1.0;    // LIQ_STI_R0: rest distance in seed spacings (repulsive below, attractive up to the cutoff)
    double  sti_rc     = 1.5;    // LIQ_STI_RC: cutoff in seed spacings (with the 0.25 h list margin it must fit in one cell)
    int     sti_freeze = 0;      // LIQ_STI_FREEZE (diagnostic): compute the surface-particle gate once (first step) and keep it, so no pair
                                 //   switches on or off between steps; tests whether the per-step 0/1 gate causes the sloshing damping
    double  stm_sigma  = 0.0;    // LIQ_STM: membrane surface tension (N/m): every free-surface particle carries a surface element of area
                                 //   stm_area h^2; energy sigma |F a x F b| inside the Newton solve (continuum surface stress, no particle pairs)
    double  stm_area   = 1.0;    // LIQ_STM_AREA: area multiplier (amode 1: of the calibrated area; amode 0: element area in seed spacings^2)
    double  stm_thr    = 0.25;   // LIQ_STM_THR: surface detection, weighted neighbour-centroid offset (within 2 h) above this many spacings
    double  stm_amax   = 2.0;    // LIQ_STM_AMAX (amode 1): element area cap in h^2 (flat / curved surfaces stay below 1.8 h^2; a 2-4 particle
                                 //   sub-cell spray cluster would otherwise get 2-4 h^2 per particle and its cell-level membrane sheds particles)
    int     stm_amode  = 1;      // LIQ_STM_AMODE: 1 = each element gets 1 / (areal density of the surface elements within 2 h), with the normal
                                 //   averaged over the same neighbourhood: the elements tile the surface whatever the detection threshold or
                                 //   jitter (flat jittered slab: area 1.00 of the true area; amode 0 = stm_area h^2 each, 0.62 at thr 0.35)
                                 // which particles: those in a cell next to an empty in-domain cell (air: films, droplets, free
                                 //   surfaces), not in a cell holding a wall-adjacent particle (no pull on the wetted wall layer);
                                 //   the bulk (all neighbour cells wet) and water lying on the ground under a deep pool are untouched.
                                 //   SPH density cannot make this cut: a disordered bulk has 20-25 % of its particles below 0.9 rho0
    double  rest_damp  = 0.0;    // rest damping rate (1/s) on slow particles only; 0 = off (demo-level 'sleeping' of still water)
    double  rest_v     = 0.003;  // rest damping fades out smoothly up to this particle speed (m/s); faster flow is untouched
    double  sleep_v    = 0.0;    // rest sleep (position-level): a particle whose solved grid velocity stays below sleep_v (m/s)
    int     sleep_n    = 60;     //   for sleep_n consecutive steps is held in place (x and J frozen, v = C = 0) until the solved
    double  wake_v     = 0.003;  //   velocity at it exceeds wake_v (m/s); 0 = off
                                 //   (per-particle holds are inconsistent with the grid solve: the awake rest speeds up; kept for the record)
    double  isl_v      = 0.0;    // island sleep: a CONNECTED water body whose particles all stay slower than isl_v (m/s, solved
    double  isl_wake   = 0.003;  //   grid velocity; isl_allow of them may be faster, none faster than isl_wake) for isl_t seconds is
    double  isl_t      = 1.0;    //   held as a whole (x, J frozen, v = C = 0) until any of its particles is solved faster than
    double  isl_allow  = 0.005;  //   isl_wake (container moved, liquid poured in, rod dipped). All-or-nothing per body keeps the
                                 //   particles together, but the post-solve rollback is NOT consistent with the coupled solid
                                 //   solution or CCD. Experimental visual stabilization only; 0 = off.
    double  rest_k     = 0.0;    // rest drag INSIDE the solve: a grid node whose transferred velocity is below rest_kv gets (1 + k)
    double  rest_kv    = 0.003;  //   times its inertia around a target pulled toward rest (the start gradient is unchanged, the
                                 //   per-step creep of soft modes shrinks 1/(1 + k)); k fades out smoothly up to rest_kv (m/s); 0 = off
    double  J_min      = 0.1;    // cell-ratio clamp
    double  J_track_min = 0.7;   // tracked-J clamp (modes 0/3): compressions beyond this are numerical, not physical at kappa 1e5
    double  J_track_max = 2.0;   // tracked-J upper clamp (1.0 = water never stores an 'expanded' state; stops free-surface creep?)
    double3 park       = make_double3(0, 0, 0);  // where inactive node slots sit
    // particle-level regularisation (post-step), see Stage A lessons:
    double  push_apart = 0.9;    // min spacing as a fraction of the seed spacing (0 = off)
    int     j_mode     = 3;      // 0 = particle-tracked J (MPM Lite Eq. 42), 1 = SPH density reinit
                                 // (Contact Proxy Eq. 8), 2 = cell density J_base = rho0 dx^3 / m_c,
                                 // 3 = tracked J re-anchored to the cell ratio in full cells (default),
                                 // 4 = cell density corrected by the geometric fill fraction of the cell
                                 //     (position-based like 2, but a surface/wall-cut cell at nominal
                                 //     packing reads J = 1 instead of "expanded": no layer loss, no creep)
    double  clamp_lo_y = -1e300; // particle floor (set by the engine from the ground plane)
    double  dhat       = 0.0;    // IPC barrier distance (absolute, set by the engine): caps the push-apart
    double  tension_ratio = 0.0; // stiffness ratio of the J > 1 branch (0 = no tension: C1 kink at J = 1)
    double  vbar   = 0.0;        // LIQ_VOX_BAR: barrier on the volume ratio J = Jb det A below this (0 = off; see vox_barrier)
    double  vbar_k = 1.0;        // LIQ_VOX_BAR_K: barrier stiffness in units of kappa
    int     invert_ok = 0;       // LIQ_VOX_INVERT_OK: do not reject det A <= 0 (inverted in-step voxel map) in the line search. The quadratic
                                 // energy already makes inverting a DENSE cell far costlier than its kinetic energy; a SPARSE cell's in-step
                                 // map inverting is harmless in MPM. The 1e300 wall made the Newton direction (which cannot see it)
                                 // stall against it: line search to 2^-64, same direction again (orifice, dam break dx 1.25, pour film).
    double  reanchor_beta = 1.0; // j_mode 3: J_p += beta (cell ratio - J_p) for interior particles (1 = hard reset)
    double  conv_scale    = 0.1; // liquid rows' move direction is scaled by this in the Newton max-move exit test
    double  push_surf_ratio = 1.2; // push-apart skips particles whose cell is under-filled beyond this ratio (0 = push all)
    int     push_film   = 0;       // LIQ_PUSH_FILM: water lying on the ground plane (centre within one cell of it, no solid triangle
                                   // within the wall search radius) is pushed apart in under-filled cells too, and its ground-contact
                                   // particles may slide along / lift off the plane (1); 2 = also drop the velocity against the push
    double  push_film_r = 0.0;     // LIQ_PUSH_FILM_R: min spacing of that water in seed spacings (0 = push_apart)
    // j_mode 4 fill reconstruction (units of the seed spacing h): water region =
    // (particles (+) fill_dilate h) (-) fill_erode h, boundary at fill_dilate - fill_erode
    // = 0.5 h beyond the particle centres (the particle cube faces), gaps below
    // 2 fill_dilate h closed; plus each particle's own ball (fill_own h = the h^3 ball)
    double  fill_dilate = 2.0;
    double  fill_erode  = 1.5;
    double  fill_own    = 0.62;
    // table adhesion stand-in: a particle whose centre comes within stick_h of the
    // ground plane freezes where it is (0 = off). Vessel floors are thicker than
    // stick_h, so water inside vessels is unaffected; spilled water forms a puddle
    // instead of an inviscid, tension-free sheet that sprays to the domain walls.
    double  stick_h     = 0.0;
    // grid-level solid wall boundary condition (see LiquidContainer): constrained grid nodes get this
    // inertia (x rho dx^3) and a kinematic target (normal: the body, tangential: the fluid), no gravity
    int     wall_bc     = 0;
    double  wall_mass   = 1e3;
    int     wall_static_mode = 1;     // LIQ_WALL_STATIC_MODE: 1 = every static band node, 2 = only band nodes behind the wall (phi < 0)
    double  wall_static_mass = 0.0;   // LIQ_WALL_STATIC_MASS: penalty of the WB band nodes of STATIC walls (0 = wall_mass): a Dirichlet-stiff band where
                                      // the wall reaction is one-sided (no fluid behind), soft where the wall moves (piston, sliding tank)
    int     wall_static_normal = 0;   // LIQ_WALL_STATIC_NORMAL: 1 = the static stiffening acts on the wall-normal component only (the pressure reaction);
                                      // the tangential penalty stays wall_mass (the stiff no-slip band freezes the wall layer: m6 water 0.45 mm off centre)
    double  wb_trim = 0.0;            // LIQ_WB_TRIM: a column body's bottom cells filled below this fraction of their geometric volume are separate water (orifice jet)
    double  wb_nrm_exact = 0.0;       // LIQ_WB_NRM_EXACT: band nodes with |phi| <= this x dx take their wall normal from the closest point on the mesh (0 = off)
    double  wall_tang = 1.0;          // LIQ_WALL_TANG: tangential penalty multiplier of the fixed band nodes of a static wall on the fluid side (phi >= 0):
                                      // 1 = soft no-slip (wall_mass), 0 = free slip (normal only); nodes behind the wall and moving walls keep 1
    int     ground_bc   = 0;     // grid nodes below the ground plane (clamp_lo_y): vertical motion locked
    // wall field constraint (WFC, LIQ_WFC=1): the grid displacement field may not move into a solid at the
    // wetted wall surface.  One wall point per particle within wfc_r*h of a solid triangle (step start): the
    // closest point on the triangle; E = K/2 min(0, n.(u_field(X_w) - u_solid(X_w)))^2 with u_field the Q1
    // field of the 8 nodes of the cell containing X_w and u_solid the triangle's barycentric displacement
    // (two-way: blocks go to the contact triplets, ABD vertices are reduced to body DOFs by the engine).
    // CPIC-style compatibility (LIQ_CPIC=1): a particle only transfers to / interpolates from grid nodes on ITS side of
    // the nearest solid surface (wall point within cpic_r dx); masked weights are renormalised. Isolates the transport
    // across walls thinner than a cell (thin plate: 12 of the 27 stencil nodes were shared by both sides).
    int     cpic   = 0;
    double  cpic_r = 1.6;        // wall-point search radius for the mask, in cells (stencil reach is 1.5 dx)
    int     cpic_ghost = 1;      // ablation switches (LIQ_CPIC_GHOST / _P2G / _C): masked-out referenced nodes fixed as ghosts;
    int     cpic_p2g   = 1;      //   masked renormalised particle->node deposit (0 = original cell route);
    int     cpic_split = 0;      //   2b-1 EXPERIMENT (rejected 2026-10-20, see audit doc §8): per-side elements with PRIVATE fixed ghost corners cut the
                                 //   load path to curved walls (vessel carries 8 % of the water weight) and destabilise plates; keep 0.
                                 //   2b: cut cells carry one volume state + one voxel element PER SIDE of the wall (private
                                 //   fixed ghost corners for the other side); 0 = single element per cell
    int     cpic_fields = 1;     //   2 = per-side node slots (two velocity fields across thin walls, 2026-10-20); 1 = single field (S2)
    int     cpic_side1 = 1;      //   ablation: 0 = never use the second ('-') deposit set
    int     cpic_pghost = 1;     //   ablation: 0 = never replace a corner by a private ghost (only slot-less corners)
    int     cpic_cell  = 0;      //   masked cell-centre deposit -- REJECTED 2026-10-20 (s2e): the phantom wall elements it removes ARE the wall
                                 //   pressure reaction; without them the wall layer collapses (flask median 0.16 -> 172 mm/s). Keep 0.
    int     cpic_c     = 0;      //   ghost-velocity substitution in the affine-velocity operator (0 = actual node velocities; the
                                 //   substitution (1) was REJECTED 2026-10-20: it drives a J ratchet near walls, ablation S2)
    int     wfc   = 0;
    int     wfc_pre = 0;         // LIQ_WFC_PRE (2026-11-13): activity of the one-sided wall constraint AT ITS BOUNDARY (g = 0). The field displacement restarts
                                 //   from zero every time step, so in the first Newton iteration every wall point sits exactly at g = 0 and the plain rule
                                 //   (active when g < 0) drops the wall from the first direction: the water moves into the wall, only the particle barrier's
                                 //   CCD stops it (step size ~0.1), and a solve that stops after one iteration (fast mode) never applies the wall at all.
                                 //   1 = at g = 0 count the point as active when it carried a wall reaction at the end of the previous step OR the inertial
                                 //       predictor moves the field into the wall there (semismooth Newton: only the Hessian changes, the energy and its
                                 //       minimiser do not); 2 = predictor only; 3 = previous step only; 0 = off (plain rule)
    int     wfc_al = 0;          // LIQ_WFC_AL (2026-11-13): augmented-Lagrangian wall constraint. Each wall point (one per near-wall particle) keeps a
                                 //   multiplier lambda >= 0 = the wall reaction it carried at the end of the previous step; energy
                                 //   K/2 min(0, g - lambda/K)^2, update lambda <- max(0, lambda - K g) once per step. The plain penalty restarts from
                                 //   g = 0 every step, so a loaded wall point lets the field move F dt^2 / K INTO the wall each step (6 um per step at
                                 //   1 kPa = 1.2 mm/s): the wall layer creeps into the wall until the particle barrier carries the load at micrometre
                                 //   distances -- which is what cuts the line search to alpha ~ 0.1. With the multiplier the reaction is already in the
                                 //   first gradient of the step and the stationary state has g = 0 (no creep). 0 = off (plain penalty)
    double  wfc_k = 100.0;       // K per wall point, x rho dx^3 (Newton/inertia units)
    double  wfc_r = 1.0;         // search radius in particle spacings
    int     wfc_ground = 0;      // LIQ_WFC_GROUND: the analytic +y ground plane is a WFC wall too (static, normal +y) for particles with
                                 //   no mesh triangle within the search radius; without it the grid flows into the plane, where only the
                                 //   particle barrier stops the particles: they pile up on it (floor sediment, flattened puddles)
    // volumetric ghost penalty (LIQ_GP > 0, CutFEM-style): every wall-cut cell's volumetric rate tr(G_c) must match each
    // active face neighbour's, E = gp*kappa*V_cell/2 (tr G_c - tr G_c')^2 -- closes the pressure-relief modes of the
    // solid-side nodes (stretching about the wall point, tangential spreading) without adding shear stiffness
    double  gp = 0.0;
    int     gp_full = 1;     // 1: full gradient jump (CutFEM ghost penalty), 0: trace only
                             // 2: gradient of the normal displacement, 3: normal stretch, 6: rest-adaptive = full gradient where
                             //    the flow is slow (stops the wall-layer creep) blended to normal stretch where it flows (no plug)
    double  gp_vlo = 0.003;  // mode 6: full gradient at pair node speeds <= gp_vlo (m/s, start-of-step transferred velocity) ...
    double  gp_vhi = 0.01;   //   ... normal stretch only at >= gp_vhi, smoothstep in between
    int     cutcell = 0;     // LIQ_CUTCELL: a wall-cut cell measures only its FLUID sub-volume: dV = sum_q a_q . u_q (a_q = area vectors of the
                             //   fluid-side face caps under the trilinear field, wall plane = mean of the cell's wall points), J = Jb (1 + dV / V_g).
                             //   Pressure then never pushes the solid-side corners into the wall; closure sum_q a_q = A_w n (static wall)
    double  cutcell_vmin = 0.25;  // LIQ_CUTCELL_VMIN: fluid sub-volume floor in dx^3 for the J denominator (slivers would be stiff)
    int     wb      = 0;      // LIQ_WB: well-balanced cut cells (liquid_wb.cuh): pressure areas, within-cell hydrostatic term and nodal
                              // weights from ONE fluid sub-volume per cell -> still water is an exact stationary point (needs LIQ_WFC, no CPIC)
    int     wb_slip = 1;      // LIQ_WB_SLIP: grid nodes on / behind the wall (phi <= wb_slip_phi dx) get the slip wall BC (normal displacement tied to
                              // the wall, k_wall_assemble) and implicit Euler: the volume-change operator gives them no normal stiffness by itself
    double  wb_slip_phi = 0.05; // LIQ_WB_SLIP_PHI
    double  wb_moving_eps = 1e-7;  // LIQ_WB_MOVING_EPS (m): wall displacement per step above which a wall counts as moving (kinematic pour ~4e-5; static ABD jitter ~1e-9)
    double  wb_slip_phi_mov = -1.0; // LIQ_WB_SLIP_PHI_MOVING: band thickness (x dx) for nodes whose wall moved this step (-1 = same as wb_slip_phi)
    int     wb_surf = 1;      // LIQ_WB_SURF: 1 = per-column free-surface plane (p = 0 boundary of the top cell); 0 = no surface cut, the column's
                              // cells above the body's full cells keep the current operator with their deposited volume as weight (diagnostic)
    int     wb_colmin_pool = 0; // LIQ_WB_COLMIN_POOL: decide the column-density guard over the 3 x 3 neighbourhood (slivers along curved walls stay in the body)
    int     wb_hsmooth_w = 0; // LIQ_WB_HSMOOTH_W: weight the column-height smoothing by each column's particles per layer
    double  wb_cap = 0.0;     // LIQ_WB_CAP: a column whose water touches a wall within cap x h above its top particle has no free surface (overhangs)
    double  wb_phi_ghost = 0.0; // LIQ_WB_PHI_GHOST (x dx, LIQ_WB_PHI_SIGN=7): dilate the fluid region into the solids (behind-wall slivers)
    double  wb_vis_r = 0.0;   // LIQ_WB_VIS_R (x dx): modes 8+ test visibility only where the nearest wall point is farther than this
    double  wb_thin = 1.2;    // LIQ_WB_THIN (x dx): a wall thinner than this keeps rule 3's dead zone (LIQ_WB_PHI_SIGN=11)
    int     wb_phi_ghost_side = 0; // LIQ_WB_PHI_GHOST_SIDE: 1 = only fluid-side nodes are lifted (thin-plate dead zones keep their depth)
    double  wb_ovh = 0.0;     // LIQ_WB_OVH: radius (x h) of the overhang test on a column's top layer; a wall-limited column takes its neighbours' surface
    double  wb_ovh_cos = 0.1; // LIQ_WB_OVH_COS: the wall is above when (x_p - q).up < -cos |x_p - q|
    int     wb_hcol_freeze = 0; // LIQ_WB_HCOL_FREEZE (diagnostic): after this step the column surface heights stop following the particles
    int     wb_phi_freeze = 0; // LIQ_WB_PHI_FREEZE (diagnostic, static containers): from this step on, nodes keep the wall distance they had
    int     wb_fullf = 0;     // LIQ_WB_FULLF: full cells carry the exact hydrostatic term rho int g.(x - x_c) grad N_q dV like cut cells (oblique / curved walls)
    double  wb_colmin = 0.0;  // LIQ_WB_COLMIN: a column whose deposit is below colmin x its geometric volume under h_col is separate water (thin film / stream / splash)
    double  wb_phi_dtol = 0.0; // LIQ_WB_PHI_DTOL: wall-side sign tolerance (x dx): a node within it of a wall point's plane is on the wall, not behind (roundoff on lattice-aligned walls)
    int     wb_hmode = 0;     // LIQ_WB_HMODE: column surface estimator 0 = top-k mean, 1 = top-K with K = the column's particles per layer, 2 = top-layer window
    double  wb_hwin = 0.5;    // LIQ_WB_HWIN: window of mode 2 (particles within hwin * h of the column's topmost)
    int     wb_topk = 4;      // LIQ_WB_TOPK: the column's surface height = mean of its top-k particles (+ h/2); 1 = the single topmost particle
    int     wb_hsmooth = 1;   // LIQ_WB_HSMOOTH: column average radius of the surface heights (1 = 3 x 3, 2 = 5 x 5, 0 = off)
    double  wb_aggr = 0.0;    // LIQ_WB_AGGR: aggregate nodes with geometric fluid share < aggr rho dx^3 to a well-supported root cell (0 = off)
    int     wb_aggr_depth = 3; // LIQ_WB_AGGR_DEPTH: root search steps away from the wall
    int     wb_aggr_wall = 1;  // LIQ_WB_AGGR_WALL: on-wall slip nodes are slaved tangentially to the interior extrapolation (always, independent of share)
    int     wb_aggr_ghost = 0; // LIQ_WB_AGGR_GHOST: 1 = also aggregate nodes behind the wall (instead of the GHOST mode)
    int     wb_aggr_debug = 0; // LIQ_WB_AGGR_DEBUG: print the constraint residual after each step's solve (every 200 steps)
    double  wb_g2p_fmin = 0.25; // LIQ_WB_G2P_FMIN: with LIQ_WB_G2P=2, minimum fill fraction of a wall-cut element for the sub-volume transfer
    double  wb_hole_lat = 0.0; // (0 = exact triangle-containment rule) // LIQ_WB_HOLE_LAT (x dx, Euclid mode): a node counts as behind the wall only if its lateral offset to the nearest wall point is within this
    int     wb_phi_euclid = 0; // LIQ_WB_PHI_EUCLID: nodal wall distance = signed Euclidean distance to the nearest wall point (frees nodes over holes / past edges)
    int     wb_phi_tri = 0;   // LIQ_WB_PHI_TRI (euclid mode): |phi| = distance to the wall point's TRIANGLE, not to the sample point itself
    double  wb_geov_fill = 0.8; // LIQ_WB_GEOV=2 fill guard (deposit x Jb / dx^3 below this keeps the deposited volume)
    int     wb_geov = 0;      // LIQ_WB_GEOV: voxel rest volume = geometric volume / Jb (force = pressure x geometric area, consistent with the WB node weights)
    int     wb_ybar = 0;      // LIQ_WB_YBAR: shift the deposited Jb from the depositing particles' mean height to the element's fluid centroid (hydrostatic gradient)
    int     g2p_jcell = 0;    // LIQ_G2P_JCELL (WB): particle J update = volume ratio of its own element over the step (1), or the deposit-weighted
                              // mean of its stencil elements' ratios (2: the transpose of the deposit; one sliver cell cannot dominate a particle;
                              // 3: also weighted by each element's fluid volume)
    int     g2p_fluid = 0;    // LIQ_G2P_FLUID (WB): G2P interpolates velocity / affine gradient only from fluid elements of the centre stencil (renormalised)
    int     wb_airgrav = 0;   // LIQ_WB_AIRGRAV (diagnostic): nodes above the free surface (deposited mass, zero geometric weight) also weigh their deposit
    double  jfilt = 0.0;      // LIQ_JFILT: hydrostatically corrected null-space filter of the tracked particle J (0 = off, 1 = full projection per step)
    int     wb_phi_sign = 3;  // LIQ_WB_PHI_SIGN (euclid mode): 3 = behind iff ANY nearby wall point's plane has the node behind it (= sti48: thin-wall exteriors behind, a plate's
                              //   neighbourhood frozen on both sides, holes blocked); 0 = sign of the nearest wall point's plane (frees nodes inside thick walls: m9 2 vs 475 outside cells);
                              // 1 = behind iff ANY containing triangle has the node behind it (sti52; frees exterior nodes of thin walls -> phantom cut-cell fluid, beaker sloshing 0.158 -> 0.25);
                              // 2 = sign of the NEAREST containing triangle, none = open space (holes open; exterior convex corners of thin walls freed)
    int     wb_g2p = 0;       // LIQ_WB_G2P: cut cells' particle transfers (velocity mean, affine/J gradient) use the fluid sub-volume's nodal weights and a_q/V_g
    double  visc = 0.0;       // LIQ_VISC: dynamic viscosity mu (Pa s; water 1e-3). Implicit cell-mean viscous potential (mu/dt) V |sym grad u|^2 on every
                              // element (full: corner-sign gradient, cut: a_q / V_g). The one physical dissipation the model lacked: grid-resolved
                              // cell-scale shear modes (surface and wall-tangential) are neutral in an inviscid one-point model and are only damped
                              // by the transfers; real water damps them in rho dx^2 / (mu pi^2) ~ 2.5 s at dx 5 mm.
    int     wb_topface = 0;   // LIQ_WB_TOPFACE: register a column surface lying exactly on a cell's top face as a surface plane (default off: worse at rest)
    int     wb_hg_wallonly = 0; // LIQ_WB_HG_WALLONLY: apply the sub-cell term only to wall-cut cells (free-surface-only cut cells keep the plain operator)
    double  wb_surfk = 0.0;   // LIQ_WB_SURFK: free-surface gravity stiffness x rho g: E = 1/2 surfk rho g int_S (u.up)^2 dA_h over the free-surface faces of the
                              // cut cells (the Lagrangian second-order potential of a lifted surface; zero at rest, conservative) -- makes the surface-wave
                              // restoring force implicit instead of appearing one step late through the re-rasterised weights
    double  wb_snap = 0.0;    // LIQ_WB_SNAP: a column surface within snap*dx of a node plane is put on it (no sliver surface cells)
    int     wb_det = 0;       // LIQ_WB_DET: cut elements use J = Jb det(I + sum u_q a_q^T / V_g) (geometric stiffness) instead of the linearised J
    double  pdev = 0.0;       // LIQ_PDEV: particle volumetric-deviation energy, x kappa. E = 1/2 pdev kappa sum_p V_p eps_p^2 with
                              // eps_p = (J_p - J~_c) + (rho/kappa) g.(x_p - x~_c): each particle's J against the LINEAR hydrostatic profile through the
                              // cell's particle mean. Zero at hydrostatic rest (keeps the well-balanced operator), ELASTIC (J_p carries the state),
                              // so it is no viscosity: resolved flow stores and returns it, the unresolvable sub-cell (hourglass) modes become
                              // kHz oscillators that the L-stable integrator removes. Multi-point quadrature of the volumetric energy's deviatoric part.
    int     pdev_all = 1;     // LIQ_PDEV_ALL: 1 = every element, 0 = cut cells only
    double  wb_hg = 0.0;      // LIQ_WB_HG: sub-cell volumetric consistency of cut cells, x kappa: E = 1/2 hg kappa sum_T V_T (delta_T - delta_c)^2 over the 6 Kuhn-tet pieces (hourglass control of the one-point energy; zero for affine motion)
    double  wb_lat = 0.0;     // LIQ_WB_LAT: lateral gate (x dx) for the nodal wall distance: a node counts as over a wall point only within this lateral offset (0 = off); nodes in a hole are then free
    double  wb_mfloor = 0.0;  // LIQ_WB_MFLOOR: inertia floor (x rho dx^3, weightless) for the FREE nodes within wb_near dx of a wall: they carry little fluid
                              // and little deposited mass, and without GP they jitter; extra inertia regularises without dissipating (0 = off)
    double  wb_wmin = 0.1;    // LIQ_WB_WMIN: a near-wall node (phi <= wb_near dx) whose geometric fluid share is below wmin rho dx^3 is a sliver corner: fixed
    double  wb_near = 0.5;    // LIQ_WB_NEAR (lattice units)
    int     wb_ghost = 2;     // LIQ_WB_GHOST: nodes strictly behind the wall (phi < -wb_slip_phi dx): 1 = slip like the wall nodes, 2 = fixed to the
                              // wall (they carry no fluid energy, only particle-penalty forces, and would slide freely along the wall)
    double  wb_gate = 0.25;   // LIQ_WB_GATE: a cell with deposited volume < gate x its geometric fluid volume is "not full": above a full cell it is
                              // kernel spill (weightless, credited to the surface cell below), otherwise it is the top of its column
    int     gm     = 0;      // LIQ_GHOST_MIRROR: every active grid node on or behind a wall (s = n.(x - X_w) <= 0 w.r.t. its closest wall point)
                             //   is slaved to the mirror image of the fluid field: u_g* = M u_f(x_g - 2 s n) + 2 n n^T u_wall, M = I - 2 n n^T
                             //   (normal: antisymmetric through the wall point -> in 1D the cut cell's J is exactly the fluid sub-cell's;
                             //   tangential: the fluid's -> free slip). E = K/2 [min(0, n.(u_g - u_g*))^2 + |P_t (u_g - u_g*)|^2]: the normal
                             //   part is one-sided (the fluid may leave the wall), u_wall from the wall triangle (two-way). Constrains only
                             //   the solid-side nodes, never the fluid's own gradients (unlike the ghost penalty).
    double  gm_k   = 100.0;  // LIQ_GHOST_MIRROR_K: penalty in rho dx^3 (Newton units, as LIQ_WFC_K)
    int     gp_rel = 1;      // LIQ_GP_REL (mode 6): node speeds measured relative to the wall the cut cell touches (mean velocity of its
                             //   wall points' triangles, previous step), so water at rest in a carried / moving vessel counts as at rest; 0 = world speed
};

// Grid-level wall boundary condition for an axisymmetric container carried by an ABD body.
// The IPC barrier only acts on the particles; without this the grid nodes on the solid side of a
// cut cell are free, the cut cells relieve their pressure through them, the wall layer loses its
// hydrostatic support and sinks (a steady roll in water that should be at rest), and with the
// tracked J the grid flows through the floor while the particles pile up (volume loss).
#define LIQ_MAX_CONT 8
#define LIQ_PROF_N 32
struct LiquidContainer
{
    int    body = -1;          // ABD body index (q = p, a1, a2, a3; x = p + A xbar)
    double Rd[9];              // design frame of the vessel profile: X = Rd xbar + td (row-major)
    double td[3];
    int    n = 0;              // cavity profile samples in the design frame (y increasing): inner radius pr(py)
    double py[LIQ_PROF_N];     // py[0] = cavity floor height
    double pr[LIQ_PROF_N];
    double y_top = 0.0;        // design heights above this are free (open mouth / spout)
    double band  = 0.0;        // thickness of the constrained shell outside the cavity (wall + stencil reach)
    int    slip  = 1;          // 1: normal follows the body, tangential follows the fluid; 0: follow the body
};

class LiquidSystem
{
  public:
    LiquidParams P;
    int          n_particles = 0;

    // engine layout (global vertex indices / collision body ids)
    int node_slot_base     = -1;
    int particle_slot_base = -1;
    int body_id_nodes      = -1;
    int body_id_particles  = -1;
    int max_voxels         = 0;  // triplet reservation: 36 blocks per voxel

    // lattice sizes
    int n_cells = 0;
    int n_nodes = 0;

    // per-step host stats
    int h_active_cells = 0;
    int h_elements     = 0;      // voxel elements this step (= active cells, or more with the per-side split)
    int n_vnodes       = 0;      // virtual node count: n_nodes (one field) or 2 n_nodes (per-side slots: side-1 copy at nid + n_nodes)
    int h_twosided     = 0;      // lattice nodes with both side copies active (last step)
    int h_ghost_slots  = 0;      // private fixed ghost corner slots this step (after the active nodes in the pool)
    int h_live_nodes   = 0;      // active nodes + private ghosts: the node slots the live-prefix PCG must cover (ghost
                                 // slots are referenced by voxel blocks; a solve truncated before them reads past its vectors)
    int h_active_nodes = 0;
    int steps_done     = 0;

    // host seed
    std::vector<double3> h_x0;
    double               particle_volume = 0.0;

    // device: particles
    double3* d_xp  = nullptr;
    double3* d_vp  = nullptr;
    double*  d_Cp  = nullptr;  // 9 per particle, row major
    double*  d_Jp  = nullptr;
    double*  d_mp  = nullptr;
    double*  d_Vp0 = nullptr;
    // device: cells (dense lattice)
    double*  d_mc   = nullptr;
    double3* d_mvc  = nullptr;  // momentum, then velocity after finalize
    double*  d_Gc   = nullptr;  // 9 per cell
    double*  d_Vc   = nullptr;
    double*  d_piVc = nullptr;
    double*  d_Jbc  = nullptr;
    int*     d_cell_flag    = nullptr;
    int*     d_cell_scan    = nullptr;
    int*     d_active_cells = nullptr;
    // device: nodes (dense lattice)
    double*  d_mi        = nullptr;
    double*  d_mcc       = nullptr;  // cell mass from particles in solid contact (wall-ghost detection)
    double*  d_mci       = nullptr;  // node mass from particles in solid contact
    double3* d_mvi       = nullptr;
    double3* d_vi        = nullptr;  // solved velocity (post-step)
    double3* d_vi_old    = nullptr;  // grid velocity the solve started from (FLIP)
    double3* d_st_n      = nullptr;  // surface-tension normals (Akinci)
    // implicit surface tension (LIQ_STI): cohesive pairs of this step (built in pre_step, used by every Newton assembly)
    int2*    d_coh_pairs = nullptr;
    double*  d_coh_w     = nullptr;  // per pair: gate weight min(s_p, s_q), fixed for the step
    double*  d_coh_s     = nullptr;  // per particle: gate (1 = surface particle, 0 = bulk / wall layer)
    int*     d_coh_cwall = nullptr;  // per cell: holds a wall-adjacent particle
    double*  d_coh_red   = nullptr;  // energy partials
    int*     d_coh_count = nullptr;
    int*     d_coh_nblk  = nullptr;  // per pair: Hessian blocks this assembly (0 beyond the cutoff), then their offsets
    int*     d_coh_off   = nullptr;
    int      coh_cap = 0, h_coh = 0;
    bool     coh_gate_done = false;   // LIQ_STI_FREEZE: the gate has been computed once
    // LIQ_BDF2 history: per particle the velocity / affine velocity of the previous step, their grid transfer, and per node slot
    // the velocity part of the BDF2 update (4/3 v - 1/3 v_prev + 2/3 dt g); d_bh_vel = the solved BDF2 node velocity
    double3* d_bh_vp  = nullptr;
    double*  d_bh_Cp  = nullptr;
    double3* d_bh_mvc = nullptr;
    double*  d_bh_Gc  = nullptr;
    double3* d_bh_mvi = nullptr;
    double3* d_bh_vel = nullptr;
    double3* d_bh_vb  = nullptr;
    double3* d_bh_xt  = nullptr;   // the BDF2 predictor per node slot (the engine rewrites xTilta before post_step)
    int*     d_bh_wall = nullptr;  // LIQ_BDF2_WALL_BE: per lattice node, 1 = near a wall (implicit Euler)
    double3* d_gp_cutv = nullptr;   // LIQ_GP_REL: per cell, summed wall velocity of its wall points (+ count)
    int*     d_gp_cutc = nullptr;
    double3* d_cc_nsum = nullptr;   // LIQ_CUTCELL: per cell, summed normals / positions of its wall points (+ count)
    double3* d_cc_xsum = nullptr;
    int*     d_cc_cnt  = nullptr;
    int*     d_vox_cc  = nullptr;   // per voxel element: 1 = cut (uses d_vox_a / d_vox_vg)
    double3* d_vox_a   = nullptr;   // 8 area vectors per element
    double*  d_vox_vg  = nullptr;   // fluid sub-volume (floored) per element
    double3* d_vox_f   = nullptr;   // LIQ_WB: 8 constant hydrostatic forces rho (b_q - beta_q) per element
    double*  d_vox_w   = nullptr;   // LIQ_WB: 8 consistent nodal weights per element
    double*  d_wb_phi  = nullptr;   // LIQ_WB: nodal signed distance to the wall (dense)
    double*  d_node_w  = nullptr;   // LIQ_WB: consistent weight per node (dense), gravity acts on it instead of the deposited mass
    int*     d_wb_cnt  = nullptr;
    double3* d_wb_n    = nullptr;   // LIQ_WB: wall normal at the node (of the wall point that set phi)
    double*  d_wb_mfl  = nullptr;   // LIQ_WB: per node extra inertia floor (x rho dx^3), 0 = none
    int*     d_wb_pmin = nullptr;   // LIQ_WB: the particle whose wall point set the node's phi (-1 none)
    int*     d_wb_cflag = nullptr;  // LIQ_WB: per cell 0 inactive / 1 fluid-side / 2 all corners outside
    int*     d_wb_dead  = nullptr;  // LIQ_WB: per cell 1 = dead element this step (no fluid energy): excluded from ghost-penalty pairs
    int*     d_wb_agroot = nullptr; // LIQ_WB_AGGR: per node root cell (-1 none)
    int*     d_wb_agtype = nullptr; int* d_wb_agt = nullptr; double3* d_wb_agnrm = nullptr;
    int*     d_wb_agflag = nullptr; int* d_wb_agscan = nullptr; int* d_wb_agn = nullptr; int* d_wb_agr = nullptr; double* d_wb_agc = nullptr; double* d_wb_agred = nullptr;
    int      h_ag = 0, ag_cap = 0;
    double*  d_wb_agdbg = nullptr;
    double*  d_wb_hg = nullptr;     // LIQ_WB_HG: per element 6 x (V_T, a^T_q[8]) = 150 doubles
    double*  d_pdev_red = nullptr;  // LIQ_PDEV energy partials
    double*  d_vox_s = nullptr;     // LIQ_WB_SURFK: per element 36 doubles, int_S N_q N_r dA_h
    double*  d_visc_red = nullptr;  // LIQ_VISC energy partials
    double*  d_vox_Vr = nullptr;   // LIQ_WB_GEOV per-element rest volume
    double*  d_vox_yc = nullptr; double* d_ybar = nullptr;   // LIQ_WB_YBAR: element fluid-centroid height, per-cell deposited mean height
    const double3* fp_xt = nullptr; const double* fp_mass = nullptr; int fp_done = -1;   // LIQ_FORCE_PROBE (per-term node forces at u = 0)
    double*  d_jf_num = nullptr; double* d_jf_den = nullptr; int* d_jf_flag = nullptr; double* d_jf_red = nullptr;   // LIQ_JFILT
    int*     d_cell2elem = nullptr; // LIQ_WB_G2P: cell -> voxel element (-1 none)
    int*     d_vox_wc = nullptr;    // LIQ_WB: per element 1 = wall-cut
    double3  last_gravity = make_double3(0.0, 0.0, 0.0);   // gravity of the current step (set in pre_step; LIQ_PDEV hydrostatic offset)
    double*  d_wb_cvgeo = nullptr;  // LIQ_WB: per cell geometric fluid volume without the surface cut
    double*  d_wb_hcol  = nullptr;  // LIQ_WB: per lattice column: water height of the body standing on the floor
    double* d_wb_hwt = nullptr;   // per-column smoothing weight (LIQ_WB_HSMOOTH_W)
    int*    d_wb_frz = nullptr;   // LIQ_WB_PHI_FREEZE: per lattice node, phi frozen
    double* d_wb_colD = nullptr;  // per-column deposit / geometric volume under h_col (LIQ_WB_COLMIN_POOL)
    double* d_wb_colG = nullptr;
    int*    d_wb_ovh = nullptr;   // LIQ_WB_OVH: per column: overhang-limited top
    int*    d_wb_thin = nullptr; int wb_thin_cap = 0;   // LIQ_WB_PHI_SIGN=11: per particle, its wall point's wall is thin
    int*    d_wb_lab = nullptr; unsigned char* d_wb_blk = nullptr; int* d_wb_pcomp = nullptr; int* d_wb_ccflag = nullptr; int wb_pcomp_cap = 0;   // LIQ_WB_PHI_SIGN=9
    int*    d_wb_ovr = nullptr; int* d_wb_ovr2 = nullptr; double* d_wb_hcol2 = nullptr;   // fill passes (resolved flags, double buffers)
    double*  d_wb_hraw  = nullptr;  // LIQ_WB: per column raw (unsmoothed) height
    int*     d_wb_blo = nullptr; int* d_wb_bhi = nullptr;   // LIQ_WB: per column: the body's cell range along up
    int      h_wb_slip = 0;
    int      h_wb = 0, h_wb_surf = 0, h_wb_allneg = 0, h_wb_empty = 0, h_wb_sep = 0;
    int      h_cc = 0;
    unsigned long long* d_gm_key = nullptr;   // LIQ_GHOST_MIRROR: per lattice node, (distance, particle) of its closest wall point
    int*     d_gm_front = nullptr;            // per lattice node: in front of some nearby wall plane (a fluid node of that side)
    int2*    d_gm_list = nullptr;             // per ghost term: (active-node index, wall-point particle)
    int*     d_gm_cnt  = nullptr;
    double*  d_gm_red  = nullptr;
    int      h_gm = 0, gm_cap = 0;
    bool     bh_init  = false;
    bool rep_on() const { return P.ug && P.ug_rep > 0.0; }                       // LIQ_UG_REP: the pair pipeline runs as repulsion
    bool coh_active() const { return (P.sti_gamma > 0.0 || rep_on()) && h_coh > 0 && d_coh_pairs && wfc_x; }
    double pair_k() const { return rep_on() ? P.ug_rep * P.kappa * h_spacing : 2.0 * P.sti_gamma * h_spacing * h_spacing; }   // pair energy scale (J/m^2 or J)
    double pair_r0() const { return (rep_on() ? P.ug_rep_r0 : P.sti_r0) * h_spacing; }
    double pair_rc() const { return (rep_on() ? P.ug_rep_r0 : P.sti_rc) * h_spacing; }
    // membrane surface tension (LIQ_STM): surface elements of this step (particle index + tangents), built in pre_step
    int*     d_stm_flag = nullptr;
    int*     d_stm_scan = nullptr;
    double3* d_stm_nrm  = nullptr;
    int*     d_stm_p    = nullptr;
    double3* d_stm_a    = nullptr;
    double3* d_stm_b    = nullptr;
    double*  d_stm_red  = nullptr;
    double*  d_stm_A    = nullptr;   // amode 1: per-particle element area
    double3* d_stm_n2   = nullptr;   // amode 1: neighbourhood-averaged normal
    int      h_stm = 0;
    bool stm_active() const { return P.stm_sigma > 0.0 && h_stm > 0 && d_stm_p && wfc_x; }
    int*     d_node_flag = nullptr;
    int*     d_node_ref   = nullptr;  // CPIC: node referenced by a particle stencil (masked or not)
    int*     d_node_ghost = nullptr;  // CPIC: referenced but massless -> fixed ghost this step
    int*     d_node_scan = nullptr;
    int*     d_node_slot = nullptr;
    int*     d_active_nodes = nullptr;
    int*     d_counts    = nullptr;  // [active_cells, active_nodes]
    // voxels (per active cell)
    int*     d_vox_nodes = nullptr;  // 8 GLOBAL vertex ids per voxel
    double*  d_vox_V     = nullptr;
    double*  d_vox_Jb    = nullptr;
    int3*    d_vox_ijk   = nullptr;
    double*  d_reduce    = nullptr;  // energy partials
    int*     d_ug_nodes  = nullptr;  // LIQ_UG: 8 vertex ids per particle (-1 = inactive node)
    double*  d_ug_hmass  = nullptr;  // LIQ_UG + LIQ_BDF2: scratch node mass / ref flags of the history P2G
    int*     d_ug_href   = nullptr;
    double*  d_ug_w      = nullptr;  //         8 weights per particle
    double3* d_ug_grad   = nullptr;  //         8 weight gradients per particle
    double*  d_ug_red    = nullptr;  //         energy partials (blocks of particles)
    double3* d_md_scaled = nullptr;  // scaled copy of moveDir for the convergence test (vertexNum)
    int      n_vertices_total = 0;
    // neighbour grid over the lattice cells (post-step push-apart / density)
    int*     d_pkey      = nullptr;  // cell key per particle (sorted)
    int*     d_pidx      = nullptr;  // particle index (sorted by key)
    int*     d_cstart    = nullptr;  // per lattice cell
    int*     d_cend      = nullptr;
    double3* d_push      = nullptr;  // push-apart displacement
    double*  d_rho       = nullptr;  // SPH density per particle
    double   h_spacing   = 0.0;      // seed spacing = particle_volume^(1/3)
    double   rho_ref     = 0.0;      // SPH density of the seed interior (calibration)
    // particle embedding (27-node stencil = separable two-hop weights) and contact chain rule
    int3*    d_pbase     = nullptr;  // centre-stencil base cell per particle
    double3* d_pfrac     = nullptr;  // fractions
    double*  d_Hpp       = nullptr;  // accumulated (p,p) contact Hessian, 9 per particle
    int*     d_pflag     = nullptr;  // particle has contact blocks this iteration
    int*     d_counters  = nullptr;  // [cross blocks, emit cursor]
    int      h_contact_particles = 0;
    long long h_expanded_blocks  = 0;
    int*     d_fallback  = nullptr;  // [8] clamp / fallback event counters: 0 cell-J clamp, 1 reanchor-J clamp, 2 fill-J clamp,
                                     //     3 non-finite J, 4 particle-J clamp, 5 push-apart domain clamp, 6 reinit-J clamp, 7 g2p domain clamp
    int      h_fallback[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   // last step's counts (zeroed in pre_step, read back in post_step)
                                     //     8 cpic regression near-singular, 9 cpic particle without a compatible node
    int*     d_pmask     = nullptr;  // per particle: 27-bit node compatibility mask (all ones away from walls)
    // 2b per-side cell state (cut cells): side of a particle w.r.t. a cell = sign of its wall normal along the cell's
    // dominant cut axis; set 0 = '+' side (or everything when the cell is not cut), set 1 = '-' side
    double3* d_cell_nabs = nullptr;  // per cell: sum |n| over the near-wall particles whose centre stencil touches it
    int*     d_cell_axis = nullptr;  // per cell: dominant axis of the cut normals (-1 = not cut)
    float*   d_corner_vote = nullptr; // per cell x 2 sides x 16: [0..8) compatible weight, [8..16) masked weight of the depositing
                                      // particles' own masks; a corner is a ghost only if NO particle of that side is compatible with it
                                      // (unanimity: a free node that some particle of the side moves with must stay the element's corner)
    double*  d_mc2 = nullptr; double* d_Vc2 = nullptr; double* d_piVc2 = nullptr; double* d_Jbc2 = nullptr;   // second deposit set
    int*     d_elem_cnt = nullptr; int* d_elem_off = nullptr; int* d_ghost_cnt = nullptr; int* d_ghost_off = nullptr;
    int3*    d_ghost_corner = nullptr;   // per private ghost slot: its lattice corner
    int*     d_slot_used    = nullptr;   // per active slot: referenced by at least one voxel element this step
    // per-side node slots: each lattice node near a wall carries a reference plane (wall point X, normal n flipped to a
    // positive dominant axis, from the lowest-index near particle referencing it); a particle addresses copy
    // side = ((x_p - X) . n < 0) of that node. Both copies of a node exist only across a thin wall.
    unsigned long long* d_nref_key = nullptr;  // per node: (quantised distance node -> wall point) << 32 | particle index; min = closest wall point
    int*     d_nref_has  = nullptr;  // per node: has a reference plane
    double3* d_nref_X    = nullptr;
    double3* d_nref_n    = nullptr;
    int*     d_corner_side = nullptr; // per cell x 2 sets x 8 corners: side copy the set's particles use at that corner
    int*     d_plabel    = nullptr;  // per particle: its side label from its OWN wall normal (dominant component < 0 -> 1), -1 = no wall point
    int*     d_nlab      = nullptr;  // per node: bitmask of the labels of the near particles referencing it (1 = label 0, 2 = label 1)
    int      h_orphans      = 0;         // active nodes anchored by no element -> fixed this step
    double*  d_pwinv     = nullptr;  // per particle: 1 / sum of the compatible stencil weights
    int      h_masked    = 0;        // particles with at least one masked node (last step)
    // j_mode 4: fill-fraction reconstruction (dense dx/4 sub-grid, 64 samples per cell)
    unsigned char* d_chi       = nullptr;  // bit0 = within fill_dilate of a particle, bit1 = within fill_own
    unsigned char* d_chi2      = nullptr;  // eroded region | own balls
    int*           d_near_flag = nullptr;  // cells within one cell of an active cell
    int*           d_near_scan = nullptr;
    int*           d_near_cells = nullptr;
    int*           d_near_count = nullptr;
    double*        d_fill      = nullptr;  // per active cell (diagnostic)
    int            h_near_cells = 0;
    // pump (dropper / pipette): particles within pump_r of the segment [a - 10 mm, b] get
    // an acceleration pump_accel along (b - a)/|b - a| before P2G (+ = up the tube, - = out
    // of the tip, 0 = off); a = tip, b = top of the tube cavity, world frame, set per step
    // Three actions on the particles of the region: (1) accel [m/s^2] along the axis added to the
    // velocity before P2G (cancels gravity on a column whose grid nodes are its own: +g = hold);
    // (2) conveyor [m/s] along the axis applied to the POSITIONS after G2P (robust against the
    // grid averaging the column with the bulk water across the thin glass wall: + = draw in,
    // - = squeeze out); (3) follow: fraction of the tube's own velocity (tip displacement between
    // consecutive set_pump calls / dt) added to the positions, so the sealed column travels
    // with the dropper.
    double3        pump_a = make_double3(0, 0, 0), pump_b = make_double3(0, 0, 1), pump_a_prev = make_double3(0, 0, 0);
    double         pump_r = 0.0, pump_accel = 0.0, pump_conveyor = 0.0, pump_follow = 0.0, pump_track = 0.0;
    // bore radius of the tube (<= 0: same as pump_r). The intake position conveyor only draws in
    // particles within bore - dhat of the axis, i.e. those under the opening, never under the glass
    // annulus (see k_pump).
    double         pump_bore = 0.0;
    bool           pump_has_prev = false;
    // chamber-pressure pump (mode 2, 2026-10-20 re-audit step 4): the ONLY input is the air-chamber volume V_cav
    // (bulb cavity + bore) prescribed or measured from the bulb; the liquid column in the bore is driven by the gas
    // pressure p = p0 V0 / V_air, V_air = V_cav - V_liq(column particles). Isothermal, 1-DOF implicit gas spring on the
    // column's mean axial velocity (stable at any dt). An unsealed tip (no liquid at the opening) resets V0 (p = p0).
    int            pump_mode = 0;              // 0 = kinematic (accel / conveyor / follow / track), 2 = chamber pressure
    double         pump_Vcav = 0.0, pump_p0 = 101325.0, pump_V0 = -1.0;
    double         h_pump_state[6] = {0, 0, 0, 0, 0, 0};   // V_air, p_gauge, V_liq, n_column, V0, sealed
    double*        d_pump_red = nullptr;       // [4] count, sum Vp0 J, sum v.axis, count at the opening
    void set_pump_chamber(double3 a, double3 b, double r, double Vcav, double p0, double bore)
    {
        pump_a_prev = pump_has_prev ? pump_a : a;
        pump_has_prev = true;
        pump_a = a; pump_b = b; pump_r = r; pump_bore = bore; pump_Vcav = Vcav; pump_p0 = p0;
        pump_accel = pump_conveyor = pump_follow = pump_track = 0.0; pump_mode = 2;
    }
    // track: per-step relaxation of the velocity of the particles INSIDE the bore toward the
    // tube's velocity (0 = off); with accel = +g this keeps the sealed column at rest in the tube
    void set_pump(double3 a, double3 b, double r, double accel, double conveyor, double follow, double track, double bore = 0.0)
    {
        pump_a_prev = pump_has_prev ? pump_a : a;
        pump_has_prev = true;
        pump_a = a; pump_b = b; pump_r = r; pump_accel = accel; pump_conveyor = conveyor; pump_follow = follow; pump_track = track;
        pump_bore = bore;
    }
    int  count_in_pump();
    // porous body (open-cell sponge): particles inside the tets of one 3D FEM body are coupled to the
    // material instead of colliding with its surface. Before P2G each inside particle (1) relaxes its
    // velocity toward the interpolated FEM vertex velocity (Darcy-like drag, rate porous_drag 1/s) and
    // (2) has gravity cancelled while its tet is not over-saturated: water volume in the tet <=
    // porosity x current tet volume (capillary retention). A squeezed tet (volume down) becomes
    // over-saturated: gravity returns and the MPM pressure pushes the excess out through the surface.
    int      porous_body = -1;                 // global collision body id, -1 = off
    int      porous_tet0 = 0, porous_tet1 = 0; // tet index range [tet0, tet1) of that body
    int      porous_v0 = 0, porous_nv = 0;     // global vertex range of that body (per-vertex saturation buffers)
    double*  d_vwater = nullptr;               // per body vertex: water volume of the incident tets (1/4 each)
    double*  d_vcap   = nullptr;               // per body vertex: porosity x current volume of the incident tets (1/4 each)
    double   porous_drag = 0.0, porous_phi = 1.0, porous_suction = 0.0;
    int*     d_ptet    = nullptr;              // per particle: containing tet (global index) or -1
    double4* d_pbary   = nullptr;              // barycentric coordinates in that tet
    int*     d_tet_cnt = nullptr;              // per porous tet: particles inside (this step)
    int*     d_porous_count = nullptr;         // [inside particles]
    double*  d_porous_vol   = nullptr;         // [current total volume of the porous body's tets]
    int      h_porous_inside = 0;
    double   h_porous_sat    = 0.0;            // global pore saturation = inside water volume / (porosity x current body volume)
    void set_porous(int body, int tet0, int tet1, int v0, int nv, double drag, double phi, double suction)
    {
        porous_body = body; porous_tet0 = tet0; porous_tet1 = tet1; porous_v0 = v0; porous_nv = nv;
        porous_drag = drag; porous_phi = phi; porous_suction = suction;
    }
    void porous_pre(device_TetraData& m, double dt, double3 gravity);
    void porous_post(device_TetraData& m, double dt);   // after G2P: inside particles follow their material point (kinematic retention)
    // grid wall boundary condition
    std::vector<LiquidContainer> h_cont;
    LiquidContainer* d_cont   = nullptr;
    const double*    d_abd_q  = nullptr;  // engine ABD state (12 doubles per body)
    const double*    d_abd_qv = nullptr;
    int*             d_wall_count = nullptr;
    int*             d_wall_t  = nullptr;  // per node slot: 0 free, 1 slip (normal pinned), 2 follow the body
    double3*         d_wall_n  = nullptr;  // wall normal (world)
    double3*         d_wall_xt = nullptr;  // body-carried target position x0 + dt v_body
    double*          d_wall_red = nullptr; // energy partials
    double           last_dt   = 0.0;
    bool wall_active() const { return d_wall_t && ((P.wall_bc && (!h_cont.empty() || P.ground_bc)) || (P.wb && P.wb_slip)); }
    int              h_wall_nodes = 0;
    void set_containers(const std::vector<LiquidContainer>& c, const double* abd_q, const double* abd_qv);
    // wall field constraint
    const uint3*   d_faces   = nullptr;   // engine collision surface (global vertex ids)
    int            n_faces   = 0;
    const double3* d_overt   = nullptr;   // step-start positions (o_vertexes)
    int*     d_tri_near  = nullptr;  // cells near fluid (dilated active flag)
    int*     d_tri_cnt   = nullptr;  // per cell triangle count / cursor
    int*     d_tri_start = nullptr;  // exclusive scan (n_cells + 1)
    int*     d_tri_list  = nullptr;  // binned triangle ids
    size_t   tri_list_cap = 0;
    int*     d_wp_valid = nullptr;   // per particle: has a wall point within wfc_r h (WFC energy)
    int*     d_wp_near  = nullptr;   // per particle: has a wall point within the (larger) mask radius
    double3* d_wp_X     = nullptr;   // the wall point (world, step start)
    int*     d_wp_tri   = nullptr;
    double3* d_wp_b     = nullptr;   // barycentric of the closest point
    double3* d_wp_n     = nullptr;   // wall normal into the fluid
    int3*    d_wp_cell  = nullptr;   // Q1 cell containing X_w
    double3* d_wp_xi    = nullptr;   // local coords of X_w in that cell
    int*     d_wp_vid   = nullptr;   // 8 node vertex ids (-1 inactive)
    int*     d_wp_act   = nullptr;   // LIQ_WFC_PRE: per particle, 1 = its wall point carried a wall reaction (g < 0) at the end of the previous step
    double*  d_wp_lam   = nullptr;   // LIQ_WFC_AL: per particle, multiplier of its wall point (units of K g: kg m)
    const double3* wfc_xt  = nullptr;   // inertial predictor (engine xTilta) and start-of-step velocities of this step, for the activity prediction
    const double3* wfc_vel = nullptr;
    int      h_wp_pre   = 0;         // wall points pre-activated in the last sampled first-iteration assembly (log)
    double*  d_wp_red   = nullptr;
    int      h_wp       = 0;
    double3* d_wfc_fsolid = nullptr;  // WFC gradient on the SOLID vertices at the last Newton assembly (force = -grad/dt^2)
    int      wfc_fsolid_n = 0;
    int      h_tri_total = 0;        // binned triangle-cell entries of this step (wall_geometry -> wfc_points)
    int*     d_gp_cut   = nullptr;   // per cell: wall-cut
    double3* d_gp_cutn  = nullptr;   // per cell: sum of the wall-point normals (gp_full 2/3)
    double*  d_gp_w     = nullptr;   // per pair: full-gradient weight (gp_full 6)
    int*     d_gp_flag  = nullptr;   // per (cell, +x/+y/+z) pair candidate flag
    int*     d_gp_scan  = nullptr;
    int2*    d_gp_pairs = nullptr;
    double*  d_gp_red   = nullptr;
    int      gp_cap     = 0;
    int      h_gp       = 0;
    void wall_geometry(device_TetraData& m);   // nearest solid surface per particle (+ compatibility masks), before P2G
    int  get_wall_forces(double* out3, int n, double dt) const;   // liquid -> solid WFC reaction (N) per solid vertex, 0 beyond
    void get_ghost_reaction(const double3* vertexes, double* out3);    // CPIC: total voxel-pressure force on the fixed ghost nodes (N)
    void wfc_points(device_TetraData& m);      // node binding of the wall points + ghost-penalty pairs, after the node slots
    bool wfc_active() const { return P.wfc && d_faces && n_faces > 0 && d_wp_valid; }
    void wfc_commit_activity(const double3* vertexes);   // LIQ_WFC_PRE: remember which wall points are loaded at the end of the step
    void completion_dir(double3* md, int vertex_num, double scale);   // LIQ_SPLIT_STEP: liquid-only remainder of a Newton direction
    double* local_step_alpha_buffer();                                 // LIQ_LOCAL_STEP: per-particle step bound (reset to 1), device
    int     local_step_filter(double3* md, const double3* vertexes, double cut, bool count_out);   // scale the node rows around limited particles
    double* d_ls_alpha = nullptr;
    double* d_ls_scale = nullptr;
    int     ls_scale_n = 0;
    bool cpic_active() const { return P.cpic && d_faces && n_faces > 0 && d_pmask; }
    // table stick
    int*           d_stuck     = nullptr;  // 1 = frozen on the table
    double3*       d_xstuck    = nullptr;  // frozen position
    int*           d_stuck_count = nullptr;
    int            h_stuck     = 0;
    // island sleep (connected water bodies)
    int*           d_isl_label   = nullptr;  // per cell: component label (min active cell index)
    int*           d_isl_cnt     = nullptr;  // per root cell: particles / not-slow / fast / min rest counter
    int*           d_isl_miss    = nullptr;
    int*           d_isl_fast    = nullptr;
    int*           d_isl_minrest = nullptr;
    int*           d_isl_flag    = nullptr;  // [0] labels changed, [1] asleep particle count
    int*           d_pisl        = nullptr;  // per particle: island root
    int*           d_prest       = nullptr;  // per particle: island rest counter (steps); >= N: held
    double3*       d_xold        = nullptr;  // start-of-step state, restored for held bodies
    double*        d_Jold        = nullptr;
    double*        d_vnmag       = nullptr;  // |solved grid velocity| at the particle
    int            isl_n_steps   = 0;
    void island_pass(double dt);
    // rest sleep
    int*           d_sleep       = nullptr;  // consecutive slow steps; >= sleep_n = asleep (held)
    int*           d_sleep_count = nullptr;

    ~LiquidSystem();

    void configure(const LiquidParams& p, const std::vector<double3>& x0, double particle_volume);
    // host-side estimate of the active node count of the seed (for the pool default)
    int  estimate_seed_nodes() const;
    void alloc_device();
    void free_device();

    // per step (SimEngine::step)
    void pre_step(device_TetraData& m, double dt, double3 gravity);
    void post_step(device_TetraData& m, double dt);
    // Optional read-only snapshot BEFORE updateVelocities overwrites o_vertexes.
    void hydrostatic_probe(const device_TetraData& m) const;

    void force_probe(const double3* vertexes, int fem_off, double dt, int global_offset, Eigen::Matrix3d* vals, int* rows, int* cols, const char* prefix);
    // Newton assembly (inside GIPC::computeGradientAndHessian elastic bracket)
    void assemble(const double3*   vertexes,
                  int              fem_global_hessian_index_offset,
                  double           dt,
                  int              global_offset,
                  Eigen::Matrix3d* triplet_values,
                  int*             row_ids,
                  int*             col_ids);
    int  ug_ns() const { return P.ug_bspline ? 27 : 8; }                 // LIQ_UG stencil nodes per particle
    bool   has_feasible_step() const { return P.ug && P.ug_feasible > 0.0; }
    double feasible_alpha(const double3* vertexes, const double3* dir);   // LIQ_UG: largest line-search step keeping every particle's det A > 0 (k_ug_feasible)
    void   feasible_alpha_device(const double3* vertexes, const double3* dir, double* out_slot);   // same, written into a device slot of the CCD alpha chain
    long long feasible_hits = 0;    // Newton iterations where the bound was below the contact alpha
    int  ug_nb() const { int k = ug_ns(); return k * (k + 1) / 2; }  // LIQ_UG upper-triangle Hessian blocks per particle (36 / 378)
    int  triplet_count() const { return P.ug ? ug_nb() * n_particles : 36 * h_elements + (wall_active() ? h_active_nodes : 0) + 45 * h_ag; }
    bool has_elements() const { return P.ug ? (n_particles > 0 && h_active_nodes > 0) : h_active_cells > 0; }
    void ug_pre(device_TetraData& m, double dt, double3 gravity);
    void ug_post(device_TetraData& m, double dt);
    // sum_c V_c psi(J_c)  (WITHOUT dt^2; the energy combine scales by dt^2)
    void energy_device_out(const double3* vertexes, double* out_slot);

    // line search: x_p = x_p(temp) + sum_i W_ip (x_i - x_i(temp))
    // (also rewrites xTilta_p = x_p so the engine's FEM kinetic term stays zero on particles)
    void apply_particle_positions(double3* vertexes, const double3* temp, double3* xTilta);
    // before CCD: moveDir_p = sum_i W_ip moveDir_i
    void apply_particle_movedir(double3* move_dir);
    // after the barrier/friction/ground assembly: gradient fb[p] -> nodes, and every
    // contact Hessian block touching a particle expanded onto its stencil nodes
    void chain_rule_contact(double3* fb, GIPCTripletMatrix& triplets);
    void chain_rule_contact_particles(double3* fb, GIPCTripletMatrix& triplets);
    const double3* wfc_x = nullptr;   // current vertex positions for the WFC assembly (set by the engine)

    // convergence test: copy of moveDir with the liquid rows scaled by conv_scale (nullptr = use moveDir)
    const double3* scaled_movedir(const double3* move_dir, int vertex_num);
    void           zero_rows(double3* move_dir, int vertex_num);   // LIQ_NEWTON_CAP: freeze the liquid rows (nodes + particle slots) of a Newton direction

    // queries (host copies)
    void get_particles(double* out_xyz) const;
    void get_particle_velocities(double* out_xyz) const;
    void get_particle_J(double* out) const;
    void set_particle_J(const double* in);
    void get_particle_density(double* out) const;

  private:
    void build_neighbor_grid();
    void push_apart_pass();
    void density_pass(bool calibrate);
    void cohesion_pairs();      // LIQ_STI: gate + pair list at the step start (after the wall pass)
    void stm_setup();           // LIQ_STM: surface elements at the step start (after the wall pass)
    void fill_fraction_pass();  // j_mode 4: rewrites d_Jbc of the active cells
};

}  // namespace gipc_liquid
