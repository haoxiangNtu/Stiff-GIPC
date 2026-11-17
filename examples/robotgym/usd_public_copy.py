#!/usr/bin/env python3
"""Copy an exported robotgym USD for public download: replace the lab-bench description on /World with the scene's own,
drop the empty Young's-modulus estimates, and save as .usdc under a public name.
    python examples/robotgym/usd_public_copy.py scene.usdc --scene sponge --out out/usd_public/robotgym_sponge.usdc"""
import argparse
from pxr import Usd

DOCS = {
    "bedsheet": "StiffGIPC bed-sheet fold: two FR3 arms pinch the hanging near edge of a 1.0 x 0.6 m sheet (IPC triangle-shell "
                "cloth, contact + friction only), lift it over the slab and release it on the far half. Rigid bodies animate by "
                "transform over static meshes; the sheet is a points animation.",
    "bottle": "StiffGIPC PET bottle squeeze: an FR3 squeezes a water-filled 0.3 mm PET shell (IPC thin shell with plastic hinges) "
              "and releases it; the dent stays. Water is the engine's MPM liquid, two-way coupled to the shell. Rigid bodies animate "
              "by transform, the bottle shell is a points animation, water is a Points prim (one point per particle).",
    "sponge": "StiffGIPC sponge water transfer: an FR3 soaks an FEM foam sponge in a basin (porous liquid coupling), carries it "
              "over a glass and wrings it. Rigid bodies animate by transform, the sponge is a points animation, water is a "
              "Points prim (one point per particle).",
    "sponge_regrasp": "StiffGIPC sponge regrasp: an FR3 runs four soak-carry-wring cycles with an FEM foam sponge in a basin "
                      "(porous liquid coupling); after each wring it puts the sponge back, regrasps it at a different position "
                      "along its length, soaks it again and wrings it over a cup. Rigid bodies animate by transform, the sponge "
                      "is a points animation, water is a Points prim (one point per particle).",
}

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--scene", required=True, choices=sorted(DOCS))
ap.add_argument("--out", required=True)
ap.add_argument("--tactile", default="", help="*_tactile_data.npz from tactile_export.py (or the recording.npz of a --tactile run): "
                "per-pad normal force, peak indentation and peak shear as time-sampled telemetry (the images stay in the npz)")
a = ap.parse_args()
stage = Usd.Stage.Open(a.src)
world = stage.GetPrimAtPath("/World")
dt = world.GetDocumentation().split("Physics dt=")[-1] if "Physics dt=" in world.GetDocumentation() else ""
world.SetDocumentation(DOCS[a.scene] + (" Physics dt=" + dt if dt else ""))
cd = dict(world.GetCustomData() or {})
if cd.get("youngs_modulus_estimates") in ("[]", None):
    cd.pop("youngs_modulus_estimates", None)
    world.ClearCustomData()
    for k, v in cd.items():
        world.SetCustomDataByKey(k, v)
water = stage.GetPrimAtPath("/World/Liquid/water")
if water:   # the lab-bench exporter predates the two-way coupling
    water.SetDocumentation("Engine MPM liquid particles, two-way coupled to the rigid/soft surfaces through IPC contact; "
                           "point = particle position, width = 0.6 x grid dx")
tel0 = stage.GetPrimAtPath("/World/Telemetry")
if tel0:   # lab-bench fields that are constant in these scenes: no Young's-modulus estimates, a single stage
    for nm, empty in (("young_estimate", ("",)), ("stage", ("0", "",))):
        at = tel0.GetAttribute(nm)
        if at and all(str(at.Get(tc)) in empty for tc in at.GetTimeSamples()):
            tel0.RemoveProperty(nm)
if a.tactile:
    import numpy as np
    from pxr import Sdf
    z = np.load(a.tactile, allow_pickle=True)
    if "normal_force_N" in z.files:          # tactile_export.py output
        names = [str(x) for x in z["pad_names"]]
        series = {"normal_force_N": z["normal_force_N"], "peak_indentation_mm": z["peak_indentation_mm"]}
        if "peak_shear_mm" in z.files:
            series["peak_shear_mm"] = z["peak_shear_mm"]
    else:                                    # raw recording: forces only
        names = [str(x) for x in z["tac_names"]]
        series = {"normal_force_N": z["tac_fn"]}
    tel = stage.GetPrimAtPath("/World/Telemetry") or stage.DefinePrim("/World/Telemetry")
    for key, arr in series.items():
        arr = np.asarray(arr, float)
        for k, nm in enumerate(names):
            at = tel.CreateAttribute(f"{nm}_{key}", Sdf.ValueTypeNames.Float)
            for fr in range(len(arr)):
                at.Set(float(arr[fr, k]), Usd.TimeCode(fr + 1))
    tel.SetDocumentation("Tactile fingertips: one VBTS gel pad (FEM, 15.9 x 14.0 x 2.5 mm) on each finger, drawn under "
                         "/World/Deformables/pad_*. Per pad: <pad>_normal_force_N = contact force along the pad normal, "
                         "<pad>_peak_indentation_mm = deepest indentation of the gel surface, <pad>_peak_shear_mm = largest "
                         "tangential displacement of the gel surface. The indentation images (200 x 176 px, 0.079375 mm/px) "
                         "and shear fields are in the companion *_tactile_data.npz.")
    world.SetDocumentation(world.GetDocumentation() + " The fingertips carry VBTS tactile gel pads (FEM) simulated in the same solve.")
stage.GetRootLayer().Export(a.out)
print(f"[usd_public] {a.src} -> {a.out}: {world.GetDocumentation()[:90]}...")
