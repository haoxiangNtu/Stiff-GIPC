"""One line per test of a liquid suite folder (run_suite.sh), from the logs / summary files the tests already write.

    python examples/liquid/suite/summarize.py out/suite/v21 [out/suite/v2 ...]
"""
import json, os, re, sys

import numpy as np


def last(pattern, text, group=1, conv=float):
    m = re.findall(pattern, text)
    return conv(m[-1] if isinstance(m[-1], str) else m[-1][group - 1]) if m else None


def read(path):
    try:
        return open(path, errors="replace").read()
    except OSError:
        return ""


def summarize(d):
    rows = []
    z = [last(r"damping ratio ([0-9.]+)", read(os.path.join(d, f"m6_r{r}.log"))) for r in (1, 2, 3)]
    z = [v for v in z if v is not None]
    if z:
        rows.append(("m6 sloshing zeta", f"{np.mean(z):.3f} +- {np.std(z):.3f} (n {len(z)}: {', '.join(f'{v:.3f}' for v in z)})"))
    s = read(os.path.join(d, "m14", "summary.json"))
    if s:
        j = json.loads(s)
        rows.append(("m14 flask still", f"median {j['median_speed_mm_s']:.3f} p90 {j['p90_speed_mm_s']:.3f} max {j['max_speed_mm_s']:.1f} mm/s"))
    for tag, name in (("m15y30", "m15 tank 30 deg"), ("m15y0", "m15 tank aligned")):
        t = read(os.path.join(d, tag + ".log"))
        m = re.findall(r"\[m15\] (\{.*\})", t)
        if m:
            j = json.loads(m[-1])
            rows.append((name, f"median {j['median_speed_mm_s']:.3f} p90 {j['p90_speed_mm_s']:.3f} max {j['max_speed_mm_s']:.1f} mm/s"))
    m = re.findall(r"\[m16\] (\{.*\})", read(os.path.join(d, "m16.log")))
    if m:
        j = json.loads(m[-1])
        rows.append(("m16 sliding tank", f"relative median {j['rel_median_speed_mm_s']:.3f} p90 {j['rel_p90_speed_mm_s']:.3f} mm/s"))
    t = read(os.path.join(d, "m9.log"))
    m = re.findall(r"\[m9\] t=2\.400 .*?B med ([0-9.]+) p90 ([0-9.]+) max ([0-9.]+) mm/s.*?crossed (\d+)", t)
    if m:
        bm, bp, bx, cr = m[-1]
        rows.append(("m9 thin divider t 2.4", f"crossed {cr} | B median {float(bm):.1f} p90 {float(bp):.1f} max {float(bx):.0f} mm/s"))
    v = last(r"= ([0-9.]+) sqrt\(gH\)", read(os.path.join(d, "m12.log")))
    if v is not None:
        rows.append(("m12 dam-break front", f"{v:.2f} sqrt(gH) (experiments 1.48-1.74)"))
    t = read(os.path.join(d, "m13.log"))
    c = last(r"effective C_d .*?median ([0-9.]+)", t)
    if c is not None:
        rows.append(("m13 orifice C_d", f"{c:.2f} (sharp-edged ~0.61) | line-search exhaustions {t.count('budget exhausted')}"))
    t = read(os.path.join(d, "m4.log"))
    nw = [float(x) for x in re.findall(r"newton +([0-9.]+)/step", t)]
    m = re.findall(r"\[m4\] onset.*?: (\S+) deg; final inside ([0-9.]+)% ground ([0-9.]+)%; wall (\d+) s", t)
    if m:
        on, fin, gr, w = m[-1]
        rows.append(("m4 beaker pour", f"onset {on} deg, final inside {fin} % ground {gr} %, Newton/step max {max(nw):.0f} mean {np.mean(nw):.0f}, wall {w} s"))
    elif nw:
        rows.append(("m4 beaker pour", f"UNFINISHED, Newton/step max {max(nw):.0f} last {nw[-1]:.0f}"))
    t = read(os.path.join(d, "m10.log"))
    m = re.findall(r"^\s+([0-9.]+) \|.*?\|\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+) \|\s+([0-9.]+) \| crossed (\d+)", t, re.M)
    if m:
        tt, n, r90, vm, vp, nn, wall, cr = m[-1]
        rows.append(("m10 table spill", f"t {tt} s: r90 {r90} mm, v_p90 {vp} mm/s, crossed {cr}, wall {wall} min"))
    t = read(os.path.join(d, "m5.log"))
    m = re.findall(r"t= *([0-9.]+)s (\w+) .*?in bore +(\d+) .*?on table +(\d+)", t)
    if m:
        after = [int(b) for tt, ph, b, tb in m if ph == "hold2"]
        lift = [int(b) for tt, ph, b, tb in m if ph == "lift"]
        rows.append(("m5 dropper (scripted pump)", f"in bore after lift {lift[-1] if lift else '-'} / after move {after[0] if after else '-'}, on table {m[-1][3]}"))
    return rows


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(f"== {d}")
        env = read(os.path.join(d, "environment.txt")).split()
        if env:
            print("   " + " ".join(env))
        for name, val in summarize(d):
            print(f"   {name:28s} {val}")
