#!/bin/bash
# bash script/internal/gate/vae_compare.sh <csvA> <csvB>
# Compares two VAE tables (per_shape.csv) on the gate arms (GATE_ARMS, default
# pretrained_pad,ours_pbr2emission) with the `seconds` timing column dropped, so a table with more arms
# (the committed reports/vae_recon_test204/per_shape.csv has six) compares on the gate's two. Each side
# must hold every gate arm, one row per (arm, shape), all with status ok; exits 0 only when the rows
# are then identical.
set -euo pipefail
python3 - "${1:?csvA}" "${2:?csvB}" "${GATE_ARMS:-pretrained_pad,ours_pbr2emission}" <<'PY'
import collections, csv, sys
a_path, b_path, arms = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
def rows(p):
    r = csv.reader(open(p)); h = next(r); keep = [i for i, c in enumerate(h) if c != "seconds"]
    arm, sha, st = h.index("arm"), h.index("sha256"), h.index("status")
    sel = [x for x in r if x[arm] in arms]
    bad = [f"{x[arm]}/{x[sha]}: {x[st]}" for x in sel if x[st] != "ok"]
    dup = [k for k, n in collections.Counter((x[arm], x[sha]) for x in sel).items() if n > 1]
    absent = arms - {x[arm] for x in sel}
    if bad or dup or absent:
        sys.exit(f"INCOMPLETE {p}: {len(bad)} rows not ok {bad[:3]}, {len(dup)} duplicates, arms absent {sorted(absent)}")
    return [h[i] for i in keep], sorted(tuple(x[i] for i in keep) for x in sel)
(ha, ra), (hb, rb) = rows(a_path), rows(b_path)
if ha != hb: sys.exit(f"HEADER DIFFERS: {ha} vs {hb}")
d = [(x, y) for x, y in zip(ra, rb) if x != y]
if len(ra) != len(rb) or d:
    print(f"VAE TABLE DIFFERS: {len(ra)} vs {len(rb)} rows, {len(d)} differing"); [print(" ", x, "\n ", y) for x, y in d[:5]]; sys.exit(1)
print(f"vae table identical: {len(ra)} rows on {sorted(arms)}, seconds ignored")
PY
