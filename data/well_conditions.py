"""
Per-well downhole-condition integration (Track A).

This is the plug-and-play hook for REAL per-well data. Right now every well uses one
assumed configuration (wellbore.REF), which makes SI/kinetics a shared deterministic
transform of surface chemistry and is why physics adds no predictive value. Supplying
measured per-well downhole pressure/temperature is the only route to testing whether
physics carries orthogonal signal.

Workflow:
  1)  python data/well_conditions.py template
        -> writes data/well_conditions_template.csv : one row per Well_Number with
           empty TVD_ft, WHP_psi, BHP_psi, WHT_C, BHT_C columns. Fill in whatever you
           obtain (leave cells blank to fall back to REF for that value).

  2)  python data/well_conditions.py build data/well_conditions_template.csv
        -> merges your filled conditions onto every sample, runs the simulator PER WELL,
           and writes outputs/with_physics_perwell.parquet.

  3)  point ml/experiment.py at with_physics_perwell.parquet and re-run — that is the
      real test of physics value under measured conditions.
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physics.wellbore import PER_WELL_KEYS, simulate_dataframe

PREPARED = os.path.join(ROOT, "outputs", "prepared.parquet")
TEMPLATE = os.path.join(ROOT, "data", "well_conditions_template.csv")
OUT = os.path.join(ROOT, "outputs", "with_physics_perwell.parquet")


def make_template():
    df = pd.read_parquet(PREPARED)
    wells = (df[["Well_Number", "Field"]].drop_duplicates()
             .sort_values(["Field", "Well_Number"]).reset_index(drop=True))
    for k in PER_WELL_KEYS:
        wells[k] = ""        # blank -> falls back to REF for that value
    wells.to_csv(TEMPLATE, index=False)
    print(f"wrote {TEMPLATE}  ({len(wells)} wells, columns: {PER_WELL_KEYS})")
    print("Fill in any downhole conditions you have, then run:")
    print(f"  python data/well_conditions.py build {os.path.relpath(TEMPLATE, ROOT)}")


def build(csv_path):
    df = pd.read_parquet(PREPARED)
    cond = pd.read_csv(csv_path)
    # Well_Number is NOT globally unique (some numbers exist in both fields), so merge
    # on (Well_Number, Field) to avoid fanning out rows.
    merge_keys = ["Well_Number"]
    if "Field" in cond.columns and "Field" in df.columns:
        merge_keys.append("Field")
    keep = merge_keys + [k for k in PER_WELL_KEYS if k in cond.columns]
    cond = cond[keep].drop_duplicates(subset=merge_keys)
    for k in PER_WELL_KEYS:
        if k in cond.columns:
            cond[k] = pd.to_numeric(cond[k], errors="coerce")  # blanks -> NaN -> REF fallback
    merged = df.merge(cond, on=merge_keys, how="left")
    assert len(merged) == len(df), f"merge changed row count {len(df)} -> {len(merged)} (check keys)"
    filled = {k: int(merged[k].notna().sum()) for k in PER_WELL_KEYS if k in merged.columns}
    print(f"merged on {merge_keys}: {len(merged)} samples; non-empty per key: {filled}")
    out = simulate_dataframe(merged, per_well=True)
    out.to_parquet(OUT)
    print(f"wrote {OUT}  -> now re-run ml/experiment.py against it")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "template"
    if cmd == "template":
        make_template()
    elif cmd == "build":
        build(sys.argv[2] if len(sys.argv) > 2 else TEMPLATE)
    else:
        print(__doc__)
