"""LightgenBench splits.json -> the two index files trellis2/datasets/lightgen_slat.py reads.

The dataset classes are used unchanged. _build_parquet_samples takes
  * a parquet whose index is the sample id (rows with success != True are dropped), and
  * a split JSON {"train"|"val"|"test": {"indices": [...]}} of POSITIONAL indices into
    that filtered parquet.
The release names its shapes by uuid and lists them in splits.json, so rows are written
train, val, test in splits.json order and the three index lists are consecutive ranges.
Same files and names as TEXGen's scripts/lightgenbench/build_index.py writes.

  python data_toolkit/build_lightgenbench_index.py --splits <splits.json> --out-dir <dir>          # write both
  python data_toolkit/build_lightgenbench_index.py --splits <splits.json> --out-dir <dir> --check  # verify them

--check re-reads the two files the way the loader does and requires that the positional
indices select exactly the uuids of splits.json, in order. Exit 0: ok; 1: mismatch.
"""
import argparse
import json
import os
import sys

import pandas as pd

PARQUET = "df_lightgenbench.parquet"
SPLIT_JSON = "data_splits_lightgenbench.json"
SPLITS = ("train", "val", "test")


def read_splits(path):
    with open(path) as f:
        splits = json.load(f)
    if sorted(splits) != sorted(SPLITS):
        sys.exit(f"{path}: keys {sorted(splits)}, expected {sorted(SPLITS)}")
    uuids = [u for s in SPLITS for u in splits[s]]
    if len(set(uuids)) != len(uuids):
        sys.exit(f"{path}: a uuid appears more than once across the splits")
    return splits


def build(splits, out_dir):
    uuids = [u for s in SPLITS for u in splits[s]]
    df = pd.DataFrame({"ditem_dir": uuids, "success": True}, index=pd.Index(uuids, name="uuid"))
    df.to_parquet(os.path.join(out_dir, PARQUET))
    indices, start = {}, 0
    for s in SPLITS:
        indices[s] = {"indices": list(range(start, start + len(splits[s])))}
        start += len(splits[s])
    with open(os.path.join(out_dir, SPLIT_JSON), "w") as f:
        json.dump(indices, f)


def check(splits, out_dir):
    # Same read and filter as lightgen_slat._build_parquet_samples / _apply_split_indices.
    df = pd.read_parquet(os.path.join(out_dir, PARQUET))
    df = df[df["success"] == True]  # noqa: E712
    ids = [str(x) for x in df.index]
    with open(os.path.join(out_dir, SPLIT_JSON)) as f:
        indices = json.load(f)
    for s in SPLITS:
        if [ids[i] for i in indices[s]["indices"]] != splits[s]:
            print(f"MISMATCH: the {s} indices do not select the {s} uuids of splits.json")
            return 1
    print("[index] ok - pandas %s: %s rows; train/val/test = %s"
          % (pd.__version__, len(ids), "/".join(str(len(splits[s])) for s in SPLITS)))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    splits = read_splits(args.splits)
    if not args.check:
        os.makedirs(args.out_dir, exist_ok=True)
        build(splits, args.out_dir)
    sys.exit(check(splits, args.out_dir))


if __name__ == "__main__":
    main()
