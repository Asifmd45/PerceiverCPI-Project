"""
prepare_davis_novel_pair.py

Builds the Davis "novel pair" 5-fold splits for Perceiver CPI in the CSV format
train.py expects:  smiles,sequence,label,protein_name

Split logic follows newcompound_newprotein() from the original
prepare_data_for_setting.py (dmis-lab/PerceiverCPI):
  - compounds = sorted unique SMILES, proteins = sorted unique sequences
  - fold i test set = i-th consecutive 20% slice of each list
  - test  = pairs whose compound AND protein are both in the test slices
  - train = pairs whose compound AND protein are both outside the test slices
  - pairs mixing a test entity with a train entity are dropped
    (this is what makes it "novel pair": no test compound/protein is seen in training)
  - validation = random 20% of the training pairs (paper: train:val = 80:20)

Deliberate differences from the original script:
  - no hardcoded author paths
  - validation split is seeded (original shuffles without a seed)
  - prints sanity checks and asserts there is no train/test overlap

Input : Davis_dataset.csv with 5 columns, in this order:
        compound CID, protein name, SMILES, protein sequence, label
        (a header row is detected and skipped automatically)

Usage :
  python prepare_davis_novel_pair.py --input Davis_dataset.csv --out_dir data/davis_novel_pair
"""
import argparse
import os

import numpy as np
import pandas as pd

OUT_COLS = ["smiles", "sequence", "label", "protein_name"]


def load_davis(path):
    # sep=None lets pandas sniff comma vs tab
    df = pd.read_csv(path, header=None, dtype=str, sep=None, engine="python",
                     keep_default_na=False)
    if df.shape[1] < 5:
        raise SystemExit(f"Expected 5 columns, found {df.shape[1]}. Check the file.")
    df = df.iloc[:, :5].copy()
    df.columns = ["cid", "protein_name", "smiles", "sequence", "label"]

    # skip header row if the first label is not a number
    try:
        float(df.iloc[0]["label"])
    except ValueError:
        print(f"Header row detected and skipped: {df.iloc[0].tolist()}")
        df = df.iloc[1:]

    for c in df.columns:
        df[c] = df[c].str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    bad = df["label"].isna() | (df["smiles"] == "") | (df["sequence"] == "")
    if bad.any():
        print(f"WARNING: dropping {int(bad.sum())} rows with missing smiles/sequence/label")
        df = df[~bad]
    return df.reset_index(drop=True)


def report(df):
    n_c = df["smiles"].nunique()
    n_name = df["protein_name"].nunique()
    n_seq = df["sequence"].nunique()
    print("\n=== Dataset sanity report ===")
    print(f"rows (pairs)             : {len(df)}")
    print(f"unique compounds         : {n_c}")
    print(f"unique protein names     : {n_name}")
    print(f"unique protein sequences : {n_seq}")
    print(f"compounds x names        : {n_c * n_name}")
    print(f"duplicate (smiles, sequence) rows: {int(df.duplicated(['smiles', 'sequence']).sum())}")
    print(f"label min/max/mean       : {df.label.min():.3f} / {df.label.max():.3f} / {df.label.mean():.3f}")
    frac5 = np.isclose(df["label"], 5.0, atol=1e-3).mean() * 100
    print(f"labels ~= 5.0            : {frac5:.2f}%   (paper reports 69.64% for Davis)")
    long_frac = (df["sequence"].str.len() > 500).mean() * 100
    print(f"sequences longer than 500: {long_frac:.1f}% of rows (model fixes protein length to 500)")


def make_folds(df, n_folds=5, frac=0.2):
    compounds = sorted(df["smiles"].unique())
    proteins = sorted(df["sequence"].unique())
    n_c = int(round(frac * len(compounds)))
    n_p = int(round(frac * len(proteins)))
    ic = ip = 0
    for i in range(n_folds):
        test_c = set(compounds[ic:ic + n_c])
        test_p = set(proteins[ip:ip + n_p])
        ic += n_c
        ip += n_p
        if not test_c or not test_p:
            raise SystemExit(f"Fold {i} has an empty test slice - too few compounds/proteins.")
        c_in = df["smiles"].isin(test_c)
        p_in = df["sequence"].isin(test_p)
        yield i, df[~c_in & ~p_in], df[c_in & p_in], len(test_c), len(test_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to Davis_dataset.csv")
    ap.add_argument("--out_dir", default="data/davis_novel_pair")
    ap.add_argument("--seed", type=int, default=0, help="seed for the train/val split")
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    df = load_davis(args.input)
    report(df)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== Novel-pair folds ===")
    print(f"{'fold':<5}{'train':>8}{'val':>7}{'test':>7}{'test_cmpds':>12}{'test_prots':>12}")
    for i, train, test, n_tc, n_tp in make_folds(df, n_folds=args.n_folds):
        val = train.sample(n=int(round(0.2 * len(train))), random_state=args.seed)
        train_final = train.drop(val.index)

        # novel-pair guarantee: nothing in test appears in train/val
        for part in (train_final, val):
            assert set(part["smiles"]).isdisjoint(test["smiles"]), "compound leak!"
            assert set(part["sequence"]).isdisjoint(test["sequence"]), "protein leak!"

        for name, part in (("train", train_final), ("val", val), ("test", test)):
            part[OUT_COLS].to_csv(os.path.join(args.out_dir, f"novel_pair_{i}_{name}.csv"),
                                  index=False)
        print(f"{i:<5}{len(train_final):>8}{len(val):>7}{len(test):>7}{n_tc:>12}{n_tp:>12}")

    print(f"\nDone. Files written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
