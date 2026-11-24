"""
split_dataset.py
-----------------
Utility to split a flat class-labeled image folder into train/val/test
sub-folders expected by src/preprocessing.py.

Before:
    raw_data/
        healthy/*.jpg
        cavity/*.jpg
        gum_disease/*.jpg
        oral_lesion/*.jpg

After (in --out_dir):
    data/train/<class>/...
    data/val/<class>/...
    data/test/<class>/...

Usage:
    python scripts/split_dataset.py --src raw_data --out data \
        --train 0.7 --val 0.15 --test 0.15
"""

import argparse
import os
import shutil
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", default="data")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    classes = [d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))]

    for cls in classes:
        files = os.listdir(os.path.join(args.src, cls))
        random.shuffle(files)
        n = len(files)
        n_train = int(n * args.train)
        n_val = int(n * args.val)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, split_files in splits.items():
            out_dir = os.path.join(args.out, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for f in split_files:
                shutil.copy2(os.path.join(args.src, cls, f), os.path.join(out_dir, f))

        print(f"{cls}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
