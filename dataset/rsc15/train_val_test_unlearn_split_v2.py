import pandas as pd
import numpy as np
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporal 0.8/0.1/0.1 split + pick multiple unlearning fractions of the TRAIN interactions."
    )
    parser.add_argument(
        "--unlearning_fractions",
        type=float,
        nargs="+",
        default=[0.00001],
        help=(
            "Space‐separated list of fractions of the TRAIN split to mark as 'unlearn'. "
            "E.g. `0.00001 0.0001 0.001`."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "popular"],
        default="random",
        help="“random” or “popular”. If “popular” is chosen, both popular‐based and unpopular‐based subsets will be produced.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed (for reproducibility).",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="rsc15.inter",
        help="Path to the full .inter file (tab‐separated).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # fix random seeds once
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 1) LOAD original full .inter file
    print(f"Loading data from '{args.input_file}'...")
    df = pd.read_csv(
        args.input_file,
        sep="\t",
        header=0,
        names=["session_id", "item_id", "timestamp"],
    )
    df["timestamp"] = df["timestamp"].astype(float)
    print(f"  → Loaded {len(df)} total interactions.")

    # 2) PER‐SESSION TEMPORAL SPLIT 0.8/0.1/0.1 (vectorized)
    print("Performing per‐session temporal split (0.8/0.1/0.1) in a vectorized way...")

    # 2.a) Sort by session_id, then timestamp
    df = df.sort_values(["session_id", "timestamp"], ascending=[True, True]).reset_index(drop=True)

    # 2.b) Compute each row’s rank (0,1,2,...) within its session
    df["session_index"] = df.groupby("session_id").cumcount()

    # 2.c) Compute session size for each row
    df["session_size"] = df.groupby("session_id")["session_id"].transform("size")

    # 2.d) Compute n_train and n_valid per session, broadcasted to every row
    #     floor(0.8 * size) → number of train‐rows in that session
    #     floor(0.1 * size) → number of validation‐rows in that session
    df["n_train"] = (0.8 * df["session_size"]).floordiv(1).astype(int)
    df["n_valid"] = (0.1 * df["session_size"]).floordiv(1).astype(int)

    # 2.e) Assign split based on session_index relative to n_train and n_valid
    #     if session_index < n_train  → "train"
    #     elif session_index < n_train + n_valid  → "validation"
    #     else → "test"
    df["split"] = np.where(
        df["session_index"] < df["n_train"],
        "train",
        np.where(
            df["session_index"] < (df["n_train"] + df["n_valid"]),
            "validation",
            "test"
        )
    )

    # 2.f) (Optional) print counts of each split
    total_train = int((df["split"] == "train").sum())
    total_val = int((df["split"] == "validation").sum())
    total_test = int((df["split"] == "test").sum())
    print(f"  → Split counts: train={total_train}, validation={total_val}, test={total_test}")

    # 2.g) Drop temporary columns so they don’t clutter our final output
    df.drop(columns=["session_index", "session_size", "n_train", "n_valid"], inplace=True)

    # 3) Initialize unlearn_flag = 0
    df["unlearn_flag"] = 0
    print("Initialized 'unlearn_flag' column to 0 for all rows.")

    # 4) Extract TRAIN indices & precompute for popular/unpopular if needed
    train_mask = df["split"] == "train"
    train_indices = df[train_mask].index.values
    n_train_interactions = len(train_indices)
    print(f"Collected {n_train_interactions} total TRAIN interactions.")

    if args.method == "popular":
        print("Computing popularity ordering (popular & unpopular) within TRAIN split...")
        train_df = df.loc[train_indices].copy()

        # 4.a) Build item → count in TRAIN split
        item_to_count = train_df["item_id"].value_counts().to_dict()
        train_df["item_popularity"] = train_df["item_id"].map(item_to_count)
        print("  → Computed item popularity for TRAIN split.")

        # 4.b) Compute average item_popularity per session in TRAIN
        df_sessions = (
            train_df.groupby("session_id")["item_popularity"]
            .mean()
            .reset_index(name="avg_popularity")
        )
        print("  → Computed average popularity per session.")

        # 4.c) Sort sessions ascending by avg_popularity (lowest popularity first)
        df_sessions.sort_values(by="avg_popularity", ascending=True, inplace=True)
        sessions_sorted_asc = df_sessions["session_id"].tolist()
        print("  → Sessions sorted by ascending average popularity.")

        # 4.d) Build session_id → list of row‐indices (in df.index) for TRAIN
        session_to_indices = {
            sess: idxs.tolist()
            for sess, idxs in train_df.groupby("session_id").indices.items()
        }

        # 4.e) Flatten into a single list of row‐indices: least‐popular first
        flatten_unpopular = []
        for sess in sessions_sorted_asc:
            flatten_unpopular.extend(session_to_indices[sess])
        print("  → Flattened TRAIN indices into 'unpopular' ordering.")

        # 4.f) For popular, reverse the above list
        flatten_popular = list(reversed(flatten_unpopular))
        print("  → Created 'popular' ordering by reversing the 'unpopular' list.")

        train_df.drop(columns=["item_popularity"], inplace=True)
        print("  → Dropped temporary 'item_popularity' column.")

    # 5) LOOP over unlearning fractions
    print("Starting loop over unlearning fractions...")
    for fraction in sorted(args.unlearning_fractions):
        print(f"\n---\nProcessing fraction = {fraction} ...")
        n_to_unlearn = int(n_train_interactions * fraction)
        print(f"  → Number to unlearn (train): {n_to_unlearn:,}")

        # Reset flags to zero before marking new fraction
        df["unlearn_flag"] = 0

        if n_to_unlearn <= 0:
            print("  → Fraction too small; no rows will be marked for unlearning.")
            # Still save splits with unlearn_flag=0
            if args.method == "random":
                suffixes = ["random"]
            else:  # popular
                suffixes = ["popular", "unpopular"]

            for suffix in suffixes:
                for s in ["train", "validation", "test"]:
                    out_df = df[df["split"] == s]
                    output_name = (
                        f"rsc15_{suffix}_fraction_{fraction}_seed_{args.seed}.{s}.inter"
                    )
                    out_df.to_csv(output_name, sep="\t", index=False, header=True)
                    print(f"   → Saved {len(out_df):,} rows to '{output_name}'")
            continue

        if args.method == "random":
            print("  → Method = random: sampling TRAIN indices at random...")
            sampled = np.random.choice(train_indices, size=n_to_unlearn, replace=False)
            df.loc[sampled, "unlearn_flag"] = 1
            print(f"  → Marked {len(sampled):,} TRAIN rows as unlearn (random).")

            # Save single set
            for s in ["train", "validation", "test"]:
                out_df = df[df["split"] == s]
                output_name = (
                    f"rsc15_random_fraction_{fraction}_seed_{args.seed}.{s}.inter"
                )
                out_df.to_csv(output_name, sep="\t", index=False, header=True)
                print(f"   → Saved {len(out_df):,} rows to '{output_name}'")

        else:  # args.method == "popular"
            # 5.a) Popular‐based unlearning
            df["unlearn_flag"] = 0
            print("  → Method = popular: marking most‐popular TRAIN rows first.")
            sampled_pop = flatten_popular[:n_to_unlearn]
            df.loc[sampled_pop, "unlearn_flag"] = 1
            print(f"  → Marked {len(sampled_pop):,} TRAIN rows as unlearn (popular).")
            for s in ["train", "validation", "test"]:
                out_df = df[df["split"] == s]
                output_name = (
                    f"rsc15_popular_fraction_{fraction}_seed_{args.seed}.{s}.inter"
                )
                out_df.to_csv(output_name, sep="\t", index=False, header=True)
                print(f"   → Saved {len(out_df):,} rows to '{output_name}'")

            # 5.b) Unpopular‐based unlearning
            df["unlearn_flag"] = 0
            print("  → Now marking least‐popular TRAIN rows (unpopular).")
            sampled_unpop = flatten_unpopular[:n_to_unlearn]
            df.loc[sampled_unpop, "unlearn_flag"] = 1
            print(f"  → Marked {len(sampled_unpop):,} TRAIN rows as unlearn (unpopular).")
            for s in ["train", "validation", "test"]:
                out_df = df[df["split"] == s]
                output_name = (
                    f"rsc15_unpopular_fraction_{fraction}_seed_{args.seed}.{s}.inter"
                )
                out_df.to_csv(output_name, sep="\t", index=False, header=True)
                print(f"   → Saved {len(out_df):,} rows to '{output_name}'")

    print("\nAll done!")


if __name__ == "__main__":
    main()
