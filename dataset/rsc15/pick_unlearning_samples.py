import pandas as pd
import numpy as np
import collections
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Process unlearning percentage.")
    parser.add_argument(
        "--unlearning_fractions", 
        type=float,
        nargs='+',
        default=[0.00001],
        help='Space-separated list of unlearning fractions (e.g., 0.00001 0.0001)'
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "popular", "unpopular"],
        default="random",
        help="how are the interactions picked"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="random seed"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    train_dataset_file = "rsc15.train.inter"
    df_train = pd.read_csv(train_dataset_file, sep="\t", header=0, names=["session_id", "item_id", "timestamp"])
    n_interactions_before_filtering = len(df_train)

    # filter out sessions with only 1 item, as they are not used for training
    session_counts = df_train["session_id"].value_counts()
    valid_sessions = session_counts[session_counts >= 2].index
    df_train = df_train[df_train["session_id"].isin(valid_sessions)].reset_index(drop=True)
    n_interactions = len(df_train)
    session_indices = df_train.groupby("session_id").indices

    print(f"length 1 session count: {n_interactions_before_filtering - n_interactions}")
    print(f"interactions left: {n_interactions}")


    if args.method in ["popular", "unpopular"]:
        item_to_session_and_timestamp = collections.defaultdict(list)
        for row in df_train.itertuples(index=False):
            item_to_session_and_timestamp[row.item_id].append((row.session_id, row.timestamp))

        item_interaction_counts = {item: len(sessions) for item, sessions in item_to_session_and_timestamp.items()}
        df_train["item_popularity"] = df_train["item_id"].map(item_interaction_counts)
        # df_sorted = df_train.sort_values(by="item_popularity", ascending=(args.method == "unpopular"))
        df_sessions = (
            df_train
            .groupby("session_id")["item_popularity"]
            .mean()
            .reset_index(name="avg_popularity")
        ).sort_values(by="avg_popularity", ascending=(args.method == "unpopular"))
        df_train = df_train.drop(columns="item_popularity")



    selected_idxs = []
    sessions_visited = 0

    for unlearning_fraction in sorted(args.unlearning_fractions):
        unlearn_count = int(n_interactions * unlearning_fraction)

        if args.method == "random":
            unlearn_rows = df_train.sample(n=unlearn_count, random_state=args.seed)
        else:
            selected_idxs = []
            row_count = 0
            
            for session in df_sessions["session_id"].iloc[sessions_visited:]:
                if row_count >= unlearn_count:
                    break
                sessions_visited += 1
                idx = session_indices.get(session, [])
                if len(idx) == 0:
                    continue

                selected_idxs.extend(idx)
                row_count += len(idx)

                # session_rows = df_train[df_train["session_id"] == session]
                # selected_rows.append(session_rows)
                # row_count += len(session_rows)

                # if row_count >= unlearn_count:
                #     break
            
            # unlearn_rows = pd.concat(selected_rows, ignore_index=True)
            unlearn_rows = df_train.iloc[selected_idxs].reset_index(drop=True)

        unlearn_rows = unlearn_rows.sort_values(by=["session_id", "timestamp"])
        unlearn_rows = unlearn_rows.rename(columns={
            "session_id": "user_id:token", 
            "item_id": "item_id:token", 
            "timestamp": "timestamp:float"
        })
        
        output_name = f"rsc15_unlearn_pairs_{args.method}_seed_{args.seed}_unlearning_fraction_{unlearning_fraction}.inter"
        unlearn_rows.to_csv(output_name, sep="\t", index=False, header=True)

if __name__ == "__main__":
    main()
