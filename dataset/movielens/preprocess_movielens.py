import sys
import os
import csv
import argparse
from datetime import datetime


def parse_timestamp(ts_str):
    """Parse MovieLens timestamp to Unix timestamp."""
    try:
        # MovieLens format: "2005-04-02 23:53:47"
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return str(int(dt.timestamp()))
    except:
        return ts_str


def main(args):
    input_file = args.input_file
    output_file = args.output_file

    print(f"Processing MovieLens ratings from {input_file}...")

    record_count = 0

    with open(input_file, 'r', encoding='utf-8') as in_f, \
         open(output_file, 'w', newline='', encoding='utf-8') as out_f:

        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')

        for row in reader:
            try:
                user_id = row['userId']
                item_id = row['movieId']
                rating = row['rating']
                timestamp = parse_timestamp(row['timestamp'])

                # Only write if all fields exist
                if all([user_id, item_id, rating, timestamp]):
                    writer.writerow({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': rating,
                        'timestamp': timestamp
                    })
                    record_count += 1

            except Exception as e:
                print(f"Error parsing row: {e}")
                continue

    print(f"Total records written: {record_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        default="rating.csv",
        help="Input MovieLens rating.csv file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="movielens_raw.inter",
        help="Output .inter file"
    )
    args = parser.parse_args()
    main(args)
