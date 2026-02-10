import sys
import os
import gzip
import json
import csv
import argparse


def main(args):
    output_file = args.output_file
    
    print("Processing records (no deduplication)...")
    
    record_count = 0
    
    with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        files = args.files if args.files is not None else sorted(os.listdir("."))
        for file in files:
            if not file.endswith(".jsonl.gz") or "meta_" in file:
                continue
            
            print(f"  Processing {file}...")
            
            with gzip.open(file, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        user_id = data.get('user_id')
                        item_id = data.get('asin')
                        rating = data.get('rating')
                        timestamp = data.get('timestamp')
                        
                        # Only check if all fields exist
                        if all([user_id is not None, item_id is not None, 
                               rating is not None, timestamp is not None]):
                            writer.writerow({
                                'user_id': user_id,
                                'item_id': item_id,
                                'rating': rating,
                                'timestamp': timestamp
                            })
                            record_count += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {file}: {e}")
                        continue
    
    print(f"\nTotal records written: {record_count}")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
    )
    parser.add_argument(
        "--files",
        nargs='+',
        default=None,
    )
    args = parser.parse_args()
    main(args)
