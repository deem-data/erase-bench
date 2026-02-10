import sys
import os
import gzip
import json
import csv


def main():
    seen = set()
    temp_file = 'temp_unsorted.tsv'
    
    # First pass: deduplicate and write to temp file
    print("Pass 1: Deduplicating and writing to temp file...")
    with open(temp_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        
        record_count = 0
        for file in sorted(os.listdir(".")):
            if not file.endswith(".jsonl.gz"):
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
                        
                        key = (user_id, item_id, rating, timestamp)
                        if all(map(lambda x: x is not None, key)) and key not in seen:
                            seen.add(key)
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
    
    print(f"Total unique records: {record_count}")
    
    # Clear seen set to free memory
    del seen
    
    # Second pass: sort the temp file
    print("Pass 2: Sorting records...")
    all_records = []
    with open(temp_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        all_records = list(reader)
    
    all_records.sort(key=lambda x: (x['user_id'], float(x['timestamp'])))
    
    # Write final sorted output
    output_file = 'amazon_reviews.inter'
    print(f"Writing sorted output to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        writer = csv.DictWriter(f, fieldnames=['user_id', 'item_id', 'rating', 'timestamp'], delimiter='\t')
        writer.writerows(all_records)
    
    # Clean up temp file
    os.unlink(temp_file)
    
    print(f"Done! Written {len(all_records)} records to {output_file}")


if __name__ == "__main__":
    main()
