#!/usr/bin/env python3
"""
Preprocess NowPlaying dataset to RecBole .inter format

Step 1 (Python): Convert CSV to .inter format
Step 2 (Bash): Sort and deduplicate using external sort
Step 3: Final nowp.inter file

Input: sessions_2018.csv with fields:
    - user_id (hash)
    - source (tweet source)
    - timestamp
    - track_title
    - artist_name
    - musicbrainz_id (item identifier)
    - session_id

Output: nowp.inter with tab-separated fields:
    - user_id:token
    - session_id:token
    - item_id:token (musicbrainz_id)
    - timestamp:float
"""

import csv
import sys
import subprocess
from datetime import datetime

def parse_timestamp(timestamp_str):
    """
    Convert timestamp string to Unix timestamp
    Format: 2018-02-04 05:39:33
    """
    try:
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        return int(dt.timestamp())
    except:
        return 0

def convert_to_inter(input_file, output_file):
    """
    Convert NowPlaying CSV to .inter format (memory efficient streaming)
    """
    print(f"Step 1: Converting CSV to .inter format")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")

    total_lines = 0
    skipped_lines = 0
    valid_lines = 0

    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # Write header
            outfile.write('user_id:token\tsession_id:token\titem_id:token\ttimestamp:float\n')

            # CSV reader (handle quoted fields)
            reader = csv.reader(infile, quotechar="'")

            for i, row in enumerate(reader, 1):
                if i % 1000000 == 0:
                    print(f"    Processed {i:,} lines...")

                total_lines += 1

                # Skip malformed lines
                if len(row) < 7:
                    skipped_lines += 1
                    continue

                user_id = row[0]
                session_id = row[6]
                timestamp_str = row[2]
                musicbrainz_id = row[5]

                # Skip header line
                if session_id == 'session_id':
                    continue

                # Skip if missing essential fields
                if not user_id or not musicbrainz_id or not session_id:
                    skipped_lines += 1
                    continue

                # Convert timestamp
                timestamp = parse_timestamp(timestamp_str)

                # Write to output file
                outfile.write(f'{user_id}\t{session_id}\t{musicbrainz_id}\t{timestamp}\n')
                valid_lines += 1

    print(f"\n  Conversion complete!")
    print(f"    Total lines: {total_lines:,}")
    print(f"    Skipped: {skipped_lines:,}")
    print(f"    Valid interactions: {valid_lines:,}")

    return valid_lines

def sort_and_deduplicate(input_file, output_file):
    """
    Sort and remove duplicates using bash sort (memory efficient)
    """
    print(f"\nStep 2: Sorting and deduplicating")
    print(f"  Using external sort (memory efficient)...")

    # Extract header
    with open(input_file, 'r') as f:
        header = f.readline()

    # Sort by all fields and remove duplicates, keeping header
    # Using external sort for memory efficiency
    try:
        # Sort everything except the header, then add header back
        cmd = f"(head -n 1 {input_file} && tail -n +2 {input_file} | sort -u) > {output_file}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"  Sorting complete!")

        # Count lines in both files
        with open(input_file, 'r') as f:
            original_count = sum(1 for _ in f) - 1  # Exclude header

        with open(output_file, 'r') as f:
            final_count = sum(1 for _ in f) - 1  # Exclude header

        duplicates = original_count - final_count
        print(f"    Original interactions: {original_count:,}")
        print(f"    After deduplication: {final_count:,}")
        print(f"    Duplicates removed: {duplicates:,}")

        return final_count

    except subprocess.CalledProcessError as e:
        print(f"  Error during sorting: {e}")
        sys.exit(1)

def main():
    input_file = 'sessions_2018.csv'
    temp_file = 'nowp_temp.inter'
    output_file = 'nowp.inter'

    print("NOWPLAYING DATASET PREPROCESSOR\n")
    print("This script converts the NowPlaying dataset to RecBole .inter format")
    print("Format: user_id, session_id, item_id (musicbrainz_id), timestamp")
    print("Using memory-efficient streaming and external sort\n")

    # Check if input file exists
    import os
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure sessions_2018.csv is in the current directory.")
        sys.exit(1)

    # Step 1: Convert CSV to .inter format
    convert_to_inter(input_file, temp_file)

    # Step 2: Sort and deduplicate
    final_count = sort_and_deduplicate(temp_file, output_file)

    # Step 3: Clean up temp file
    print(f"\nStep 3: Cleaning up")
    try:
        os.remove(temp_file)
        print(f"  Removed temporary file: {temp_file}")
    except:
        pass

    print(f"\nSuccessfully created {output_file}")
    print(f"Final dataset: {final_count:,} unique interactions")

if __name__ == "__main__":
    main()
