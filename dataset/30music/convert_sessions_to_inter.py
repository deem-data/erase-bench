#!/usr/bin/env python3
"""
Convert 30music sessions.idomaar to RecBole .inter format

Output format: user_id:token, session_id:token, item_id:token, timestamp:float
"""

import json
import sys
from pathlib import Path

def convert_sessions_to_inter(input_file, output_file):
    """
    Convert sessions.idomaar to .inter format

    Args:
        input_file: Path to sessions.idomaar file
        output_file: Path to output .inter file
    """
    print("Converting 30music sessions to .inter format")
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")
    print()

    interactions = []
    session_count = 0
    interaction_count = 0
    skipped_sessions = 0

    print("Processing sessions...")

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"  Processed {line_num:,} sessions, {interaction_count:,} interactions...")

            try:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    skipped_sessions += 1
                    continue

                session_id = parts[1]
                base_timestamp = int(parts[2])

                # The metadata and relations are in part 3, separated by space
                # Format: {"numtracks":N,"playtime":X} {"subjects":[...],"objects":[...]}
                json_parts = parts[3].split('} {')
                if len(json_parts) != 2:
                    skipped_sessions += 1
                    continue

                # Parse the relations part (second JSON object)
                relations_json = '{' + json_parts[1]
                relations = json.loads(relations_json)

                # Extract user ID
                if not relations.get('subjects') or len(relations['subjects']) == 0:
                    skipped_sessions += 1
                    continue

                user_id = relations['subjects'][0]['id']

                # Extract tracks (items)
                tracks = relations.get('objects', [])
                if not tracks:
                    skipped_sessions += 1
                    continue

                session_count += 1

                # Create an interaction for each track in the session
                for track in tracks:
                    track_id = track['id']
                    playstart = track.get('playstart', 0)

                    # Calculate timestamp: base session timestamp + playstart offset
                    timestamp = base_timestamp + playstart

                    # Store interaction: (user_id, session_id, item_id, timestamp)
                    interactions.append((user_id, session_id, track_id, timestamp))
                    interaction_count += 1

            except Exception as e:
                print(f"\nError processing line {line_num}: {e}")
                skipped_sessions += 1
                continue

    print(f"\nTotal sessions processed: {session_count:,}")
    print(f"Total interactions: {interaction_count:,}")
    print(f"Skipped sessions: {skipped_sessions:,}")
    if session_count > 0:
        print(f"Average tracks per session: {interaction_count/session_count:.2f}")

    # Sort by user_id, then by timestamp
    print("\nSorting interactions by user and timestamp...")
    interactions.sort(key=lambda x: (x[0], x[3]))

    # Write to .inter file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        # Write header
        f.write("user_id:token\tsession_id:token\titem_id:token\ttimestamp:float\n")

        # Write interactions
        for user_id, session_id, item_id, timestamp in interactions:
            f.write(f"{user_id}\t{session_id}\t{item_id}\t{timestamp}\n")

    print("Conversion complete!")
    print(f"\nOutput file: {output_file}")
    print(f"Total interactions: {interaction_count:,}")

    # Show sample of the output
    print("\nFirst 10 lines of output:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 11:  # Header + 10 lines
                print(line.rstrip())
            else:
                break

    return interaction_count

def main():
    # Set paths
    input_file = Path(__file__).parent / 'relations' / 'sessions.idomaar'
    output_file = Path(__file__).parent / '30music.inter'

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Convert
    convert_sessions_to_inter(input_file, output_file)

if __name__ == "__main__":
    main()
