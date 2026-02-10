import csv
import argparse
from collections import defaultdict


def create_sessions_from_time_windows(user_interactions, time_window_seconds=3600):
    """
    Create sessions by grouping user interactions within time windows.

    Args:
        user_interactions: List of interactions for a user, sorted by timestamp
        time_window_seconds: Maximum time gap between interactions in same session (default: 1 hour)

    Returns:
        List of interactions with session_id added
    """
    if not user_interactions:
        return []

    sessions = []
    current_session = []
    session_id = 0

    for interaction in user_interactions:
        if not current_session:
            # Start new session
            current_session.append(interaction)
        else:
            # Check time gap from last interaction
            last_timestamp = float(current_session[-1]['timestamp'])
            current_timestamp = float(interaction['timestamp'])
            time_gap = current_timestamp - last_timestamp

            if time_gap <= time_window_seconds * 1000:  # Convert to milliseconds
                # Continue current session
                current_session.append(interaction)
            else:
                # Start new session
                # Assign session_id to all interactions in completed session
                for inter in current_session:
                    inter['session_id'] = f"{inter['user_id']}_session_{session_id}"
                sessions.extend(current_session)

                session_id += 1
                current_session = [interaction]

    # Handle last session
    if current_session:
        for inter in current_session:
            inter['session_id'] = f"{inter['user_id']}_session_{session_id}"
        sessions.extend(current_session)

    return sessions


def main(args):
    input_file = args.input_file
    output_file = args.output_file
    time_window_hours = args.time_window_hours
    time_window_seconds = time_window_hours * 3600

    print(f"Step 3: Adding sessions to cleaned data (time window: {time_window_hours} hours)...")

    # Load all interactions by user (already sorted by user_id and timestamp from deduplicate.py)
    user_interactions = defaultdict(list)
    total_interactions = 0

    print(f"Loading interactions from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline().strip()

        # Parse field names from header
        fields = [field.split(':')[0] for field in header.split('\t')]

        reader = csv.DictReader(f, fieldnames=fields, delimiter='\t')

        for row in reader:
            user_interactions[row['user_id']].append({
                'user_id': row['user_id'],
                'item_id': row['item_id'],
                'rating': row['rating'],
                'timestamp': row['timestamp']
            })
            total_interactions += 1

    print(f"Total interactions loaded: {total_interactions}")
    print(f"Total users: {len(user_interactions)}")

    # Create sessions for each user (data already sorted by timestamp within each user)
    print(f"\nCreating sessions (time window: {time_window_hours} hours)...")

    all_interactions_with_sessions = []
    total_sessions = 0

    for user_id, interactions in user_interactions.items():
        # Data is already sorted by timestamp from deduplicate.py
        # Create sessions
        sessions = create_sessions_from_time_windows(interactions, time_window_seconds)
        all_interactions_with_sessions.extend(sessions)

        # Count unique sessions for this user
        unique_sessions = set(inter['session_id'] for inter in sessions)
        total_sessions += len(unique_sessions)

    print(f"Total sessions created: {total_sessions}")
    print(f"Average interactions per session: {len(all_interactions_with_sessions) / total_sessions:.2f}")

    # Write to output file
    print(f"\nWriting to {output_file}...")

    with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
        # Write header with type annotations
        out_f.write('user_id:token\tsession_id:token\titem_id:token\trating:float\ttimestamp:float\n')

        writer = csv.DictWriter(
            out_f,
            fieldnames=['user_id', 'session_id', 'item_id', 'rating', 'timestamp'],
            delimiter='\t'
        )
        writer.writerows(all_interactions_with_sessions)

    print(f"\nStep 3 complete!")
    print(f"Total records written: {len(all_interactions_with_sessions)}")
    print(f"\nFinal output: {output_file}")
    print(f"\nNext steps:")
    print(f"  - Identify sensitive items: python identify_sensitive_items.py --files meta_Books.jsonl.gz")
    print(f"  - Generate forget sets: python generate_forget_sets.py --dataset {output_file} --sensitive-items sensitive_asins_health.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Step 3: Add sessions to deduplicated data based on time windows'
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default='amazon_reviews_books_clean.inter',
        help='Input deduplicated .inter file (default: amazon_reviews_books_clean.inter)'
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='amazon_reviews_books.inter',
        help='Output .inter file with sessions (default: amazon_reviews_books.inter)'
    )
    parser.add_argument(
        "--time_window_hours",
        type=float,
        default=1.0,
        help='Time window for session creation in hours (default: 1.0)'
    )
    args = parser.parse_args()
    main(args)
