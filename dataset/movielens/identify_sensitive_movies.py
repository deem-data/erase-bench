import csv
import os
import argparse
import re
from collections import defaultdict, Counter


# Predefined sensitive category keywords (similar to NowPlaying approach)
SENSITIVE_CATEGORIES = {
    'violence': {
        'title_keywords': {
            'exact': ['kill', 'murder', 'death', 'blood', 'gun', 'weapon', 'war', 'fight'],
            'relaxed': [
                'violence', 'violent', 'brutal', 'torture', 'massacre', 'slaughter',
                'revenge', 'attack', 'shoot', 'assassin', 'gangster', 'mafia',
                'criminal', 'crime', 'homicide', 'execution', 'terrorist', 'battle',
                'combat', 'warfare', 'destruction', 'explosion', 'bomb', 'hostage'
            ]
        },
        'genre_keywords': ['Horror', 'Thriller', 'War', 'Crime'],
        'tag_keywords': [
            'violent', 'violence', 'gory', 'brutal', 'disturbing', 'graphic',
            'bloody', 'torture', 'murder', 'killing', 'war', 'combat', 'weapons'
        ],
        'description': 'Violent content and themes',
        'min_signals': 2  # Require at least 2 signals (genre + title, or title + tag, etc.)
    },
    'extreme_violence': {
        'title_keywords': {
            'exact': ['torture', 'massacre', 'slaughter', 'execution', 'homicide'],
            'relaxed': [
                'extreme violence', 'graphic violence', 'brutal murder', 'serial killer',
                'psychopath', 'sadistic', 'gore', 'cannibal', 'saw', 'hostel',
                'torture porn', 'snuff', 'beheading', 'dismemberment'
            ]
        },
        'genre_keywords': ['Horror'],  # More restrictive
        'tag_keywords': [
            'extreme violence', 'graphic violence', 'gory', 'torture porn',
            'disturbing', 'brutal', 'sadistic', 'serial killer', 'psychopath'
        ],
        'description': 'Extreme or graphic violent content',
        'min_signals': 2  # Require stricter matching
    },
    'health': {
        'title_keywords': {
            'exact': ['suicide', 'depression', 'ptsd', 'schizophrenia'],
            'relaxed': [
                'mental health', 'mental illness', 'psychiatric', 'psychiatrist',
                'bipolar', 'self harm', 'self-harm',
                'overdose', 'addiction', 'rehab',
                'insanity', 'insane', 'asylum', 'institutionalized', 'psychosis',
                'psychotic', 'mental hospital', 'mental institution'
            ]
        },
        'genre_keywords': [],  # Drama is too broad - removed to prevent false positives
        'tag_keywords': [
            'mental health', 'mental illness', 'suicide', 'suicidal', 'depression',
            'depressing', 'addiction', 'alcoholism', 'drug abuse',
            'psychological', 'psychiatry', 'therapy', 'ptsd', 'trauma',
            'schizophrenia', 'bipolar'
        ],
        'description': 'Mental health related content',
        'min_signals': 1  # Match on title or tag (without Drama genre catching everything)
    },
    'explicit': {
        'title_keywords': {
            'exact': ['sex', 'nude', 'erotic', 'porn'],
            'relaxed': [
                'explicit', 'adult', 'mature', 'sexual', 'xxx', 'nsfw',
                'uncensored', 'erotic', 'sensual', 'nudity'
            ]
        },
        'genre_keywords': [],
        'tag_keywords': [
            'explicit', 'adult', 'sexual', 'nudity', 'erotic', 'mature',
            'nsfw', 'xxx', 'uncensored'
        ],
        'description': 'Explicit sexual content',
        'min_signals': 1
    }
}


def extract_sensitive_movies_advanced(
    movie_file='movie.csv',
    tag_file='tag.csv',
    category='violence',
    output_file='sensitive_movies.txt',
    use_genres=True,
    use_titles=True,
    use_tags=True,
    min_signals=None
):
    """
    Extract movie IDs based on multiple signals: genres, titles, and user tags.
    Uses a scoring system that requires multiple signals to reduce false positives.

    Args:
        movie_file: Path to movie.csv (movieId, title, genres)
        tag_file: Path to tag.csv (userId, movieId, tag, timestamp)
        category: Predefined category name or 'custom'
        output_file: File to write sensitive movie IDs to
        use_genres: Whether to search in movie genres
        use_titles: Whether to search in movie titles
        use_tags: Whether to search in user tags
        min_signals: Minimum number of signals required (overrides category default)
    """

    # Get category configuration
    if category in SENSITIVE_CATEGORIES:
        cat_config = SENSITIVE_CATEGORIES[category]
    else:
        print(f"Warning: Category '{category}' not found. Using 'violence' as default.")
        cat_config = SENSITIVE_CATEGORIES['violence']

    min_signals = min_signals if min_signals is not None else cat_config.get('min_signals', 1)

    sensitive_movies = {}  # movie_id -> set of signals matched
    movie_titles = {}  # movie_id -> title (for reporting)

    # Compile regex patterns for title keywords
    title_exact_patterns = []
    title_relaxed_patterns = []
    
    if use_titles and 'title_keywords' in cat_config:
        title_kw = cat_config['title_keywords']
        if title_kw.get('exact'):
            exact_pattern = '|'.join(r'\b' + re.escape(kw.lower()) + r'\b' for kw in title_kw['exact'])
            title_exact_patterns = [re.compile(exact_pattern, re.IGNORECASE)]
        if title_kw.get('relaxed'):
            relaxed_pattern = '|'.join(re.escape(kw.lower()) for kw in title_kw['relaxed'])
            title_relaxed_patterns = [re.compile(relaxed_pattern, re.IGNORECASE)]

    # Compile regex patterns for genre keywords
    genre_patterns = []
    if use_genres and cat_config.get('genre_keywords'):
        genre_patterns = [re.compile(r'\b' + re.escape(kw.lower()) + r'\b', re.IGNORECASE)
                         for kw in cat_config['genre_keywords']]

    # Compile regex patterns for tag keywords
    tag_patterns = []
    if use_tags and cat_config.get('tag_keywords'):
        tag_patterns = [re.compile(re.escape(kw.lower()), re.IGNORECASE)
                       for kw in cat_config['tag_keywords']]

    # Load movie data and check genres/titles
    if os.path.exists(movie_file):
        print(f"Processing {movie_file}...")
        with open(movie_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = row['movieId']
                title = row.get('title', '')
                genres = row.get('genres', '')
                
                movie_titles[movie_id] = title
                signals = set()

                # Check genres
                if use_genres and genre_patterns:
                    for pattern in genre_patterns:
                        if pattern.search(genres):
                            signals.add('genre')
                            break

                # Check titles
                if use_titles and title:
                    title_lower = title.lower()
                    # Check exact patterns
                    for pattern in title_exact_patterns:
                        if pattern.search(title_lower):
                            signals.add('title_exact')
                            break
                    # Check relaxed patterns
                    if 'title_exact' not in signals:
                        for pattern in title_relaxed_patterns:
                            if pattern.search(title_lower):
                                signals.add('title_relaxed')
                                break

                if signals:
                    sensitive_movies[movie_id] = signals

        print(f"  Found {len(sensitive_movies)} movies with genre/title matches")

    # Load and check user tags
    if use_tags and tag_patterns and os.path.exists(tag_file):
        print(f"Processing {tag_file}...")
        tag_matches = defaultdict(set)  # movie_id -> set of matched tags
        
        with open(tag_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = row['movieId']
                tag_str = row.get('tag', '').lower()

                # Check if any sensitive tag keyword matches
                for pattern in tag_patterns:
                    if pattern.search(tag_str):
                        tag_matches[movie_id].add('tag')
                        break

        # Add tag signals to existing movies or create new entries
        tag_count = 0
        for movie_id, tag_signals in tag_matches.items():
            if movie_id in sensitive_movies:
                sensitive_movies[movie_id].update(tag_signals)
            else:
                sensitive_movies[movie_id] = tag_signals
                if movie_id not in movie_titles:
                    movie_titles[movie_id] = f"Movie {movie_id}"  # Placeholder
            tag_count += 1

        print(f"  Found {tag_count} movies with tag matches")

    # Filter by minimum signals required
    filtered_movies = {}
    signal_counts = Counter()
    
    for movie_id, signals in sensitive_movies.items():
        signal_count = len(signals)
        signal_counts[signal_count] += 1
        
        if signal_count >= min_signals:
            filtered_movies[movie_id] = signals

    # Print statistics
    print(f"\nSignal distribution:")
    for count in sorted(signal_counts.keys()):
        print(f"  {count} signal(s): {signal_counts[count]} movies")
    
    print(f"\nTotal movies matching criteria (min_signals={min_signals}): {len(filtered_movies)}")
    
    if len(filtered_movies) == 0:
        print("\nWarning: No sensitive movies found!")
        print("Possible reasons:")
        print("  1. The keywords don't match any movies")
        print("  2. The minimum signals threshold is too high")
        print("  3. The movie.csv or tag.csv files are not in the expected format")
        print("\nTry:")
        print("  - Lowering --min-signals")
        print("  - Using a different category")
        print("  - Checking that movie.csv and tag.csv exist")
    else:
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for movie_id in sorted(filtered_movies.keys(), key=int):
                f.write(f"{movie_id}\n")

        print(f"\nWritten sensitive movie IDs to {output_file}")
        
        # Show sample matches
        print(f"\nSample matches (first 10):")
        for i, (movie_id, signals) in enumerate(list(filtered_movies.items())[:10]):
            title = movie_titles.get(movie_id, f"Movie {movie_id}")
            print(f"  {movie_id}: {title[:60]:<60} [signals: {', '.join(sorted(signals))}]")

    return filtered_movies


def extract_sensitive_movies_by_genre(
    movie_file='movie.csv',
    tag_file='tag.csv',
    genres=None,
    tags=None,
    output_file='sensitive_movies.txt',
    use_genres=True,
    use_tags=False
):
    """
    Legacy function for backward compatibility.
    Extract movie IDs based on sensitive genres or user tags.
    """
    sensitive_movies = set()

    # Compile regex patterns for whole-word matching (case-insensitive)
    if genres:
        genre_patterns = [re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
                          for keyword in genres]
    else:
        genre_patterns = []

    if tags:
        tag_patterns = [re.compile(re.escape(keyword.lower()), re.IGNORECASE)
                        for keyword in tags]
    else:
        tag_patterns = []

    # Search in movie genres
    if use_genres and genre_patterns and os.path.exists(movie_file):
        print(f"Searching for movies with genres: {genres}")
        print(f"Processing {movie_file}...")

        genre_count = 0
        with open(movie_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = row['movieId']
                genre_str = row['genres']

                # Check if any sensitive genre keyword is in the genres
                for pattern in genre_patterns:
                    if pattern.search(genre_str):
                        sensitive_movies.add(movie_id)
                        genre_count += 1
                        break

        print(f"  Found {genre_count} movies with sensitive genres")

    # Search in user tags
    if use_tags and tag_patterns and os.path.exists(tag_file):
        print(f"Searching for movies with tags: {tags}")
        print(f"Processing {tag_file}...")

        tag_count = 0
        with open(tag_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = row['movieId']
                tag_str = row['tag']

                # Check if any sensitive tag keyword is in the tag
                for pattern in tag_patterns:
                    if pattern.search(tag_str):
                        if movie_id not in sensitive_movies:
                            sensitive_movies.add(movie_id)
                            tag_count += 1
                        break

        print(f"  Found {tag_count} additional movies with sensitive tags")

    # Write to output file
    print(f"\nTotal unique sensitive movies: {len(sensitive_movies)}")

    if len(sensitive_movies) == 0:
        print("\nWarning: No sensitive movies found!")
        print("Possible reasons:")
        print("  1. The genre/tag keywords don't match any movies")
        print("  2. The movie.csv or tag.csv files are not in the expected format")
        print("\nMake sure to:")
        print("  - Use appropriate genre/tag keywords")
        print("  - Check that movie.csv and tag.csv exist in the current directory")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for movie_id in sorted(sensitive_movies, key=int):
                f.write(f"{movie_id}\n")

        print(f"Written sensitive movie IDs to {output_file}")

    return sensitive_movies


def main():
    parser = argparse.ArgumentParser(
        description='Identify sensitive movies from MovieLens dataset using multi-signal detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined category with multi-signal detection (RECOMMENDED)
  python identify_sensitive_movies.py --category violence --output sensitive_movies_violence.txt
  
  # Use extreme_violence category (stricter criteria)
  python identify_sensitive_movies.py --category extreme_violence --output sensitive_movies_extreme.txt
  
  # Use health category
  python identify_sensitive_movies.py --category health --output sensitive_movies_health.txt
  
  # Customize minimum signals required
  python identify_sensitive_movies.py --category violence --min-signals 3 --output sensitive_movies_strict.txt
  
  # Disable title search (only use genres and tags)
  python identify_sensitive_movies.py --category violence --no-titles --output sensitive_movies.txt
  
  # Legacy mode: Find movies by genre only
  python identify_sensitive_movies.py --genres Horror Thriller --output sensitive_movies_horror.txt
  
  # Legacy mode: Find movies by user tags
  python identify_sensitive_movies.py --tags "violent" "disturbing" --use-tags --output sensitive_movies_violent.txt

Available categories:
  - violence: General violent content (requires 2+ signals)
  - extreme_violence: Extreme/graphic violence (requires 2+ signals, stricter)
  - health: Mental health related content (requires 1+ signal)
  - explicit: Explicit sexual content (requires 1+ signal)

Common MovieLens genres:
  Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama,
  Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller,
  War, Western, IMAX
        """
    )
    
    # New advanced mode arguments
    parser.add_argument(
        '--category',
        choices=list(SENSITIVE_CATEGORIES.keys()),
        default=None,
        help='Predefined sensitive category to use (violence, extreme_violence, health, explicit)'
    )
    parser.add_argument(
        '--min-signals',
        type=int,
        default=None,
        help='Minimum number of signals required (genre, title, tag). Overrides category default.'
    )
    parser.add_argument(
        '--no-titles',
        action='store_true',
        help='Disable title keyword search'
    )
    parser.add_argument(
        '--no-genres',
        action='store_true',
        help='Disable genre search'
    )
    parser.add_argument(
        '--no-tags',
        action='store_true',
        help='Disable tag search'
    )
    
    # Legacy mode arguments
    parser.add_argument(
        '--genres',
        nargs='+',
        default=None,
        help='[Legacy] Genre keywords to search for (case-insensitive, whole-word matching)'
    )
    parser.add_argument(
        '--tags',
        nargs='+',
        default=None,
        help='[Legacy] Tag keywords to search for in user tags (case-insensitive, partial matching)'
    )
    parser.add_argument(
        '--use-tags',
        action='store_true',
        help='[Legacy] Search in user tags (requires --tags)'
    )
    parser.add_argument(
        '--output',
        default='sensitive_movies.txt',
        help='Output file for sensitive movie IDs (default: sensitive_movies.txt)'
    )
    parser.add_argument(
        '--movie-file',
        default='movie.csv',
        help='Path to movie.csv (default: movie.csv)'
    )
    parser.add_argument(
        '--tag-file',
        default='tag.csv',
        help='Path to tag.csv (default: tag.csv)'
    )

    args = parser.parse_args()

    # Determine which mode to use
    if args.category:
        # Advanced multi-signal mode
        if not os.path.exists(args.movie_file):
            print(f"Error: Movie file '{args.movie_file}' not found!")
            return

        extract_sensitive_movies_advanced(
            movie_file=args.movie_file,
            tag_file=args.tag_file,
            category=args.category,
            output_file=args.output,
            use_genres=not args.no_genres,
            use_titles=not args.no_titles,
            use_tags=not args.no_tags,
            min_signals=args.min_signals
        )
    else:
        # Legacy mode
        if not args.genres and not (args.tags and args.use_tags):
            print("Error: Must specify either --category or (--genres or both --tags and --use-tags)")
            print("\nRecommended: Use --category for better results")
            print("Example: python identify_sensitive_movies.py --category violence")
            return

        if not os.path.exists(args.movie_file) and args.genres:
            print(f"Error: Movie file '{args.movie_file}' not found!")
            return

        if not os.path.exists(args.tag_file) and args.use_tags:
            print(f"Warning: Tag file '{args.tag_file}' not found!")
            print("Continuing with genre search only...")
            args.use_tags = False

        extract_sensitive_movies_by_genre(
            movie_file=args.movie_file,
            tag_file=args.tag_file,
            genres=args.genres,
            tags=args.tags,
            output_file=args.output,
            use_genres=args.genres is not None,
            use_tags=args.use_tags
        )


if __name__ == "__main__":
    main()
