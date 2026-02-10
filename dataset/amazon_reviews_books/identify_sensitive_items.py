import gzip
import json
import os
import argparse
import re
from collections import defaultdict, Counter


# Predefined sensitive category keywords
SENSITIVE_CATEGORIES = {
    'health': {
        'keywords': {
            'exact': ['suicide', 'depression', 'anxiety', 'trauma', 'ptsd'],
            'relaxed': [
                'mental health', 'mental illness', 'self harm', 'self-harm',
                'therapy', 'psychiatric', 'breakdown', 'bipolar', 'schizophrenia',
                'medication', 'overdose', 'cutting', 'grief', 'bereavement'
            ]
        },
        'description': 'Mental health related content',
        'min_signals': 2  # Require title + description, or category + title, etc.
    },
    'violence': {
        'keywords': {
            'exact': ['murder', 'violence', 'war', 'torture', 'abuse', 'assault'],
            'relaxed': [
                'weapon', 'blood', 'kill', 'death', 'gun', 'shoot', 'attack',
                'revenge', 'massacre', 'slaughter', 'rape', 'sexual assault'
            ]
        },
        'description': 'Violent content and themes',
        'min_signals': 2
    },
    'explicit': {
        'keywords': {
            'exact': ['sex', 'erotic', 'porn', 'nude'],
            'relaxed': [
                'explicit', 'adult', 'sexual', 'xxx', 'nsfw', 'uncensored',
                'sensual', 'nudity', 'erotica'
            ]
        },
        'description': 'Explicit sexual content',
        'min_signals': 1
    }
}


def extract_sensitive_items_advanced(
    category='health',
    output_file='sensitive_items.txt',
    input_files=None,
    min_signals=None,
    use_title=True,
    use_description=True,
    use_categories=True,
    use_features=True
):
    """
    Extract item IDs for books on sensitive topics using multi-signal detection.
    Works with Amazon Books metadata files.

    Args:
        category: Predefined category name ('health', 'violence', 'explicit')
        output_file: File to write sensitive item IDs to
        input_files: List of specific metadata files to process
        min_signals: Minimum number of signals required (overrides category default)
        use_title: Whether to search in titles
        use_description: Whether to search in descriptions
        use_categories: Whether to search in categories
        use_features: Whether to search in features/details
    """
    
    # Get category configuration
    if category in SENSITIVE_CATEGORIES:
        cat_config = SENSITIVE_CATEGORIES[category]
    else:
        print(f"Warning: Category '{category}' not found. Using 'health' as default.")
        cat_config = SENSITIVE_CATEGORIES['health']
    
    min_signals = min_signals if min_signals is not None else cat_config.get('min_signals', 2)
    
    sensitive_items = {}  # item_id -> set of signals matched
    item_titles = {}  # item_id -> title (for reporting)
    
    # Compile regex patterns
    exact_patterns = []
    relaxed_patterns = []
    
    keywords = cat_config['keywords']
    if keywords.get('exact'):
        exact_pattern = '|'.join(r'\b' + re.escape(kw.lower()) + r'\b' for kw in keywords['exact'])
        exact_patterns = [re.compile(exact_pattern, re.IGNORECASE)]
    
    if keywords.get('relaxed'):
        relaxed_pattern = '|'.join(re.escape(kw.lower()) for kw in keywords['relaxed'])
        relaxed_patterns = [re.compile(relaxed_pattern, re.IGNORECASE)]
    
    # Determine which files to process
    if input_files:
        files_to_process = input_files
        print(f"Processing specified files: {files_to_process}")
    else:
        files_to_process = [f for f in sorted(os.listdir(".")) if f.startswith("meta_") and f.endswith(".jsonl.gz")]
        if not files_to_process:
            files_to_process = [f for f in sorted(os.listdir(".")) if f.endswith(".jsonl.gz")]
        print(f"Processing metadata files in current directory")
    
    if not files_to_process:
        print("Error: No metadata files found!")
        print("Please ensure you have downloaded the metadata files (meta_*.jsonl.gz)")
        return set()
    
    print(f"\nSearching for books in category: {category}")
    print(f"  Description: {cat_config['description']}")
    print(f"  Minimum signals required: {min_signals}")
    print(f"  Signals: title={use_title}, description={use_description}, categories={use_categories}, features={use_features}")
    
    # Process files
    for file in files_to_process:
        if not os.path.exists(file):
            print(f"Warning: File '{file}' not found, skipping...")
            continue
        
        if not file.endswith(".jsonl.gz"):
            print(f"Warning: File '{file}' is not a .jsonl.gz file, skipping...")
            continue
        
        print(f"\nProcessing {file}...")
        processed = 0
        
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed += 1
                    
                    # Get item ID
                    item_id = data.get('parent_asin') or data.get('asin')
                    if not item_id:
                        continue
                    
                    signals = set()
                    
                    # Signal 1: Title
                    if use_title:
                        title = data.get('title', '') or ''
                        if title:
                            item_titles[item_id] = title
                            title_lower = title.lower()
                            # Check exact patterns
                            for pattern in exact_patterns:
                                if pattern.search(title_lower):
                                    signals.add('title_exact')
                                    break
                            # Check relaxed patterns
                            if 'title_exact' not in signals:
                                for pattern in relaxed_patterns:
                                    if pattern.search(title_lower):
                                        signals.add('title_relaxed')
                                        break
                    
                    # Signal 2: Description
                    if use_description:
                        description = data.get('description', [])
                        if isinstance(description, list):
                            description = ' '.join(str(d) for d in description if d is not None)
                        else:
                            description = str(description) if description else ''
                        
                        if description:
                            desc_lower = description.lower()
                            for pattern in exact_patterns:
                                if pattern.search(desc_lower):
                                    signals.add('description_exact')
                                    break
                            if 'description_exact' not in signals:
                                for pattern in relaxed_patterns:
                                    if pattern.search(desc_lower):
                                        signals.add('description_relaxed')
                                        break
                    
                    # Signal 3: Categories
                    if use_categories:
                        main_category = data.get('main_category', '') or ''
                        categories = data.get('categories', [])
                        if isinstance(categories, list):
                            categories = ' '.join(str(c) for c in categories if c is not None)
                        else:
                            categories = str(categories) if categories else ''
                        
                        cat_text = f"{main_category} {categories}".lower()
                        if cat_text.strip():
                            for pattern in exact_patterns:
                                if pattern.search(cat_text):
                                    signals.add('category')
                                    break
                    
                    # Signal 4: Features/Details
                    if use_features:
                        features = data.get('features', [])
                        if isinstance(features, list):
                            features = ' '.join(str(f) for f in features if f is not None)
                        else:
                            features = str(features) if features else ''
                        
                        details = data.get('details', {})
                        if isinstance(details, dict):
                            details = ' '.join(str(v) for v in details.values() if v is not None)
                        else:
                            details = str(details) if details else ''
                        
                        feat_text = f"{features} {details}".lower()
                        if feat_text.strip():
                            for pattern in exact_patterns:
                                if pattern.search(feat_text):
                                    signals.add('features')
                                    break
                    
                    if signals:
                        if item_id in sensitive_items:
                            sensitive_items[item_id].update(signals)
                        else:
                            sensitive_items[item_id] = signals
                
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
        
        print(f"  Processed {processed} products")
    
    # Filter by minimum signals required
    filtered_items = {}
    signal_counts = Counter()
    
    for item_id, signals in sensitive_items.items():
        signal_count = len(signals)
        signal_counts[signal_count] += 1
        
        if signal_count >= min_signals:
            filtered_items[item_id] = signals
    
    # Print statistics
    print(f"\nSignal distribution:")
    for count in sorted(signal_counts.keys()):
        print(f"  {count} signal(s): {signal_counts[count]} items")
    
    print(f"\nTotal items matching criteria (min_signals={min_signals}): {len(filtered_items)}")
    
    if len(filtered_items) == 0:
        print("\nWarning: No sensitive items found!")
        print("Possible reasons:")
        print("  1. The keywords don't match any books")
        print("  2. The minimum signals threshold is too high")
        print("  3. The metadata files don't contain books on those topics")
        print("\nTry:")
        print("  - Lowering --min-signals")
        print("  - Using a different category")
        print("  - Checking that metadata files exist")
    else:
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for item_id in sorted(filtered_items):
                f.write(f"{item_id}\n")
        
        print(f"\nWritten sensitive item IDs to {output_file}")
        
        # Show sample matches
        print(f"\nSample matches (first 10):")
        for i, (item_id, signals) in enumerate(list(filtered_items.items())[:10]):
            title = item_titles.get(item_id, f"Book {item_id}")
            print(f"  {item_id}: {title[:60]:<60} [signals: {', '.join(sorted(signals))}]")
    
    return filtered_items


def extract_sensitive_items(sensitive_keywords, output_file='sensitive_items.txt', input_files=None):
    """
    Legacy function for backward compatibility.
    Extract item IDs for books on sensitive topics using keyword matching.
    """
    sensitive_items = set()
    
    # Compile regex patterns for whole-word matching (case-insensitive)
    keyword_patterns = [
        (keyword, re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE))
        for keyword in sensitive_keywords
    ]
    
    print(f"Searching for books with keywords: {sensitive_keywords}")
    print("Note: This script processes METADATA files (meta_*.jsonl.gz), not review files")
    print("Note: Using whole-word matching")
    
    # Determine which files to process
    if input_files:
        files_to_process = input_files
        print(f"Processing specified files: {files_to_process}")
    else:
        files_to_process = [f for f in sorted(os.listdir(".")) if f.startswith("meta_") and f.endswith(".jsonl.gz")]
        if not files_to_process:
            files_to_process = [f for f in sorted(os.listdir(".")) if f.endswith(".jsonl.gz")]
        print(f"Processing metadata files in current directory")
    
    if not files_to_process:
        print("Error: No metadata files found!")
        print("Please ensure you have downloaded the metadata files (meta_*.jsonl.gz)")
        return set()
    
    # Process files
    for file in files_to_process:
        if not os.path.exists(file):
            print(f"Warning: File '{file}' not found, skipping...")
            continue
        
        if not file.endswith(".jsonl.gz"):
            print(f"Warning: File '{file}' is not a .jsonl.gz file, skipping...")
            continue
        
        print(f"Processing {file}...")
        item_count = 0
        processed = 0
        
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed += 1
                    
                    item_id = data.get('parent_asin') or data.get('asin')
                    if not item_id:
                        continue
                    
                    # Build searchable text from various fields
                    main_category = data.get('main_category', '') or ''
                    categories = data.get('categories', [])
                    if isinstance(categories, list):
                        categories = ' '.join(str(c) for c in categories if c is not None)
                    else:
                        categories = str(categories) if categories is not None else ''
                    
                    title = data.get('title', '') or ''
                    description = data.get('description', [])
                    if isinstance(description, list):
                        description = ' '.join(str(d) for d in description if d is not None)
                    else:
                        description = str(description) if description is not None else ''
                    
                    features = data.get('features', [])
                    if isinstance(features, list):
                        features = ' '.join(str(f) for f in features if f is not None)
                    else:
                        features = str(features) if features is not None else ''
                    
                    details = data.get('details', {})
                    if isinstance(details, dict):
                        details = ' '.join(str(v) for v in details.values() if v is not None)
                    else:
                        details = str(details) if details is not None else ''
                    
                    store = data.get('store', '') or ''
                    
                    # Combine all text fields
                    all_text = ' '.join([
                        main_category,
                        categories,
                        title,
                        description,
                        features,
                        details,
                        store
                    ])
                    
                    # Check if any sensitive keyword is in the combined text
                    for keyword, pattern in keyword_patterns:
                        if pattern.search(all_text):
                            sensitive_items.add(item_id)
                            item_count += 1
                            break
                
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
        
        print(f"  Processed {processed} products, found {item_count} sensitive items in {file}")
    
    # Write to output file
    print(f"\nTotal unique sensitive items: {len(sensitive_items)}")
    
    if len(sensitive_items) == 0:
        print("\nWarning: No sensitive items found!")
        print("Possible reasons:")
        print("  1. You're processing review files instead of metadata files")
        print("  2. The keywords don't match the book descriptions")
        print("  3. The metadata files don't contain books on those topics")
        print("\nMake sure to:")
        print("  - Download metadata files (meta_*.jsonl.gz)")
        print("  - Use appropriate keywords")
        print("  - For Books dataset, use: meta_Books.jsonl.gz")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item_id in sorted(sensitive_items):
                f.write(f"{item_id}\n")
        
        print(f"Written sensitive item IDs to {output_file}")
    
    return sensitive_items


def main():
    parser = argparse.ArgumentParser(
        description='Identify sensitive books from Amazon Books metadata files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined category with multi-signal detection (RECOMMENDED)
  python identify_sensitive_items.py --category health --output sensitive_asins_health.txt
  
  # Use violence category
  python identify_sensitive_items.py --category violence --output sensitive_asins_violence.txt
  
  # Customize minimum signals required
  python identify_sensitive_items.py --category health --min-signals 3 --output sensitive_asins_strict.txt
  
  # Disable specific signals
  python identify_sensitive_items.py --category health --no-features --output sensitive_asins.txt
  
  # Legacy mode: Use custom keywords
  python identify_sensitive_items.py --keywords suicide depression anxiety --output sensitive_asins_health.txt

Available categories:
  - health: Mental health related content (requires 2+ signals)
  - violence: Violent content and themes (requires 2+ signals)
  - explicit: Explicit sexual content (requires 1+ signal)

Important:
  - This script requires METADATA files (meta_*.jsonl.gz), NOT review files
  - Multi-signal detection reduces false positives by requiring matches in multiple fields
  - Use --category for better results than --keywords
        """
    )
    
    # New advanced mode arguments
    parser.add_argument(
        '--category',
        choices=list(SENSITIVE_CATEGORIES.keys()),
        default=None,
        help='Predefined sensitive category to use (health, violence, explicit)'
    )
    parser.add_argument(
        '--min-signals',
        type=int,
        default=None,
        help='Minimum number of signals required (title, description, categories, features). Overrides category default.'
    )
    parser.add_argument(
        '--no-title',
        action='store_true',
        help='Disable title search'
    )
    parser.add_argument(
        '--no-description',
        action='store_true',
        help='Disable description search'
    )
    parser.add_argument(
        '--no-categories',
        action='store_true',
        help='Disable category search'
    )
    parser.add_argument(
        '--no-features',
        action='store_true',
        help='Disable features/details search'
    )
    
    # Legacy mode arguments
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=None,
        help='[Legacy] Keywords to search for (case-insensitive, whole-word matching)'
    )
    parser.add_argument(
        '--output',
        default='sensitive_items.txt',
        help='Output file for sensitive item IDs'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=None,
        help='Specific metadata .jsonl.gz files to process (default: all meta_*.jsonl.gz files)'
    )
    
    args = parser.parse_args()
    
    # Determine which mode to use
    if args.category:
        # Advanced multi-signal mode
        extract_sensitive_items_advanced(
            category=args.category,
            output_file=args.output,
            input_files=args.files,
            min_signals=args.min_signals,
            use_title=not args.no_title,
            use_description=not args.no_description,
            use_categories=not args.no_categories,
            use_features=not args.no_features
        )
    elif args.keywords:
        # Legacy mode
        extract_sensitive_items(args.keywords, args.output, args.files)
    else:
        print("Error: Must specify either --category or --keywords")
        print("\nRecommended: Use --category for better results")
        print("Example: python identify_sensitive_items.py --category health")
        return


if __name__ == "__main__":
    main()
