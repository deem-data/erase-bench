import gzip
import json
import os
import argparse
import re

def extract_sensitive_items(sensitive_categories, output_file='sensitive_items.txt', input_files=None):
    """
    Extract item IDs from sensitive categories (e.g., meat products).
    Works with Amazon metadata files that contain product information.
    
    Args:
        sensitive_categories: List of category keywords to search for (case-insensitive, whole-word matching)
        output_file: File to write sensitive item IDs to
        input_files: List of specific metadata files to process. If None, process all metadata files
    """
    sensitive_items = set()
    
    # Compile regex patterns for whole-word matching (case-insensitive)
    # \b ensures we match whole words only
    keyword_patterns = [re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE) 
                        for keyword in sensitive_categories]
    
    print(f"Searching for items in categories: {sensitive_categories}")
    print("Note: This script processes METADATA files (meta_*.jsonl.gz), not review files")
    print("Note: Using whole-word matching (e.g., 'ipa' won't match 'hipa')")
    
    # Determine which files to process
    if input_files:
        files_to_process = input_files
        print(f"Processing specified files: {files_to_process}")
    else:
        # Process all metadata files in current directory (meta_*.jsonl.gz)
        files_to_process = [f for f in sorted(os.listdir(".")) if f.startswith("meta_") and f.endswith(".jsonl.gz")]
        if not files_to_process:
            # Fallback to any .jsonl.gz file
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
                    
                    # Get item ID (prefer parent_asin, fallback to asin)
                    item_id = data.get('parent_asin') or data.get('asin')
                    if not item_id:
                        continue  # Silently skip items without IDs
                    
                    # Build searchable text from various fields
                    
                    # 1. Main category (most important field)
                    main_category = data.get('main_category', '') or ''
                    
                    # 2. Categories field (list of category paths)
                    categories = data.get('categories', [])
                    if isinstance(categories, list):
                        categories = ' '.join(str(c) for c in categories if c is not None)
                    else:
                        categories = str(categories) if categories is not None else ''
                    
                    # 3. Title field
                    title = data.get('title', '') or ''
                    
                    # 4. Description field
                    description = data.get('description', [])
                    if isinstance(description, list):
                        description = ' '.join(str(d) for d in description if d is not None)
                    else:
                        description = str(description) if description is not None else ''
                    
                    # 5. Features field
                    features = data.get('features', [])
                    if isinstance(features, list):
                        features = ' '.join(str(f) for f in features if f is not None)
                    else:
                        features = str(features) if features is not None else ''
                    
                    # 6. Details field
                    details = data.get('details', {})
                    if isinstance(details, dict):
                        details = ' '.join(str(v) for v in details.values() if v is not None)
                    else:
                        details = str(details) if details is not None else ''
                    
                    # 7. Store name
                    store = data.get('store', '') or ''
                    
                    # Combine all text fields (main_category is included for better matching)
                    all_text = ' '.join([
                        main_category,
                        categories,
                        title,
                        description,
                        features,
                        details,
                        store
                    ])
                    
                    # Check if any sensitive category keyword is in the combined text (whole-word matching)
                    for pattern in keyword_patterns:
                        if pattern.search(all_text):
                            sensitive_items.add(item_id)
                            item_count += 1
                            break
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {file}: {e}")
                    continue
                except Exception:
                    # Silently skip lines with errors (e.g., None values in unexpected places)
                    continue
        
        print(f"  Processed {processed} products, found {item_count} sensitive items in {file}")
    
    # Write to output file
    print(f"\nTotal unique sensitive items: {len(sensitive_items)}")
    
    if len(sensitive_items) == 0:
        print("\nWarning: No sensitive items found!")
        print("Possible reasons:")
        print("  1. You're processing review files instead of metadata files")
        print("  2. The category keywords don't match the product descriptions")
        print("  3. The metadata files don't contain products in those categories")
        print("\nMake sure to:")
        print("  - Download metadata files (meta_*.jsonl.gz)")
        print("  - Use appropriate category keywords")
        print("  - Specify the correct metadata files with --files")
        print("  - For meat items, use: meta_Grocery_and_Gourmet_Food.jsonl.gz")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item_id in sorted(sensitive_items):
                f.write(f"{item_id}\n")
        
        print(f"Written sensitive item IDs to {output_file}")
    
    return sensitive_items

def main():
    parser = argparse.ArgumentParser(
        description='Identify sensitive items from Amazon metadata files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all metadata files (meta_*.jsonl.gz) with default meat categories
  python identify_sensitive_items.py
  
  # Process only specific metadata files
  python identify_sensitive_items.py --files meta_Grocery_and_Gourmet_Food.jsonl.gz
  
  # Process multiple specific metadata files
  python identify_sensitive_items.py --files meta_Grocery_and_Gourmet_Food.jsonl.gz meta_Prime_Pantry.jsonl.gz
  
  # Custom categories and specific files
  python identify_sensitive_items.py --categories meat beef pork seafood --files meta_Grocery_and_Gourmet_Food.jsonl.gz
  
  # Look for different sensitive categories (e.g., alcohol)
  python identify_sensitive_items.py --categories wine beer alcohol liquor vodka whiskey --output alcohol_items.txt
  
  # Search for IPA beer (won't match words containing 'ipa' like 'participant')
  python identify_sensitive_items.py --categories ipa --output ipa_items.txt

Important:
  - This script requires METADATA files (meta_*.jsonl.gz), NOT review files
  - Metadata files contain product information: title, description, categories
  - Review files only contain user reviews and ratings (not useful for this purpose)
  - Download metadata files from Amazon dataset separately if you don't have them
  - Keywords use whole-word matching (e.g., 'ipa' won't match 'hipa' or 'participant')
        """
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['meat', 'beef', 'pork', 'chicken', 'lamb', 'turkey'],
        help='Category keywords to search for (case-insensitive, whole-word matching)'
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
        help='Specific metadata .jsonl.gz files to process (default: all meta_*.jsonl.gz files in current directory)'
    )
    
    args = parser.parse_args()
    
    extract_sensitive_items(args.categories, args.output, args.files)

if __name__ == "__main__":
    main()
