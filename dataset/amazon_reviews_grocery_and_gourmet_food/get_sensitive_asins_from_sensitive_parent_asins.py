import gzip
import json
import os
import argparse


def extract_asins_from_reviews(sensitive_parents_file, output_file='sensitive_asins.txt', input_files=None):
    """
    Extract ASINs from review files where the item has a sensitive parent_asin.
    This scans the actual interaction data (reviews) not metadata.
    
    Args:
        sensitive_parents_file: File containing sensitive parent_asin values (one per line)
        output_file: File to write asin values to
        input_files: List of specific review files to process. If None, process all review files
    """
    
    # Load sensitive parent ASINs
    print(f"Loading sensitive parent ASINs from {sensitive_parents_file}...")
    sensitive_parents = set()
    with open(sensitive_parents_file, 'r', encoding='utf-8') as f:
        for line in f:
            parent_asin = line.strip()
            if parent_asin:
                sensitive_parents.add(parent_asin)
    
    print(f"Loaded {len(sensitive_parents)} sensitive parent ASINs")
    
    if len(sensitive_parents) == 0:
        print("Error: No parent ASINs found in input file!")
        return set()
    
    # Determine which files to process
    if input_files:
        files_to_process = input_files
        print(f"Processing specified files: {files_to_process}")
    else:
        # Process all review .jsonl.gz files (excluding meta_ files)
        files_to_process = [f for f in sorted(os.listdir(".")) 
                          if f.endswith(".jsonl.gz") and not f.startswith("meta_")]
        print(f"Processing review files in current directory")
    
    if not files_to_process:
        print("Error: No review files found!")
        print("Make sure you have downloaded the review files (*.jsonl.gz, not meta_*.jsonl.gz)")
        return set()
    
    # Extract ASINs from reviews with sensitive parent ASINs
    sensitive_asins = set()
    
    for file in files_to_process:
        if not os.path.exists(file):
            print(f"Warning: File '{file}' not found, skipping...")
            continue
        
        if not file.endswith(".jsonl.gz"):
            print(f"Warning: File '{file}' is not a .jsonl.gz file, skipping...")
            continue
        
        print(f"Processing {file}...")
        matched_count = 0
        processed = 0
        
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed += 1
                    
                    # Get parent_asin and asin from review
                    parent_asin = data.get('parent_asin')
                    asin = data.get('asin')
                    
                    # If this review is for an item with a sensitive parent_asin, save its asin
                    if parent_asin in sensitive_parents and asin:
                        sensitive_asins.add(asin)
                        matched_count += 1
                    
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
        
        print(f"  Processed {processed} reviews, found {matched_count} reviews with sensitive parent ASINs")
    
    # Write to output file
    print(f"\nTotal unique sensitive ASINs found in review data: {len(sensitive_asins)}")
    
    if len(sensitive_asins) == 0:
        print("\nWarning: No matching ASINs found!")
        print("Possible reasons:")
        print("  1. No reviews exist for products with these parent ASINs")
        print("  2. The review files don't contain 'parent_asin' field")
        print("  3. There's a mismatch between metadata and review files")
        print("\nNote: Some review datasets may not include parent_asin field.")
        print("If reviews only have 'asin', you may need to use the metadata mapping approach.")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for asin in sorted(sensitive_asins):
                f.write(f"{asin}\n")
        
        print(f"Written {len(sensitive_asins)} sensitive ASINs to {output_file}")
        print(f"\nSummary:")
        print(f"  Input parent ASINs: {len(sensitive_parents)}")
        print(f"  Output ASINs (from actual reviews): {len(sensitive_asins)}")
    
    return sensitive_asins


def main():
    parser = argparse.ArgumentParser(
        description='Extract ASINs from review data that have sensitive parent ASINs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract ASINs from all review files
  python extract_asins_from_reviews.py --input sensitive_items.txt --output sensitive_asins.txt
  
  # Use specific review files
  python extract_asins_from_reviews.py \\
      --input sensitive_items.txt \\
      --output sensitive_asins.txt \\
      --files Grocery_and_Gourmet_Food.jsonl.gz

Workflow:
  1. Identify sensitive parent ASINs from metadata:
     python identify_sensitive_items.py --files meta_Grocery_and_Gourmet_Food.jsonl.gz
     → Creates: sensitive_items.txt (parent ASINs)
  
  2. Extract ASINs from actual review data:
     python extract_asins_from_reviews.py --input sensitive_items.txt --output sensitive_asins.txt
     → Creates: sensitive_asins.txt (ASINs actually used in your dataset)
  
  3. Generate forget sets:
     python generate_forget_sets.py --sensitive-items sensitive_asins.txt --forget-ratio 0.001
        """
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input file with sensitive parent ASINs (one per line)'
    )
    parser.add_argument(
        '--output',
        default='sensitive_asins.txt',
        help='Output file for ASINs from review data (default: sensitive_asins.txt)'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=None,
        help='Specific review .jsonl.gz files to process (default: all non-meta .jsonl.gz files)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    extract_asins_from_reviews(args.input, args.output, args.files)


if __name__ == "__main__":
    main()
