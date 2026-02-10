#!/usr/bin/env python3
"""
Generate sensitive forget sets for test datasets.

This script automates the process of:
1. Copying necessary files from main dataset directories
2. Identifying sensitive items
3. Generating forget sets

Usage:
    python generate_test_forget_sets.py --dataset movielens_test
    python generate_test_forget_sets.py --dataset amazon_reviews_books_test
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def copy_file_if_needed(src, dst, description=""):
    """Copy file if destination doesn't exist."""
    if os.path.exists(dst):
        print(f"  {dst} already exists, skipping...")
        return True
    
    if not os.path.exists(src):
        print(f"  Warning: {src} not found, cannot copy {description}")
        return False
    
    shutil.copy2(src, dst)
    print(f"  Copied {os.path.basename(dst)}")
    return True


def setup_movielens_test(test_dir, main_dir):
    """Set up files needed for MovieLens test dataset."""
    print("Setting up MovieLens test dataset files...")
    
    files_to_copy = [
        ("identify_sensitive_movies.py", "identify_sensitive_movies.py"),
        ("generate_forget_sets.py", "generate_forget_sets.py"),
        ("movie.csv", "movie.csv"),
        ("tag.csv", "tag.csv"),
    ]
    
    # Try to get generate_forget_sets.py from movielens first, then fallback
    if not os.path.exists(os.path.join(main_dir, "generate_forget_sets.py")):
        alt_source = os.path.join(main_dir, "..", "amazon_reviews_grocery_and_gourmet_food", "generate_forget_sets.py")
        if os.path.exists(alt_source):
            files_to_copy[1] = (alt_source, "generate_forget_sets.py")
    
    for src_name, dst_name in files_to_copy:
        src_path = os.path.join(main_dir, src_name) if not os.path.isabs(src_name) else src_name
        dst_path = os.path.join(test_dir, dst_name)
        copy_file_if_needed(src_path, dst_path, dst_name)
    
    return True


def setup_amazon_reviews_books_test(test_dir, main_dir):
    """Set up files needed for Amazon Reviews Books test dataset."""
    print("Setting up Amazon Reviews Books test dataset files...")
    
    files_to_copy = [
        ("identify_sensitive_items.py", "identify_sensitive_items.py"),
        ("generate_forget_sets.py", "generate_forget_sets.py"),
        ("meta_Books.jsonl.gz", "meta_Books.jsonl.gz"),
    ]
    
    for src_name, dst_name in files_to_copy:
        src_path = os.path.join(main_dir, src_name)
        dst_path = os.path.join(test_dir, dst_name)
        copy_file_if_needed(src_path, dst_path, dst_name)
    
    return True


def run_shell_script(script_path, description):
    """Run a shell script and return success status."""
    if not os.path.exists(script_path):
        print(f"Error: {description} script not found: {script_path}")
        return False
    
    if not os.access(script_path, os.X_OK):
        os.chmod(script_path, 0o755)
    
    print(f"\nRunning {description}...")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["bash", script_path],
            cwd=os.path.dirname(script_path),
            check=False,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n{description} completed successfully!")
            return True
        else:
            print(f"\n{description} completed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"Error running {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate sensitive forget sets for test datasets'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['movielens_test', 'amazon_reviews_books_test'],
        help='Test dataset name'
    )
    parser.add_argument(
        '--skip-setup',
        action='store_true',
        help='Skip file setup (assume files are already copied)'
    )
    parser.add_argument(
        '--skip-sensitive-identification',
        action='store_true',
        help='Skip sensitive item identification (assume sensitive_asins_*.txt files exist)'
    )
    
    args = parser.parse_args()
    
    # Determine directories
    dataset_name = args.dataset.replace('_test', '')
    base_dir = Path(__file__).parent
    test_dir = base_dir / args.dataset
    main_dir = base_dir / dataset_name
    
    if not test_dir.exists():
        print(f"Error: Test dataset directory not found: {test_dir}")
        return 1
    
    if not main_dir.exists():
        print(f"Error: Main dataset directory not found: {main_dir}")
        return 1
    
    print(f"Generating forget sets for {args.dataset}")
    print(f"Test directory: {test_dir}")
    print(f"Main directory: {main_dir}")
    print("=" * 60)
    
    # Step 1: Set up necessary files
    if not args.skip_setup:
        print("\nStep 1: Setting up necessary files...")
        if args.dataset == 'movielens_test':
            setup_movielens_test(str(test_dir), str(main_dir))
        elif args.dataset == 'amazon_reviews_books_test':
            setup_amazon_reviews_books_test(str(test_dir), str(main_dir))
    else:
        print("\nSkipping file setup (--skip-setup)")
    
    # Step 2: Run the generation script
    script_path = test_dir / "generate_sensitive_forget_sets.sh"
    
    if not script_path.exists():
        print(f"\nError: Generation script not found: {script_path}")
        print("Please run the script manually or create it first.")
        return 1
    
    success = run_shell_script(str(script_path), "Forget set generation")
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS! Forget sets have been generated.")
        print(f"Check {test_dir} for output files.")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("Generation completed with warnings/errors.")
        print("Check the output above for details.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
