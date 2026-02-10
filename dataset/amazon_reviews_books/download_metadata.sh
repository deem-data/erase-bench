#!/bin/bash

# Amazon Reviews 2023 Dataset - Books Metadata Download Script
# This script downloads the Books metadata file from the official McAuley Lab UCSD repository

echo "==============================================="
echo "Amazon Reviews 2023 Books Metadata - Download Script"
echo "==============================================="
echo "Downloading from: mcauleylab.ucsd.edu"
echo ""

# Base URL for the metadata dataset
BASE_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"

# Function to download a single file
download_file() {
    local filename="$1"
    local category="$2"
    local url="${BASE_URL}/${filename}"

    echo "Downloading: $category ($filename)"
    wget -c --show-progress -O "$filename" -T 60 --tries=3 "$url"

    if [ $? -eq 0 ]; then
        echo "Successfully downloaded: $filename"
    else
        echo "Failed to download: $filename"
        return 1
    fi
    echo "----------------------------------------"
}

# Download Books metadata file
download_file "meta_Books.jsonl.gz" "Books"

echo ""
echo "==============================================="
echo "Download process completed!"
echo ""
echo "Summary of downloaded metadata files:"
echo "-------------------------------------"
ls -lh meta_Books.jsonl.gz 2>/dev/null || echo "No files found. Please check download errors above."

total_size=$(du -sh meta_Books.jsonl.gz 2>/dev/null | cut -f1)

echo ""
echo "Total disk space used: $total_size"
echo ""
echo "Next steps:"
echo "-----------"
echo "1. Identify sensitive items (mental health - default):"
echo "   python identify_sensitive_items.py --files meta_Books.jsonl.gz"
echo ""
echo "2. Identify sensitive items (violence - alternative):"
echo "   python identify_sensitive_items.py --keywords violence murder war torture abuse assault --output sensitive_asins_violence.txt --files meta_Books.jsonl.gz"
echo ""
echo "3. Preprocess data (3-step memory-efficient pipeline):"
echo "   python convert_to_inter.py --output_file amazon_reviews_books_raw.inter --files Books.jsonl.gz"
echo "   python deduplicate.py --input amazon_reviews_books_raw.inter --output amazon_reviews_books_clean.inter"
echo "   python add_sessions.py --input_file amazon_reviews_books_clean.inter --output_file amazon_reviews_books.inter"
echo ""
echo "4. Generate forget sets:"
echo "   python generate_forget_sets.py --dataset amazon_reviews_books.inter --sensitive-items sensitive_asins_health.txt --forget-ratio 0.0001 --seed 2"
echo ""
