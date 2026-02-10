#!/bin/bash
# Amazon Reviews 2023 Dataset - Metadata Download Script
# This script downloads all metadata files from the official McAuley Lab UCSD repository

echo "==============================================="
echo "Amazon Reviews 2023 Metadata - Download Script"
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
        echo "✓ Successfully downloaded: $filename"
    else
        echo "✗ Failed to download: $filename"
        return 1
    fi
    echo "----------------------------------------"
}

# Download all metadata files
download_file "meta_All_Beauty.jsonl.gz" "All Beauty"
download_file "meta_Amazon_Fashion.jsonl.gz" "Amazon Fashion"
download_file "meta_Appliances.jsonl.gz" "Appliances"
download_file "meta_Arts_Crafts_and_Sewing.jsonl.gz" "Arts, Crafts and Sewing"
download_file "meta_Automotive.jsonl.gz" "Automotive"
download_file "meta_Baby_Products.jsonl.gz" "Baby Products"
download_file "meta_Beauty_and_Personal_Care.jsonl.gz" "Beauty and Personal Care"
download_file "meta_Books.jsonl.gz" "Books"
download_file "meta_CDs_and_Vinyl.jsonl.gz" "CDs and Vinyl"
download_file "meta_Cell_Phones_and_Accessories.jsonl.gz" "Cell Phones and Accessories"
download_file "meta_Clothing_Shoes_and_Jewelry.jsonl.gz" "Clothing, Shoes and Jewelry"
download_file "meta_Digital_Music.jsonl.gz" "Digital Music"
download_file "meta_Electronics.jsonl.gz" "Electronics"
download_file "meta_Gift_Cards.jsonl.gz" "Gift Cards"
download_file "meta_Grocery_and_Gourmet_Food.jsonl.gz" "Grocery and Gourmet Food"
download_file "meta_Handmade_Products.jsonl.gz" "Handmade Products"
download_file "meta_Health_and_Household.jsonl.gz" "Health and Household"
download_file "meta_Health_and_Personal_Care.jsonl.gz" "Health and Personal Care"
download_file "meta_Home_and_Kitchen.jsonl.gz" "Home and Kitchen"
download_file "meta_Industrial_and_Scientific.jsonl.gz" "Industrial and Scientific"
download_file "meta_Kindle_Store.jsonl.gz" "Kindle Store"
download_file "meta_Magazine_Subscriptions.jsonl.gz" "Magazine Subscriptions"
download_file "meta_Movies_and_TV.jsonl.gz" "Movies and TV"
download_file "meta_Musical_Instruments.jsonl.gz" "Musical Instruments"
download_file "meta_Office_Products.jsonl.gz" "Office Products"
download_file "meta_Patio_Lawn_and_Garden.jsonl.gz" "Patio, Lawn and Garden"
download_file "meta_Pet_Supplies.jsonl.gz" "Pet Supplies"
download_file "meta_Software.jsonl.gz" "Software"
download_file "meta_Sports_and_Outdoors.jsonl.gz" "Sports and Outdoors"
download_file "meta_Subscription_Boxes.jsonl.gz" "Subscription Boxes"
download_file "meta_Tools_and_Home_Improvement.jsonl.gz" "Tools and Home Improvement"
download_file "meta_Toys_and_Games.jsonl.gz" "Toys and Games"
download_file "meta_Video_Games.jsonl.gz" "Video Games"
download_file "meta_Unknown.jsonl.gz" "Unknown"

echo ""
echo "==============================================="
echo "Download process completed!"
echo ""
echo "Summary of downloaded metadata files:"
echo "-------------------------------------"
ls -lh meta_*.jsonl.gz 2>/dev/null || echo "No files found. Please check download errors above."

total_size=$(du -sh meta_*.jsonl.gz 2>/dev/null | awk '{sum+=$1} END {print sum}')
file_count=$(ls -1 meta_*.jsonl.gz 2>/dev/null | wc -l)

echo ""
echo "Total metadata files downloaded: $file_count / 34"
echo "Total disk space used: $(du -sh meta_*.jsonl.gz 2>/dev/null | tail -1 | cut -f1)"
echo ""
echo "Next steps:"
echo "-----------"
echo "1. Identify sensitive items:"
echo "   python identify_sensitive_items.py --files meta_Grocery_and_Gourmet_Food.jsonl.gz"
echo ""
echo "2. Process reviews and create dataset:"
echo "   python preprocess_fast.py"
echo "   python deduplicate.py --input temp_unsorted.tsv --output amazon_reviews.inter --no-header"
echo ""
echo "3. Generate forget sets:"
echo "   python generate_forget_sets.py --forget-ratio 0.001 --seed 42"
