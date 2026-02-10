#!/bin/bash

# Amazon Reviews 2023 Dataset - Working Download Script
# This script downloads all review files from the official McAuley Lab UCSD repository

echo "==============================================="
echo "Amazon Reviews 2023 Dataset - Download Script"
echo "==============================================="
echo "Downloading from: mcauleylab.ucsd.edu"
echo ""

# Base URL for the dataset
BASE_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories"

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

# Download all review files
download_file "All_Beauty.jsonl.gz" "All Beauty"
download_file "Amazon_Fashion.jsonl.gz" "Amazon Fashion"
download_file "Appliances.jsonl.gz" "Appliances"
download_file "Arts_Crafts_and_Sewing.jsonl.gz" "Arts, Crafts and Sewing"
download_file "Automotive.jsonl.gz" "Automotive"
download_file "Baby_Products.jsonl.gz" "Baby Products"
download_file "Beauty_and_Personal_Care.jsonl.gz" "Beauty and Personal Care"
download_file "Books.jsonl.gz" "Books"
download_file "CDs_and_Vinyl.jsonl.gz" "CDs and Vinyl"
download_file "Cell_Phones_and_Accessories.jsonl.gz" "Cell Phones and Accessories"
download_file "Clothing_Shoes_and_Jewelry.jsonl.gz" "Clothing, Shoes and Jewelry"
download_file "Digital_Music.jsonl.gz" "Digital Music"
download_file "Electronics.jsonl.gz" "Electronics"
download_file "Gift_Cards.jsonl.gz" "Gift Cards"
download_file "Grocery_and_Gourmet_Food.jsonl.gz" "Grocery and Gourmet Food"
download_file "Handmade_Products.jsonl.gz" "Handmade Products"
download_file "Health_and_Household.jsonl.gz" "Health and Household"
download_file "Health_and_Personal_Care.jsonl.gz" "Health and Personal Care"
download_file "Home_and_Kitchen.jsonl.gz" "Home and Kitchen"
download_file "Industrial_and_Scientific.jsonl.gz" "Industrial and Scientific"
download_file "Kindle_Store.jsonl.gz" "Kindle Store"
download_file "Magazine_Subscriptions.jsonl.gz" "Magazine Subscriptions"
download_file "Movies_and_TV.jsonl.gz" "Movies and TV"
download_file "Musical_Instruments.jsonl.gz" "Musical Instruments"
download_file "Office_Products.jsonl.gz" "Office Products"
download_file "Patio_Lawn_and_Garden.jsonl.gz" "Patio, Lawn and Garden"
download_file "Pet_Supplies.jsonl.gz" "Pet Supplies"
download_file "Software.jsonl.gz" "Software"
download_file "Sports_and_Outdoors.jsonl.gz" "Sports and Outdoors"
download_file "Subscription_Boxes.jsonl.gz" "Subscription Boxes"
download_file "Tools_and_Home_Improvement.jsonl.gz" "Tools and Home Improvement"
download_file "Toys_and_Games.jsonl.gz" "Toys and Games"
download_file "Video_Games.jsonl.gz" "Video Games"
download_file "Unknown.jsonl.gz" "Unknown"

echo ""
echo "==============================================="
echo "Download process completed!"
echo ""
echo "Summary of downloaded files:"
echo "-----------------------------"
ls -lh *.jsonl.gz 2>/dev/null || echo "No files found. Please check download errors above."

total_size=$(du -sh . 2>/dev/null | cut -f1)
file_count=$(ls -1 *.jsonl.gz 2>/dev/null | wc -l)

echo ""
echo "Total files downloaded: $file_count / 34"
echo "Total disk space used: $total_size"

