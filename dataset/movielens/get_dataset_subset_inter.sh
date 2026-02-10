#!/bin/bash

# MovieLens dataset preprocessing script
# Converts rating.csv to RecBole .inter format

rating_file="rating.csv"
dataset_name="movielens"

# Step 1: Convert CSV to raw .inter format
python preprocess_movielens.py --input-file "${rating_file}" --output-file "${dataset_name}_raw.inter"

# Step 2: Deduplicate and filter (using the same deduplicate.py from amazon_reviews)
# Copy deduplicate.py if not present
if [ ! -f "deduplicate.py" ]; then
    echo "Copying deduplicate.py from amazon_reviews_grocery_and_gourmet_food..."
    cp ../amazon_reviews_grocery_and_gourmet_food/deduplicate.py .
fi

python deduplicate.py --input "${dataset_name}_raw.inter" --output "${dataset_name}.inter"

echo "Done! Generated ${dataset_name}.inter"
