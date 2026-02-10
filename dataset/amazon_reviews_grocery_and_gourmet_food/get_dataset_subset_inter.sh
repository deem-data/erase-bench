#!/bin/bash

files_for_asins="../amazon_reviews/Grocery_and_Gourmet_Food.jsonl.gz"
subset="grocery_and_gourmet_food"

python preprocess_data.py --output_file "amazon_reviews_${subset}_raw.inter" --files ${files_for_asins}

python deduplicate.py --input "amazon_reviews_${subset}_raw.inter" --output "amazon_reviews_${subset}.inter"
