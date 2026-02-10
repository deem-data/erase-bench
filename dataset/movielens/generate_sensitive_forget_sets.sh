#!/bin/bash

# Configuration
dataset="movielens.inter"
rating_threshold=3.5

# Define sensitive movie categories
# Available categories: violence, extreme_violence, mental_health, explicit
# The new multi-signal detection uses titles, genres, and tags for better accuracy
declare -a categories=("health")

# For stricter matching, you can use:
# declare -a categories=("extreme_violence")
# Or customize min_signals per category in the script

# Random seeds for reproducibility
seeds=(2 3 5 7 11)

# Forget ratios (fraction of dataset to unlearn)
forget_ratios=(0.0001 0.00001 0.000001)

# Step 1: Identify sensitive movies for each category using multi-signal detection
echo "Step 1: Identifying sensitive movies using multi-signal detection..."
echo "  (Searching in titles, genres, and user tags)"
for category in "${categories[@]}"; do
    echo "  Processing category: ${category}"
    
    # Use the new category-based approach with multi-signal detection
    # This requires at least 2 signals (e.g., genre + title, or title + tag)
    # to reduce false positives from popular movies
    python identify_sensitive_movies.py \
        --category "${category}" \
        --output "sensitive_asins_${category}.txt"
    
    # For even stricter matching, you can increase min_signals:
    # python identify_sensitive_movies.py \
    #     --category "${category}" \
    #     --min-signals 3 \
    #     --output "sensitive_asins_${category}.txt"
done

# Step 2: Copy generate_forget_sets.py if not present
if [ ! -f "generate_forget_sets.py" ]; then
    echo "Copying generate_forget_sets.py from amazon_reviews_grocery_and_gourmet_food..."
    cp ../amazon_reviews_grocery_and_gourmet_food/generate_forget_sets.py .
fi

# Step 3: Generate forget sets for each combination of category, seed, and forget ratio
echo ""
echo "Step 2: Generating forget sets..."
for category in "${categories[@]}"; do
    sensitive_file="sensitive_asins_${category}.txt"

    # Check if sensitive movies file exists and has content
    if [ ! -f "$sensitive_file" ]; then
        echo "Warning: ${sensitive_file} not found, skipping category ${category}"
        continue
    fi

    if [ ! -s "$sensitive_file" ]; then
        echo "Warning: ${sensitive_file} is empty, skipping category ${category}"
        continue
    fi

    for seed in "${seeds[@]}"; do
        for forget_ratio in "${forget_ratios[@]}"; do
            echo "  Generating: category=${category}, seed=${seed}, ratio=${forget_ratio}"
            python generate_forget_sets.py \
                --sensitive-items "$sensitive_file" \
                --seed $seed \
                --forget-ratio $forget_ratio \
                --dataset $dataset \
                --rating-threshold $rating_threshold
        done
    done
done

echo ""
echo "Done! Generated forget sets for MovieLens dataset"
echo "Output files follow the pattern: ${dataset%.inter}_unlearn_pairs_sensitive_category_*_seed_*_unlearning_fraction_*.inter"
