#!/bin/bash

# Script to generate fraud baskets for NBR datasets with multiple seeds
# Usage: ./create_fraud_baskets_nbr_multiple_seeds.sh [dataset] [poisoning_ratio] [config_file]

# Default values
DATASET=${1:-"dunnhumby"}  # Default to dunnhumby if not provided
POISONING_RATIO=${2:-0.01}  # Default to 0.01 if not provided
CONFIG_FILE=${3:-""}       # Optional config file
N_TARGET_ITEMS=10
ATTACK_TYPE="bandwagon"
TARGET_STRATEGY="unpopular"
SEEDS=(2 3 5 7 11)

echo "=========================================="
echo "Generating NBR fraud baskets"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Seeds: ${SEEDS[@]}"
echo "Poisoning ratio: $POISONING_RATIO"
echo "Number of target items: $N_TARGET_ITEMS"
echo "Attack type: $ATTACK_TYPE"
echo "Target strategy: $TARGET_STRATEGY"
if [ -n "$CONFIG_FILE" ]; then
    echo "Config file: $CONFIG_FILE"
fi
echo "=========================================="
echo ""

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

for SEED in "${SEEDS[@]}"
do
    echo ""
    echo "----------------------------------------"
    echo "Running with seed: $SEED"
    echo "----------------------------------------"
    
    # Build command
    CMD="python create_fraud_baskets_nbr.py \
        --dataset $DATASET \
        --attack $ATTACK_TYPE \
        --target_strategy $TARGET_STRATEGY \
        --poisoning_ratio $POISONING_RATIO \
        --n_target_items $N_TARGET_ITEMS \
        --seed $SEED"
    
    # Add config file if provided
    if [ -n "$CONFIG_FILE" ]; then
        CMD="$CMD --config_file $CONFIG_FILE"
    fi
    
    # Execute command
    $CMD
    
    if [ $? -eq 0 ]; then
        echo "✓ Seed $SEED completed successfully"
    else
        echo "✗ Seed $SEED failed"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All seeds completed!"
echo "=========================================="
echo ""
echo "Generated files for dataset '$DATASET':"
for SEED in "${SEEDS[@]}"
do
    echo "  - ${DATASET}_fraud_baskets_${ATTACK_TYPE}_${TARGET_STRATEGY}_ratio_${POISONING_RATIO}_seed_${SEED}.json"
    echo "  - ${DATASET}_fraud_metadata_${ATTACK_TYPE}_${TARGET_STRATEGY}_ratio_${POISONING_RATIO}_seed_${SEED}.json"
    echo ""
done
