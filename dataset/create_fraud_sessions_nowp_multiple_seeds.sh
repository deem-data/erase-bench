#!/bin/bash

# Script to generate fraud sessions for nowp dataset with multiple seeds
# Similar to create_spam_sessions_multiple_seeds.sh but uses create_fraud_sessions_sbr.py

SEEDS=(2 3 5 7 11)
POISONING_RATIO=0.01
N_TARGET_ITEMS=10
ATTACK_TYPE="bandwagon"
TARGET_STRATEGY="unpopular"

echo "Running fraud session generation for nowp dataset with ${#SEEDS[@]} different seeds"
echo "Seeds: ${SEEDS[@]}"
echo "Poisoning ratio: $POISONING_RATIO"
echo "Number of target items: $N_TARGET_ITEMS"
echo "Attack type: $ATTACK_TYPE"
echo "Target strategy: $TARGET_STRATEGY"

for SEED in "${SEEDS[@]}"
do
    echo ""
    echo "Running with seed: $SEED"
    
    python create_fraud_sessions_sbr.py \
        --dataset nowp \
        --attack $ATTACK_TYPE \
        --target_strategy $TARGET_STRATEGY \
        --poisoning_ratio $POISONING_RATIO \
        --n_target_items $N_TARGET_ITEMS \
        --seed $SEED
    
    if [ $? -eq 0 ]; then
        echo "Seed $SEED completed successfully"
    else
        echo "Seed $SEED failed"
        exit 1
    fi
done

echo ""
echo "All seeds completed!"
echo ""
echo "Generated files:"
for SEED in "${SEEDS[@]}"
do
    echo "  - nowp_fraud_sessions_${ATTACK_TYPE}_${TARGET_STRATEGY}_ratio_${POISONING_RATIO}_seed_${SEED}.inter"
    echo "  - nowp_fraud_metadata_${ATTACK_TYPE}_${TARGET_STRATEGY}_ratio_${POISONING_RATIO}_seed_${SEED}.json"
    echo ""
done
