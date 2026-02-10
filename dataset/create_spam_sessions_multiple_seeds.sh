#!/bin/bash

SEEDS=(2 3 5 7 11)
UNLEARNING_FRACTION=0.0001
N_TARGET_ITEMS=10

echo "Running spam session generation with ${#SEEDS[@]} different seeds"
echo "Seeds: ${SEEDS[@]}"
echo "Unlearning fraction: $UNLEARNING_FRACTION"
echo "Number of target items: $N_TARGET_ITEMS"

for SEED in "${SEEDS[@]}"
do
    echo ""
    echo "Running with seed: $SEED"
    
    python create_spam_sessions.py \
        --dataset rsc15 \
        --seed $SEED \
        --unlearning_fraction $UNLEARNING_FRACTION \
        --n_target_items $N_TARGET_ITEMS
    
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
    echo "  - rsc15_spam_sessions_dataset_rsc15_unlearning_fraction_${UNLEARNING_FRACTION}_n_target_items_${N_TARGET_ITEMS}_seed_${SEED}.inter"
    echo "  - rsc15_with_spam_dataset_rsc15_unlearning_fraction_${UNLEARNING_FRACTION}_n_target_items_${N_TARGET_ITEMS}_seed_${SEED}.inter"
    echo "  - spam_metadata_dataset_rsc15_unlearning_fraction_${UNLEARNING_FRACTION}_n_target_items_${N_TARGET_ITEMS}_seed_${SEED}.json"
    echo ""
done