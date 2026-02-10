#!/bin/bash

# Run baseline models for implicit collaborative filtering
# These models are deterministic/memory-based, so seed doesn't matter
# We use seed=2 for consistency with other experiments
#
# IMPORTANT: Activate the conda environment BEFORE running this script:
#   conda activate env
#   ./run_baselines.sh

models=("Random" "Pop" "ItemKNN")
datasets=("amazon_reviews_grocery_and_gourmet_food" "goodreads" "movielens")
seed=2

# Get the Python executable from the current environment
PYTHON_PATH=$(which python)

echo "Running baseline models with seed=${seed}"
echo "Models: ${models[@]}"
echo "Dataset: ${datasets[@]}"
echo "Python: ${PYTHON_PATH}"
echo ""

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
        config_file="config_${model_lower}.yaml"
        session_name="${model_lower}_${dataset}"
        log_file="baselines_log/${model_lower}_${dataset}_seed${seed}.log"

        echo "Starting tmux session '${session_name}' for model: $model"

        tmux new -d -s "${session_name}" "cd $(pwd) && ${PYTHON_PATH} run_recbole.py --model $model --dataset $dataset --seed $seed --config_files $config_file 2>&1 | tee $log_file"
    done
done
