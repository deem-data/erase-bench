#!/bin/bash

# Run baseline models for session-based recommendation (SBR)
# These models are deterministic/memory-based, so seed doesn't matter much
# We use seed=2 for consistency with other experiments
#
# IMPORTANT: Activate the conda environment BEFORE running this script:
#   conda activate env
#   ./run_baselines_sbr.sh

models=("Pop_SBR" "SKNN_SBR")
datasets=("amazon_reviews_books" "30music" "nowp")
unlearning_fractions=("0.0001")
seed=2

# Map datasets to their sensitive categories
declare -A dataset_to_sensitive_categories=(
    ["amazon_reviews_books"]="health"
    ["30music"]="explicit"
    ["nowp"]="explicit"
)

# Get the Python executable from the current environment
PYTHON_PATH=$(which python)

echo "Running SBR baseline models with seed=${seed}"
echo "Models: ${models[@]}"
echo "Datasets: ${datasets[@]}"
echo "Unlearning fractions: ${unlearning_fractions[@]}"
echo "Python: ${PYTHON_PATH}"
echo ""

# Create log directory if it doesn't exist
mkdir -p baselines_log

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        # Get the sensitive category for this dataset
        sensitive_category="${dataset_to_sensitive_categories[$dataset]}"

        for unlearning_fraction in "${unlearning_fractions[@]}"; do
            model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
            config_file="config_${model_lower}.yaml"

            # Create session name and log file with sensitive category and unlearning fraction
            session_name="${model_lower}_${dataset}_${sensitive_category}_${unlearning_fraction}"
            log_file="baselines_log/${model_lower}_${dataset}_${sensitive_category}_uf${unlearning_fraction}_seed${seed}.log"

            echo "Starting tmux session '${session_name}' for model: $model on dataset: $dataset (sensitive: ${sensitive_category}, unlearning_fraction: ${unlearning_fraction})"

            # Run with task_type=SBR, sensitive_category, and unlearning_fraction
            tmux new -d -s "${session_name}" "cd $(pwd) && ${PYTHON_PATH} run_recbole.py --model $model --dataset $dataset --seed $seed --task_type SBR --sensitive_category $sensitive_category --unlearning_fraction $unlearning_fraction --config_files $config_file 2>&1 | tee $log_file"
        done
    done
done

echo ""
echo "All jobs started in tmux sessions."
echo "To monitor a session, use: tmux attach -t <session_name>"
echo "To list all sessions, use: tmux ls"
echo "To kill a session, use: tmux kill-session -t <session_name>"
