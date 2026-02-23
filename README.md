## ERASE – A Real-World Aligned Benchmark for Unlearning in Recommender Systems

This repository contains a **RecBole-based implementation** of the ERASE benchmark for machine unlearning in recommender systems. Experiments can be replicated locally (no external APIs required) using two entry points:

- **Training / evaluation (base + retrained baselines)**: `run_recbole.py`
- **Unlearning + evaluation of intermediate unlearning checkpoints**: `unlearn.py`

If you want to **avoid re-training** or compare against reference checkpoints, see the benchmark artifacts and dataset locations in:
- `https://github.com/deem-data/erase-bench/blob/main/artifacts.md`

### Code foundation

- **RecBole**: `https://github.com/RUCAIBox/RecBole`
- **Next-basket recommendation structure**: `https://github.com/liming-7/A-Next-Basket-Recommendation-Reality-Check/`

### Hardware notes

Our runs were performed with a wide range of CPU memory (dataset-dependent) and modern NVIDIA GPUs (e.g., A100 80GB). In most settings, the **GPU is the bottleneck**, while the CPU mainly needs enough memory to hold dataset structures.

## Project setup (local Python)

### Environment

- **Python**: 3.8+ (we used 3.8.20)

### Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
pip install -r requirements.txt
```

Notes:
- `requirements.txt` includes `torch_geometric`-related packages. If installation fails due to CUDA / wheel compatibility, install PyTorch first, then follow the official PyG installation instructions for your CUDA/PyTorch combination and re-run `pip install -r requirements.txt`.


### Container for reproducability

To get the exact environment used for experiments you can build a container using apptainer. Run ```apptainer build --force --fakeroot ERASE.sif ERASE.def``` to build the container ```ERASE.sif```. You can use this container python files, for example to unlearn:

```
        apptainer run  --nv --bind <path_to_saved>/:/opt/erase-bench/saved/ --bind <path_to_log>/:/opt/erase-bench/log/ --bind <path_to_dataset>/:/opt/erase-bench/dataset/ --bind <path_to_log_tensorboard>/:/opt/erase-bench/log_tensorboard/ --bind <path_to_configs>/:/opt/erase-bench/configs/ ERASE.sif unlearn.py <params_for_the_python_script>
```


## Re-running the benchmark

Runs are parameterized by:

- **YAML configs** (e.g., `config_*.yaml`)
- **Command-line overrides** (seed, task type, unlearning fraction, scenario flags, algorithm hyperparameters)

To reproduce a run, reuse the same config file(s) and CLI arguments.

### Outputs

- **Logs**: by default under `./logs/` (depending on logging config)
- **Checkpoints**: by default under `./saved/` (RecBole’s `checkpoint_dir`)

## Training a model (base model)

Use `run_recbole.py` to train a model on a dataset. Example (session-based recommendation):

```bash
python run_recbole.py \
  --model SRGNN \
  --dataset amazon_reviews_books \
  --task_type SBR \
  --config_files "configs/config_srgnn.yaml" \
  --seed 2
```

This produces a checkpoint in `./saved/` and evaluation metrics in the logs.

### Training in the spam/poisoning scenario

Add `--spam` (and optionally tune `--n_target_items`). When tuning `--n_target_items` make sure you created the corresponding forget sets (i.e. spam data):

```bash
python run_recbole.py \
  --model SRGNN \
  --dataset amazon_reviews_books \
  --task_type SBR \
  --config_files "configs/config_srgnn.yaml" \
  --seed 2 \
  --spam \
  --n_target_items 10
```

## Retraining baseline (train from scratch on retain data)

The retraining baseline trains from scratch **after removing the forget set** from the training data. In this codebase, this is triggered by providing the same forget-set specification to `run_recbole.py`.

Example (sensitive-category forgetting; remove all forget requests):

```bash
python run_recbole.py \
  --model SRGNN \
  --dataset amazon_reviews_books \
  --task_type SBR \
  --config_files "configs/config_srgnn.yaml" \
  --seed 2 \
  --unlearning_fraction 0.0001 \
  --unlearning_sample_selection_method sensitive_category_health \
  --sensitive_category health \
  --retrain_checkpoint_idx_to_match 3
```

The retrained baseline checkpoint is saved as:

- `./saved/model_<model>_seed_<seed>_dataset_<dataset>_retrained_best.pth`

Checkpoint matching:
- `--retrain_checkpoint_idx_to_match <idx>` is used to match evaluation against unlearning after \(\frac{\texttt{idx}+1}{4}\) of requests.
- The common “remove the full forget set” case is `--retrain_checkpoint_idx_to_match 3`.

Use `--spam` as well when retraining in the spam scenario.

## Unlearning a model

To unlearn from an already trained checkpoint, run `unlearn.py` with:
- the forget-set specification (`--unlearning_fraction`, `--unlearning_sample_selection_method`, and optionally `--sensitive_category`)
- the unlearning method (`--unlearning_algorithm`)
- optional method-specific hyperparameters

Example (SCIF on sensitive-category unlearning, SBR):

```bash
python unlearn.py \
  --model SRGNN \
  --dataset amazon_reviews_books \
  --task_type SBR \
  --config_files "configs/config_srgnn.yaml" \
  --seed 2 \
  --unlearning_fraction 0.0001 \
  --unlearning_sample_selection_method sensitive_category_health \
  --sensitive_category health \
  --unlearning_algorithm scif \
  --max_norm 10.0
```

Supported unlearning algorithms (CLI choices):
- `scif`, `kookmin`, `fanchuan`, `gif`, `ceu`, `idea`, `seif`

Method-specific examples:

```bash
# Kookmin: tune init rate
python unlearn.py ... --unlearning_algorithm kookmin --kookmin_init_rate 0.0001
```

### Intermediate checkpoints

Unlearning is evaluated at **four intermediate checkpoints** (approximately 1/4, 2/4, 3/4, 4/4 of requests). Checkpoints are saved under `./saved/` with filenames derived from the base model plus suffixes like:

- `_unlearn_epoch_<request_idx>_retrain_checkpoint_idx_to_match_<0..3>.pth`

If you change batching via `--unlearning_batchsize`, filenames may additionally include a `_bs<batchsize>` suffix.

### Spam/poisoning unlearning

Add `--spam` and `--n_target_items` consistently between training and unlearning runs.

## Evaluation-only modes (no (re-)training / no unlearning)

### Evaluate a trained model checkpoint

Use `run_recbole.py --eval_only` to skip training and only evaluate a saved checkpoint:

```bash
python run_recbole.py \
  --model SRGNN \
  --dataset amazon_reviews_books \
  --task_type SBR \
  --config_files "configs/config_srgnn.yaml" \
  --seed 2 \
  --eval_only
```

Optional:
- `--sensitive_eval_only` (only sensitive evaluation, no utility evaluation; requires `--eval_only`)
- `--hf_model_path hf://...` (load checkpoint from HuggingFace in eval-only mode; see `run_recbole.py`)

### Evaluate existing unlearned checkpoints

Use `unlearn.py --eval_only` to evaluate already-produced unlearned checkpoints:

```bash
python unlearn.py \
  --model SRGNN \
  --dataset amazon_reviews_books \
  --task_type SBR \
  --config_files "configs/config_srgnn.yaml" \
  --seed 2 \
  --unlearning_fraction 0.0001 \
  --unlearning_sample_selection_method sensitive_category_health \
  --sensitive_category health \
  --unlearning_algorithm scif \
  --eval_only
```

## Adding/extending the benchmark

### Adding a new dataset

Datasets and derived artifacts (test subsets, sensitive-item lists, forget sets) live under `dataset/` and are documented per dataset (see `dataset/*/README.md`).

To add a dataset:

- Create `dataset/<name>/`
- Provide data in the expected format:
  - **CF/SBR**: RecBole `.inter` with typed headers (e.g., `user_id:token`, `item_id:token`, `timestamp:float`)
  - **NBR**: merged basket JSON (e.g., `<name>_merged.json`) consumed by the next-basket dataset loader
- Add a dataset-specific YAML config declaring schema and loading columns (e.g., `USER_ID_FIELD`, `ITEM_ID_FIELD`, `TIME_FIELD`, `load_col`)

If the dataset supports sensitive-category unlearning:
- Add sensitive-item lists under `dataset/<name>/` (e.g., `sensitive_asins_<category>.txt`)
- Add (or reuse) scripts that generate forget sets (this repository contains examples under `dataset/`)

### Adding a new model

Implement the model as a standard RecBole model class under:

- `recbole/model/<type>_recommender/<model_name>.py` (module name lowercase; class name `ModelName`)

Optionally:
- Add default hyperparameters in `recbole/properties/model/ModelName.yaml`
- Add an experiment config (e.g., `config_<model>.yaml`) for benchmark settings

The model becomes runnable via `--model ModelName` in both `run_recbole.py` and `unlearn.py`.

### Adding a new unlearning algorithm

Expose the method by extending:

- the CLI in `unlearn.py` (add a new `--unlearning_algorithm` choice + method-specific hyperparameters)
- the unlearning pipeline in `recbole/quick_start/quick_start.py` (invoked via `recbole.quick_start.unlearn_recbole`)

The method should take a trained checkpoint and a forget request as input and produce updated checkpoints for evaluation at fixed intermediate checkpoints.

### Adding a new scenario

A scenario specifies what is forgotten (forget set) and how it is evaluated. In this repository, scenarios are selected via flags/config such as:

- `--task_type {CF,SBR,NBR}`
- `--unlearning_fraction`
- `--unlearning_sample_selection_method` (e.g., `sensitive_category_<cat>`)
- `--spam` (poisoning/spam removal scenario; requires corresponding dataset artifacts under `dataset/`)

To add a scenario, implement a generator that produces the required forget-set artifacts for a dataset and add the corresponding loading/selection logic so it can be invoked via the same runner interface.

## Using ERASE locally with private data

You can run the benchmark on private datasets entirely locally:

- Clone the repository
- Add your datasets (and optionally scenarios/models/unlearning algorithms)
- Run `run_recbole.py` and `unlearn.py` as above

No external APIs are required once dependencies are installed.

