## Segger Command Line Interface



This section will simulate typing the `segger --help` command output.

```console
// Run the Segger CLI helper
$ segger --help

Usage: segger [OPTIONS] COMMAND [ARGS]...

╭─ Commands ─────────────────────────────────────────────────────╮
│ create_dataset  Create a dataset for spatial transcriptomics     │
│ train           Train the model using the prepared dataset       │
│ predict         Run predictions using a trained model            │
╰────────────────────────────────────────────────────────────────╯
```


### 1. Creating a Dataset

The `create_dataset` command helps you build a dataset for spatial transcriptomics. Here’s a breakdown of the options available:

```console
// Example: Create a dataset for spatial transcriptomics
$ segger create_dataset \
    --dataset_type xenium \
    --dataset_dir data_raw/xenium \
    --sample_tag Xenium_FFPE_Human_Breast_Cancer_Rep1 \
    --transcripts_file transcripts.parquet \
    --boundaries_file nucleus_boundaries.parquet \
    --data_dir data_tidy/pyg_datasets \
    --x_size 200 \
    --y_size 200 \
    --d_x 150 \
    --d_y 150 \
    --margin_x 20 \
    --margin_y 20 \
    --r_tx 5 \
    --val_prob 0.1 \
    --test_prob 0.2 \
    --compute_labels true \
    --sampling_rate 1 \
    --workers 1 \
    --method kd_tree \
    --gpu false
```

#### Parameters

| Parameter          | Description                                                                            | Default Value |
|--------------------|----------------------------------------------------------------------------------------|---------------|
| `dataset_type`      | Specifies the type of dataset (e.g., `xenium`, `merfish`).                             | `xenium`      |
| `dataset_dir`       | Path to the directory where raw data is stored.                                        | None          |
| `sample_tag`        | Tag to identify the dataset, useful for version control.                               | None          |
| `transcripts_file`  | File path to the transcript data in Parquet format.                                    | None          |
| `boundaries_file`   | File path to the nucleus boundaries data in Parquet format.                            | None          |
| `data_dir`          | Directory to store processed datasets (used during model training).                    | None          |
| `x_size`, `y_size`  | Size of the bounding box in the x and y directions.                                    | `200`         |
| `d_x`, `d_y`        | Step size in the x and y directions for overlapping patches.                           | `150`         |
| `margin_x`, `margin_y` | Additional margins added to the bounding box.                                       | `20`          |
| `r_tx`              | Radius for transcript detection.                                                       | `5`           |
| `val_prob`          | Proportion of the dataset used for validation.                                         | `0.1`         |
| `test_prob`         | Proportion of the dataset used for testing.                                            | `0.2`         |
| `compute_labels`    | Flag to enable or disable the computation of labels.                                   | `true`        |
| `sampling_rate`     | Proportion of the dataset to sample (useful for large datasets).                       | `1` (no sampling) |
| `workers`           | Number of CPU cores to use for parallel processing.                                    | `1`           |
| `method`            | Method for creating the graph (e.g., `kd_tree`).                                       | `kd_tree`     |
| `gpu`               | Whether to use a GPU for processing (`true` or `false`).                               | `false`       |

!!! note "Customizing Your Dataset"
    - **dataset_type**: Defines the type of spatial transcriptomics data.
    - **x_size, y_size**: Bounding box dimensions are important for memory efficiency.
    - **val_prob, test_prob**: Adjust these probabilities based on the need for model validation and testing.

!!! tip "Faster Dataset Creation"
    You can reduce the **sampling_rate** to process only a subset of your dataset, which is useful for large datasets.

---

### 2. Training a Model

The `train` command initializes and trains a model using the dataset created. Here are the key parameters:

```console
// Example: Train the model using SLURM
$ segger train slurm \
    --data_dir data_tidy/pyg_datasets \
    --batch_size_train 32 \
    --batch_size_val 16 \
    --init_emb 128 \
    --hidden_channels 256 \
    --out_channels 3 \
    --heads 8 \
    --aggr mean \
    --accelerator gpu \
    --strategy ddp \
    --precision 16 \
    --devices 4 \
    --epochs 100 \
    --model_dir /path/to/save/model/checkpoints
```

#### Parameters

| Parameter          | Description                                                                            | Default Value |
|--------------------|----------------------------------------------------------------------------------------|---------------|
| `data_dir`         | Directory containing the dataset to be used for training.                               | None          |
| `batch_size_train` | Number of samples to process per training batch.                                        | `32`          |
| `batch_size_val`   | Number of samples to process per validation batch.                                      | `16`          |
| `init_emb`         | Size of the initial embedding for the input data.                                       | `128`         |
| `hidden_channels`  | Number of hidden units in each layer of the neural network.                             | `256`         |
| `out_channels`     | Number of output channels.                                                              | `3`           |
| `heads`            | Number of attention heads used in graph attention layers.                               | `8`           |
| `aggr`             | Aggregation method for attention layers (e.g., `mean`, `sum`).                          | `mean`        |
| `accelerator`      | Device used for training (e.g., `gpu` or `cpu`).                                        | `gpu`         |
| `strategy`         | Strategy for distributed training (e.g., `ddp` for Distributed Data Parallel).          | `ddp`         |
| `precision`        | Floating-point precision for training (e.g., `16` for FP16).                            | `16`          |
| `devices`          | Number of devices (GPUs or CPUs) to use during training.                                | `4`           |
| `epochs`           | Number of training epochs.                                                              | `100`         |
| `model_dir`        | Directory to save model checkpoints.                                                    | None          |

!!! tip "Adjusting for Your Hardware"
    - **batch_size_train**: For larger datasets, you might need to decrease this value based on your GPU memory.
    - **epochs**: Increasing the number of epochs can lead to better model performance but will take longer to train.

!!! warning "Ensure Correct GPU Setup"
    Before using the `--accelerator gpu` flag, make sure your system supports GPU computation and that CUDA is properly installed.

---

### 3. Making Predictions

After training the model, use the `predict` command to make predictions on new data.

```console
// Example: Make predictions using a trained model
$ segger predict \
    --dataset_path /path/to/new/dataset \
    --checkpoint_path /path/to/saved/checkpoint \
    --output_path /path/to/save/predictions.csv \
    --batch_size 16 \
    --workers 4 \
    --score_cut 0.5 \
    --use_cc true
```

#### Parameters

| Parameter          | Description                                                                            | Default Value |
|--------------------|----------------------------------------------------------------------------------------|---------------|
| `dataset_path`      | Path to the dataset for which predictions will be made.                                | None          |
| `checkpoint_path`   | Path to the saved model checkpoint from training.                                      | None          |
| `output_path`       | File where the predictions will be saved.                                              | None          |
| `batch_size`        | Number of samples processed simultaneously during prediction.                          | `16`          |
| `workers`           | Number of CPU cores used for parallel processing during prediction.                    | `4`           |
| `score_cut`         | Cutoff threshold for confidence scores in predictions.                                 | `0.5`         |
| `use_cc`            | Enable connected component analysis to refine predictions.                             | `true`        |

!!! tip "Improve Prediction Efficiency"
    - **batch_size**: Adjust this based on the size of the dataset and available GPU memory.
    - **use_cc**: Enabling connected component analysis can improve the accuracy of transcript assignments

.

---

### 4. Utility Commands and Reports

Segger includes utility commands for checking dataset and model setup as well as generating reports.

```console
// Example: Check dataset and model setup
$ segger check \
    --dataset_dir data_raw/xenium \
    --model_dir /path/to/model/checkpoints

// Example: Generate a report
$ segger report \
    --dataset_path /path/to/dataset \
    --output_path /path/to/report.html
```

#### Parameters for `check`

| Parameter        | Description                                          |
|------------------|------------------------------------------------------|
| `dataset_dir`     | Path to the raw dataset.                            |
| `model_dir`      | Path to the directory where model checkpoints are saved. |

#### Parameters for `report`

| Parameter        | Description                                          |
|------------------|------------------------------------------------------|
| `dataset_path`   | Path to the dataset for which the report will be generated. |
| `output_path`    | Path where the HTML report will be saved.            |

!!! info "Utility Commands"
    - Use `check` to verify that your dataset and model are correctly set up.
    - The `report` command provides a detailed HTML output of your model's performance.
