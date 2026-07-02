# Brain Network-Enriched Graph Machine Learning for Alzheimer's Disease Prediction

This repository provides the source code for our work, **"Brain Network-Enriched Graph Machine Learning for Alzheimer's Disease Prediction."**

An overview of the proposed deep learning framework is shown in the figure below.

<img width="1618" height="1058" alt="framework" src="https://github.com/user-attachments/assets/c10c535e-6f0e-42f0-a71d-5b3edd713ced" />

## Running the code

Install the Python dependencies used by PyTorch, PyTorch Geometric, scikit-learn, pandas, NumPy, and matplotlib before running any script.
The main training entry point is `main.py`, which accepts command-line arguments for the dataset path, cross-validation split file, task, model branches, and training hyperparameters.
For a direct run, call `python main.py --dataset_path <path-to-pt-file> --cross_val_pkl <path-to-splits-pkl> --dataset adni --task diagnosis` and add any branch or optimization flags you want to test.
The script saves fold-level metrics, prediction tables, model weights, and training curves in a timestamped results directory.
To run multiple seeds from a JSON configuration, use `runner_main.py` and pass `--json_config <config.json>`.
`runner_main.py` forwards the JSON values to the training script, allows optional CLI overrides such as `--dataset`, `--run_dir`, `--epochs`, and `--lr`, and launches each seed in parallel.
After all seeds finish, the runner calls `utils/aggregate.py` to combine the results under the chosen output folder.
If you are starting from one of the provided configs in `configs/`, copy it, adjust the paths and model flags, and use it as the input to the runner.
