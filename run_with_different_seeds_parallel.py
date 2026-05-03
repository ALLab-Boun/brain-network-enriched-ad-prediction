import subprocess
import threading
import time
import sys

seeds = [7] #[42, 123, 2021, 7, 99]

import json
import shlex


def json_config_to_cli_string(config, script_path="train.py", include_none=False):
    """
    Convert a JSON-style config dict into a command-line string.

    Rules:
      - bool True  -> --flag
      - bool False -> omitted
      - None/null  -> omitted by default
      - list/tuple -> --arg item1 item2 item3
      - other      -> --arg value

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from JSON.
    script_path : str
        Python script path to call.
    include_none : bool
        If True, converts None values to the string "None".
        Usually keep this False for argparse.

    Returns
    -------
    str
        Full command string, e.g.:
        python train.py --epochs 1 --lr 5e-05 --include_transformer
    """

    parts = ["python", script_path]

    for key, value in config.items():
        arg_name = f"--{key}"

        # Boolean flags for argparse action="store_true"
        if isinstance(value, bool):
            if value:
                parts.append(arg_name)
            continue

        # Skip null values unless explicitly requested
        if value is None:
            if include_none:
                parts.extend([arg_name, "None"])
            continue

        # Expand list values for nargs="+"
        if isinstance(value, (list, tuple)):
            parts.append(arg_name)
            parts.extend(str(v) for v in value)
            continue

        # Normal scalar values
        parts.extend([arg_name, str(value)])

    # Quote safely for shell usage
    return " ".join(shlex.quote(p) for p in parts)


print("Starting experiments with different seeds in parallel...")

base_command = [
    # "python",
    # r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
    r"C:\Users\efeka\Documents\thesis_colab_match\Scripts\python.exe",
    "exp4_main_deterministic.py",

    # ADNI
    # "--dataset", "adni",
    # "--dataset_path", "./data/adni/CT_Vol_graphs_complete_features_filtered_negative.pt",
    # "--cross_val_pkl", "./data/adni/splits/reporting_cv_splits.pkl",

    # OASIS 
    "--dataset", "oasis", 
    "--dataset_path", "./data/oasis3/CTVOL_all_graphs_relabeled_6m_filtered_negative.pt",
    "--cross_val_pkl", "./data/oasis3/splits/oasis_cv_1foldval.pkl",

    # Feature settings
    "--excluded_node_features", "std_min_max",
    "--node_feature_set", "ct_vol_sa_mc_sd",
    # "--add_adj_row_as_node_feature", # for adj models
    # "--separate_adj_features_instead_of_concat", # for adj models

    "--epochs", "2",
    "--lr", "5e-5",
    "--batch_size", "32",
    "--weight_decay", "0.05",
    

    "--dropout", "0.0",


    # ----------------------------------------
    # ------ Cognitive Model Configs ---------
    # ----------------------------------------
    "--include_cog_mlp",
    "--cog_hidden_dim", "128",
    "--cog_mlp_dropout", "0.0",
    "--cog_mlp_width_mode", "shrink",
    "--cog_mlp_num_layers", "3",
    # "--cog_mlp_use_residual_to_last", "false",


    # ----------------------------------------
    # ----------------------------------------
    # ----------------------------------------

    # ****************************************

    # ----------------------------------------
    # ------ Early Stopping Configs ---------
    # ----------------------------------------
    "--early_stopping",
    "--es_monitor", "es_f1_weighted",
    "--es_mode", "max",
    "--es_patience", "20",
    "--es_min_delta", "0.0025",

]



def stream_output(seed, process):
    try:
        for line in process.stdout:
            print(f"[seed {seed}] {line}", end="")
    except Exception as e:
        print(f"[seed {seed}] Error while reading output: {e}")


processes = []
threads = []

for i, seed in enumerate(seeds):
    cmd = base_command + ["--seed", str(seed)]
    print(f"Launching seed {seed}...")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    thread = threading.Thread(
        target=stream_output,
        args=(seed, process),
        daemon=True
    )
    thread.start()

    processes.append((seed, process))
    threads.append(thread)

    # 5-second stagger after each launch except the last one
    if i < len(seeds) - 1:
        time.sleep(5)

exit_codes = {}
for seed, process in processes:
    exit_codes[seed] = process.wait()

for thread in threads:
    thread.join()

print("\nAll runs completed.\n")
for seed in seeds:
    print(f"Seed {seed} finished with exit code {exit_codes[seed]}")

failed = [seed for seed, code in exit_codes.items() if code != 0]
if failed:
    print(f"\nThese seeds failed: {failed}")
    sys.exit(1)
else:
    print("\nAll seeds finished successfully.")