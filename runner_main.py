import subprocess
import threading
import time
import sys
import json
import argparse
from pathlib import Path


# ------------------------------------------------------------
# JSON -> CLI ARG LIST
# ------------------------------------------------------------

def json_config_to_cli_args(config, include_none=False):
    """
    Convert a JSON-style config dict into a subprocess-friendly CLI argument list.

    Rules:
      - bool True  -> --flag
      - bool False -> omitted
      - None/null  -> omitted by default
      - list/tuple -> --arg item1 item2 item3
      - other      -> --arg value
    """

    cli_args = []

    for key, value in config.items():
        arg_name = f"--{key}"

        if isinstance(value, bool):
            if value:
                cli_args.append(arg_name)
            continue

        if value is None:
            if include_none:
                cli_args.extend([arg_name, "None"])
            continue

        if isinstance(value, (list, tuple)):
            cli_args.append(arg_name)
            cli_args.extend(str(v) for v in value)
            continue

        cli_args.extend([arg_name, str(value)])

    return cli_args


def load_json_config(json_path):
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON config file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("JSON config must contain a single dictionary/object at the top level.")

    return config


# ------------------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------------------

def parse_runner_args():
    parser = argparse.ArgumentParser(
        description="Run exp4_main_deterministic.py using arguments loaded from a JSON config."
    )

    parser.add_argument(
        "--json_config",
        type=str,
        required=True,
        help="Path to the JSON config file."
    )

    parser.add_argument(
        "--python_exe",
        type=str,
        # default=r"C:\dev\GitHub\graph-based-dementia-prediction\mind_env\Scripts\python.exe",
        default=r"C:\Users\efeka\Documents\thesis_colab_match\Scripts\python.exe",
        help="Python executable used to launch the training script."
    )

    parser.add_argument(
        "--script_path",
        type=str,
        default="exp4_main_deterministic.py",
        help="Path to the training script."
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7],
        help="One or more seeds to run."
    )

    parser.add_argument(
        "--stagger_seconds",
        type=float,
        default=5.0,
        help="Delay between launching each seed."
    )

    # Optional overrides for selected JSON fields
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override dataset_path from the JSON config."
    )

    parser.add_argument(
        "--cross_val_pkl",
        type=str,
        default=None,
        help="Override cross_val_pkl from the JSON config."
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Override run_dir from the JSON config."
    )

    parser.add_argument(
        "--include_none",
        action="store_true",
        help="If set, pass None/null values as the string 'None'. Usually leave this off."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from the JSON config."
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override lr from the JSON config."
    )

    # For temporal experiments, allow overriding temporal_type from the command line
    parser.add_argument("--temporal_type", type=str, default=None, help="Override temporal_type from the JSON config.")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout from the JSON config.") # dropout for temporal classifier
    parser.add_argument("--temporal_hidden_dim", type=int, default=None, help="Override temporal_hidden_dim from the JSON config.")
    parser.add_argument("--pretrained_encoder_path", type=str, default=None, help="Path to pretrained encoder checkpoint (optional)")
    parser.add_argument("--es_delta", type=float, default=None, help="Override early stopping delta from the JSON config.")


    return parser.parse_args()


def apply_overrides(config, runner_args):
    """
    Override selected config values only if the corresponding runner argument
    was explicitly provided.
    """

    config = dict(config)

    if runner_args.dataset_path is not None:
        config["dataset_path"] = runner_args.dataset_path

    if runner_args.cross_val_pkl is not None:
        config["cross_val_pkl"] = runner_args.cross_val_pkl

    if runner_args.run_dir is not None:
        config["run_dir"] = runner_args.run_dir

    if runner_args.epochs is not None:
        config["epochs"] = runner_args.epochs

    if runner_args.temporal_type is not None:
        config["temporal_type"] = runner_args.temporal_type
    if runner_args.dropout is not None:
        config["dropout"] = runner_args.dropout
    if runner_args.temporal_hidden_dim is not None:
        config["temporal_hidden_dim"] = runner_args.temporal_hidden_dim
    if runner_args.pretrained_encoder_path is not None:
        config["pretrained_encoder_path"] = runner_args.pretrained_encoder_path

    if runner_args.lr is not None:
        config["lr"] = runner_args.lr

    if runner_args.es_delta is not None:
        config["es_min_delta"] = runner_args.es_delta

    return config


# ------------------------------------------------------------
# OUTPUT STREAMING
# ------------------------------------------------------------

def stream_output(seed, process):
    try:
        for line in process.stdout:
            print(f"[seed {seed}] {line}", end="")
    except Exception as e:
        print(f"[seed {seed}] Error while reading output: {e}")


def format_command_for_print(cmd):
    return " ".join(f'"{x}"' if " " in x else x for x in cmd)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    runner_args = parse_runner_args()

    print("Reading JSON config...")
    base_config = load_json_config(runner_args.json_config)

    print("Starting experiments with different seeds in parallel...")

    processes = []
    threads = []

    for i, seed in enumerate(runner_args.seeds):
        config = apply_overrides(base_config, runner_args)

        # Always override seed for this specific run
        config["seed"] = seed

        cli_args = json_config_to_cli_args(
            config,
            include_none=runner_args.include_none
        )

        cmd = [
            runner_args.python_exe,
            runner_args.script_path,
            *cli_args,
        ]

        print(f"\nLaunching seed {seed}...")
        print("Command:")
        print(format_command_for_print(cmd))

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

        if i < len(runner_args.seeds) - 1:
            time.sleep(runner_args.stagger_seconds)

    exit_codes = {}

    for seed, process in processes:
        exit_codes[seed] = process.wait()

    for thread in threads:
        thread.join()

    print("\nAll runs completed.\n")

    for seed in runner_args.seeds:
        print(f"Seed {seed} finished with exit code {exit_codes[seed]}")

    failed = [seed for seed, code in exit_codes.items() if code != 0]

    if failed:
        print(f"\nThese seeds failed: {failed}")
        sys.exit(1)

    print("\nAll seeds finished successfully.")

    # run aggregate.py to combine results
    print("\nAggregating results with aggregate.py...")
    aggregate_cmd = [
        runner_args.python_exe,
        "utils/aggregate.py",
        "--root_folder", runner_args.run_dir if runner_args.run_dir else "."
    ]

    subprocess.run(aggregate_cmd)
    print("Aggregation complete.")

if __name__ == "__main__":
    main()