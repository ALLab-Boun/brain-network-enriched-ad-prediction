import subprocess
import threading
import time
import sys
import json
import argparse
import itertools
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
        description="Run a tuning grid using arguments loaded from a JSON config."
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
        default=r"C:\Users\efeka\Documents\thesis_colab_match\Scripts\python.exe",
        help="Python executable used to launch the training script."
    )

    parser.add_argument(
        "--script_path",
        type=str,
        default="exp4_temporal.py",
        help="Path to the training script."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Single seed used for every tuning configuration."
    )

    parser.add_argument(
        "--max_parallel",
        type=int,
        default=5,
        help="Maximum number of experiments running in parallel."
    )

    parser.add_argument(
        "--stagger_seconds",
        type=float,
        default=5.0,
        help="Delay between launching each run."
    )

    parser.add_argument(
        "--include_none",
        action="store_true",
        help="If set, pass None/null values as the string 'None'. Usually leave this off."
    )

    # --------------------------------------------------------
    # Optional base-config overrides
    # --------------------------------------------------------

    parser.add_argument("--dataset", type=str, default=None, help="Override dataset from the JSON config.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override dataset_path from the JSON config.")
    parser.add_argument("--cross_val_pkl", type=str, default=None, help="Override cross_val_pkl from the JSON config.")
    parser.add_argument("--task", type=str, default=None, help="Override task from the JSON config.")

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Override run_dir from the JSON config."
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

    parser.add_argument(
        "--pretrained_encoder_path",
        type=str,
        default=None,
        help="Path to pretrained encoder checkpoint."
    )

    parser.add_argument(
        "--es_min_delta",
        type=float,
        default=None,
        help="Override es_min_delta from the JSON config."
    )

    parser.add_argument(
        "--es_patience",
        type=float,
        default=None,
        help="Override es_patience from the JSON config."
    )

    # --------------------------------------------------------
    # Single-value overrides
    # --------------------------------------------------------
    # These force one value. They are separate from the tuning grid.
    # If tune_* arguments are provided, grid values override these.

    parser.add_argument(
        "--temporal_type",
        type=str,
        default=None,
        help="Override temporal_type from the JSON config."
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override dropout from the JSON config."
    )

    parser.add_argument(
        "--temporal_hidden_dim",
        type=int,
        default=None,
        help="Override temporal_hidden_dim from the JSON config."
    )

    # --------------------------------------------------------
    # Tuning grid arguments
    # --------------------------------------------------------

    parser.add_argument(
        "--tune_temporal_type",
        type=str,
        nargs="+",
        default=None,
        help="Grid values for temporal_type, e.g. --tune_temporal_type rnn gru lstm"
    )

    parser.add_argument(
        "--tune_dropout",
        type=float,
        nargs="+",
        default=None,
        help="Grid values for dropout, e.g. --tune_dropout 0.1 0.2 0.3 0.4"
    )

    parser.add_argument(
        "--tune_pre_recurrent_dropout",
        type=float,
        nargs="+",
        default=None,
        help="Grid values for pre_recurrent_dropout, e.g. --tune_pre_recurrent_dropout 0.1 0.2 0.3 0.4"
    )

    parser.add_argument(
        "--tune_recurrent_dropout",
        type=float,
        nargs="+",
        default=None,
        help="Grid values for recurrent_dropout, e.g. --tune_recurrent_dropout 0.1 0.2 0.3 0.4"
    )

    parser.add_argument(
        "--tune_lr",
        type=float,
        nargs="+",
        default=None,
        help="Grid values for lr, e.g. --tune_lr 0.001 0.0005 0.0001"
    )

    parser.add_argument(
        "--tune_temporal_hidden_dim",
        type=int,
        nargs="+",
        default=None,
        help="Grid values for temporal_hidden_dim, e.g. --tune_temporal_hidden_dim 64 128 256"
    )

    early_stopping_group = parser.add_mutually_exclusive_group()

    early_stopping_group.add_argument(
        "--early_stopping",
        dest="early_stopping",
        action="store_true",
        help="Override JSON config and enable early stopping."
    )

    early_stopping_group.add_argument(
        "--no_early_stopping",
        dest="early_stopping",
        action="store_false",
        help="Override JSON config and disable early stopping."
    )

    parser.set_defaults(early_stopping=None)

    return parser.parse_args()


def apply_overrides(config, runner_args):
    """
    Override selected config values only if the corresponding runner argument
    was explicitly provided.
    """

    config = dict(config)

    if runner_args.dataset is not None:
        config["dataset"] = runner_args.dataset

    if runner_args.dataset_path is not None:
        config["dataset_path"] = runner_args.dataset_path

    if runner_args.cross_val_pkl is not None:
        config["cross_val_pkl"] = runner_args.cross_val_pkl

    if runner_args.task is not None:
        config["task"] = runner_args.task

    if runner_args.run_dir is not None:
        config["run_dir"] = runner_args.run_dir

    if runner_args.epochs is not None:
        config["epochs"] = runner_args.epochs

    if runner_args.lr is not None:
        config["lr"] = runner_args.lr

    if runner_args.temporal_type is not None:
        config["temporal_type"] = runner_args.temporal_type

    if runner_args.dropout is not None:
        config["dropout"] = runner_args.dropout

    if runner_args.temporal_hidden_dim is not None:
        config["temporal_hidden_dim"] = runner_args.temporal_hidden_dim

    if runner_args.pretrained_encoder_path is not None:
        config["pretrained_encoder_path"] = runner_args.pretrained_encoder_path

    if runner_args.es_min_delta is not None:
        config["es_min_delta"] = runner_args.es_min_delta

    if runner_args.es_patience is not None:
        config["es_patience"] = runner_args.es_patience

    if runner_args.early_stopping is not None:
        config["early_stopping"] = runner_args.early_stopping

    return config


# ------------------------------------------------------------
# PARAMETER GRID
# ------------------------------------------------------------

def build_param_grid(runner_args):
    """
    Build tuning parameter combinations from tune_* arguments.

    If no tune_* arguments are provided, this returns one empty configuration,
    meaning the base JSON config is run once with the selected seed.
    """

    param_grid = {}

    if runner_args.tune_temporal_type is not None:
        param_grid["temporal_type"] = runner_args.tune_temporal_type

    if runner_args.tune_dropout is not None:
        param_grid["dropout"] = runner_args.tune_dropout
    
    if runner_args.tune_pre_recurrent_dropout is not None:
        param_grid["pre_recurrent_dropout"] = runner_args.tune_pre_recurrent_dropout

    if runner_args.tune_recurrent_dropout is not None:
        param_grid["recurrent_dropout"] = runner_args.tune_recurrent_dropout

    if runner_args.tune_lr is not None:
        param_grid["lr"] = runner_args.tune_lr

    if runner_args.tune_temporal_hidden_dim is not None:
        param_grid["temporal_hidden_dim"] = runner_args.tune_temporal_hidden_dim

    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    grid_configs = [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]

    return grid_configs


# ------------------------------------------------------------
# OUTPUT STREAMING
# ------------------------------------------------------------

def stream_output(run_label, process):
    try:
        for line in process.stdout:
            print(f"[{run_label}] {line}", end="")
    except Exception as e:
        print(f"[{run_label}] Error while reading output: {e}")


def format_command_for_print(cmd):
    return " ".join(f'"{x}"' if " " in x else x for x in cmd)


def make_run_label(index, total, grid_params):
    if not grid_params:
        return f"run_{index}_of_{total}_base_config"

    parts = []
    for key, value in grid_params.items():
        parts.append(f"{key}={value}")

    return f"run_{index}_of_{total}_" + ",".join(parts)


# ------------------------------------------------------------
# PROCESS MANAGEMENT
# ------------------------------------------------------------

def launch_run(
    run_label,
    config,
    runner_args,
):
    cli_args = json_config_to_cli_args(
        config,
        include_none=runner_args.include_none
    )

    cmd = [
        runner_args.python_exe,
        runner_args.script_path,
        *cli_args,
    ]

    print(f"\nLaunching {run_label}")
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
        args=(run_label, process),
        daemon=True
    )
    thread.start()

    return {
        "run_label": run_label,
        "process": process,
        "thread": thread,
        "config": config,
        "cmd": cmd,
    }


def finalize_finished_runs(active_runs, finished_runs):
    """
    Move completed runs from active_runs to finished_runs.
    Returns the remaining active runs.
    """

    still_active = []

    for run in active_runs:
        process = run["process"]
        return_code = process.poll()

        if return_code is None:
            still_active.append(run)
            continue

        run["return_code"] = return_code
        finished_runs.append(run)

        status = "OK" if return_code == 0 else "FAILED"
        print(f"\n[{status}] {run['run_label']} finished with exit code {return_code}")

    return still_active


def wait_until_slot_available(active_runs, finished_runs, max_parallel):
    while len(active_runs) >= max_parallel:
        time.sleep(1)
        active_runs[:] = finalize_finished_runs(active_runs, finished_runs)


def wait_for_all(active_runs, finished_runs):
    while active_runs:
        time.sleep(1)
        active_runs[:] = finalize_finished_runs(active_runs, finished_runs)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    runner_args = parse_runner_args()

    if runner_args.max_parallel < 1:
        raise ValueError("--max_parallel must be at least 1.")

    print("Reading JSON config...")
    base_config = load_json_config(runner_args.json_config)

    print("Building tuning grid...")
    grid_configs = build_param_grid(runner_args)

    print(f"Total tuning runs: {len(grid_configs)}")
    print(f"Single seed used for all runs: {runner_args.seed}")
    print(f"Maximum parallel runs: {runner_args.max_parallel}")
    print(f"Launch stagger: {runner_args.stagger_seconds} seconds")

    active_runs = []
    finished_runs = []

    total_runs = len(grid_configs)

    for i, grid_params in enumerate(grid_configs, start=1):
        wait_until_slot_available(
            active_runs=active_runs,
            finished_runs=finished_runs,
            max_parallel=runner_args.max_parallel
        )

        config = apply_overrides(base_config, runner_args)

        # Grid params override JSON config and single-value CLI overrides.
        config.update(grid_params)

        # Every grid instance uses the same single seed.
        config["seed"] = runner_args.seed

        run_label = make_run_label(i, total_runs, grid_params)

        run = launch_run(
            run_label=run_label,
            config=config,
            runner_args=runner_args,
        )

        active_runs.append(run)

        if i < total_runs:
            time.sleep(runner_args.stagger_seconds)

    print("\nAll runs have been submitted. Waiting for remaining runs to finish...")

    wait_for_all(active_runs, finished_runs)

    for run in finished_runs:
        run["thread"].join()

    print("\n============================================================")
    print("Tuning completed.")
    print(f"Total runs:      {len(finished_runs)}")

    failed_runs = [
        run for run in finished_runs
        if run.get("return_code", 1) != 0
    ]

    print(f"Successful runs: {len(finished_runs) - len(failed_runs)}")
    print(f"Failed runs:     {len(failed_runs)}")
    print("============================================================")

    if failed_runs:
        print("\nFailed runs:")
        for run in failed_runs:
            print(f"  - {run['run_label']} | exit code {run['return_code']}")
        sys.exit(1)

    print("\nAll tuning runs finished successfully.")

if __name__ == "__main__":
    main()